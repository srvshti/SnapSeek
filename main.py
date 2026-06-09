import logging
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import clip
except ImportError:  # pragma: no cover - exercised only when dependencies are missing.
    clip = None


LOGGER = logging.getLogger("snapseek")
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
DEFAULT_CACHE_NAME = ".snapseek_index.sqlite3"

_MODEL = None
_PREPROCESS = None
_DEVICE = None


@dataclass(frozen=True)
class ImageRecord:
    path: str
    embedding: np.ndarray
    mtime: float
    size: int


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model():
    """Load CLIP lazily so tests, CLI help, and imports stay fast."""
    global _MODEL, _PREPROCESS, _DEVICE
    if clip is None:
        raise RuntimeError("OpenAI CLIP is not installed. Run: pip install -r requirements.txt")
    if _MODEL is None or _PREPROCESS is None:
        _DEVICE = get_device()
        LOGGER.info("Loading CLIP model on %s", _DEVICE)
        _MODEL, _PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
    return _MODEL, _PREPROCESS, _DEVICE


def discover_images(image_folder: str) -> List[Path]:
    folder = Path(image_folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def _cache_path(image_folder: str, cache_path: Optional[str] = None) -> Path:
    if cache_path:
        return Path(cache_path).expanduser().resolve()
    return Path(image_folder).expanduser().resolve() / DEFAULT_CACHE_NAME


def _connect_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS image_embeddings (
            path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS index_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder TEXT NOT NULL,
            indexed_count INTEGER NOT NULL,
            reused_count INTEGER NOT NULL,
            failed_count INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return connection


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    return pickle.dumps(embedding.astype(np.float32), protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return pickle.loads(blob)


def _load_cached_record(connection: sqlite3.Connection, path: Path) -> Optional[ImageRecord]:
    stat = path.stat()
    row = connection.execute(
        "SELECT mtime, size, embedding FROM image_embeddings WHERE path = ?",
        (str(path),),
    ).fetchone()
    if not row:
        return None
    mtime, size, embedding_blob = row
    if float(mtime) != stat.st_mtime or int(size) != stat.st_size:
        return None
    return ImageRecord(str(path), _deserialize_embedding(embedding_blob), stat.st_mtime, stat.st_size)


def _save_record(connection: sqlite3.Connection, record: ImageRecord) -> None:
    connection.execute(
        """
        INSERT INTO image_embeddings(path, mtime, size, embedding, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            mtime = excluded.mtime,
            size = excluded.size,
            embedding = excluded.embedding,
            updated_at = excluded.updated_at
        """,
        (
            record.path,
            record.mtime,
            record.size,
            _serialize_embedding(record.embedding),
            time.time(),
        ),
    )


def _encode_image(path: Path) -> np.ndarray:
    model, preprocess, device = load_clip_model()
    image = Image.open(path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy()


def index_images(
    image_folder: str,
    cache_path: Optional[str] = None,
    force_reindex: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, np.ndarray]:
    """
    Index images with CLIP embeddings and persist unchanged files in SQLite.

    Returns a path -> embedding dictionary for compatibility with the original GUI.
    """
    start = time.time()
    image_paths = discover_images(image_folder)
    connection = _connect_cache(_cache_path(image_folder, cache_path))
    embeddings: Dict[str, np.ndarray] = {}
    indexed_count = 0
    reused_count = 0
    failed_count = 0

    try:
        for index, path in enumerate(image_paths, start=1):
            if progress_callback:
                progress_callback(index, len(image_paths), str(path))

            try:
                record = None if force_reindex else _load_cached_record(connection, path)
                if record:
                    reused_count += 1
                else:
                    stat = path.stat()
                    record = ImageRecord(
                        path=str(path),
                        embedding=_encode_image(path),
                        mtime=stat.st_mtime,
                        size=stat.st_size,
                    )
                    _save_record(connection, record)
                    indexed_count += 1
                embeddings[record.path] = record.embedding
            except Exception:
                failed_count += 1
                LOGGER.exception("Failed to index image: %s", path)

        connection.execute(
            """
            INSERT INTO index_runs(folder, indexed_count, reused_count, failed_count, duration_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(Path(image_folder).expanduser().resolve()),
                indexed_count,
                reused_count,
                failed_count,
                time.time() - start,
                time.time(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    LOGGER.info(
        "Index complete: %s total, %s new/changed, %s reused, %s failed",
        len(embeddings),
        indexed_count,
        reused_count,
        failed_count,
    )
    return embeddings


def search_images(
    query: str,
    image_embeddings: Dict[str, np.ndarray],
    top_k: int = 10,
    threshold: float = 0.22,
) -> List[Tuple[str, float]]:
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    if not image_embeddings:
        return []

    model, _, device = load_clip_model()
    text_tokens = clip.tokenize(query).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)

    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    text_embedding = text_embedding.cpu().numpy()

    similarities = []
    for path, embedding in image_embeddings.items():
        similarity = float(np.dot(embedding, text_embedding.T).item())
        if similarity >= threshold:
            similarities.append((path, similarity))

    similarities.sort(key=lambda result: result[1], reverse=True)
    return similarities[:top_k]


def load_cached_embeddings(image_folder: str, cache_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    image_paths = discover_images(image_folder)
    connection = _connect_cache(_cache_path(image_folder, cache_path))
    try:
        records = [_load_cached_record(connection, path) for path in image_paths]
        return {record.path: record.embedding for record in records if record is not None}
    finally:
        connection.close()


if __name__ == "__main__":
    configure_logging()
    folder = input("Image folder: ").strip()
    embeddings = index_images(folder)
    query = input("Search query: ").strip()
    for image_path, score in search_images(query, embeddings, top_k=5):
        print(f"{score:.4f} {image_path}")
