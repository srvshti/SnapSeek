import sqlite3
import types
from pathlib import Path

import numpy as np
from PIL import Image

import main


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def to(self, _device):
        return self

    def norm(self, dim=-1, keepdim=True):
        return FakeTensor(np.linalg.norm(self.array, axis=dim, keepdims=keepdim))

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    @property
    def T(self):
        return self.array.T

    def __truediv__(self, other):
        other_array = other.array if isinstance(other, FakeTensor) else other
        return FakeTensor(self.array / other_array)


class FakeModel:
    def encode_text(self, _tokens):
        return FakeTensor([[1.0, 0.0]])


def test_discover_images_recurses_and_filters(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "notes.txt").write_text("skip")
    Image.new("RGB", (1, 1)).save(tmp_path / "a.jpg")
    Image.new("RGB", (1, 1)).save(nested / "b.png")

    paths = main.discover_images(str(tmp_path))

    assert [path.name for path in paths] == ["a.jpg", "b.png"]


def test_index_images_reuses_sqlite_cache(tmp_path, monkeypatch):
    image_path = tmp_path / "a.jpg"
    Image.new("RGB", (1, 1)).save(image_path)
    calls = {"count": 0}

    def fake_encode(path):
        calls["count"] += 1
        assert path == image_path
        return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(main, "_encode_image", fake_encode)

    first = main.index_images(str(tmp_path))
    second = main.index_images(str(tmp_path))

    assert calls["count"] == 1
    assert list(first) == [str(image_path)]
    assert np.allclose(second[str(image_path)], [[1.0, 0.0]])

    with sqlite3.connect(tmp_path / main.DEFAULT_CACHE_NAME) as connection:
        count = connection.execute("SELECT COUNT(*) FROM index_runs").fetchone()[0]
    assert count == 2


def test_search_images_sorts_and_applies_threshold(monkeypatch):
    monkeypatch.setattr(main, "load_clip_model", lambda: (FakeModel(), None, "cpu"))
    monkeypatch.setattr(main, "clip", types.SimpleNamespace(tokenize=lambda _query: FakeTensor([[0.0, 0.0]])))
    embeddings = {
        "strong.jpg": np.array([[0.90, 0.10]], dtype=np.float32),
        "weak.jpg": np.array([[0.10, 0.90]], dtype=np.float32),
    }

    results = main.search_images("red shirt", embeddings, threshold=0.2)

    assert results[0][0] == "strong.jpg"
    assert np.isclose(results[0][1], 0.9)
