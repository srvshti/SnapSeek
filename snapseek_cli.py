import argparse
from pathlib import Path

from main import configure_logging, index_images, load_cached_embeddings, search_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SnapSeek image indexing and search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index images in a folder")
    index_parser.add_argument("folder", help="Folder containing images")
    index_parser.add_argument("--cache", help="Optional SQLite cache path")
    index_parser.add_argument("--force", action="store_true", help="Re-index every image")

    search_parser = subparsers.add_parser("search", help="Search an indexed folder")
    search_parser.add_argument("folder", help="Folder containing images")
    search_parser.add_argument("query", help="Text prompt to search for")
    search_parser.add_argument("--cache", help="Optional SQLite cache path")
    search_parser.add_argument("--top-k", type=int, default=10, help="Maximum results to return")
    search_parser.add_argument("--threshold", type=float, default=0.22, help="Minimum similarity score")
    search_parser.add_argument(
        "--index-if-empty",
        action="store_true",
        help="Build the index if no valid cached embeddings are found",
    )

    return parser


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()

    if args.command == "index":
        embeddings = index_images(args.folder, cache_path=args.cache, force_reindex=args.force)
        print(f"Indexed {len(embeddings)} images in {Path(args.folder).resolve()}")
        return 0

    if args.command == "search":
        embeddings = load_cached_embeddings(args.folder, cache_path=args.cache)
        if not embeddings and args.index_if_empty:
            embeddings = index_images(args.folder, cache_path=args.cache)
        results = search_images(args.query, embeddings, top_k=args.top_k, threshold=args.threshold)
        for path, score in results:
            print(f"{score:.4f}\t{path}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
