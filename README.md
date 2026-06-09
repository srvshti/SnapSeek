# SnapSeek Ops Edition

SnapSeek is an AI-powered local image search tool built with Python, PyQt5, and OpenAI CLIP. This fork keeps the original desktop search idea and adds production-style engineering improvements: cached indexing, SQLite persistence, CLI automation, structured logs, cross-platform file opening, and CI-friendly tests.

## Why this version

This version is shaped for a System Development Engineer portfolio story:

- Automates a manual operational workflow: finding specific assets across large folders.
- Uses Linux-friendly Python tooling, CLI commands, and logging.
- Persists indexed metadata in SQLite so unchanged files are reused instead of reprocessed.
- Separates GUI, indexing, and search logic for easier testing and maintenance.
- Includes tests that mock the model layer, so CI can validate core behavior without downloading CLIP.

## Features

- Text-to-image search using CLIP embeddings.
- Recursive folder indexing for PNG, JPG, JPEG, BMP, GIF, and WEBP files.
- SQLite cache stored as `.snapseek_index.sqlite3` inside the indexed folder by default.
- Lazy CLIP model loading to keep imports, tests, and CLI help fast.
- PyQt5 desktop GUI.
- CLI for scripted indexing/search workflows.
- Structured index run history for basic observability.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the GUI

```bash
python gui.py
```

## Use the CLI

Index a folder:

```bash
python snapseek_cli.py index /path/to/images
```

Search an existing cache:

```bash
python snapseek_cli.py search /path/to/images "person wearing a red jacket"
```

Index automatically when the cache is empty:

```bash
python snapseek_cli.py search /path/to/images "warehouse dashboard screenshot" --index-if-empty
```

Tune result filtering:

```bash
python snapseek_cli.py search /path/to/images "database architecture diagram" --top-k 5 --threshold 0.25
```

## Run tests

```bash
python -m pytest
```

## Project structure

```text
SnapSeek
├── gui.py              # PyQt5 desktop application
├── main.py             # Indexing, SQLite cache, CLIP search logic
├── snapseek_cli.py     # Automation-friendly command-line interface
├── tests/              # CI-friendly unit tests
├── requirements.txt
├── logo.png
├── arrow.png
├── open-folder.png
└── search_icon.png
```

## Resume bullets

- Re-engineered an AI-powered local image search tool with Python, CLIP, SQLite, and PyQt5, adding persistent embedding cache reuse to avoid repeated indexing of unchanged files.
- Built a CLI automation layer for indexing and searching image folders, enabling scripted operational workflows alongside the desktop UI.
- Added structured logging and index run history to improve diagnosability, supportability, and production-style observability.
- Added unit tests around image discovery, SQLite cache reuse, and similarity filtering using mocked model calls for CI reliability.

## Credits

Original project: [avgvcoding/SnapSeek](https://github.com/avgvcoding/SnapSeek)
