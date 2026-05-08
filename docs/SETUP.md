# Setup Guide

## Prerequisites

- Python 3.11 (managed via [pyenv](https://github.com/pyenv/pyenv))
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- FFmpeg (`brew install ffmpeg`)
- Docker (for PostgreSQL and the optional backend)

## Project layout

```
shazam-for-movies/
├── pyproject.toml      # single source of truth for Python deps
├── uv.lock             # locked transitive deps
├── engine/             # indexing + matching library
├── backend/            # FastAPI service exposing the engine
├── mobile/             # Expo app (separate Node project)
└── data/               # FAISS indexes (gitignored)
```

## Install (CLI / engine only)

```bash
# 1. Pyenv venv (if you don't already have one)
pyenv virtualenv 3.11.6 movie_fingerprint
pyenv local movie_fingerprint    # autoselects the venv when you cd here

# 2. Install the project + locked deps with uv
uv sync                          # installs the engine deps from uv.lock

# 3. Start PostgreSQL
docker compose up -d postgres

# 4. Verify
python -c "from engine.db import Database; db = Database(); print('OK'); db.close()"
```

## Install (CLI + backend)

```bash
uv sync --extra backend          # adds FastAPI / uvicorn / multipart
```

## Adding a dependency

```bash
uv add <package>                 # adds to [project].dependencies + updates uv.lock
uv add --optional backend <pkg>  # adds to backend extra
uv lock --upgrade-package <pkg>  # bump a single locked dep
```

## CLI usage

### Index a movie

```bash
python main.py index --file movie.mp4 --title "Movie Name"
```

A 2-hour movie indexes in ~4-5 minutes on CPU.

### Query a clip

```bash
python main.py query --file clip.mp4
python main.py query --file clip.mp4 --screen   # for laptop/monitor captures
```

### Self-test

```bash
python main.py test --file movie.mp4 --title "Movie Name" --duration 10
```

## Backend (FastAPI)

See `backend/README.md` for endpoint reference. To run locally:

```bash
# Everything (Postgres + backend) in Docker
docker compose up --build

# Or backend on host, Postgres in Docker
docker compose up -d postgres
uv sync --extra backend
uvicorn backend.main:app --reload
```

## Configuration

All tunable matching parameters live in `engine/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VISUAL_INDEX_FPS` | 2 | Frames/sec when indexing |
| `VISUAL_QUERY_FPS` | 4 | Frames/sec when querying |
| `CLIP_MODEL_NAME` | `ViT-B-32` | CLIP variant |
| `VISUAL_TOP_K` | 20 | Nearest neighbors per query frame |
| `OFFSET_BIN_WIDTH` | 1.5 | Seconds per histogram bin |
| `MIN_VISUAL_MATCHES` | 4 | Minimum matches required |

## GPU acceleration

CLIP runs on CPU by default. To use Apple Silicon GPU:

```bash
CLIP_DEVICE=mps python main.py index --file movie.mp4 --title "Movie"
```

## Troubleshooting

**OpenMP segfault** — `OMP_NUM_THREADS=1` is set automatically in `main.py` and
in the backend to avoid a FAISS/PyTorch OMP runtime conflict on macOS.

**`uv` complains about Python version** — your `.python-version` likely holds a
pyenv venv name (which uv can't read). Either run `uv` with `--python` pointing
at the pyenv interpreter, or set `UV_PYTHON=3.11` in your shell.

**Postgres connection refused** — `docker compose up -d postgres` and retry.

**CLIP model download** — happens once, ~350 MB. Cached under `~/.cache/clip`
on the host, or in the `clip_cache` Docker volume in container mode.
