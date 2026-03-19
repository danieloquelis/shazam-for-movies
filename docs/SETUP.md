# Setup Guide

## Prerequisites

- Python 3.11+ (via pyenv)
- FFmpeg
- Docker (for PostgreSQL)

## Installation

```bash
# 1. Create Python environment
pyenv virtualenv 3.11.6 movie_fingerprint
pyenv local movie_fingerprint

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL
docker compose up -d

# 4. Verify
python -c "from engine.db import Database; db = Database(); print('OK'); db.close()"
```

## Usage

### Index a movie

```bash
python main.py index --file movie.mp4 --title "Movie Name"
```

Indexing a 2-hour movie takes ~4-5 minutes on CPU.

### Query a clip

```bash
python main.py query --file clip.mp4
```

Returns the movie title, timestamp, and confidence in ~1 second.

### Run a self-test

```bash
# Random 10-second clip
python main.py test --file movie.mp4 --title "Movie Name"

# Specific timestamp, 5-second clip
python main.py test --file movie.mp4 --title "Movie Name" --start 2400 --duration 5
```

## Configuration

All tunable parameters are in `engine/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VISUAL_INDEX_FPS` | 2 | Frames/sec when indexing |
| `VISUAL_QUERY_FPS` | 4 | Frames/sec when querying |
| `CLIP_MODEL_NAME` | ViT-B-32 | CLIP model variant |
| `VISUAL_TOP_K` | 20 | Nearest neighbors per query frame |
| `OFFSET_BIN_WIDTH` | 0.5 | Seconds per histogram bin |
| `MIN_AUDIO_MATCHES` | 8 | Min audio hash matches |
| `MIN_VISUAL_MATCHES` | 4 | Min visual frame matches |

## GPU Acceleration

By default CLIP runs on CPU for stability. To use Apple Silicon GPU:

```bash
CLIP_DEVICE=mps python main.py index --file movie.mp4 --title "Movie"
```

## Troubleshooting

**OpenMP segfault**: The `OMP_NUM_THREADS=1` env var is set automatically in `main.py` to avoid a conflict between FAISS and PyTorch's OMP runtimes on macOS.

**PostgreSQL connection refused**: Make sure Docker is running: `docker compose up -d`

**CLIP model download**: The first run downloads the CLIP model (~350 MB). Subsequent runs use the cached model.
