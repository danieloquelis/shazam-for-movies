# syntax=docker/dockerfile:1.7

# Build context is the repo root (see docker-compose.yml: context: .)

# Stage 1: install deps with uv into a clean venv-style prefix.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=0

# uv is a single static binary — copy it from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /usr/local/bin/

WORKDIR /app

# Cache dep installation: copy only the manifest first.
# README.md is referenced by pyproject.toml's `readme = "README.md"` and
# Hatchling demands it exists during the editable build, even at the
# --no-install-project step.
COPY pyproject.toml uv.lock README.md /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra backend

# Now copy the source and install the project itself (engine + backend).
COPY engine /app/engine
COPY backend /app/backend
COPY main.py /app/main.py
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra backend


# Stage 2: lean runtime image.
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the resolved venv and the source.
COPY --from=builder /app /app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
