# Backend

FastAPI service wrapping the visual fingerprinting engine. Phase 1: a single
record-and-upload endpoint (`POST /query`) for the mobile app.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/healthz` | none | Liveness probe |
| POST | `/query` | `x-api-key` | Match an uploaded clip against the index |

### `POST /query`

Multipart upload of a short video clip (5-10 s). Returns the matched movie and
timestamp.

```bash
curl -X POST http://localhost:8000/query \
  -H "x-api-key: dev-key-change-me" \
  -F "file=@clip.mov"
```

Successful response:

```json
{
  "movie_id": 1,
  "title": "Project Almanac",
  "timestamp_sec": 3653.4,
  "timestamp_human": "01:00:53.40",
  "confidence": 0.886,
  "visual_score": 0.731,
  "match_details": { "...": "..." }
}
```

Other responses:

| Status | Meaning |
|--------|---------|
| 401 | `x-api-key` missing or wrong |
| 413 | Upload exceeds `MAX_UPLOAD_BYTES` (default 20 MB) |
| 415 | Unsupported content type |
| 422 | No confident match found |
| 503 | Database not ready |

## Environment

| Var | Default | Purpose |
|-----|---------|---------|
| `API_KEY` | `dev-key-change-me` | Required header value for `/query` |
| `MAX_UPLOAD_BYTES` | `20971520` (20 MB) | Hard cap on upload size |
| `POSTGRES_HOST` | `localhost` | Postgres host (set to `postgres` in compose) |
| `POSTGRES_PORT` | `5432` | |
| `POSTGRES_DB` | `movie_fp` | |
| `POSTGRES_USER` | `fingerprint` | |
| `POSTGRES_PASSWORD` | `fingerprint` | |
| `CLIP_DEVICE` | `cpu` | Set to `mps` on Apple Silicon for GPU |

## Running locally with Docker

From the repo root (the `Dockerfile` references `engine/` so build context must
be the repo root):

```bash
docker compose up --build
```

The compose file at the repo root brings up Postgres and the backend together.
Backend listens on `http://localhost:8000`.

## Running locally without Docker

```bash
pyenv activate movie_fingerprint
uv sync --extra backend
uvicorn backend.main:app --reload
```

The CLIP model downloads on first start (~350 MB). Subsequent starts use the
cached weights.

## Notes

- This service holds the CLIP model and FAISS index in memory and assumes a
  long-lived process. Do not deploy it as a serverless function.
- Streaming endpoints (Phase 2) will live next to `/query` once the streaming
  matcher exists. See `docs/CLIENT.md`.
