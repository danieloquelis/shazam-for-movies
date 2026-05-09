"""
FastAPI service wrapping the visual fingerprinting engine.

Phase 1: record-and-upload. The mobile client records a short clip (5-10s),
uploads it to /query, and receives the matched movie + timestamp.

Streaming (Phase 2) lives at a different endpoint that doesn't exist yet.
"""
import os

# Match the env tweaks that engine/main.py applies before importing the engine.
# OMP/MKL must be set BEFORE PyTorch and FAISS are imported.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
import math
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engine.db import Database, DatabaseBackend
from engine.matcher import MatchResult, match_clip
from engine.visual_fingerprint import _get_clip


logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


API_KEY = os.environ.get("API_KEY", "dev-key-change-me")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))  # 20 MB
ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v", "application/octet-stream"}

# When set (any truthy string), every successful upload is also copied to
# /app/data/debug-uploads/ before being deleted. The data/ dir is mounted from
# the host, so the saved clip is inspectable from the Mac. Off by default.
DEBUG_KEEP_UPLOADS = os.environ.get("DEBUG_KEEP_UPLOADS", "").lower() in {"1", "true", "yes", "on"}
DEBUG_UPLOADS_DIR = Path("/app/data/debug-uploads")


_db: DatabaseBackend | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm CLIP and open a long-lived Database connection on boot."""
    global _db
    logger.info("Warming up CLIP model")
    _get_clip()
    logger.info("Opening database")
    _db = Database()
    logger.info("Backend ready")
    try:
        yield
    finally:
        if _db is not None:
            _db.close()
            _db = None


app = FastAPI(title="Shazam for Movies", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Phase 1: open for LAN dev. Tighten before any public deploy.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")


def _format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def _json_safe(value):
    """Recursively coerce NaN/inf to None so the response is valid JSON.

    The matcher uses float('inf') as a sentinel for "no runner-up", which is
    fine internally but blows up json.dumps. Sanitize at the response edge.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _serialize(result: MatchResult) -> dict:
    return _json_safe(
        {
            "movie_id": result.movie_id,
            "title": result.movie_title,
            "timestamp_sec": result.timestamp_sec,
            "timestamp_human": _format_timestamp(result.timestamp_sec),
            "confidence": result.confidence,
            "visual_score": result.visual_score,
            "match_details": result.match_details,
        }
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "db_ready": _db is not None}


@app.post("/query", dependencies=[Depends(require_api_key)])
async def query(
    file: UploadFile = File(...),
    # Default ON: every request is a phone capture of a screen, so screen
    # detection helps far more often than it hurts. Override per-request
    # via ?detect_screen=false if needed.
    detect_screen: bool = True,
) -> JSONResponse:
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": "unsupported_content_type", "got": file.content_type},
        )

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    if suffix not in {".mp4", ".mov", ".m4v"}:
        suffix = ".mp4"

    # Stream the upload to a temp file with a hard size cap.
    bytes_written = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail={"error": "file_too_large", "limit_bytes": MAX_UPLOAD_BYTES},
                )
            tmp.write(chunk)

    try:
        if _db is None:
            raise HTTPException(status_code=503, detail="db_not_ready")

        logger.info("Matching clip (%d bytes, screen=%s)", bytes_written, detect_screen)

        if DEBUG_KEEP_UPLOADS:
            try:
                DEBUG_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d-%H%M%S")
                dest = DEBUG_UPLOADS_DIR / f"upload-{ts}{suffix}"
                shutil.copyfile(tmp_path, dest)
                logger.info("DEBUG: saved upload to %s", dest)
            except Exception as e:  # noqa: BLE001 — debug path; never fail the request
                logger.warning("DEBUG: could not save upload: %s", e)

        result = match_clip(tmp_path, db=_db, detect_screen=detect_screen)

        if result is None:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={"error": "no_match"},
            )

        return JSONResponse(content=_serialize(result))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
