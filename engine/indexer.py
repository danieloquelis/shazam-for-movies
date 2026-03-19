"""
Movie indexer: extracts visual fingerprints and stores in DB.
Streams fingerprints in batches to handle long movies without OOM.
"""

import subprocess
import time
import gc
import numpy as np
from engine.visual_fingerprint import (
    extract_frames_raw, _normalize_frame, compute_phash,
    compute_clip_embeddings, _get_clip,
)
from engine.db import Database
from engine.config import VISUAL_INDEX_FPS, FRAME_BATCH_SIZE


def get_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def index_movie(video_path: str, title: str, db: Database = None) -> int:
    """
    Index a movie: extract visual fingerprints and store in DB + FAISS.
    Returns movie_id.
    """
    own_db = db is None
    if own_db:
        db = Database()

    try:
        existing = db.movie_exists(title)
        if existing:
            print(f"Movie '{title}' already indexed (id={existing}). Skipping.")
            return existing

        duration = get_duration(video_path)
        print(f"\nIndexing: {title}")
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

        movie_id = db.create_movie(title, video_path, duration)
        print(f"Movie ID: {movie_id}")

        # --- Visual fingerprinting (streamed to DB in batches) ---
        print("\n[Visual Fingerprinting]")
        t0 = time.time()
        fps = VISUAL_INDEX_FPS
        window_sec = FRAME_BATCH_SIZE / fps
        total_visual = 0

        _get_clip()  # Warm up CLIP model before FAISS loads

        current = 0.0
        while current < duration:
            chunk_dur = min(window_sec, duration - current)

            raw_frames = extract_frames_raw(video_path, fps, start_sec=current, duration_sec=chunk_dur)
            if not raw_frames:
                current += chunk_dur
                continue

            n_frames = len(raw_frames)
            timestamps = [f[0] for f in raw_frames]
            raw_arrays = [f[1] for f in raw_frames]
            del raw_frames

            frame_arrays = [_normalize_frame(f) for f in raw_arrays]
            del raw_arrays

            phashes = [compute_phash(f) for f in frame_arrays]
            embeddings = compute_clip_embeddings(frame_arrays)
            del frame_arrays

            batch = [(ts, ph, embeddings[i]) for i, (ts, ph) in enumerate(zip(timestamps, phashes))]
            db.store_visual_fingerprints(movie_id, batch, silent=True)
            del batch, embeddings, phashes, timestamps
            gc.collect()

            total_visual += n_frames
            pct = min(100, (current + chunk_dur) / duration * 100)
            print(f"  Visual: {total_visual} frames indexed ({pct:.0f}%)", end="\r")

            current += chunk_dur

        db.save()
        elapsed = time.time() - t0
        print(f"  Visual: {total_visual} frames indexed (100%)   ")
        print(f"  Time: {elapsed:.1f}s")

        print(f"\n{'='*50}")
        print(f"Indexing complete: {title} (id={movie_id})")
        print(f"  Visual frames: {total_visual:,}")
        print(f"  Time:          {elapsed:.1f}s")
        print(f"{'='*50}\n")

        return movie_id
    finally:
        if own_db:
            db.close()
