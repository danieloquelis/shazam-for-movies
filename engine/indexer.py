"""
Movie indexer: extracts visual fingerprints and stores in DB.

Two-level indexing:
1. Keyframe-level: deduplicated frames stored in FAISS (skips near-duplicates)
2. Scene-level: mean embeddings per scene stored in separate FAISS index

Streams in batches to handle long movies without OOM.
"""

import subprocess
import time
import gc
import numpy as np
from engine.visual_fingerprint import (
    extract_frames_raw, _normalize_frame, compute_phash,
    compute_clip_embeddings, _get_clip, deduplicate_keyframes,
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


def _flush_scene(scene_embs, scene_start, scene_end, movie_id, db, scene_list):
    """Compute mean embedding for a completed scene and queue it for storage."""
    if not scene_embs:
        return
    mean_emb = np.mean(scene_embs, axis=0).astype(np.float32)
    scene_list.append((scene_start, scene_end, len(scene_embs), mean_emb))


def index_movie(video_path: str, title: str, db: Database = None) -> int:
    """
    Index a movie: extract visual fingerprints with deduplication and scene detection.
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
        total_frames = 0
        total_keyframes = 0
        total_scenes = 0

        _get_clip()  # Warm up CLIP model before FAISS loads

        # Cross-batch state for deduplication and scene tracking
        prev_kept_embedding = None
        pending_scene_embs = []
        pending_scene_start = None
        pending_scene_end = None
        scene_buffer = []  # accumulated scenes to store in bulk

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

            # --- Deduplication and scene detection ---
            kept_indices, completed_scenes, scene_boundary_at_start = deduplicate_keyframes(
                embeddings, prev_embedding=prev_kept_embedding
            )

            # Handle cross-batch scene boundary
            if scene_boundary_at_start and pending_scene_embs:
                _flush_scene(pending_scene_embs, pending_scene_start, pending_scene_end,
                             movie_id, db, scene_buffer)
                pending_scene_embs = []
                pending_scene_start = None

            # Flush completed scenes (all except the last ongoing one)
            for scene_indices in completed_scenes:
                # First, flush any pending scene from previous batch
                if pending_scene_embs:
                    _flush_scene(pending_scene_embs, pending_scene_start, pending_scene_end,
                                 movie_id, db, scene_buffer)
                    pending_scene_embs = []

                # Start this scene
                for idx in scene_indices:
                    pending_scene_embs.append(embeddings[idx])
                pending_scene_start = timestamps[scene_indices[0]]
                pending_scene_end = timestamps[scene_indices[-1]]

                # This scene is complete, flush it
                _flush_scene(pending_scene_embs, pending_scene_start, pending_scene_end,
                             movie_id, db, scene_buffer)
                pending_scene_embs = []
                pending_scene_start = None

            # Accumulate ongoing scene (frames after last completed scene)
            # These are the kept frames that aren't in any completed scene
            all_completed = set()
            for scene_indices in completed_scenes:
                all_completed.update(scene_indices)

            for idx in kept_indices:
                if idx not in all_completed:
                    pending_scene_embs.append(embeddings[idx])
                    if pending_scene_start is None:
                        pending_scene_start = timestamps[idx]
                    pending_scene_end = timestamps[idx]

            # Store kept keyframes
            if kept_indices:
                batch = [(timestamps[i], phashes[i], embeddings[i]) for i in kept_indices]
                db.store_visual_fingerprints(movie_id, batch, silent=True)
                prev_kept_embedding = embeddings[kept_indices[-1]]
                total_keyframes += len(kept_indices)
                del batch

            # Store accumulated scenes in bulk periodically
            if len(scene_buffer) >= 50:
                db.store_scene_descriptors(movie_id, scene_buffer, silent=True)
                total_scenes += len(scene_buffer)
                scene_buffer = []

            del embeddings, phashes, timestamps
            gc.collect()

            total_frames += n_frames
            skipped = n_frames - len(kept_indices)
            pct = min(100, (current + chunk_dur) / duration * 100)
            print(f"  Indexing: {total_frames} frames, {total_keyframes} keyframes, "
                  f"{total_scenes} scenes ({pct:.0f}%)", end="\r")

            current += chunk_dur

        # Flush remaining pending scene
        if pending_scene_embs:
            _flush_scene(pending_scene_embs, pending_scene_start, pending_scene_end,
                         movie_id, db, scene_buffer)

        # Store remaining scenes
        if scene_buffer:
            db.store_scene_descriptors(movie_id, scene_buffer, silent=True)
            total_scenes += len(scene_buffer)

        db.save()
        elapsed = time.time() - t0

        dedup_pct = (1 - total_keyframes / max(total_frames, 1)) * 100
        print(f"  Indexing: {total_frames} frames, {total_keyframes} keyframes, "
              f"{total_scenes} scenes (100%)   ")
        print(f"  Deduplication: {total_frames} -> {total_keyframes} "
              f"({dedup_pct:.0f}% reduction)")
        print(f"  Time: {elapsed:.1f}s")

        print(f"\n{'='*50}")
        print(f"Indexing complete: {title} (id={movie_id})")
        print(f"  Total frames:  {total_frames:,}")
        print(f"  Keyframes:     {total_keyframes:,} ({100-dedup_pct:.0f}% kept)")
        print(f"  Scenes:        {total_scenes:,}")
        print(f"  Time:          {elapsed:.1f}s")
        print(f"{'='*50}\n")

        return movie_id
    finally:
        if own_db:
            db.close()
