"""
Visual-only query matching via FAISS nearest neighbors + offset histogram voting.

Design:
- Resolution-agnostic (frames normalized to 224x224)
- Dub-invariant (purely visual — no audio dependency)
- Noise-tolerant (CLAHE, center-crop, letterbox removal)

Core mechanism: offset histogram voting
- For each FAISS match: offset = db_time - query_time
- Tight cluster of offsets = correct movie + timestamp
- Random false positives produce uniform offsets (filtered out)
"""

import time
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from engine.visual_fingerprint import fingerprint_visual
from engine.db import Database
from engine.config import (
    VISUAL_QUERY_FPS, VISUAL_TOP_K,
    OFFSET_BIN_WIDTH, MIN_VISUAL_MATCHES,
    OFFSET_STDDEV_THRESHOLD, CONFIDENCE_RATIO,
)


@dataclass
class MatchResult:
    movie_id: int
    movie_title: str
    timestamp_sec: float
    confidence: float
    visual_score: float
    match_details: dict = field(default_factory=dict)


def _offset_histogram(offsets: list[float], bin_width: float) -> tuple[float, int, float]:
    """
    Build histogram of offsets, find peak bin.
    Returns (peak_center_sec, count_in_peak_region, stddev).
    """
    if not offsets:
        return 0.0, 0, float("inf")

    offsets = np.array(offsets)
    min_off, max_off = offsets.min(), offsets.max()
    bins = np.arange(min_off - bin_width, max_off + 2 * bin_width, bin_width)
    counts, edges = np.histogram(offsets, bins=bins)

    peak_idx = np.argmax(counts)
    peak_center = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0

    # Count peak + neighbors for robustness
    total_count = counts[peak_idx]
    if peak_idx > 0:
        total_count += counts[peak_idx - 1]
    if peak_idx < len(counts) - 1:
        total_count += counts[peak_idx + 1]

    in_peak = offsets[
        (offsets >= edges[max(0, peak_idx - 1)]) &
        (offsets < edges[min(len(edges) - 1, peak_idx + 2)])
    ]
    stddev = float(np.std(in_peak)) if len(in_peak) > 1 else 0.0

    return float(peak_center), int(total_count), stddev


def match_clip(clip_path: str, db: Database = None) -> MatchResult | None:
    """
    Match a short video clip against the indexed database.
    Returns MatchResult or None if no confident match.
    """
    own_db = db is None
    if own_db:
        db = Database()

    try:
        print(f"\nMatching clip: {clip_path}")
        t0 = time.time()

        # --- Visual fingerprinting ---
        print("\n[Visual Matching]")
        t1 = time.time()
        visual_fps = fingerprint_visual(clip_path, fps=VISUAL_QUERY_FPS, normalize=True)
        print(f"  Query: {len(visual_fps)} frames ({time.time()-t1:.1f}s)")

        if not visual_fps:
            print("\nNo frames extracted from clip.")
            return None

        timestamps = [ts for ts, _, _ in visual_fps]
        embeddings = np.array([emb for _, _, emb in visual_fps], dtype=np.float32)

        # --- FAISS search ---
        t1 = time.time()
        search_results = db.search_visual(embeddings, VISUAL_TOP_K)
        print(f"  FAISS search: {time.time()-t1:.2f}s")

        # --- Offset voting per movie ---
        movie_data = defaultdict(lambda: {"offsets": [], "similarities": [], "query_order": []})
        for q_idx, (q_ts, matches) in enumerate(zip(timestamps, search_results)):
            for movie_id, db_ts, similarity in matches:
                offset = db_ts - q_ts
                movie_data[movie_id]["offsets"].append(offset)
                movie_data[movie_id]["similarities"].append(similarity)
                movie_data[movie_id]["query_order"].append((q_idx, db_ts))

        # --- Score candidates ---
        candidates = []
        for movie_id, data in movie_data.items():
            offsets = data["offsets"]
            peak_sec, peak_count, stddev = _offset_histogram(offsets, OFFSET_BIN_WIDTH)

            if peak_count < MIN_VISUAL_MATCHES:
                continue

            offset_arr = np.array(offsets)
            sim_arr = np.array(data["similarities"])
            in_peak_mask = np.abs(offset_arr - peak_sec) < OFFSET_BIN_WIDTH * 2
            mean_sim = float(np.mean(sim_arr[in_peak_mask])) if in_peak_mask.any() else 0.0

            cluster_frac = min(1.0, peak_count / len(visual_fps))

            # Temporal order consistency
            peak_pairs = [(qi, dt) for (qi, dt), m in zip(data["query_order"], in_peak_mask) if m]
            if len(peak_pairs) >= 2:
                sorted_by_q = sorted(peak_pairs, key=lambda x: x[0])
                db_times = [p[1] for p in sorted_by_q]
                in_order = sum(1 for i in range(len(db_times) - 1) if db_times[i] <= db_times[i + 1])
                temporal_consistency = in_order / (len(db_times) - 1)
            else:
                temporal_consistency = 0.5

            visual_score = 0.45 * mean_sim + 0.35 * cluster_frac + 0.20 * temporal_consistency

            candidates.append({
                "movie_id": movie_id,
                "offset_sec": peak_sec,
                "visual_score": visual_score,
                "mean_similarity": mean_sim,
                "cluster_size": peak_count,
                "temporal_consistency": temporal_consistency,
                "offset_stddev": stddev,
            })

        candidates.sort(key=lambda c: c["visual_score"], reverse=True)

        for c in candidates[:3]:
            movie = db.get_movie(c["movie_id"])
            name = movie["title"] if movie else f"id={c['movie_id']}"
            print(f"  -> {name} | offset={c['offset_sec']:.1f}s | "
                  f"score={c['visual_score']:.3f} | sim={c['mean_similarity']:.3f} | "
                  f"temporal={c['temporal_consistency']:.3f}")

        if not candidates:
            print("\nNo match found.")
            return None

        # --- Confidence ---
        best = candidates[0]
        if len(candidates) >= 2:
            ratio = best["visual_score"] / max(candidates[1]["visual_score"], 1e-10)
        else:
            ratio = float("inf")

        confidence = min(1.0, best["visual_score"])
        if ratio < CONFIDENCE_RATIO:
            confidence *= 0.5

        movie = db.get_movie(best["movie_id"])
        title = movie["title"] if movie else "Unknown"
        total_time = time.time() - t0

        ts = best["offset_sec"]
        ts_fmt = f"{int(ts//3600):02d}:{int(ts%3600//60):02d}:{ts%60:05.2f}"

        print(f"\n{'='*50}")
        print(f"MATCH RESULT")
        print(f"  Movie:      {title} (id={best['movie_id']})")
        print(f"  Timestamp:  {ts:.1f}s ({ts_fmt})")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Similarity: {best['mean_similarity']:.3f}")
        print(f"  Temporal:   {best['temporal_consistency']:.3f}")
        print(f"  Cluster:    {best['cluster_size']} matches")
        print(f"  Time:       {total_time:.1f}s")
        print(f"{'='*50}\n")

        return MatchResult(
            movie_id=best["movie_id"],
            movie_title=title,
            timestamp_sec=best["offset_sec"],
            confidence=confidence,
            visual_score=best["visual_score"],
            match_details={
                "mean_similarity": best["mean_similarity"],
                "temporal_consistency": best["temporal_consistency"],
                "cluster_size": best["cluster_size"],
                "offset_stddev": best["offset_stddev"],
                "confidence_ratio": ratio,
                "n_candidates": len(candidates),
            },
        )
    finally:
        if own_db:
            db.close()
