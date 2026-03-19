"""
Visual-only query matching via FAISS nearest neighbors + offset histogram voting.

Design:
- Resolution-agnostic (frames normalized to 224x224)
- Dub-invariant (purely visual — no audio dependency)
- Noise-tolerant (CLAHE, center-crop, letterbox removal)

Matching pipeline:
1. Rank-weighted, best-per-frame voting — suppresses false positives
2. Offset histogram voting with concentration scoring
3. Temporal order enforcement — rejects random scatter
4. Two-pass verification — re-score at higher FPS with focus on offset density
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
    SIMILARITY_GATE, RANK_WEIGHT_DECAY, MIN_TEMPORAL_ORDER,
    VERIFY_FPS, VERIFY_WINDOW, VERIFY_TOP_N,
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


def _temporal_order_score(peak_pairs: list[tuple[int, float]]) -> float:
    """
    Measure how well the matched frames preserve temporal order.
    Returns fraction of consecutive pairs that are in correct order.
    """
    if len(peak_pairs) < 2:
        return 0.5

    sorted_by_q = sorted(peak_pairs, key=lambda x: x[0])
    db_times = [p[1] for p in sorted_by_q]
    in_order = sum(1 for i in range(len(db_times) - 1) if db_times[i] <= db_times[i + 1])
    return in_order / (len(db_times) - 1)


def _gather_votes(timestamps, search_results, similarity_gate: float) -> dict:
    """
    Collect offset votes from FAISS results with rank-weighted, best-per-frame voting.

    Two key mechanisms to suppress false positives:
    1. Rank weighting: top-ranked matches get exponentially more weight
    2. Best-per-frame: each query frame gives each movie at most ONE vote
       (the highest-ranked match). This prevents a wrong movie from
       accumulating many low-rank votes that inflate its offset histogram.
    """
    # First pass: find best (lowest rank) match per (query_frame, movie) pair
    best_per_frame_movie = {}  # (q_idx, movie_id) -> (rank, db_ts, similarity)
    gated_count = 0
    total_count = 0

    for q_idx, (q_ts, matches) in enumerate(zip(timestamps, search_results)):
        for rank, (movie_id, db_ts, similarity) in enumerate(matches):
            total_count += 1
            if similarity < similarity_gate:
                gated_count += 1
                continue
            key = (q_idx, movie_id)
            if key not in best_per_frame_movie or rank < best_per_frame_movie[key][0]:
                best_per_frame_movie[key] = (rank, db_ts, similarity)

    # Second pass: build movie data from best-per-frame matches
    movie_data = defaultdict(lambda: {"offsets": [], "similarities": [], "weights": [], "query_order": []})

    for (q_idx, movie_id), (rank, db_ts, similarity) in best_per_frame_movie.items():
        q_ts = timestamps[q_idx]
        weight = RANK_WEIGHT_DECAY ** rank
        offset = db_ts - q_ts
        movie_data[movie_id]["offsets"].append(offset)
        movie_data[movie_id]["similarities"].append(similarity)
        movie_data[movie_id]["weights"].append(weight)
        movie_data[movie_id]["query_order"].append((q_idx, db_ts))

    return movie_data, gated_count, total_count


def _score_candidates(movie_data: dict, n_query_frames: int) -> list[dict]:
    """
    Score candidate movies using rank-weighted offset voting + temporal consistency.

    Key scoring factors:
    - mean_sim: how similar are the matched frames (weighted by rank)
    - cluster_frac: what fraction of query frames have matches at the peak offset
    - concentration: what fraction of ALL this movie's votes land in the peak
      (true match = high concentration, false positive = scattered votes)
    - temporal_consistency: do matched frames preserve playback order
    """
    candidates = []
    for movie_id, data in movie_data.items():
        offsets = data["offsets"]
        peak_sec, peak_count, stddev = _offset_histogram(offsets, OFFSET_BIN_WIDTH)

        if peak_count < MIN_VISUAL_MATCHES:
            continue

        offset_arr = np.array(offsets)
        sim_arr = np.array(data["similarities"])
        weight_arr = np.array(data["weights"])
        in_peak_mask = np.abs(offset_arr - peak_sec) < OFFSET_BIN_WIDTH * 2

        # Weighted similarity: high-rank matches contribute more
        peak_weights = weight_arr[in_peak_mask]
        peak_sims = sim_arr[in_peak_mask]
        if peak_weights.sum() > 0:
            mean_sim = float(np.average(peak_sims, weights=peak_weights))
        else:
            mean_sim = 0.0

        # Cluster fraction: what fraction of query frames contributed to peak
        cluster_frac = min(1.0, peak_count / n_query_frames)

        # Concentration: what fraction of this movie's total votes are in the peak
        # True match: most votes cluster at one offset -> high concentration
        # False positive: votes scatter across many offsets -> low concentration
        # BUT: only meaningful when there are enough total votes (prevents lucky small clusters)
        concentration = peak_count / len(offsets) if len(offsets) > 0 else 0.0

        # Temporal order consistency
        peak_pairs = [(qi, dt) for (qi, dt), m in zip(data["query_order"], in_peak_mask) if m]
        temporal_consistency = _temporal_order_score(peak_pairs)

        # Strict temporal order enforcement: reject candidates with random ordering
        if temporal_consistency < MIN_TEMPORAL_ORDER and peak_count >= 6:
            continue

        # Combined scoring:
        # - cluster_frac: absolute signal strength (how many query frames matched at peak)
        # - concentration: relative signal quality (what fraction of votes are useful)
        # - Use geometric mean of cluster_frac and concentration to require BOTH
        #   a large cluster AND high concentration
        cluster_quality = (cluster_frac * concentration) ** 0.5

        visual_score = (0.35 * mean_sim
                        + 0.30 * cluster_quality
                        + 0.15 * cluster_frac
                        + 0.20 * temporal_consistency)

        candidates.append({
            "movie_id": movie_id,
            "offset_sec": peak_sec,
            "visual_score": visual_score,
            "mean_similarity": mean_sim,
            "cluster_size": peak_count,
            "concentration": concentration,
            "temporal_consistency": temporal_consistency,
            "offset_stddev": stddev,
        })

    candidates.sort(key=lambda c: c["visual_score"], reverse=True)
    return candidates


def _verify_candidate(candidate: dict, query_embeddings: np.ndarray,
                       query_timestamps: list[float], db: Database) -> dict:
    """
    Two-pass verification: re-check candidate at higher FPS.

    The key insight: more query frames should produce MORE matches at the
    SAME offset if the match is real. If it's noise, more frames just scatter.
    We measure this via concentration — fraction of votes in the peak.
    """
    movie_id = candidate["movie_id"]
    predicted_offset = candidate["offset_sec"]

    # Search with more neighbors for deeper verification
    search_results = db.search_visual(query_embeddings, VISUAL_TOP_K * 2)

    # Gather votes using best-per-frame for this specific movie
    all_offsets = []
    peak_sims = []
    peak_weights = []
    peak_pairs = []

    for q_idx, (q_ts, matches) in enumerate(zip(query_timestamps, search_results)):
        # Find best rank for this movie among all matches
        best_rank = None
        best_match = None
        for rank, (mid, db_ts, similarity) in enumerate(matches):
            if mid != movie_id:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_match = (db_ts, similarity)

        if best_match is None:
            continue

        db_ts, similarity = best_match
        offset = db_ts - q_ts
        all_offsets.append(offset)

        # Check if this vote agrees with predicted offset
        if abs(offset - predicted_offset) < OFFSET_BIN_WIDTH * 3:
            weight = RANK_WEIGHT_DECAY ** best_rank
            peak_sims.append(similarity)
            peak_weights.append(weight)
            peak_pairs.append((q_idx, db_ts))

    if not peak_sims:
        candidate["visual_score"] *= 0.3
        candidate["verified"] = False
        return candidate

    # Concentration: of all frames that matched this movie, how many agree on offset?
    concentration = len(peak_sims) / len(all_offsets) if all_offsets else 0.0

    weight_arr = np.array(peak_weights)
    sim_arr = np.array(peak_sims)
    verified_sim = float(np.average(sim_arr, weights=weight_arr))
    verified_temporal = _temporal_order_score(peak_pairs)
    cluster_frac = min(1.0, len(peak_sims) / len(query_timestamps))

    cluster_quality = (cluster_frac * concentration) ** 0.5

    verified_score = (0.35 * verified_sim
                      + 0.30 * cluster_quality
                      + 0.15 * cluster_frac
                      + 0.20 * verified_temporal)

    # Use the better of original and verified scores
    candidate["visual_score"] = max(candidate["visual_score"], verified_score)
    candidate["mean_similarity"] = max(candidate["mean_similarity"], verified_sim)
    candidate["temporal_consistency"] = max(candidate["temporal_consistency"], verified_temporal)
    candidate["concentration"] = max(candidate.get("concentration", 0), concentration)
    candidate["verified"] = True
    candidate["verified_matches"] = len(peak_sims)

    return candidate


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

        # --- Pass 1: Visual fingerprinting ---
        print("\n[Pass 1: Candidate Selection]")
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

        # --- Rank-weighted offset voting ---
        movie_data, gated, total = _gather_votes(timestamps, search_results, SIMILARITY_GATE)
        print(f"  Votes: {total} total, {gated} gated")

        # --- Score candidates ---
        candidates = _score_candidates(movie_data, len(visual_fps))

        for c in candidates[:3]:
            movie = db.get_movie(c["movie_id"])
            name = movie["title"] if movie else f"id={c['movie_id']}"
            print(f"  -> {name} | offset={c['offset_sec']:.1f}s | "
                  f"score={c['visual_score']:.3f} | conc={c['concentration']:.2f} | "
                  f"temporal={c['temporal_consistency']:.3f}")

        if not candidates:
            print("\nNo match found.")
            return None

        # --- Pass 2: Verification on top candidates ---
        if len(candidates) >= 2:
            gap = candidates[0]["visual_score"] / max(candidates[1]["visual_score"], 1e-10)
            needs_verification = gap < CONFIDENCE_RATIO * 1.5
        else:
            needs_verification = False

        if needs_verification:
            n_verify = min(VERIFY_TOP_N, len(candidates))
            print(f"\n[Pass 2: Verification — re-scoring top {n_verify} at {VERIFY_FPS}fps]")
            t1 = time.time()

            # Extract higher-FPS embeddings once for all candidates
            verify_fps_data = fingerprint_visual(clip_path, fps=VERIFY_FPS, normalize=True)
            if verify_fps_data:
                v_timestamps = [ts for ts, _, _ in verify_fps_data]
                v_embeddings = np.array([emb for _, _, emb in verify_fps_data], dtype=np.float32)
                print(f"  Verify query: {len(verify_fps_data)} frames")

                for i in range(n_verify):
                    movie = db.get_movie(candidates[i]["movie_id"])
                    name = movie["title"] if movie else f"id={candidates[i]['movie_id']}"
                    old_score = candidates[i]["visual_score"]
                    candidates[i] = _verify_candidate(
                        candidates[i], v_embeddings, v_timestamps, db
                    )
                    new_score = candidates[i]["visual_score"]
                    verified = candidates[i].get("verified", False)
                    v_matches = candidates[i].get("verified_matches", 0)
                    conc = candidates[i].get("concentration", 0)
                    print(f"  -> {name} | {old_score:.3f} -> {new_score:.3f} | "
                          f"conc={conc:.2f} | verified={verified} ({v_matches} matches)")

            # Re-sort after verification
            candidates.sort(key=lambda c: c["visual_score"], reverse=True)
            print(f"  Verification: {time.time()-t1:.1f}s")
        else:
            print(f"\n[Pass 2: Skipped — clear winner]")

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
        print(f"  Conc:       {best.get('concentration', 0):.3f}")
        print(f"  Temporal:   {best['temporal_consistency']:.3f}")
        print(f"  Cluster:    {best['cluster_size']} matches")
        if best.get("verified"):
            print(f"  Verified:   yes ({best.get('verified_matches', 0)} matches)")
        print(f"  Ratio:      {ratio:.2f}x over runner-up")
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
                "concentration": best.get("concentration", 0),
                "cluster_size": best["cluster_size"],
                "offset_stddev": best["offset_stddev"],
                "confidence_ratio": ratio,
                "n_candidates": len(candidates),
                "verified": best.get("verified", False),
                "similarity_gated": gated,
            },
        )
    finally:
        if own_db:
            db.close()
