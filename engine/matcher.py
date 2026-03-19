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
    SCENE_TOP_K, SCENE_CANDIDATE_MOVIES,
)


@dataclass
class MatchResult:
    movie_id: int
    movie_title: str
    timestamp_sec: float
    confidence: float
    visual_score: float
    match_details: dict = field(default_factory=dict)


def _offset_histogram(offsets: list[float], bin_width: float,
                       weights: np.ndarray = None) -> tuple[float, int, float, float]:
    """
    Build weighted histogram of offsets, find peak bin.
    Returns (peak_center_sec, count_in_peak_region, stddev, peak_weight).
    When weights are provided, the peak is selected by total weight, not count.
    """
    if not offsets:
        return 0.0, 0, float("inf"), 0.0

    offsets = np.array(offsets)
    if weights is None:
        weights = np.ones(len(offsets))

    min_off, max_off = offsets.min(), offsets.max()
    bins = np.arange(min_off - bin_width, max_off + 2 * bin_width, bin_width)

    # Weighted histogram: sum weights per bin instead of counting
    bin_weights = np.zeros(len(bins) - 1)
    bin_indices = np.digitize(offsets, bins) - 1
    for i, bi in enumerate(bin_indices):
        if 0 <= bi < len(bin_weights):
            bin_weights[bi] += weights[i]

    peak_idx = np.argmax(bin_weights)
    peak_center = (bins[peak_idx] + bins[peak_idx + 1]) / 2.0

    # Peak + neighbors
    lo = max(0, peak_idx - 1)
    hi = min(len(bins) - 1, peak_idx + 2)
    in_peak_mask = (offsets >= bins[lo]) & (offsets < bins[hi])

    in_peak = offsets[in_peak_mask]
    total_count = int(in_peak_mask.sum())
    peak_weight = float(weights[in_peak_mask].sum())
    stddev = float(np.std(in_peak)) if len(in_peak) > 1 else 0.0

    return float(peak_center), total_count, stddev, peak_weight


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


def _compute_frame_distinctiveness(search_results) -> dict[int, float]:
    """
    For each query frame, compute how distinctive it is — i.e., how much
    its FAISS results favor one movie over others.

    A distinctive frame has its top matches concentrated in one movie.
    A generic frame has matches spread equally across many movies.

    Returns: {query_idx: distinctiveness_score} where 0.0 = generic, 1.0 = highly distinctive.

    Method: for each frame, find the best similarity per movie, then measure
    the gap between the best movie and the second-best movie.
    """
    distinctiveness = {}

    for q_idx, matches in enumerate(search_results):
        if not matches:
            distinctiveness[q_idx] = 0.0
            continue

        # Best similarity per movie for this frame
        best_per_movie = {}
        for movie_id, db_ts, similarity in matches:
            if movie_id not in best_per_movie or similarity > best_per_movie[movie_id]:
                best_per_movie[movie_id] = similarity

        if len(best_per_movie) <= 1:
            distinctiveness[q_idx] = 1.0
            continue

        # Sort by similarity descending
        sorted_sims = sorted(best_per_movie.values(), reverse=True)
        top1 = sorted_sims[0]
        top2 = sorted_sims[1]

        # Gap between best and second-best movie
        # Typical range: 0.00 (identical) to 0.05+ (very distinctive)
        # Normalize: gap of 0.02+ is considered fully distinctive
        gap = top1 - top2
        distinctiveness[q_idx] = min(1.0, gap / 0.02)

    return distinctiveness


def _gather_votes(timestamps, search_results, similarity_gate: float) -> tuple:
    """
    Collect offset votes from FAISS results with distinctiveness-weighted,
    rank-weighted, best-per-frame voting.

    Three mechanisms to suppress false positives:
    1. Frame distinctiveness: generic frames (dark scenes, transitions) that
       match every movie equally get low weight. Distinctive frames that
       clearly favor one movie get high weight. Self-calibrating — no
       hardcoded brightness or content thresholds.
    2. Rank weighting: top-ranked matches get exponentially more weight.
    3. Best-per-frame: each query frame gives each movie at most ONE vote.
    """
    # Compute per-frame distinctiveness from the raw FAISS results
    distinctiveness = _compute_frame_distinctiveness(search_results)

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
        # Weight = rank decay × frame distinctiveness
        # Distinctive frames get full weight, generic frames get reduced weight
        # but not silenced (they still contribute to offset clustering)
        rank_weight = RANK_WEIGHT_DECAY ** rank
        frame_dist = distinctiveness.get(q_idx, 0.5)
        weight = rank_weight * (0.2 + 0.8 * frame_dist)

        offset = db_ts - q_ts
        movie_data[movie_id]["offsets"].append(offset)
        movie_data[movie_id]["similarities"].append(similarity)
        movie_data[movie_id]["weights"].append(weight)
        movie_data[movie_id]["query_order"].append((q_idx, db_ts))

    n_distinctive = sum(1 for d in distinctiveness.values() if d > 0.5)
    n_generic = sum(1 for d in distinctiveness.values() if d <= 0.5)

    return movie_data, gated_count, total_count, n_distinctive, n_generic


def _longest_ordered_subsequence(pairs: list[tuple[int, float]],
                                   max_db_span: float = None) -> list[tuple[int, float]]:
    """
    Find the longest subsequence of (query_idx, db_time) pairs where both
    query_idx and db_time are monotonically increasing, AND the total db_time
    span is bounded (to prevent random matches across a whole movie from
    forming a spurious long chain).

    max_db_span: if set, the db_time range of the chain must be <= this value.
    """
    if len(pairs) <= 1:
        return pairs

    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    if max_db_span is None:
        # No constraint — standard LIS
        db_times = [p[1] for p in sorted_pairs]
        tails = []
        indices = []
        parents = [-1] * len(db_times)

        for i, dt in enumerate(db_times):
            pos = np.searchsorted(tails, dt, side='left')
            if pos == len(tails):
                tails.append(dt)
                indices.append(i)
            else:
                tails[pos] = dt
                indices[pos] = i
            if pos > 0:
                parents[i] = indices[pos - 1]

        result = []
        idx = indices[len(tails) - 1]
        while idx >= 0:
            result.append(sorted_pairs[idx])
            idx = parents[idx]
        result.reverse()
        return result

    # Span-constrained: try each pair as the start of a chain
    best_chain = []
    for start_i in range(len(sorted_pairs)):
        start_db = sorted_pairs[start_i][1]
        max_db = start_db + max_db_span

        # Filter to pairs within db_time range
        valid = [(qi, dt) for qi, dt in sorted_pairs[start_i:]
                 if dt >= start_db and dt <= max_db]

        if len(valid) <= len(best_chain):
            continue

        # Standard LIS within this window
        db_times = [p[1] for p in valid]
        tails = []
        indices = []
        parents = [-1] * len(db_times)

        for i, dt in enumerate(db_times):
            pos = np.searchsorted(tails, dt, side='left')
            if pos == len(tails):
                tails.append(dt)
                indices.append(i)
            else:
                tails[pos] = dt
                indices[pos] = i
            if pos > 0:
                parents[i] = indices[pos - 1]

        if len(tails) > len(best_chain):
            chain = []
            idx = indices[len(tails) - 1]
            while idx >= 0:
                chain.append(valid[idx])
                idx = parents[idx]
            chain.reverse()
            best_chain = chain

    return best_chain


def _score_candidates(movie_data: dict, n_query_frames: int,
                       query_duration: float = 10.0) -> list[dict]:
    """
    Score candidate movies using multi-scale offset histogram + temporal order.

    Tries multiple bin widths to find the best cluster for each movie,
    handling both tight matches (small bins) and dedup-spread matches (larger bins).
    """
    candidates = []
    for movie_id, data in movie_data.items():
        offsets = data["offsets"]
        sim_arr = np.array(data["similarities"])
        weight_arr = np.array(data["weights"])
        offset_arr = np.array(offsets)

        # Try multiple bin widths — pick the one that gives the best cluster
        best_result = None
        for bin_w in [1.5, 5.0, 15.0]:
            peak_sec, peak_count, stddev, _ = _offset_histogram(offsets, bin_w)
            if peak_count < MIN_VISUAL_MATCHES:
                continue

            in_peak_mask = np.abs(offset_arr - peak_sec) < bin_w * 2

            # Weighted similarity: distinctive frames contribute more
            peak_weights = weight_arr[in_peak_mask]
            peak_sims = sim_arr[in_peak_mask]
            if peak_weights.sum() > 0:
                mean_sim = float(np.average(peak_sims, weights=peak_weights))
            else:
                mean_sim = 0.0

            cluster_frac = min(1.0, peak_count / n_query_frames)
            concentration = peak_count / len(offsets) if len(offsets) > 0 else 0.0

            # Temporal order of peak matches
            peak_pairs = [(qi, dt) for (qi, dt), m in zip(data["query_order"], in_peak_mask) if m]
            temporal_consistency = _temporal_order_score(peak_pairs)

            # Reject poor temporal order
            if temporal_consistency < MIN_TEMPORAL_ORDER and peak_count >= 6:
                continue

            cluster_quality = (cluster_frac * concentration) ** 0.5
            visual_score = (0.30 * mean_sim
                            + 0.20 * cluster_quality
                            + 0.15 * cluster_frac
                            + 0.35 * temporal_consistency)

            if best_result is None or visual_score > best_result["visual_score"]:
                best_result = {
                    "movie_id": movie_id,
                    "offset_sec": peak_sec,
                    "visual_score": visual_score,
                    "mean_similarity": mean_sim,
                    "cluster_size": peak_count,
                    "concentration": concentration,
                    "temporal_consistency": temporal_consistency,
                    "offset_stddev": stddev,
                }

        if best_result:
            candidates.append(best_result)

    candidates.sort(key=lambda c: c["visual_score"], reverse=True)
    return candidates


def _verify_candidate(candidate: dict, query_embeddings: np.ndarray,
                       query_timestamps: list[float], db: Database) -> dict:
    """
    Two-pass verification: re-check candidate at higher FPS with multi-scale bins.
    """
    movie_id = candidate["movie_id"]
    predicted_offset = candidate["offset_sec"]

    search_results = db.search_visual(query_embeddings, VISUAL_TOP_K * 2)

    # Gather best-per-frame matches for this movie
    all_offsets = []
    all_sims = []
    all_weights = []
    all_pairs = []

    for q_idx, (q_ts, matches) in enumerate(zip(query_timestamps, search_results)):
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
        weight = RANK_WEIGHT_DECAY ** best_rank
        all_offsets.append(offset)
        all_sims.append(similarity)
        all_weights.append(weight)
        all_pairs.append((q_idx, db_ts))

    if not all_offsets:
        candidate["visual_score"] *= 0.3
        candidate["verified"] = False
        return candidate

    offset_arr = np.array(all_offsets)
    sim_arr = np.array(all_sims)
    weight_arr = np.array(all_weights)

    # Multi-scale: try bin widths, pick best
    best_score = 0
    best_result = None
    for bin_w in [1.5, 5.0, 15.0]:
        peak_sec, peak_count, stddev, _ = _offset_histogram(all_offsets, bin_w)
        if peak_count < 2:
            continue

        in_peak = np.abs(offset_arr - peak_sec) < bin_w * 2
        peak_weights = weight_arr[in_peak]
        peak_sims = sim_arr[in_peak]

        if peak_weights.sum() > 0:
            verified_sim = float(np.average(peak_sims, weights=peak_weights))
        else:
            continue

        peak_pairs = [(qi, dt) for (qi, dt), m in zip(all_pairs, in_peak) if m]
        verified_temporal = _temporal_order_score(peak_pairs)
        cluster_frac = min(1.0, peak_count / len(query_timestamps))
        concentration = peak_count / len(all_offsets)
        cluster_quality = (cluster_frac * concentration) ** 0.5

        score = (0.35 * verified_sim + 0.30 * cluster_quality
                 + 0.15 * cluster_frac + 0.20 * verified_temporal)

        if score > best_score:
            best_score = score
            best_result = (verified_sim, verified_temporal, concentration, peak_count)

    if best_result is None:
        candidate["verified"] = False
        return candidate

    verified_sim, verified_temporal, concentration, v_matches = best_result

    candidate["visual_score"] = max(candidate["visual_score"], best_score)
    candidate["mean_similarity"] = max(candidate["mean_similarity"], verified_sim)
    candidate["temporal_consistency"] = max(candidate["temporal_consistency"], verified_temporal)
    candidate["concentration"] = max(candidate.get("concentration", 0), concentration)
    candidate["verified"] = True
    candidate["verified_matches"] = v_matches

    return candidate


def _scene_candidate_movies(query_embeddings: np.ndarray, db, top_k: int, max_movies: int) -> list[int] | None:
    """
    Coarse pass: search scene index, return top movie IDs by vote count.
    Returns None if scene index is empty (fallback to frame-only search).
    """
    scene_results = db.search_scenes(query_embeddings, top_k)

    # Check if we got any results
    if not any(scene_results):
        return None

    # Count votes per movie
    movie_votes = defaultdict(float)
    for matches in scene_results:
        for movie_id, start, end, similarity in matches:
            movie_votes[movie_id] += similarity

    # Return top movies by total similarity
    sorted_movies = sorted(movie_votes.items(), key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in sorted_movies[:max_movies]]


def _filter_to_movies(search_results: list, candidate_movie_ids: set) -> list:
    """Filter FAISS search results to only include matches from candidate movies."""
    filtered = []
    for frame_matches in search_results:
        filtered.append([m for m in frame_matches if m[0] in candidate_movie_ids])
    return filtered


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

        # --- Scene-level coarse search (if available) ---
        candidate_movie_ids = _scene_candidate_movies(
            embeddings, db, SCENE_TOP_K, SCENE_CANDIDATE_MOVIES
        )
        if candidate_movie_ids:
            print(f"  Scene search: narrowed to {len(candidate_movie_ids)} candidate movies")

        # --- FAISS frame-level search ---
        t1 = time.time()
        search_results = db.search_visual(embeddings, VISUAL_TOP_K)
        if candidate_movie_ids:
            search_results = _filter_to_movies(search_results, set(candidate_movie_ids))
        print(f"  FAISS search: {time.time()-t1:.2f}s")

        # --- Rank-weighted offset voting ---
        movie_data, gated, total, n_distinctive, n_generic = _gather_votes(timestamps, search_results, SIMILARITY_GATE)
        print(f"  Votes: {total} total | Frames: {n_distinctive} distinctive, {n_generic} generic")

        # --- Score candidates ---
        query_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 10.0
        candidates = _score_candidates(movie_data, len(visual_fps), query_duration)

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
        # Scoring (visual_score) ranks candidates. Confidence answers a different
        # question: "should we trust this result?" — using only signals that are
        # independent of index density and number of indexed movies.
        #
        # Scale-independent signals:
        # 1. Mean similarity of peak matches (CLIP cosine) — how visually close?
        # 2. Temporal consistency — do matches preserve playback order?
        # 3. Peak match ratio — what fraction of query frames found a match
        #    at the winning offset? Normalized by expected hit rate (index_fps / query_fps)
        #    so deduplication doesn't artificially lower it.
        best = candidates[0]
        if len(candidates) >= 2:
            ratio = best["visual_score"] / max(candidates[1]["visual_score"], 1e-10)
        else:
            ratio = float("inf")

        # 1. Similarity quality: CLIP cosine 0.82 = weak, 0.92+ = strong
        sim_norm = min(1.0, max(0.0, (best["mean_similarity"] - 0.82) / 0.10))

        # 2. Temporal order
        temporal_conf = best["temporal_consistency"]

        # 3. Match coverage: what fraction of query frames matched at the peak?
        #    Normalize by expected coverage given index density.
        #    At 2fps index / 4fps query, we expect ~50% coverage at best.
        #    With dedup (~60% kept), expect ~30% coverage.
        #    So 20%+ coverage is good signal.
        coverage = min(1.0, best["cluster_size"] / max(len(visual_fps) * 0.20, 1))

        confidence = 0.40 * sim_norm + 0.30 * temporal_conf + 0.30 * coverage

        # Only penalize if runner-up is within a tight margin
        # This is genuinely suspicious regardless of scale
        if ratio < 1.05:
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
