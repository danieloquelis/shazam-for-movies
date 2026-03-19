# Changelog

## MVP v0.3 — 3-Movie Index + Cross-Movie Tests (Current)

**Date:** 2026-03-19

Indexed 3 movies (49,448 frames total) and ran comprehensive tests including clean extractions and real phone captures across all movies. **7/7 correct identifications.**

### Index

| Movie | Duration | Frames | Index Time |
|-------|----------|--------|------------|
| Project Almanac | 106 min | 12,746 | 240s |
| Harry Potter and the Goblet of Fire | 157 min | 18,850 | 347s |
| X-Men: Days of Future Past | 149 min | 17,852 | 402s |
| **Total** | **412 min** | **49,448** | |

### Storage (3 movies)

| Component | Size |
|-----------|------|
| FAISS index | 97.2 MB |
| PostgreSQL | 14.4 MB |
| **Total** | **111.6 MB** (~0.27 MB/min) |

### Clean extraction tests (10s clips)

| Movie | Timestamp | Delta | Confidence | Time |
|-------|-----------|-------|------------|------|
| Project Almanac @54:05 | Correct | 0.1s | 0.902 | 1.2s |
| Harry Potter @57:18 | Correct | 0.0s | 0.882 | 1.0s |
| X-Men @01:57 | Correct | 0.1s | 0.917 | 1.1s |

### Real phone capture tests (all no audio)

| Clip | Conditions | Matched | Confidence | Cluster | Time |
|------|-----------|---------|------------|---------|------|
| test1.MOV | Vertical, partial screen (Project Almanac) | Project Almanac @07:30 | 0.360 | 27 | 2.4s |
| test2.MOV | Landscape, good (Project Almanac) | Project Almanac @07:51 | 0.861 | 90 | 1.7s |
| test1-vertical.mov | Vertical (Harry Potter) | Harry Potter @01:05:09 | 0.861 | 62 | 1.6s |
| test2-vertical.MOV | Vertical, less screen (X-Men) | X-Men @00:54:52 | 0.406 | 35 | 1.9s |
| test2-horizontal.MOV | Horizontal, good (X-Men) | X-Men @00:54:38 | 0.882 | 61 | 1.4s |
| test1-horizontal (Harry Potter) | Horizontal, good | Harry Potter @38:27 | 0.808 | 31 | 1.4s |

### Key findings

- **7/7 correct identifications** — the engine always picks the right movie
- **Orientation matters significantly:** same scene vertical (0.406) vs horizontal (0.882). More screen = more CLIP features = better separation from false positives
- **Confidence drops with more movies:** test1.MOV went from 0.731 (1 movie) to 0.360 (3 movies) because runner-up movies now score closer. The correct movie still wins, but the gap narrows.
- **Runner-up problem:** With 3 movies, the 2nd-best candidate often scores 0.6-0.7, triggering the confidence penalty. This will worsen with more movies.
- **Resolution differences handled well:** Harry Potter is 1280x536 while others are 1920x800. No impact on matching quality.

### Scaling concerns

As more movies are indexed:
- **What stays stable:** Correct movie identification (offset voting), timestamp accuracy, query speed (FAISS is fast)
- **What degrades:** Confidence scores (more runner-ups), vertical/partial captures (first to fail)
- **Estimated limits:** ~100 movies before vertical captures start failing, horizontal should hold to 500+
- **Mitigation:** Screen detection/perspective rectification, stricter re-ranking, FAISS IVF+PQ

---

## MVP v0.2.1 — Initial Phone Capture Tests

**Date:** 2026-03-19

First real-world tests with phone recordings against a single indexed movie (Project Almanac). Both clips had no audio.

| Clip | Capture | Confidence | Similarity | Cluster |
|------|---------|------------|------------|---------|
| test1.MOV | Vertical, partial screen | 0.731 | 0.848 | 29 |
| test2.MOV | Landscape, more screen | 0.861 | 0.885 | 90 |

Confirmed visual-only approach works on phone captures with no audio. Frame normalization pipeline (letterbox removal, CLAHE, center-crop) is critical for robustness.

---

## MVP v0.2 — Visual-Only Engine

**Date:** 2026-03-19

Stripped audio fingerprinting in favor of a visual-only approach. Audio was contributing almost nothing (0.075 score vs visual's 0.896) while consuming 5.5x more storage.

### What changed

- Removed audio fingerprinting from the matching pipeline
- Visual is now the sole signal (CLIP embeddings + FAISS + offset voting)
- Dropped `audio_fingerprints` table from PostgreSQL (was 114 MB per movie)
- Removed `scipy` and `tqdm` from dependencies
- Dynamic fusion logic replaced with direct visual scoring
- Confidence scores improved significantly (no longer diluted by weak audio)

### Results — Project Almanac (106 min)

| Test | Clip | Expected | Got | Delta | Confidence | Time |
|------|------|----------|-----|-------|------------|------|
| Fixed @50:00 | 10s | 3000.0s | 3000.0s | **0.0s** | 0.876 | 1.1s |
| Fixed @40:00 | 10s | 2400.0s | 2400.0s | **0.0s** | 0.896 | 1.0s |
| Random @01:26:43 | 10s | 5203.9s | 5203.8s | **0.1s** | 0.899 | 1.0s |
| Random @01:00:53 | 5s | 3653.6s | 3653.3s | **0.2s** | 0.886 | 0.7s |

### Storage per movie

| Component | v0.1 (audio+visual) | v0.2 (visual-only) |
|-----------|--------------------|--------------------|
| PostgreSQL | 123 MB | **9.6 MB** |
| FAISS index | 25 MB | 25 MB |
| FAISS ID map | 162 KB | 162 KB |
| **Total** | **~148 MB** | **~27 MB** |

### Scaling projections (visual-only)

| Movies | FAISS index | PostgreSQL | Total |
|--------|-------------|------------|-------|
| 1 | 25 MB | 10 MB | ~35 MB |
| 10 | 250 MB | 100 MB | ~350 MB |
| 100 | 2.5 GB | 1 GB | ~3.5 GB |
| 1,000 | 25 GB | 10 GB | ~35 GB |

At 1,000+ movies, switch FAISS from `IndexFlatIP` to `IndexIVFPQ` for sub-linear search.

---

## MVP v0.1 — Audio + Visual Engine

**Date:** 2026-03-19

Initial working prototype with dual audio+visual fingerprinting.

### What was built

- **Audio:** Shazam-style constellation maps (Wang 2003) — spectrogram peak detection, combinatorial hashing, offset histogram voting. Stored 1.7M hashes per movie in PostgreSQL.
- **Visual:** CLIP ViT-B/32 embeddings (512-dim) + pHash, searched via FAISS IndexFlatIP. Frame normalization pipeline (letterbox removal, CLAHE, center-crop).
- **Fusion:** Dynamic late fusion — when audio and visual agree, 40/60 weighting. When they disagree (dub scenario), 10/90 favoring visual.
- **DB:** PostgreSQL (metadata + hashes) + FAISS (vectors). Abstract `DatabaseBackend` interface for future portability.

### Key design decisions

1. **Visual-primary architecture** — visual is dub-invariant, resolution-agnostic, encode-invariant. Audio is a bonus signal.
2. **Frame normalization** — letterbox detection, center-crop 90%, CLAHE contrast normalization, resize to 224x224. Handles phone-camera captures.
3. **Offset histogram voting** — same mechanism for both audio and visual matching. Peak in the offset histogram = movie + timestamp.
4. **Streaming indexer** — processes frames in batches of 120 to avoid OOM on long movies.
5. **Lazy FAISS loading** — avoids OMP thread conflict with PyTorch on macOS.

### Results

All tests passed with <0.5s timestamp accuracy on a 106-minute movie. Query time <1.5s for 10-second clips.

### Issues encountered

- **OMP segfault:** FAISS and PyTorch both bundle OpenMP. Loading FAISS index from disk then PyTorch model causes a deadlock. Fixed by setting `OMP_NUM_THREADS=1` and lazy-loading FAISS after CLIP.
- **pHash overflow:** 64-bit perceptual hashes exceeded PostgreSQL BIGINT range. Fixed by storing as hex TEXT.
- **Audio peak detection:** Absolute dB threshold (10 dB) found zero peaks because spectrogram values were all negative. Fixed with adaptive percentile-based threshold (80th percentile).
- **Memory on long movies:** Accumulating 12K+ fingerprints in memory caused crashes. Fixed by streaming batches to DB incrementally.

---

## Next steps

- [ ] Test with a second movie (cross-movie discrimination)
- [ ] Test with degraded clips (lower resolution, re-encoded, simulated phone capture)
- [ ] Benchmark with 10+ movies indexed
- [ ] Evaluate IVF+PQ index for FAISS at scale
- [ ] Consider adding audio back as optional secondary signal for non-dub scenarios
