# Architecture

## Overview

Visual-only fingerprinting engine with two-pass matching.

```
                     INDEX PIPELINE
                     ==============

  Movie file (.mp4)
       |
       +---> ffmpeg (frame extraction, 2 fps, scaled to 640px max)
               |
               +---> Frame normalization pipeline:
               |       1. Letterbox detection & crop
               |       2. Center-crop 90%
               |       3. CLAHE contrast normalization
               |       4. Resize to 224x224
               |
               +---> pHash (64-bit DCT-based perceptual hash)
               |       +---> Store in PostgreSQL
               |
               +---> CLIP ViT-B/32 embedding (512-dim, L2-normalized)
                       +---> Store in FAISS IndexFlatIP


                     QUERY PIPELINE (Two-Pass)
                     =========================

  Query clip (5-10 seconds)
       |
       +---> [Pass 1: Candidate Selection] (4 fps)
       |       |
       |       +---> FAISS top-20 nearest neighbors per frame
       |       +---> Rank-weighted, best-per-frame voting
       |       |       (each frame gives each movie at most 1 vote,
       |       |        weighted by rank: 1.0, 0.5, 0.25, ...)
       |       +---> Offset histogram voting (1.5s bins)
       |       +---> Concentration scoring (peak votes / total votes)
       |       +---> Temporal order enforcement
       |       +---> Score = 0.35*sim + 0.30*cluster_quality + 0.15*cluster_frac + 0.20*temporal
       |
       +---> [Pass 2: Verification] (8 fps, only if top-2 are close)
               |
               +---> Re-extract frames at higher FPS
               +---> Re-score top candidates with same pipeline
               +---> Verify offset prediction holds with more data
               |
               +---> Return: movie_id, timestamp, confidence
```

## Why Visual is Primary

| Scenario | Audio | Visual |
|----------|-------|--------|
| Same encode, same language | Works | Works |
| Different resolution (480p vs 1080p) | Works | Works (normalized to 224x224) |
| Dubbed audio (different language) | **Fails** | Works |
| Phone recording of screen | Degraded | Works (CLAHE + center-crop) |
| Re-encoded (different codec/bitrate) | Works | Works |
| Different aspect ratio / letterboxing | Works | Works (letterbox detection) |
| Ambient noise | **Degraded** | Works |
| Different cut (director's cut) | Mostly works | May diverge at added scenes |

## Offset Histogram Voting

The core matching mechanism (used by both audio and visual):

```
Query frame at 0.0s  --> DB match at 2400.0s  --> offset = 2400.0
Query frame at 0.5s  --> DB match at 2400.5s  --> offset = 2400.0
Query frame at 1.0s  --> DB match at 2401.0s  --> offset = 2400.0
Query frame at 1.5s  --> DB match at 2401.5s  --> offset = 2400.0
Query frame at 2.0s  --> DB match at 987.3s   --> offset = 985.3  (false positive)

Histogram:
  offset=2400.0  |████████████████|  count=4  <-- PEAK (true match)
  offset=985.3   |████|              count=1  (noise)
```

The peak in the offset histogram gives both the movie and the exact timestamp. False positives produce uniformly distributed offsets; true matches cluster tightly.

## Frame Normalization Pipeline

Handles real-world capture conditions:

```
Raw frame (any resolution)
    |
    +---> Letterbox detection: find rows/cols with mean brightness < 15
    |     Crop to content area (removes black bars from aspect ratio mismatch)
    |
    +---> Center-crop 90%: removes edge artifacts from phone capture
    |     (glare, partial screen, phone bezels)
    |
    +---> CLAHE (Contrast Limited Adaptive Histogram Equalization)
    |     Normalizes brightness/contrast across different screens,
    |     ambient lighting, and camera exposure settings
    |
    +---> Resize to 224x224: resolution normalization
          480p, 720p, 1080p, 4K all produce identical-size inputs
```

## Database Design

```
PostgreSQL                           FAISS (file-based)
==========                           ==================

movies                               IndexFlatIP (512-dim)
  movie_id (PK)                        |
  title                                +-- vector[0] -> id_map[0] = (movie_id, timestamp)
  file_path                            +-- vector[1] -> id_map[1] = (movie_id, timestamp)
  duration_sec                         +-- ...
  created_at                           +-- vector[N] -> id_map[N] = (movie_id, timestamp)

audio_fingerprints                   Stored as:
  hash_value (indexed)                 data/faiss_visual.index
  time_offset                          data/faiss_id_map.pkl
  movie_id (FK)

visual_fingerprints
  id (PK)
  movie_id (FK)
  timestamp_sec
  phash (TEXT, hex-encoded)
```

The `DatabaseBackend` abstract class allows swapping to any storage:
- PostgreSQL + pgvector (vectors in DB)
- Redis (fast hash lookups) + FAISS
- SQLite (zero-dependency) + FAISS
- MongoDB + Milvus (managed scale)

## Performance (3 movies, 412 min total)

| Metric | Value |
|--------|-------|
| Movies indexed | 3 |
| Total frames indexed | 49,448 (at 2 fps) |
| Indexing time per movie | ~240-400s (CLIP on CPU) |
| FAISS index size | 97.2 MB |
| PostgreSQL size | 14.4 MB |
| **Total storage** | **111.6 MB (~0.27 MB/min)** |
| Query time (10s clip, clear winner) | 1.4-2.2s |
| Query time (10s clip, with verification) | 4.7-7.2s |
| FAISS search time | 0.03-0.04s |
| Timestamp accuracy (clean) | < 0.2s delta |
| Identification accuracy | 6/6 (100%) phone captures + clean extractions |
| Confidence (horizontal phone) | 0.661-0.861 |
| Confidence (vertical phone) | 0.306-0.805 |
| Runner-up ratio (horizontal) | 1.47x - inf (3/3 have no runner-up) |
| Runner-up ratio (vertical) | 1.03x - 1.30x |
