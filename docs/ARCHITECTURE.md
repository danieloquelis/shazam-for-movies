# Architecture

## Overview

Visual-primary, audio-bonus fingerprinting engine with dynamic late fusion.

```
                     INDEX PIPELINE
                     ==============

  Movie file (.mp4)
       |
       +---> ffmpeg (audio extraction, mono 11025 Hz)
       |       |
       |       +---> STFT spectrogram (1024-sample Hamming window)
       |       +---> Peak detection (adaptive percentile threshold)
       |       +---> Constellation map (sparse time-frequency peaks)
       |       +---> Combinatorial hashing (anchor + target pairs)
       |       +---> Store: hash_value -> (movie_id, time_offset) in PostgreSQL
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


                     QUERY PIPELINE
                     ==============

  Query clip (5-10 seconds)
       |
       +---> Audio fingerprinting (same pipeline as indexing)
       |       |
       |       +---> Hash lookup in PostgreSQL
       |       +---> Offset voting: db_time - query_time = alignment
       |       +---> Histogram peak = candidate (movie_id, timestamp)
       |       +---> Audio score = peak_matches / total_query_hashes
       |
       +---> Visual fingerprinting (same pipeline, 4 fps)
       |       |
       |       +---> FAISS top-20 nearest neighbors per frame
       |       +---> Offset voting (same mechanism)
       |       +---> Visual score = 0.45*similarity + 0.35*cluster_frac + 0.20*temporal_order
       |
       +---> Dynamic fusion
               |
               +---> If audio & visual agree: 40% audio + 60% visual
               +---> If they disagree (dub): 10% audio + 90% visual
               +---> If only one signal: use what's available
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
| Query time (10s clip) | 1.0-1.4s |
| Query time (5s clip) | ~0.7s |
| FAISS search time | 0.03-0.04s |
| Timestamp accuracy (clean) | < 0.2s delta |
| Identification accuracy | 7/7 (100%) across clean + phone captures |
| Confidence (horizontal phone) | 0.808-0.882 |
| Confidence (vertical phone) | 0.360-0.861 (depends on screen visibility) |
