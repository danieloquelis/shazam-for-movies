# Architecture

A technical reference for how the system is wired up. The user-facing
explanation lives in the top-level [README](../README.md); this document is
where you go when you want to know which process talks to which datastore,
why there are two of them, and how the pieces deploy.

For the **mobile + backend product plan** (build phases, scan modes,
streaming-vs-record-and-upload), see [`CLIENT.md`](CLIENT.md).

---

## System overview

Three components, one data plane:

```
                ┌─────────────────────────────────────────────┐
                │                  Engine                     │
                │  (Python library — engine/)                 │
                │                                             │
                │   indexer.py  ─┐                            │
                │                ├── visual_fingerprint.py    │
                │   matcher.py  ─┘                            │
                │                                             │
                │   db.py  (storage abstraction)              │
                └──────────────┬──────────────┬───────────────┘
                               │              │
                               ▼              ▼
                ┌──────────────────┐    ┌─────────────────────┐
                │   PostgreSQL     │    │   FAISS (file)      │
                │   metadata +     │    │   data/*.index      │
                │   pHashes        │    │   vectors           │
                └──────────────────┘    └─────────────────────┘
                          ▲
                          │ used by
                          │
              ┌───────────┴────────────┐
              │                        │
       ┌──────────────┐          ┌─────────────┐
       │     CLI      │          │  Backend    │
       │   main.py    │          │  FastAPI    │
       │              │          │  backend/   │
       │  index /     │          │             │
       │  query /     │          │  /healthz   │
       │  test        │          │  /query     │
       └──────────────┘          └──────┬──────┘
                                        │
                                        ▼
                                 ┌──────────────┐
                                 │  Mobile app  │
                                 │  (Expo)      │
                                 │  mobile/     │
                                 └──────────────┘
```

The **engine** is a regular Python library. Both the **CLI** (`main.py`) and
the **HTTP backend** (`backend/main.py`) import it directly — they're two
different drivers around the same matching code.

---

## Storage: why two stores

This is the question that surprises most readers, so it gets its own section.

The system uses **PostgreSQL** *and* **FAISS** simultaneously. They are not
redundant — each is good at a different operation:

| Operation | Right tool | Why |
|---|---|---|
| Find the 20 most visually similar frames to this query frame | **FAISS** | Vector search on hundreds of thousands of 512-dim embeddings is what FAISS exists for. |
| Resolve `movie_id = 13` → "Harry Potter and the Goblet of Fire" | **Postgres** | Standard relational lookup. |
| Look up an exact perceptual-hash match (debugging, dedup) | **Postgres** | Indexed text column, microsecond lookups. |
| Persist the catalogue across restarts; coordinate concurrent writes | **Postgres** | Transactions and replication, not a `.pkl` file. |

The pattern is: **FAISS finds the vector, the FAISS id-map tells you which
`(movie_id, timestamp)` it came from, and Postgres tells you what `movie_id 13`
means.**

### What's stored where

```
┌─────────────────────────────────────────────┐
│              PostgreSQL                     │  ← the catalogue
├─────────────────────────────────────────────┤
│                                             │
│  movies                                     │
│    movie_id  (PK)                           │
│    title                                    │
│    file_path                                │
│    duration_sec                             │
│    created_at                               │
│                                             │
│  visual_fingerprints                        │
│    id        (PK)                           │
│    movie_id  (FK)                           │
│    timestamp_sec                            │
│    phash     (TEXT, hex-encoded 64-bit)     │
│                                             │
│  scenes                                     │
│    id           (PK)                        │
│    movie_id     (FK)                        │
│    start_sec                                │
│    end_sec                                  │
│    n_keyframes                              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│            FAISS (file-based)               │  ← the vectors
├─────────────────────────────────────────────┤
│                                             │
│  data/faiss_visual.index                    │
│    one 512-float vector per indexed frame   │
│    (IndexFlatIP — exact inner-product       │
│     search, L2-normalized → cosine sim)     │
│                                             │
│  data/faiss_id_map.pkl                      │
│    parallel list, index_position → tuple    │
│    (movie_id, timestamp_sec)                │
│                                             │
│  data/faiss_scene.index                     │
│    one 512-float vector per scene           │
│    (mean of frames in the scene)            │
│                                             │
│  data/faiss_scene_id_map.pkl                │
│    index_position → (movie_id, start, end)  │
└─────────────────────────────────────────────┘
```

### Sizes from a real index (7 movies)

| Store | Rows / vectors | Size on disk |
|---|---|---|
| Postgres `movies` | 7 | trivial |
| Postgres `visual_fingerprints` | 116,198 | ~14 MB |
| Postgres `scenes` | 29,370 | ~3 MB |
| FAISS `faiss_visual.index` | 116,198 vectors × 512 floats | ~238 MB |
| FAISS `faiss_id_map.pkl` | 116,198 tuples | ~1.5 MB |
| FAISS `faiss_scene.index` | ~29,370 vectors × 512 floats | ~60 MB |
| FAISS `faiss_scene_id_map.pkl` | ~29,370 tuples | ~650 KB |

Vectors dominate the storage budget, by ~20×. That's why FAISS exists.

### `data/` is gitignored on purpose

The folder is reproducible: re-run `python main.py index ...` against your
source movies and you'll get a functionally equivalent index. Committing 286 MB
of binary FAISS files would be wasteful and would also embed an implicit list
of which movies were indexed.

The `DatabaseBackend` abstract class in `engine/db.py` is designed so the whole
storage stack can be swapped — Redis + FAISS, SQLite + FAISS, Postgres +
pgvector, MongoDB + Milvus, etc. The current implementation is the
Postgres-plus-FAISS one.

---

## Index pipeline

```
Movie file (.mp4)
  │
  ├─ ffmpeg: extract frames at 2 fps, scale to 640px max
  │
  ├─ Frame normalization (see "Frame normalization" below)
  │
  ├─ For each frame, compute:
  │     • pHash (64-bit DCT-based perceptual hash)
  │     • CLIP ViT-B/32 embedding (512-dim, L2-normalized)
  │
  ├─ Streaming write in batches (FRAME_BATCH_SIZE=120 frames):
  │     • pHash + (movie_id, timestamp) → Postgres visual_fingerprints
  │     • Embedding + (movie_id, timestamp) → FAISS visual.index + id_map
  │
  ├─ Scene boundaries detected from embedding similarity drops:
  │     • Scene metadata → Postgres scenes
  │     • Scene mean-embedding → FAISS scene.index + scene_id_map
  │
  └─ Final save: faiss.write_index() persists both index files to data/
```

A 2-hour movie indexes in ~4-5 minutes on CPU and adds ~30 MB of vectors plus
~3 MB of metadata.

---

## Query pipeline (two-pass)

```
Query clip (5-10 seconds)
  │
  ├─ ffmpeg: extract at 4 fps, normalize identically to indexing
  │
  ├─ CLIP-embed each frame
  │
  │  ┌───────────────────────────────────────────┐
  │  │  Pass 1 — Candidate selection             │
  │  │                                           │
  │  │  1. (Optional) FAISS scene index          │
  │  │     Top-K scenes → narrow to ~5 movies    │
  │  │                                           │
  │  │  2. FAISS frame index, top-20 per frame   │
  │  │     (filtered to scene candidates if any) │
  │  │                                           │
  │  │  3. Rank-weighted, best-per-frame voting  │
  │  │     Each frame contributes at most one    │
  │  │     vote per movie, weighted by rank      │
  │  │     (1.0, 0.5, 0.25, ...).                │
  │  │                                           │
  │  │  4. Frame-distinctiveness weighting       │
  │  │     Generic frames (dark scenes, etc.)    │
  │  │     down-weighted; distinctive frames     │
  │  │     get full vote weight.                 │
  │  │                                           │
  │  │  5. Multi-scale offset histogram          │
  │  │     Try bin widths 1.5 / 5 / 15 sec;      │
  │  │     pick the bin width with the strongest │
  │  │     cluster per candidate.                │
  │  │                                           │
  │  │  6. Temporal-order check                  │
  │  │     Reject candidates where matches are   │
  │  │     out of playback order.                │
  │  │                                           │
  │  │  7. Score per candidate movie             │
  │  │     0.30·sim + 0.20·cluster_quality       │
  │  │     + 0.15·cluster_frac + 0.35·temporal   │
  │  └───────────────────────────────────────────┘
  │                  │
  │                  ▼
  │  ┌───────────────────────────────────────────┐
  │  │  Pass 2 — Verification                    │
  │  │  (only if top-2 are within ~1.875×)       │
  │  │                                           │
  │  │  1. Re-extract clip at 8 fps              │
  │  │  2. Re-score top N candidates with the    │
  │  │     same machinery; require the predicted │
  │  │     offset to hold up with more data.     │
  │  └───────────────────────────────────────────┘
  │                  │
  │                  ▼
  └─ Compute confidence from scale-independent signals:
       0.40·similarity_norm + 0.30·temporal + 0.30·coverage
       (halved if runner-up is within 1.05× — suspicious tie)
```

### Why the offset histogram works

```
Query frame at 0.0s  →  DB match at 2400.0s  →  offset = 2400.0
Query frame at 0.5s  →  DB match at 2400.5s  →  offset = 2400.0
Query frame at 1.0s  →  DB match at 2401.0s  →  offset = 2400.0
Query frame at 1.5s  →  DB match at 2401.5s  →  offset = 2400.0
Query frame at 2.0s  →  DB match at  987.3s  →  offset =  985.3   (noise)

Histogram:
  offset = 2400.0  ████████████████   count=4   ← PEAK (true match)
  offset =  985.3  ████               count=1   (false positive)
```

The peak gives the movie *and* the timestamp simultaneously. False positives
produce scattered offsets; true matches cluster tightly. This is the same
trick Shazam uses for audio, applied to dense visual embeddings.

---

## Frame normalization

```
Raw frame (any resolution)
    │
    ├─ Letterbox detection: drop rows/cols with mean brightness < 15
    │     Removes black bars from aspect-ratio mismatch.
    │
    ├─ Center-crop 90%: removes edge artifacts from phone capture
    │     (glare, partial screen, phone bezels).
    │
    ├─ CLAHE (Contrast Limited Adaptive Histogram Equalization)
    │     Normalizes brightness/contrast across different screens,
    │     ambient lighting, and camera exposure settings.
    │
    └─ Resize to 224×224
          480p, 720p, 1080p, 4K all produce identical-size inputs.
```

Optionally, `--screen` enables a screen-detection pre-pass that finds and
crops the laptop/monitor region before the standard normalization. Useful
for phone captures of off-axis screens.

---

## Backend: request lifecycle

```
Mobile app              Backend (FastAPI)              Engine + datastores
──────────              ─────────────────              ───────────────────

POST /query          ──► Validate x-api-key
multipart            ──► Stream upload to temp file
file=clip.mov            (reject at 20 MB cap)
                     ──► engine.matcher.match_clip(temp_path)
                            │
                            ├─► extract & normalize frames (ffmpeg)
                            ├─► CLIP-embed (model held in RAM)
                            ├─► faiss_scenes.search(...)        ──► FAISS file
                            ├─► faiss_visual.search(...)        ──► FAISS file
                            ├─► offset voting + scoring
                            ├─► [verification pass]
                            └─► db.get_movie(winner.movie_id)   ──► Postgres
                     ◄── MatchResult
                     ──► serialize → JSON
                     ──► delete temp file
                ◄──  200 OK { title, timestamp_sec, ... }
```

Two things worth knowing:

1. **CLIP and the FAISS indexes are held in RAM** by the backend process for
   its entire lifetime. The first request after boot pays the load cost
   (~10-15s on cold start, including the one-time CLIP model download).
   Every subsequent request reuses the warm state and runs in ~1-2s.

2. **The engine code path is the same as the CLI.** `match_clip` doesn't
   know whether it was invoked from a shell or an HTTP handler. This means
   any matcher improvement automatically benefits both.

---

## Deployment shape

The engine has two characteristics that constrain where it can run:

- **CLIP model takes ~10-15s to load** (first-time download is ~350 MB).
- **FAISS indexes live in RAM** for fast search (~300 MB for a 7-film index;
  scales linearly with frames).

Together, these rule out **serverless functions** (Vercel, Lambda):
per-request cold starts would dominate latency and the 50 MB function bundle
limit can't fit PyTorch.

The right shape is a **long-lived container** holding the model and indexes
warm. Concretely:

```
┌────────────────────────────────────────┐
│ Long-lived VM / container (Fly.io)     │
│   ~2 GB RAM                            │
│  ┌──────────────────────────────────┐  │
│  │ FastAPI process                  │  │
│  │  • CLIP loaded once at boot      │  │
│  │  • FAISS indexes mmap'd          │  │
│  │  • imports engine.matcher        │  │
│  └──────────────────────────────────┘  │
└──────────────────┬─────────────────────┘
                   │ private network
                   ▼
        ┌──────────────────────┐
        │ Managed PostgreSQL   │
        │ (Fly Postgres etc.)  │
        └──────────────────────┘
```

A persistent volume mounted at `/app/data` keeps the FAISS files across
redeploys.

---

## Why visual is primary (not audio)

The system used to combine audio + visual fingerprinting (v0.1). Audio was
dropped in v0.2. The matrix below explains why:

| Scenario | Audio | Visual |
|---|---|---|
| Same encode, same language | Works | Works |
| Different resolution (480p vs 1080p) | Works | Works (224×224 normalize) |
| **Dubbed audio (different language)** | **Fails** | **Works** |
| **Phone recording of screen** | Degraded | Works |
| Re-encoded (different codec/bitrate) | Works | Works |
| Different aspect ratio / letterboxing | Works | Works (letterbox crop) |
| Ambient noise on capture | Degraded | Works |
| Different cut (director's vs theatrical) | Mostly works | May diverge at added scenes |

The bolded rows are the use cases that motivated the project. Audio simply
can't serve them.

---

## Performance reference

Numbers from the v0.4 benchmark run (3 movies, 49,448 frames). See
[`CHANGELOG.md`](CHANGELOG.md) for newer numbers as the index grows.

| Metric | Value |
|---|---|
| Frames indexed | 49,448 (at 2 fps) |
| Indexing time per movie | 240–400 s (CLIP on CPU) |
| FAISS visual index size | 97 MB (3 movies) |
| Postgres size | ~14 MB |
| Total storage per indexed minute | ~0.27 MB |
| Query time, clear winner | 1.4–2.2 s |
| Query time, with verification pass | 4.7–7.2 s |
| FAISS search time (excluded) | 0.03–0.04 s |
| Timestamp accuracy on clean clips | <0.2 s |
| Identification accuracy | 6/6 phone captures + clean extractions |
| Confidence (horizontal phone) | 0.66–0.86 |
| Confidence (vertical phone) | 0.31–0.81 |
