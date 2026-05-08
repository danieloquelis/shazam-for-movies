# Shazam for Movies

> Point your phone at any screen playing a movie. In about a second, get back the title and the exact timestamp of the scene.

This project is a visual fingerprinting engine that does for movies what Shazam does for songs. Capture a 5–10 second clip with any camera — phone, webcam, screen recording — and the engine identifies which film is playing and where in the runtime you are, down to a fraction of a second.

It is **language-agnostic** by design: dubs, subtitles, and silenced playback all work, because the matching is done from pixels, not audio.

---

## Why this exists

Audio fingerprinting (Shazam, AcoustID) is a solved problem for music. Movies are harder:

- The same film ships with **dozens of dubs and subtitle tracks**. Audio fingerprints don't transfer between them.
- People watch movies **muted** — on planes, in waiting rooms, on a TV across a noisy room.
- Phone-recording a screen introduces **glare, skew, moiré, partial framing, and weird aspect ratios** that wreck audio and challenge naive image search.

A purely visual approach sidesteps all of that. The frames of a given film are the same whether you're watching the German dub, the French dub, or a silent screening — so one visual index covers every release of a given cut.

---

## How it works

The system has two pipelines: an offline **indexer** that ingests movies, and an online **matcher** that identifies clips.

### 1. Indexing (offline, once per movie)

```
movie.mp4
  │
  ├─ ffmpeg samples 2 frames/sec
  │
  ├─ Each frame is normalized:
  │     letterbox crop → 90% center crop → CLAHE contrast → 224×224
  │
  ├─ Each frame produces:
  │     • a 64-bit perceptual hash (pHash)
  │     • a 512-dim CLIP ViT-B/32 embedding
  │
  └─ Embeddings are pushed into a FAISS vector index;
     metadata (movie_id, timestamp) goes into PostgreSQL.
```

A 2-hour movie indexes in roughly 4–5 minutes on CPU and adds ~30 MB to the index.

### 2. Querying (online, ~1–2 seconds)

```
clip.mp4 (5–10 seconds)
  │
  ├─ Sample 4 frames/sec, normalize identically
  │
  ├─ For each query frame: FAISS top-20 nearest neighbors
  │
  ├─ Offset voting:
  │     For every match, compute  offset = movie_time − query_time
  │     The true match shows up as a tight cluster at one offset;
  │     false positives scatter randomly across the runtime.
  │
  ├─ Rank-weighted, best-per-frame voting
  │     (each frame gives each candidate movie at most one vote,
  │      weighted by how high it ranked: 1.0, 0.5, 0.25, …)
  │
  ├─ Two-pass verification:
  │     If the top two candidates are close, re-extract at 8 fps
  │     and re-score. The right answer holds up; the wrong one collapses.
  │
  └─ Return: (movie_id, timestamp, confidence)
```

The intuition behind offset voting: a real match isn't a single frame that looks similar — it's *a sequence of frames that are similar in the right temporal order*. That sequence is the actual fingerprint of a scene. Random visual collisions don't survive that constraint.

---

## What's actually been built

The repo has been through five iterations, each documented in `docs/CHANGELOG.md`. The short version:

- **v0.1** — Combined audio (Shazam-style constellation maps) + visual matching. Worked, but audio added storage cost without much accuracy benefit and broke on dubs.
- **v0.2** — Dropped audio entirely. Storage went from ~148 MB/movie to ~27 MB/movie with no accuracy loss.
- **v0.3** — Real phone-camera tests across 3 movies. 6/6 correct identifications.
- **v0.4** — Rank-weighted voting + concentration scoring + two-pass verification. Eliminated runner-up false positives.
- **v0.5** — Scene-level coarse index, frame distinctiveness weighting, multi-scale matching.
- Latest — Optional `--screen` flag detects and rectifies a laptop/monitor in the frame for higher accuracy on screen captures.

### Real-world results (3 commercial feature films indexed, ~412 minutes total)

Tested across a sci-fi film (~106 min), a fantasy film (~157 min), and an action film (~149 min) — chosen for visual diversity:

| Capture style                | Accuracy | Confidence range |
|------------------------------|---------:|-----------------:|
| Clean digital extraction     |     6/6  | 0.88–0.92        |
| Phone, horizontal, full screen |   3/3  | 0.66–0.86        |
| Phone, vertical, partial screen | 3/3  | 0.31–0.81        |

Timestamp accuracy is consistently under **0.2 seconds** on clean clips and within a few seconds on phone captures.

Vertical, partial-screen phone captures are the genuinely hard case — the engine still picks the right film, but with a tighter confidence margin. That's a known limitation, not a bug: when only a fraction of the screen is visible, there's less visual signal to work with.

---

## Quick start

### Requirements

- Python 3.11+ (managed via [pyenv](https://github.com/pyenv/pyenv))
- FFmpeg
- Docker (for the bundled PostgreSQL container)

### Install

```bash
pyenv virtualenv 3.11.6 movie_fingerprint
pyenv local movie_fingerprint
pip install -r requirements.txt
docker compose up -d
```

### Index a movie

```bash
python main.py index --file movie.mp4 --title "Movie Name"
```

### Identify a clip

```bash
python main.py query --file clip.mp4

# For laptop/monitor captures (rectifies the visible screen):
python main.py query --file clip.mp4 --screen
```

### Self-test (extracts a random clip from a movie and tries to identify it)

```bash
python main.py test --file movie.mp4 --title "Movie Name" --duration 10
```

A more detailed setup walkthrough lives in [`docs/SETUP.md`](docs/SETUP.md).

---

## Repository layout

```
shazam-for-movies/
├── main.py                  # CLI entry: index / query / test / reset
├── engine/
│   ├── visual_fingerprint.py  # Frame extraction, normalization, CLIP embedding
│   ├── indexer.py             # Offline indexing pipeline
│   ├── matcher.py             # Two-pass query pipeline & offset voting
│   ├── db.py                  # PostgreSQL + FAISS storage layer
│   └── config.py              # All tunable parameters
├── docs/
│   ├── IDEATION.md          # Original design rationale
│   ├── ARCHITECTURE.md      # Detailed system diagrams
│   ├── RESEARCH.md          # Survey of fingerprinting literature
│   ├── CHANGELOG.md         # Version-by-version evolution
│   └── SETUP.md             # Setup walkthrough
├── docker-compose.yml       # PostgreSQL container
└── requirements.txt
```

---

## How it differs from Shazam (audio)

| | Shazam (audio) | This project (visual) |
|---|---|---|
| Fingerprint unit | Frequency-peak pairs | CLIP frame embeddings |
| Per-track entries | Millions of tiny hashes | Thousands of dense vectors |
| Matching signal | Hash collisions + offset histogram | Nearest-neighbor + offset histogram |
| Robust to | Noise, compression, ambient sound | Dubs, mute playback, screen capture, dub-only releases |
| Breaks on | Different dub/language | Different cut (director's vs. theatrical) |

The **offset histogram voting** mechanism is the same trick Shazam uses — applied to vector similarities instead of hash collisions. That part of the algorithm is the deepest reason the system works at all; everything else is plumbing around it.

---

## Limitations and honest tradeoffs

- **Different cuts diverge.** Theatrical and director's cuts share most frames, but inserted/removed scenes break offset clustering. The architecture supports a `version_id` field for handling this, but the current code treats one cut per movie.
- **Vertical, partial-screen phone captures are the weakest case.** Confidence margins shrink as more movies are added; identification still works, but the runner-up gets closer.
- **CPU-only by default.** CLIP runs on CPU for stability across platforms. Apple Silicon GPU works via `CLIP_DEVICE=mps`.
- **`IndexFlatIP` is exact but linear.** Past ~1,000 movies, switch FAISS to `IndexIVFPQ` for sublinear search. The interface is already there; it's a one-line index swap.

---

## Testing & content

This repository contains **no video content** — no movies, no clips, no test fixtures. Everything in `data/` and the various `test*.MOV` files referenced during development is gitignored and stays local.

The accuracy results above were measured against commercial films from the maintainer's own legally-obtained personal library, used solely for offline benchmarking. Titles are intentionally omitted from the results table because they're incidental to what's being demonstrated — the engine works on whatever you index it with.

If you want to reproduce the benchmarks, bring your own copies of any films you have the right to use (your own library, public-domain cinema, films you've licensed for research, footage you produced). The engine treats every video as opaque pixels; it has no preference about the source.

This project is intended for use cases where you have the rights to the content you're indexing — content owners, archivists, internal media tooling, research, and personal libraries.

## What this project is *not*

- Not a piracy or content-moderation tool. It matches a clip against an index *you* have built from content *you* control. It has no public database of films and no opinion about what you choose to index.
- Not a production service. It's a working MVP and a study of how an audio-style fingerprinting trick generalizes to video.
- Not a CLIP-finetune project. The whole point is that off-the-shelf CLIP embeddings, plus the right matching logic around them, are already enough.

---

## License & contributing

This is an open-source MVP. If you want to extend it — version-aware matching, a web demo, IVF+PQ scaling, GPU-accelerated indexing — issues and PRs are welcome.

The code aims to stay small and readable: the entire engine is under 1,000 lines of Python, deliberately. The interesting work is in the algorithm, not the abstraction layer.
