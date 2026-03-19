# Video Fingerprinting & Content-Based Video Identification: Research Summary

## Table of Contents
1. [Audio Fingerprinting Techniques](#1-audio-fingerprinting-techniques)
2. [Visual Fingerprinting Techniques](#2-visual-fingerprinting-techniques)
3. [Combined Audio+Visual Approaches](#3-combined-audiovisual-approaches)
4. [Database & Indexing Strategies](#4-database--indexing-strategies)
5. [Key Papers & References](#5-key-papers--references)
6. [Recommended MVP Architecture](#6-recommended-mvp-architecture)

---

## 1. Audio Fingerprinting Techniques

### 1.1 Shazam's Approach (Constellation Maps + Combinatorial Hashing)

**Seminal Paper**: Avery Li-Chun Wang, "An Industrial-Strength Audio Search Algorithm," ISMIR 2003.
- PDF: https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf
- Semantic Scholar: https://www.semanticscholar.org/paper/An-Industrial-Strength-Audio-Search-Algorithm-Wang/2f58f6b34c6cd08ca901433949275e4a42368036

**How it works:**

1. **Spectrogram Generation**: Audio is downsampled to 11,025 Hz. A Short-Time Fourier Transform (STFT) is applied with a Hamming window of 1024 samples (~0.1s per frame), producing a time-frequency spectrogram. Frequencies above 5 kHz are filtered out.

2. **Constellation Map**: Local maxima (peaks) are identified in the spectrogram using a neighborhood maximum filter. These sparse high-energy points in time-frequency space form a "constellation map" -- a compact, noise-resistant representation.

3. **Combinatorial Hashing**: Pairs of peaks are combined into hashes:
   - Each hash encodes: frequency of anchor point (fA), frequency of target point (fB), time delta between them (deltaT)
   - With 10-bit frequencies and 10-bit time delta = 30-bit hash (~1 billion possibilities)
   - Each hash is stored with the anchor point's absolute timestamp and track ID

4. **Matching**: Query hashes are looked up in the database. Matches are grouped by track ID. For each candidate track, the algorithm computes (track_time - sample_time) for all matching hashes and builds a histogram. The track with the tallest histogram bin wins. This temporal coherence check is what makes the algorithm robust.

**Strengths**: Noise-resistant, works with as little as 2-5 seconds of audio, scales to millions of tracks, handles compression artifacts well.

### 1.2 Dejavu (Open-Source Python Implementation)

**Repository**: https://github.com/worldveil/dejavu
**Blog post**: https://willdrevo.com/fingerprinting-and-audio-recognition-with-python/

Dejavu is a Python re-implementation of the Shazam-style approach:

- **Database**: MySQL or PostgreSQL with two tables: `songs` (metadata) and `fingerprints` (hash + time offset + song reference)
- **Process**: FFT spectrogram -> peak detection (with configurable amplitude threshold and neighborhood size) -> combinatorial hashing -> storage
- **Accuracy**: 100% from disk, ~96% from 2s microphone recording, 100% at 5+ seconds
- **Storage**: ~377 MB for 5.4 million fingerprints (45 songs)
- **Key config params**: `FINGERPRINT_REDUCTION`, `DEFAULT_AMP_MIN`, `PEAK_NEIGHBORHOOD_SIZE`

**Best for MVP**: Dejavu provides the most accessible starting point for the audio fingerprinting component.

### 1.3 Chromaprint / AcoustID

**Website**: https://acoustid.org/chromaprint
**Technical details**: https://oxygene.sk/2011/01/how-does-chromaprint-work/
**GitHub**: https://github.com/acoustid/chromaprint
**Python bindings**: https://github.com/beetbox/pyacoustid

Chromaprint takes a different approach based on chroma (musical note) features:

1. **Preprocessing**: Audio resampled to 11,025 Hz, frames of 4096 samples (0.371s), 2/3 overlap
2. **STFT** applied to create spectrogram
3. **Chroma features**: Frequencies mapped to 12 musical note bins (one per semitone)
4. **Filter bank**: 16 pre-trained filters process a sliding 16x12 window over the chroma image
5. **Quantization**: Each filter outputs 0-3 (2 bits, Gray coded) -> 16 filters x 2 bits = 32-bit integer per window position
6. **Comparison**: Bit error rate between fingerprints

**Performance**: <100ms to process 2 minutes of audio; fingerprint is ~2.5 KB for a full song.

**Trade-off vs Shazam approach**: Chromaprint is optimized for identifying entire songs (music metadata lookup). Shazam's constellation approach is better for short, noisy clips -- which is what we need for a "Shazam for movies" use case.

---

## 2. Visual Fingerprinting Techniques

### 2.1 Perceptual Hashing (pHash, dHash, aHash)

**pHash library**: https://www.phash.org/
**Wikipedia**: https://en.wikipedia.org/wiki/Perceptual_hashing

**Types of perceptual hashes:**

| Algorithm | Method | Speed | Robustness | Best For |
|-----------|--------|-------|------------|----------|
| **aHash** (Average Hash) | Compare each pixel to average brightness | Fastest | Low | Quick pre-filter |
| **dHash** (Difference Hash) | Compare adjacent pixel brightness gradients | Fast | Medium | Real-time applications |
| **pHash** (Perceptual Hash) | DCT-based frequency domain analysis | Medium | High | Robust matching under transformations |
| **wHash** (Wavelet Hash) | Discrete Wavelet Transform | Medium | High | Alternative to pHash |

**pHash process for images:**
1. Reduce image to 32x32 grayscale
2. Apply Discrete Cosine Transform (DCT)
3. Keep top-left 8x8 DCT coefficients (lowest frequencies)
4. Compute mean of coefficients
5. Generate 64-bit hash: each bit = 1 if coefficient > mean, else 0
6. Compare hashes using Hamming distance

**For video**: Extract keyframes at regular intervals, compute perceptual hash for each frame, compare sequences.

### 2.2 Videohash (Python Library)

**GitHub**: https://github.com/akamhy/videohash

Generates a 64-bit comparable hash for any video:

1. Extract frames at 1-second intervals, resize to 144x144
2. **Bitlist 1**: Assemble frames into collage, compute wavelet hash
3. **Bitlist 2**: Stitch frames horizontally, divide into 64 segments, compare dominant color against pattern
4. **Final hash**: XOR of both bitlists -> 64-bit hash
5. Compare using Hamming distance

**Resilient to**: resolution changes, transcoding, watermarks, stabilization, color adjustments, frame rate changes, aspect ratio changes, cropping.

**Limitation**: Cannot detect partial video matches (only full-video comparison).

### 2.3 Advanced Visual Fingerprinting

**Haar Wavelet + LSH approach** (used in commercial systems like Emysound):
- Source: https://emysound.com/blog/open-source/2021/08/01/video-fingerprinting.html

1. **Dimensionality reduction**: Downsample from 1080p to 128x72 grayscale (675x reduction)
2. **Feature extraction**: 2D Discrete Haar Wavelet Transform, retain top 4% of coefficients
3. **Encoding**: Locality Sensitive Hashing with min-hash permutations
4. **Matching**: Two-stage -- LSH for candidate retrieval, then Difference of Gaussians for pixel-level verification

This approach processes 30 frames/second and can match against databases of 5,000-10,000 reference items across 1,000+ sources.

### 2.4 Color Histograms and Temporal Patterns

**Color histograms**: Compute color distribution for each frame in HSV or LAB color space. Comparing histogram sequences captures the "visual rhythm" of a video. Robust to resolution changes but sensitive to color grading.

**Temporal patterns**: Track how visual features change over time:
- Scene cut detection (abrupt changes in frame similarity)
- Average brightness curves
- Motion vectors / optical flow patterns
- These temporal signatures are highly discriminative for video identification

---

## 3. Combined Audio+Visual Approaches

### 3.1 Joint Audio-Video Fingerprinting

**Key paper**: "Joint Audio-Video Fingerprint Media Retrieval" (2016)
- arXiv: https://arxiv.org/pdf/1609.01331

**Key paper**: "Short video fingerprint extraction: from audio-visual fingerprint fusion to multi-index hashing"
- Springer: https://link.springer.com/article/10.1007/s00530-022-01031-4

### 3.2 Fusion Strategies

**Early Fusion**: Combine raw audio and visual features into a single feature vector before hashing. Simpler but loses modality-specific structure.

**Late Fusion (Recommended for MVP)**: Compute audio and visual fingerprints independently, then combine match scores:
- Audio fingerprint -> candidate list with confidence scores
- Visual fingerprint -> candidate list with confidence scores
- Merge: weighted combination of scores, or require agreement from both modalities

**Advantages of combined approach:**
- Audio alone fails when: audio track is replaced, muted, or heavily mixed
- Video alone fails when: video is cropped/letterboxed, overlaid with graphics, or color-graded
- Combined approach covers both failure modes
- Multi-channel fingerprints (US Patent 9275427) merge audio+video data at common time offsets

### 3.3 Practical Fusion Architecture

```
Input clip (5-10 seconds)
    |
    +---> Audio extraction --> Spectrogram --> Constellation map --> Audio hashes
    |                                                                    |
    +---> Frame extraction --> Perceptual hash per frame --> Visual hash sequence
    |                                                                    |
    v                                                                    v
    Audio candidate lookup (hash table)          Visual candidate lookup (LSH)
    |                                                                    |
    +----------------------> Score fusion <------------------------------+
                                |
                                v
                        Final ranked results
```

---

## 4. Database & Indexing Strategies

### 4.1 Hash Table (for Audio Fingerprints -- Shazam style)

The simplest and most effective approach for constellation-map hashes:
- **Schema**: `hash_value (30-bit int) -> [(song_id, time_offset), ...]`
- **Lookup**: O(1) per hash, then histogram voting across matches
- **Storage**: In-memory hash map or database index (B-tree on hash column)
- **Scale**: Works well up to millions of tracks with standard databases

### 4.2 Locality Sensitive Hashing (LSH)

**Wikipedia**: https://en.wikipedia.org/wiki/Locality-sensitive_hashing
**Pinecone guide**: https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing/

For perceptual hashes and high-dimensional feature vectors:
- Hashes similar items into the same buckets with high probability
- Enables sub-linear search time (avoid comparing against entire database)
- Common variant: Random projection LSH for cosine similarity

**Application to video fingerprinting**:
- Paper: "A robust and fast video fingerprinting based on 3D-DCT and LSH" (https://www.researchgate.net/publication/252035285)
- 3D-DCT extracts spatial+temporal video features; LSH indexes them for fast retrieval

### 4.3 FAISS (Facebook AI Similarity Search)

**GitHub**: https://github.com/facebookresearch/faiss
**Documentation**: https://faiss.ai/
**Paper**: https://arxiv.org/pdf/2401.08281

Meta's library for efficient similarity search on dense vectors:
- Scales to **billions** of vectors (1.5 trillion in Meta's internal use)
- GPU-accelerated implementations
- Key index types:
  - `IndexFlatL2`: Brute-force, exact (small datasets)
  - `IndexIVFFlat`: Inverted file index (medium datasets)
  - `IndexIVFPQ`: Product quantization + inverted file (large datasets, memory-efficient)
  - `IndexHNSW`: Graph-based, high accuracy (recommended for quality-sensitive applications)

**For video fingerprinting**: Convert perceptual hashes or CNN features into vectors, index with FAISS for sub-millisecond nearest-neighbor lookup.

### 4.4 Recommended Database Architecture for MVP

```
PostgreSQL
  |
  +-- audio_fingerprints table
  |     hash_value (INT, indexed) -> content_id, time_offset
  |
  +-- visual_fingerprints table
  |     frame_hash (BIGINT) -> content_id, timestamp
  |
  +-- content table
        content_id -> title, year, type (movie/show), metadata

For scale: migrate visual search to FAISS or Milvus vector DB
```

---

## 5. Key Papers & References

### Foundational Papers

| Paper | Authors | Year | Topic | URL |
|-------|---------|------|-------|-----|
| An Industrial-Strength Audio Search Algorithm | Avery Li-Chun Wang | 2003 | Shazam algorithm | [PDF](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf) |
| Video Google: A Text Retrieval Approach to Object Matching in Videos | Sivic & Zisserman | 2003 | Visual word vocabulary for video search | Referenced in FAISS |
| Digital Fingerprinting on Multimedia: A Survey | Various | 2024 | Comprehensive survey of all fingerprinting modalities | [arXiv](https://arxiv.org/html/2408.14155v1) |
| Video Fingerprinting: Past, Present, and Future | Various | 2022 | Survey of video fingerprinting evolution | [Frontiers](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2022.984169/full) |
| Video Fingerprinting for Copy Identification: From Research to Industry | Lu | 2009 | Industry applications | [SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/7254/1/Video-fingerprinting-for-copy-identification--from-research-to-industry/10.1117/12.805709.short) |
| Joint Audio-Video Fingerprint Media Retrieval | Various | 2016 | Multimodal fusion | [arXiv](https://arxiv.org/pdf/1609.01331) |
| Short Video Fingerprint Extraction: Audio-Visual Fusion to Multi-Index Hashing | Various | 2022 | Modern fusion approach | [Springer](https://link.springer.com/article/10.1007/s00530-022-01031-4) |
| A Robust and Fast Video Fingerprinting Based on 3D-DCT and LSH | Various | - | LSH for video | [ResearchGate](https://www.researchgate.net/publication/252035285) |
| SpectroMap: Peak Detection Algorithm for Audio Fingerprinting | Various | 2022 | Improved peak detection | [arXiv](https://arxiv.org/pdf/2211.00982) |
| The FAISS Library | Douze et al. | 2024 | Vector similarity search at scale | [arXiv](https://arxiv.org/pdf/2401.08281) |

### Open-Source Implementations

| Project | Language | Purpose | URL |
|---------|----------|---------|-----|
| Dejavu | Python | Audio fingerprinting (Shazam-style) | [GitHub](https://github.com/worldveil/dejavu) |
| Chromaprint | C++ | Audio fingerprinting (chroma-based) | [GitHub](https://github.com/acoustid/chromaprint) |
| pyacoustid | Python | Python bindings for Chromaprint | [GitHub](https://github.com/beetbox/pyacoustid) |
| videohash | Python | Perceptual video hashing (64-bit) | [GitHub](https://github.com/akamhy/videohash) |
| pHash | C++ | Perceptual hashing library | [Website](https://www.phash.org/) |
| FAISS | C++/Python | Vector similarity search | [GitHub](https://github.com/facebookresearch/faiss) |
| SoundFingerprinting | C# | Audio+video fingerprinting | [Emysound](https://emysound.com/blog/open-source/2021/08/01/video-fingerprinting.html) |

### Tutorials & Explainers

- How Does Shazam Work: https://www.cameronmacleod.com/blog/how-does-shazam-work
- How Does Chromaprint Work: https://oxygene.sk/2011/01/how-does-chromaprint-work/
- Dejavu blog post: https://willdrevo.com/fingerprinting-and-audio-recognition-with-python/
- Audio Identification (academic tutorial): https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html
- LSH for Audio Fingerprinting: https://santhoshhari.github.io/Locality-Sensitive-Hashing/
- The Five-Second Fingerprint (Shazam): https://towardsdatascience.com/the-five-second-fingerprint-inside-shazams-instant-song-id/

---

## 6. Recommended MVP Architecture

### Phase 1: Audio-First MVP

Start with audio fingerprinting since it is the most mature, well-documented approach and provides the highest accuracy for the least effort.

**Stack:**
- Python 3.10+
- FFmpeg for audio/video extraction
- Dejavu (or custom implementation based on Wang 2003 algorithm)
- PostgreSQL for fingerprint storage
- FastAPI for the recognition API

**Process:**
1. **Ingest pipeline**: For each movie/show, extract audio track -> generate spectrogram -> detect peaks -> create constellation map -> compute combinatorial hashes -> store in database
2. **Recognition**: User records 5-10 seconds of audio -> same fingerprinting pipeline -> lookup hashes in database -> histogram voting -> return match with confidence score
3. **Target**: 100% accuracy from clean audio, >90% from ambient recording at 5+ seconds

### Phase 2: Add Visual Fingerprinting

Layer visual identification on top for cases where audio alone is insufficient.

**Additional stack:**
- OpenCV or Pillow for frame extraction
- imagehash Python library (provides aHash, dHash, pHash, wHash)
- FAISS for visual fingerprint indexing

**Process:**
1. **Ingest**: Extract keyframes (1 per second) -> compute pHash for each -> store with timestamp
2. **Recognition**: Extract frames from user clip -> compute pHash -> query FAISS for nearest neighbors -> temporal alignment check
3. **Fusion**: Combine audio and visual candidate scores with weighted voting

### Phase 3: Scale & Optimize

- Migrate to FAISS IVF+PQ index for visual fingerprints at scale
- Add GPU acceleration for fingerprint computation
- Implement caching layer (Redis) for hot fingerprints
- Add scene-level indexing (detect scene boundaries, fingerprint per scene)
- Consider CNN-based features (ResNet, EfficientNet) for visual fingerprinting if perceptual hashes prove insufficient

### MVP Data Flow

```
                    INGESTION
                    =========
Movie file
  |
  +---> ffmpeg -i movie.mp4 -vn audio.wav
  |       |
  |       +---> Spectrogram (STFT, 1024-sample window, 11025 Hz)
  |       +---> Peak detection (local maxima in time-freq space)
  |       +---> Constellation map
  |       +---> Combinatorial hashing (anchor + target pairs)
  |       +---> Store: hash -> (movie_id, time_offset) in PostgreSQL
  |
  +---> ffmpeg -i movie.mp4 -vf fps=1 frame_%04d.png
          |
          +---> For each frame: compute pHash (64-bit)
          +---> Store: (movie_id, timestamp, phash) in PostgreSQL/FAISS


                    RECOGNITION
                    ===========
User recording (5-10 sec audio + optional video)
  |
  +---> Audio fingerprinting pipeline (same as above)
  |       |
  |       +---> Hash lookup in DB -> candidate movies + time offsets
  |       +---> Histogram voting -> top audio candidates with scores
  |
  +---> Visual fingerprinting pipeline (if video available)
  |       |
  |       +---> Frame pHash computation
  |       +---> FAISS nearest-neighbor search -> visual candidates
  |
  +---> Score fusion
          |
          +---> final_score = 0.7 * audio_score + 0.3 * visual_score
          +---> Return: movie title, timestamp, confidence
```

### Key Design Decisions for MVP

1. **Audio is primary, visual is supplementary**: Audio fingerprinting alone gives >95% accuracy. Visual adds robustness for edge cases.

2. **Use Dejavu as starting point**: Fork and adapt rather than building from scratch. Key modifications needed:
   - Adapt for long-form content (movies are 1.5-3 hours vs. 3-5 minute songs)
   - Optimize for segment matching (return timestamp within movie, not just movie ID)
   - Tune peak detection parameters for movie audio (dialogue + music + effects vs. music only)

3. **Fingerprint granularity**: Index in 5-second chunks with 2.5-second overlap for temporal precision.

4. **Database sizing estimate**:
   - Audio: ~120,000 hashes per minute of audio -> ~14.4M hashes per 2-hour movie
   - Visual: 1 frame/sec = 7,200 hashes per 2-hour movie (64 bits each, negligible)
   - For 10,000 movies: ~144 billion audio hashes (requires optimization -- see below)
   - Optimization: Use hash table with collision lists; most lookups are O(1)

5. **Start small**: Index 100 movies, validate accuracy, then scale. Audio fingerprint storage for 100 movies is ~1.44 billion hashes, manageable with PostgreSQL + proper indexing.
