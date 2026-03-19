## **Goal**

User points phone camera at a screen for up to **10 seconds**.  
System returns:

- **movie_id**
- **estimated timestamp in the movie**
- **confidence**

For this, the cleanest MVP is:

**many timestamped visual fingerprints per movie \+ nearest-neighbor search \+ offset voting**

Perceptual hashes are useful as a cheap prefilter, but for the main matching signal you want **dense visual embeddings** plus a fast vector index such as **Faiss**, which is built for similarity search over large vector sets. ([faiss.ai](https://faiss.ai/index.html?utm_source=chatgpt.com))

## **High-level architecture**

### **Offline indexing**

For every movie:

1. Decode frames
2. Sample frames at a fixed rate
3. Compute:
   - a **cheap perceptual hash**
   - a **strong visual embedding**
4. Store each fingerprint with:
   - `movie_id`
   - `timestamp_sec`
   - `frame_no`
5. Put embeddings into a vector index

### **Online query**

For the user’s captured clip:

1. Extract frames from the 10-second recording
2. Compute the same hashes and embeddings
3. Search nearest neighbors in the index
4. Collect candidate `(movie_id, movie_timestamp, query_timestamp)` matches
5. Compute **offset \= movie_timestamp \- query_timestamp**
6. Find the movie and offset with the strongest consistent votes
7. Return the best match and timestamp

That offset-voting idea is the core of “how do we know the timestamp.” If many query frames agree on roughly the same offset, the clip is aligned to that point in the movie.

## **Representation choice**

For visual embeddings, a very practical starting point is a CLIP-style image encoder. CLIP was designed to learn transferable visual representations and is widely used as a general-purpose image embedding model. ([arXiv](https://arxiv.org/abs/2103.00020?utm_source=chatgpt.com))

For the cheap hash layer, perceptual hashing libraries such as ImageHash expose pHash, dHash, wavelet hash, and related methods that are useful for fast coarse filtering and duplicate-like matching. ([GitHub](https://github.com/JohannesBuchner/imagehash?utm_source=chatgpt.com))

## **Recommended MVP spec**

### **1\) Sampling strategy**

Do **not** index every frame.

Start with:

- sample **2 frames/sec** for the movie index
- sample **4 frames/sec** for the query clip

So for a 2-hour movie:

- 2 hr \= 7200 sec
- at 2 fps → about **14,400 indexed frames**

That is very manageable for an MVP.

Why this works:

- enough density to localize a scene
- not so dense that storage/search becomes painful
- multiple nearby matches improve robustness

### **2\) Per-frame data**

Store one row per indexed frame:

movie_id  
version_id  
timestamp_sec  
frame_no  
phash_64  
embedding_vector

`version_id` matters because theatrical cut and director’s cut may have different timelines even if most frames are visually similar.

### **3\) Two-stage retrieval**

Use a two-stage design:

**Stage A: coarse candidate generation**

- compare perceptual hashes
- or directly search embeddings for top-K nearest neighbors

**Stage B: robust verification**

- for candidate movies, gather all frame matches
- cluster by offset
- score temporal consistency

Faiss is the natural fit for the embedding-search stage because it is built for efficient similarity search on dense vectors and supports large collections and GPU acceleration. ([faiss.ai](https://faiss.ai/index.html?utm_source=chatgpt.com))

## **Matching logic**

Let each query frame produce top-K matches.

Example:

query frame at 0.5s \-\> Movie A at 4532.0s  
query frame at 1.0s \-\> Movie A at 4532.5s  
query frame at 1.5s \-\> Movie A at 4533.0s  
query frame at 2.0s \-\> Movie A at 4533.6s

Offsets:

4531.5  
4531.5  
4531.5  
4531.6

That is a strong cluster.

So estimated movie timestamp for query start is about:

offset_cluster_center ≈ 4531.5 sec

Then if the user captured 10 seconds, the clip spans roughly:

Movie A: 01:15:31.5 to 01:15:41.5

## **Scoring**

Use a score like:

final_score \=  
 0.45 \* mean_similarity \+  
 0.35 \* offset_cluster_size \+  
 0.20 \* temporal_order_consistency

Where:

- **mean_similarity** \= average embedding similarity of in-cluster matches
- **offset_cluster_size** \= how many query frames support the same offset
- **temporal_order_consistency** \= whether matched timestamps increase in the same order as the query frames

That last term is important. Random similar-looking frames should not beat a true scene sequence.

## **Confidence rules**

Return a result only if all are true:

- at least **N matched query frames** support the same movie
- offset variance is below a threshold
- similarity exceeds a threshold
- no second-best movie is too close

Example MVP thresholds:

- at least **6 supporting matches**
- offset stddev \< **0.8 sec**
- best score at least **1.25×** second-best

If not, return:

- “No confident match”

That is better than forcing a wrong answer.

## **Why visual-only solves the dub problem**

Because you are not depending on dialogue or soundtrack.  
If the same movie is dubbed into five languages, the **frames are still mostly the same**, so one visual index can usually serve all those dubs. You only need separate indexing when the **visual timeline changes**, such as different cuts or hard-burned edits.

## **Handling phone-camera distortions**

Phone capture of a TV introduces:

- skew/perspective
- glare
- blur
- moiré
- partial cropping
- subtitles

To make the system more robust:

1. **Resize and center-crop consistently**
2. **Normalize brightness/contrast**
3. Optionally detect the screen quadrilateral and rectify perspective
4. Sample multiple frames, not just one
5. Trust sequence consistency more than any single-frame match

This is why using a deep embedding is stronger than relying only on a hash.

## **Database design**

You can split storage into two parts.

### **Metadata table**

movies(  
 movie_id,  
 title,  
 runtime_sec,  
 version_id  
)

### **Frame fingerprints table**

frame_fingerprints(  
 movie_id,  
 version_id,  
 timestamp_sec,  
 frame_no,  
 phash_64,  
 embedding_id  
)

### **Vector index**

Faiss stores:

embedding_vector \-\> embedding_id

Then `embedding_id` maps back to the fingerprint row.

## **Recommended MVP stack**

- **Frame extraction:** FFmpeg
- **Image preprocessing:** OpenCV
- **Perceptual hash:** pHash via ImageHash-like library
- **Visual embedding:** CLIP image encoder
- **Vector search:** Faiss
- **Database:** Postgres or SQLite for metadata; Faiss for vectors

CLIP gives you general-purpose image embeddings, perceptual hash gives you a simple content fingerprint, and Faiss gives you the scalable similarity search layer. ([arXiv](https://arxiv.org/abs/2103.00020?utm_source=chatgpt.com))

## **What I would choose for version 1**

Very concrete MVP:

- Index movies at **2 fps**
- Query at **4 fps**
- For every frame:
  - store **64-bit pHash**
  - store **CLIP embedding**
- Search top **20** neighbors per query frame
- Vote on:
  - movie_id
  - timestamp offset
- Verify:
  - at least **6** aligned matches
  - increasing temporal order
- Return:
  - best movie
  - start timestamp
  - confidence

## **What I would avoid in v1**

Do **not** start with:

- full scene segmentation
- optical flow
- fancy shot-boundary detection
- custom trained models
- multimodal fusion

Those can help later, but they are not necessary to prove the concept.

## **Biggest practical limitation**

Some frames are just not distinctive:

- black frames
- extreme closeups
- end credits
- generic skies/walls
- very dark scenes

That is why the system should always use **multiple frames over several seconds** and score **sequence consistency**, not single-frame identity.

---

## **So what is the fingerprint _in your system_?**

For each frame:

fingerprint \= {  
 phash: 64-bit binary (fast filter)  
 embedding: 512-dim vector (main match signal)  
}

And crucially:

\+ timestamp  
\+ movie_id

---

## **Why this is different from Shazam**

Shazam fingerprint:

- one type (audio landmarks)
- highly structured (frequency peaks)

Your movie fingerprint:

- **many fingerprints per movie**
- each fingerprint \= **frame-level representation**
- matching happens over **time sequences**, not single points

---

## **Important insight**

A single fingerprint (one frame) is **not enough**.

What makes it powerful is:

**a sequence of fingerprints over time**

So your real “fingerprint” of a scene is:

Frame 1 → embedding A  
Frame 2 → embedding B  
Frame 3 → embedding C  
...

That sequence becomes **unique to that moment in the movie**.

---

## **Analogy**

- Perceptual hash \= like a blurry thumbnail comparison
- Embedding \= like a “semantic understanding” of the frame
- Sequence of embeddings \= like a **DNA strand of the scene**

---
