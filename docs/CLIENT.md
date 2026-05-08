# Client Architecture & Plan

Plan and architectural decisions for the mobile client (`mobile/`), the backend that serves it, and the streaming-aware engine refactor that makes it all work.

This document captures decisions made during the design conversation on 2026-05-08. It's a living plan — update as choices are revisited.

---

## Goal

A Shazam-like UX for movies:

> Open the app, point at a screen (or scan your own phone screen). Within 1–3 seconds, see the title and the timestamp of the scene currently playing.

No "record 10 seconds, wait, get answer." The match fires the moment the engine is confident — exactly like Shazam does for audio.

---

## Two scan modes

The app supports two distinct frame sources, both feeding the same matching engine:

### Mode 1 — Camera scan (point at external screen)

Standard use case: TV, laptop, friend's phone across the table. Use `expo-camera` to grab frames from the rear camera at ~4 fps.

### Mode 2 — Screen scan (capture this device's own screen)

Killer use case: identifying movie clips in social-media feeds (Instagram Reels, TikTok, Twitter, YouTube Shorts). You can't point a phone at its own screen, so the app has to *be* the screen recorder.

This requires platform-native screen capture:

- **iOS:** `ReplayKit` via a **Broadcast Upload Extension**. Requires a separate app target. Strict ~50 MB memory limit, restricted process. Frames pass to the main app via App Group / shared container.
- **Android:** `MediaProjection` API. User grants permission once; whole-device frames stream to the app. Significantly easier than iOS.

**Why this matters for architecture:** sending raw frames of a user's social-media feed to a remote server is a real privacy concern. This is the strongest argument for eventually moving CLIP on-device (see *Option B* below).

---

## The streaming engine (Phase 2)

> **Status: deferred to Phase 2.** Phase 1 ships with the existing batch `match_clip`. This section documents the intended streaming design so we don't paint ourselves into a corner with the Phase 1 protocol. Skim and skip on first read.

### Why

Today, `engine/matcher.py:match_clip` operates batch-style: collect all frames → embed all → search all → vote → decide. Latency is fixed by the longest path.

For a Shazam-like UX the engine must be **incremental**:

```
loop:
  receive next frame
  embed it (CLIP)
  FAISS top-K search
  add weighted votes to per-movie offset histograms
  check if any movie has crossed the confidence threshold
  if yes → emit match, stop
  if no → keep going
  if 10 s elapsed → emit best-effort or "no match"
```

A clean capture should fire a match in **1.5–2.5 seconds** (~6–10 frames). A noisy/partial capture might run to the 10-second budget. Either way, no fixed-length recording.

### How (no code duplication)

The streaming interface IS the foundation. The existing batch entry point becomes a one-line wrapper.

```python
# engine/matcher.py — new class
class StreamingMatcher:
    def __init__(self, db: Database):
        self.db = db
        self.histograms: dict[int, dict[float, float]] = {}  # movie_id -> offset_bin -> weight
        self.frame_count = 0
        # ...rank-weighted voting state, two-pass borderline tracker, etc.

    def add_frame(self, frame: np.ndarray, query_timestamp: float) -> None:
        """Embed, search, vote. Mutates internal state."""

    def is_confident(self) -> bool:
        """Apply v0.4 confidence rules (ratio >= 1.25x, >= 6 matches, etc.)."""

    def result(self) -> MatchResult | None:
        """Return current best candidate. None if no movie has any votes yet."""

    def best_effort(self) -> MatchResult | None:
        """Called at timeout. Returns result only if minimum thresholds met."""

    def reset(self) -> None:
        ...
```

The existing batch function then becomes:

```python
def match_clip(clip_path: str, db: Database, detect_screen: bool = False) -> MatchResult | None:
    matcher = StreamingMatcher(db)
    for frame, ts in extract_frames_lazy(clip_path, fps=VISUAL_QUERY_FPS):
        matcher.add_frame(frame, ts)
        if matcher.is_confident():
            return matcher.result()
    return matcher.best_effort()
```

**Net effect:** the CLI continues to work byte-for-byte. The streaming backend uses the same `StreamingMatcher` and feeds frames as they arrive over the wire. Estimated effort: ~100 lines net new, **0 lines duplicated**.

### What needs care

- **All existing v0.4 / v0.5 logic still applies incrementally** — rank-weighted voting, concentration scoring, temporal-order enforcement. The math doesn't change; the bookkeeping becomes incremental.
- **Two-pass verification reshapes.** Today: "if borderline at the end, re-extract at 8 fps and re-score." In streaming: "if borderline at frame N, request denser sampling for the next K frames" (or have the client bump fps on demand). Same idea, different driver.
- **Test parity.** Every existing test clip must produce the same `(movie_id, timestamp, confidence)` when run through `StreamingMatcher` as it does through today's `match_clip`. This is the regression suite.

---

## Backend

### Decision: defer

We are NOT building the backend yet. Engine streaming work comes first. This section captures the *intended* design so the engine refactor doesn't paint us into a corner.

### Backend will be in TypeScript (eventually)

User preference: NestJS for the production backend. But NestJS can't directly call PyTorch / FAISS, which forces a choice:

#### Pattern A — NestJS gateway, Python worker (production-grade)

```
Mobile ──WebSocket──► NestJS gateway ──gRPC stream──► Python StreamingMatcher service
                                                       (PyTorch + FAISS + Postgres)
```

NestJS handles auth, rate limiting, WebSocket sessions, observability. Python handles the actual ML. This is how Spotify, Discord, and many ML-heavy products are built.

**Cost:** designing the gRPC schema, writing two services, two deployments, two log streams.

#### Pattern B — NestJS-only via subprocess / sidecar HTTP

NestJS spawns Python as a subprocess (or calls a localhost FastAPI). Less efficient, more fragile, but only one "service" to deploy. Not recommended.

#### Pattern C — FastAPI all the way (MVP-grade)

Drop NestJS for now. FastAPI has first-class WebSocket support and lives in the same Python process as the engine. ~80 lines for a working endpoint.

**Decision: Pattern C for MVP, Pattern A if/when traffic justifies it.**

The MVP backend is one file:

```python
# backend/main.py (sketch)
@app.websocket("/query/stream")
async def query_stream(ws: WebSocket):
    await ws.accept()
    matcher = StreamingMatcher(db)
    try:
        async for msg in ws.iter_bytes():  # or iter_json for embedding mode
            frame = decode(msg)
            matcher.add_frame(frame, time.monotonic())
            if matcher.is_confident():
                await ws.send_json(matcher.result().asdict())
                return
            await ws.send_json({"status": "scanning", "frames": matcher.frame_count})
        # client disconnected before match
    finally:
        await ws.close()
```

### Transport: WebSocket

- Bidirectional, persistent, supported natively by both FastAPI and React Native.
- Backend can push `progress`, `match`, or `nope` events anytime.
- Only realistic alternative is gRPC streaming, which is overkill at this stage.

---

## On-device vs. server-side CLIP

This is the dial that controls bandwidth, latency, privacy, and complexity.

### Option A — Server-side CLIP (MVP path)

```
Phone ──streams JPEG frames──► Backend (CLIP + FAISS) ──► result
```

- Mobile app stays simple. No native ML modules.
- Bandwidth: ~30–50 KB per JPEG × 4 fps = ~150 KB/s, with early-stop typically ending at ~300–500 KB total.
- Privacy: raw frames leave the device. Acceptable for camera scan of someone's TV; not acceptable for screen scan of their Instagram feed.

### Option B — On-device CLIP (production path)

```
Phone (CLIP locally) ──streams 512-float vectors──► Backend (FAISS only) ──► result
```

- Tiny upload (~2 KB per frame × ~10 frames = ~20 KB total).
- Privacy: only embeddings leave the device. Vectors are not reversible to images.
- Backend is much cheaper (no GPU/CPU CLIP cost).
- Cost: real upfront work.
  - **iOS:** convert CLIP ViT-B/32 to **CoreML** via `coremltools`. Run via `Vision` / `MLModel`. Expected ~30 ms/frame on A15+.
  - **Android:** convert to **TensorFlow Lite** or **ONNX Runtime Mobile**. NNAPI / GPU delegate. Expected ~50 ms/frame on Snapdragon 8 Gen 1.
  - Wrap both in a custom Expo Module: TypeScript API → `encodeFrame(image): Promise<Float32Array>`.

### Plan: A first, B second

Ship streaming with Option A. The backend protocol is designed so swapping to Option B requires only a client-side change — the wire format becomes vectors instead of JPEGs, but the matcher is the same.

For the **screen-scan** mode, prioritize Option B before launch — it's effectively a privacy requirement once anyone reads the policy.

---

## Mobile app structure

Existing scaffold (already in `mobile/`):

- Expo SDK 55, React Native 0.83.6, Expo Router, TypeScript, dev-build (NOT Expo Go — required for native modules).
- App slug: `whomie`.

Planned additions (in build order):

```
mobile/src/
├── app/
│   ├── (scan)/
│   │   ├── camera.tsx       # camera-scan screen
│   │   └── screen.tsx       # screen-scan screen
│   └── result/[id].tsx      # match-result detail
├── lib/
│   ├── streaming-client.ts  # WebSocket client + reconnect
│   └── frame-encoder.ts     # JPEG encode (Option A) / vector encode (Option B)
└── modules/
    ├── screen-capture/      # Expo Module: ReplayKit (iOS) + MediaProjection (Android)
    └── clip-encoder/        # Expo Module: CoreML (iOS) + TFLite (Android) — Option B
```

### Native modules required

| Module | iOS | Android | Phase |
|---|---|---|---|
| Camera frame grab | `expo-camera` (built-in) | `expo-camera` (built-in) | 3 |
| Screen capture | ReplayKit + Broadcast Upload Extension (Swift) | MediaProjection (Kotlin) | 4 |
| On-device CLIP | CoreML via `Vision` (Swift) | TFLite / ONNX Runtime Mobile (Kotlin) | 5 |

Both custom modules require `npx expo prebuild` to materialize the native projects, then standard Expo Modules API (`requireNativeModule`).

---

## Build order

Revised on 2026-05-08 after surveying the existing matcher. The original plan started with a streaming-engine refactor; on inspection, the matcher uses batch-wide computations (frame distinctiveness, multi-scale histograms, scene-level pre-filtering) that don't translate trivially to per-frame incremental updates. A streaming refactor without a regression suite would be hard to validate.

**Pivot:** ship a **record-and-upload** loop first using the existing `match_clip` unchanged. Streaming becomes a v2 optimization once the rest of the system exists and can be regression-tested against it.

### Phase 1 — Working end-to-end

1. ✅ **Validation backend** — `backend/` FastAPI service with `GET /healthz` and `POST /query` (multipart video upload, `x-api-key` header, 20 MB cap). Wraps `match_clip` directly. Runs in Docker via the multi-stage `backend/Dockerfile` and the top-level `docker-compose.yml`.
2. ✅ **Smoke-tested end-to-end** — Harry Potter and X-Men test clips both match through the API with identical results to the CLI. See `docs/CHANGELOG.md` v0.6.
3. ⏳ **Mobile: camera scan (record-and-upload)** — `expo-camera` records a fixed 5-second clip, uploads as multipart, displays the result.
4. ⏳ **End-to-end device test** — dev client build, real device, real screen.

### Phase 2 — Streaming (deferred)

4. **Streaming engine** — Add `StreamingMatcher` to `engine/matcher.py`. Refit `match_clip` as a thin wrapper. Validate against the test clips already passing in Phase 1, so any drift is detectable.
5. **Streaming backend** — WebSocket endpoint wrapping `StreamingMatcher`. The HTTP `/query` from Phase 1 stays for clients that don't want streaming.
6. **Mobile: streaming camera scan** — replace the fixed-duration record with continuous frame capture and WebSocket stream.

### Phase 3 — Screen scan + privacy (later)

7. **Mobile: screen scan** — Custom Expo module wrapping ReplayKit (iOS) + MediaProjection (Android). Reuses whichever endpoint shape Phase 2 settled on.
8. **On-device CLIP (Option B)** — Custom Expo module wrapping CoreML (iOS) + TFLite (Android). Swap wire protocol from JPEGs/video to embedding vectors. Required for the screen-scan privacy story.
9. **(Optional) NestJS gateway** — If/when scale or developer preference demands it, put NestJS in front of the Python service over gRPC. Defer until needed.

We can stop after step 3 and have a complete working demo to show people.

---

## Open questions

- **Confidence thresholds in streaming mode.** Today's thresholds (ratio ≥ 1.25×, ≥ 6 matches, stddev < 1.5s) were tuned for batch. They may need re-calibration for the streaming case where decisions are made on partial data. *Address during step 1.*
- **Two-pass verification in streaming mode.** Need a concrete answer: does the client bump fps on borderline cases, or does the matcher just keep going at fixed fps until confident or timed-out? *Lean toward fixed-fps for simplicity, revisit if accuracy drops.*
- **Auth model.** Anonymous use vs. account-required. Probably anonymous + per-device API key for the MVP.
- **Index synchronization.** As the catalogue of indexed films grows, how (if at all) does the mobile app know what's in the index? Probably irrelevant — server just returns "no match" — but worth confirming the UX.

---

## What this document is not

Not a final API spec. Not a deployment plan. Not a UI design. Those come later. This is the **architectural plan and decision record** so future changes can be argued against an explicit baseline.
