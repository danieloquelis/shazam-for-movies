"""
Visual fingerprinting using perceptual hashing + CLIP embeddings.

Resolution-agnostic: all frames are normalized to 224x224 before processing,
so 480p, 720p, 1080p, 4K all produce comparable embeddings.

Noise-robust: frames are preprocessed with contrast normalization and
center-cropping (avoiding letterbox bars) before embedding.

Dub-invariant: purely visual — works regardless of audio language.

References:
- CLIP: Radford et al. (2021) https://arxiv.org/abs/2103.00020
- pHash: http://www.phash.org/
- Video fingerprinting survey: https://www.frontiersin.org/articles/10.3389/frsip.2022.984169
"""

import os
import subprocess
import numpy as np
from PIL import Image, ImageFilter
import imagehash
import cv2
import torch
from engine.config import (
    CLIP_MODEL_NAME, CLIP_PRETRAINED, CLIP_EMBEDDING_DIM,
    PHASH_SIZE, FRAME_BATCH_SIZE, FRAME_TARGET_SIZE,
    DEDUP_SIMILARITY_THRESHOLD, SCENE_BOUNDARY_THRESHOLD,
)

# Module-level CLIP model singleton
_clip_model = None
_clip_preprocess = None
_clip_device = None


def _get_clip():
    """Lazy-load CLIP model (singleton)."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        import open_clip
        # Use CPU for stability — MPS can segfault under sustained CLIP load on macOS.
        # Set CLIP_DEVICE=mps env var to use Apple Silicon GPU if your system is stable.
        device = os.environ.get("CLIP_DEVICE", "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
        )
        model = model.to(device).eval()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_device = device
        print(f"  CLIP model loaded on {device}")
    return _clip_model, _clip_preprocess, _clip_device


def _detect_screen_region(frame: np.ndarray) -> np.ndarray | None:
    """
    Detect a screen/monitor region in the frame and extract it.

    Works for: laptop screens, TV screens, monitors, projected screens.
    Returns the extracted screen content, perspective-corrected to a rectangle.
    Returns None if no clear screen region is found (frame used as-is).

    Conservative detection — only triggers when there's a clear rectangular
    screen boundary with darker surroundings. Avoids false positives on
    direct TV captures where the movie fills the whole frame.

    Validation: the region inside the quad must be significantly brighter
    than the region outside (screen vs desk/wall).
    """
    h, w = frame.shape[:2]
    frame_area = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise, then edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        # Screen should be 20-85% of frame (not nearly full-frame)
        if area < frame_area * 0.20 or area > frame_area * 0.85:
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        # Looking for roughly rectangular shapes (4-8 vertices — screens
        # may have rounded corners, webcam notches, or cables that add vertices)
        if 4 <= len(approx) <= 8 and area > best_area:
            # Validate: check that the shape is roughly rectangular
            # Use min area rect for aspect ratio check (works for 4-6 vertices)
            min_rect = cv2.minAreaRect(approx)
            box_pts = cv2.boxPoints(min_rect).astype(np.float32)
            ordered = _order_points(box_pts)
            w_top = np.linalg.norm(ordered[1] - ordered[0])
            h_left = np.linalg.norm(ordered[3] - ordered[0])
            if w_top < 1 or h_left < 1:
                continue
            aspect = w_top / h_left
            if aspect < 0.3 or aspect > 3.0:
                continue

            best_quad = approx
            best_area = area

    if best_quad is not None:
        # Validate: inside the contour should have different brightness than outside
        # Screens create a clear brightness boundary (either brighter or darker
        # than surroundings, depending on content and room lighting)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, best_quad.reshape(-1, 2).astype(np.int32), 255)
        inside_mean = cv2.mean(gray, mask=mask)[0]
        outside_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]

        # Require some brightness difference (either direction)
        if abs(inside_mean - outside_mean) < 8:
            return None

        # Use bounding rectangle of the contour as the screen crop.
        # Simpler and more robust than perspective transform for slightly
        # angled captures. The subsequent CLAHE + center-crop handles
        # any remaining edge artifacts.
        x, y, bw, bh = cv2.boundingRect(best_quad)

        # Inset slightly (5%) to avoid capturing screen bezel
        inset_x = int(bw * 0.05)
        inset_y = int(bh * 0.05)
        x += inset_x
        y += inset_y
        bw -= 2 * inset_x
        bh -= 2 * inset_y

        if bw > 50 and bh > 50:
            return frame[y:y+bh, x:x+bw]

    # Fallback: brightness-based detection using Otsu thresholding.
    # Screens create a clear bright/dark boundary against surroundings.
    _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if frame_area * 0.20 < area < frame_area * 0.85:
            x, y, bw, bh = cv2.boundingRect(largest)
            # Inset 5%
            ix, iy = int(bw * 0.05), int(bh * 0.05)
            x, y, bw, bh = x + ix, y + iy, bw - 2*ix, bh - 2*iy
            if bw > 50 and bh > 50:
                return frame[y:y+bh, x:x+bw]

    return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
    rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference
    return rect


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a frame for robust fingerprinting.
    Handles: resolution differences, letterboxing, brightness/contrast variation,
    noise, and mild perspective distortion.

    Steps:
    1. Detect and crop out letterbox bars (black bars top/bottom or sides)
    2. Center-crop to remove edge noise (phone capture often has partial screen)
    3. Normalize brightness and contrast (CLAHE)
    4. Resize to standard size
    """
    h, w = frame.shape[:2]

    # 1. Detect and remove letterbox bars
    # Convert to grayscale for bar detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Find rows/cols that are mostly black (letterbox)
    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    threshold = 15  # pixels below this are "black"

    # Find content boundaries
    active_rows = np.where(row_means > threshold)[0]
    active_cols = np.where(col_means > threshold)[0]

    if len(active_rows) > 10 and len(active_cols) > 10:
        y1, y2 = active_rows[0], active_rows[-1] + 1
        x1, x2 = active_cols[0], active_cols[-1] + 1
        # Only crop if we're removing significant bars (>5% of dimension)
        if (y1 > h * 0.05 or y2 < h * 0.95 or x1 > w * 0.05 or x2 < w * 0.95):
            frame = frame[y1:y2, x1:x2]
            h, w = frame.shape[:2]

    # 2. Center-crop to 90% to remove edge artifacts
    crop_pct = 0.90
    ch, cw = int(h * crop_pct), int(w * crop_pct)
    y_off = (h - ch) // 2
    x_off = (w - cw) // 2
    frame = frame[y_off:y_off + ch, x_off:x_off + cw]

    # 3. Contrast normalization using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This handles brightness differences between captures
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_channel)
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 4. Resize to standard size
    frame = cv2.resize(frame, (FRAME_TARGET_SIZE, FRAME_TARGET_SIZE),
                       interpolation=cv2.INTER_AREA)

    return frame


def extract_frames_raw(video_path: str, fps: float,
                       start_sec: float = 0, duration_sec: float = None) -> list[tuple[float, np.ndarray]]:
    """
    Extract raw frames from video at given FPS using ffmpeg.
    Returns frames at their original resolution (no resizing here — normalization happens later).
    """
    # Get video dimensions
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path,
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    parts = [p for p in probe.stdout.strip().split(",") if p]
    w, h = int(parts[0]), int(parts[1])

    # Scale down large videos for faster extraction (keep aspect ratio)
    # Max dimension 640 — enough detail for embeddings, fast to decode
    max_dim = 640
    if max(w, h) > max_dim:
        if w > h:
            new_w = max_dim
            new_h = int(h * max_dim / w)
        else:
            new_h = max_dim
            new_w = int(w * max_dim / h)
        # Ensure even dimensions
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        scale_filter = f"scale={new_w}:{new_h},"
    else:
        new_w, new_h = w, h
        scale_filter = ""

    cmd = ["ffmpeg"]
    if start_sec > 0:
        cmd += ["-ss", str(start_sec)]
    cmd += ["-i", video_path]
    if duration_sec is not None:
        cmd += ["-t", str(duration_sec)]
    cmd += [
        "-vf", f"{scale_filter}fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-loglevel", "error",
        "pipe:1",
    ]

    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    frame_size = new_w * new_h * 3
    n_frames = len(raw) // frame_size

    frames = []
    for i in range(n_frames):
        frame_data = raw[i * frame_size:(i + 1) * frame_size]
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(new_h, new_w, 3)
        timestamp = start_sec + i / fps
        frames.append((timestamp, frame))

    return frames


def compute_phash(frame: np.ndarray) -> int:
    """Compute 64-bit perceptual hash of a normalized frame."""
    img = Image.fromarray(frame)
    h = imagehash.phash(img, hash_size=PHASH_SIZE)
    return int(str(h), 16)


def compute_clip_embeddings(frames: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    """
    Compute CLIP embeddings for a batch of frames.
    Frames should already be normalized (224x224).
    Returns (N, 512) float32 array, L2-normalized.
    """
    model, preprocess, device = _get_clip()

    all_embeddings = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        images = torch.stack([
            preprocess(Image.fromarray(f)) for f in batch_frames
        ]).to(device)

        with torch.no_grad():
            embeddings = model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().float().numpy())

    return np.vstack(all_embeddings)


def fingerprint_visual(video_path: str, fps: float,
                       start_sec: float = 0, duration_sec: float = None,
                       normalize: bool = True,
                       detect_screen: bool = False) -> list[tuple[float, int, np.ndarray]]:
    """
    Full visual fingerprinting pipeline.
    Returns list of (timestamp_sec, phash_int, embedding_512d).

    When normalize=True (default), frames are preprocessed for robustness
    against resolution, brightness, letterboxing, and noise differences.

    When detect_screen=True, attempts to find and extract the screen region
    from each frame before normalization. Use for phone captures of laptop/TV screens.
    """
    if duration_sec is None:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        total_duration = float(probe.stdout.strip())
        duration_sec = total_duration - start_sec

    results = []
    window_sec = FRAME_BATCH_SIZE / fps
    current = start_sec
    end = start_sec + duration_sec
    total_frames = 0

    while current < end:
        chunk_dur = min(window_sec, end - current)
        raw_frames = extract_frames_raw(video_path, fps, start_sec=current, duration_sec=chunk_dur)

        if not raw_frames:
            current += chunk_dur
            continue

        timestamps = [f[0] for f in raw_frames]

        # Screen detection: try to extract screen content from each frame.
        # If detection succeeds on most frames (>50%), only keep those frames
        # where it worked — the others would contain desk/wall noise.
        if detect_screen:
            screen_results = [(f[1], _detect_screen_region(f[1])) for f in raw_frames]
            n_detected = sum(1 for _, s in screen_results if s is not None)

            if n_detected > len(raw_frames) * 0.5:
                # Screen found in most frames — use detected screens, skip failures
                kept = [(i, s) for i, (_, s) in enumerate(screen_results) if s is not None]
                processed_raws = [s for _, s in kept]
                timestamps = [timestamps[i] for i, _ in kept]
            elif n_detected > 0:
                # Some detected — use screen where found, full frame otherwise
                processed_raws = [s if s is not None else orig for orig, s in screen_results]
            else:
                # No screen detected — use frames as-is
                processed_raws = [f[1] for f in raw_frames]
        else:
            processed_raws = [f[1] for f in raw_frames]

        # Normalize frames for robustness
        if normalize:
            frame_arrays = [_normalize_frame(f) for f in processed_raws]
        else:
            frame_arrays = [
                cv2.resize(f, (FRAME_TARGET_SIZE, FRAME_TARGET_SIZE))
                for f in processed_raws
            ]

        # Compute pHashes on normalized frames
        phashes = [compute_phash(f) for f in frame_arrays]

        # Compute CLIP embeddings
        embeddings = compute_clip_embeddings(frame_arrays)

        for ts, ph, emb in zip(timestamps, phashes, embeddings):
            results.append((ts, ph, emb))

        total_frames += len(raw_frames)
        pct = min(100, (current + chunk_dur - start_sec) / duration_sec * 100)
        print(f"  Visual: {total_frames} frames processed ({pct:.0f}%)", end="\r")

        current += chunk_dur

    print(f"  Visual: {total_frames} frames processed (100%)   ")
    return results


def deduplicate_keyframes(
    embeddings: np.ndarray,
    dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    scene_threshold: float = SCENE_BOUNDARY_THRESHOLD,
    prev_embedding: np.ndarray = None,
) -> tuple[list[int], list[list[int]], bool]:
    """
    Given N L2-normalized embeddings, identify keyframes and scene boundaries.

    Returns:
    - kept_indices: indices of keyframes to store (deduplicated)
    - scenes: list of completed scenes (each is a list of original indices)
    - scene_boundary_at_start: whether a new scene started at the beginning
      (used for cross-batch scene flushing)

    A frame is skipped if cosine similarity to the last kept frame > dedup_threshold.
    A new scene starts when similarity drops below scene_threshold.
    """
    if len(embeddings) == 0:
        return [], [], False

    kept = []
    scenes = []
    current_scene = []
    scene_boundary_at_start = False

    # Use prev_embedding from previous batch for continuity
    last_kept = prev_embedding

    for i in range(len(embeddings)):
        if last_kept is None:
            # First frame ever
            kept.append(i)
            current_scene.append(i)
            last_kept = embeddings[i]
            continue

        sim = float(embeddings[i] @ last_kept)

        if sim > dedup_threshold:
            # Near-duplicate — skip, but still part of current scene
            continue
        elif sim < scene_threshold:
            # Scene boundary
            if i == 0:
                scene_boundary_at_start = True
            if current_scene:
                scenes.append(current_scene)
            current_scene = [i]
            kept.append(i)
            last_kept = embeddings[i]
        else:
            # Same scene, different enough to keep
            kept.append(i)
            current_scene.append(i)
            last_kept = embeddings[i]

    # Don't append current_scene to scenes — it may continue in next batch
    # Caller decides when to flush it
    return kept, scenes, scene_boundary_at_start
