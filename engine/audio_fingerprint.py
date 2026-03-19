"""
Shazam-style audio fingerprinting using constellation maps.

Based on: Wang, A. (2003) "An Industrial-Strength Audio Search Algorithm"
https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf

Algorithm:
1. Extract mono audio at low sample rate
2. Compute spectrogram via STFT
3. Find spectral peaks (constellation map)
4. Generate combinatorial hashes from peak pairs (anchor + target)
5. Each hash is paired with its time offset for alignment voting

Processes audio in chunks to handle long movies without excessive memory usage.
"""

import subprocess
import numpy as np
from scipy.signal import stft
from scipy.ndimage import maximum_filter
from engine.config import (
    SAMPLE_RATE, FFT_WINDOW_SIZE, FFT_HOP_SIZE,
    PEAK_NEIGHBORHOOD_SIZE, FAN_VALUE,
    TARGET_T_DELTA_MIN, TARGET_T_DELTA_MAX,
)

# Audio chunk size in seconds for processing long files
AUDIO_CHUNK_SEC = 60


def extract_audio(video_path: str) -> np.ndarray:
    """Extract mono audio from video as float32 numpy array at SAMPLE_RATE Hz."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def extract_audio_segment(video_path: str, start_sec: float, duration_sec: float) -> np.ndarray:
    """Extract a segment of audio from video."""
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration_sec),
        "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "-loglevel", "error",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def compute_spectrogram(samples: np.ndarray) -> np.ndarray:
    """Compute log-magnitude spectrogram via STFT."""
    _, _, Zxx = stft(
        samples,
        fs=SAMPLE_RATE,
        window="hamming",
        nperseg=FFT_WINDOW_SIZE,
        noverlap=FFT_WINDOW_SIZE - FFT_HOP_SIZE,
    )
    magnitude = np.abs(Zxx)
    return 10 * np.log10(magnitude + 1e-10)


def find_peaks(spectrogram: np.ndarray) -> list[tuple[int, int]]:
    """
    Find local maxima in the spectrogram (constellation map points).
    Returns list of (time_frame, freq_bin) tuples.

    Uses maximum_filter with a small integer size for efficiency,
    plus adaptive percentile thresholding for robustness across
    different audio levels/encodings.
    """
    # Use size tuple instead of footprint — much more memory efficient
    local_max = maximum_filter(spectrogram, size=PEAK_NEIGHBORHOOD_SIZE) == spectrogram

    # Adaptive threshold: top 20% of energy
    threshold = np.percentile(spectrogram, 80)
    peaks_mask = local_max & (spectrogram > threshold)

    freq_bins, time_frames = np.where(peaks_mask)
    peaks = list(zip(time_frames.tolist(), freq_bins.tolist()))
    peaks.sort(key=lambda p: (p[0], p[1]))
    return peaks


def generate_hashes(peaks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Generate combinatorial hashes from constellation map peaks.
    Hash = (freq_anchor, freq_target, time_delta) packed into 32-bit int.
    Returns list of (hash_value, anchor_time_frame).
    """
    hashes = []
    n_peaks = len(peaks)

    for i in range(n_peaks):
        anchor_t, anchor_f = peaks[i]
        targets_found = 0
        for j in range(i + 1, n_peaks):
            target_t, target_f = peaks[j]
            t_delta = target_t - anchor_t

            if t_delta < TARGET_T_DELTA_MIN:
                continue
            if t_delta > TARGET_T_DELTA_MAX:
                break

            hash_val = (anchor_f & 0x3FF) << 20 | (target_f & 0x3FF) << 10 | (t_delta & 0xFFF)
            hashes.append((hash_val, anchor_t))

            targets_found += 1
            if targets_found >= FAN_VALUE:
                break

    return hashes


def _fingerprint_chunk(samples: np.ndarray, time_offset_frames: int) -> list[tuple[int, int]]:
    """Fingerprint a chunk of audio samples, offsetting times by time_offset_frames."""
    spec = compute_spectrogram(samples)
    peaks = find_peaks(spec)
    hashes = generate_hashes(peaks)
    # Offset the anchor times to be relative to the full audio
    return [(h, t + time_offset_frames) for h, t in hashes]


def fingerprint_audio(video_path: str) -> list[tuple[int, int]]:
    """
    Full audio fingerprinting pipeline for a video file.
    Processes in chunks to handle long movies efficiently.
    Returns list of (hash_value, time_offset_frame).
    """
    print("  Extracting audio...")
    samples = extract_audio(video_path)
    total_sec = len(samples) / SAMPLE_RATE
    print(f"  Audio: {len(samples)} samples ({total_sec:.1f}s)")

    chunk_samples = AUDIO_CHUNK_SEC * SAMPLE_RATE
    # Add overlap between chunks so peaks at boundaries aren't missed
    overlap_samples = int(2 * SAMPLE_RATE)  # 2 second overlap
    frames_per_sec = SAMPLE_RATE / FFT_HOP_SIZE

    all_hashes = []
    n_chunks = max(1, int(np.ceil(len(samples) / chunk_samples)))

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_samples
        end = min(start + chunk_samples + overlap_samples, len(samples))
        chunk = samples[start:end]

        # Time offset in STFT frames
        time_offset = int(start / FFT_HOP_SIZE)

        hashes = _fingerprint_chunk(chunk, time_offset)
        all_hashes.extend(hashes)

        pct = min(100, (chunk_idx + 1) / n_chunks * 100)
        print(f"  Audio: chunk {chunk_idx+1}/{n_chunks} ({pct:.0f}%) - {len(hashes)} hashes", end="\r")

    # Deduplicate (overlap can cause dupes)
    all_hashes = list(set(all_hashes))
    print(f"  Audio: {len(all_hashes)} total hashes from {n_chunks} chunks   ")
    return all_hashes


def fingerprint_audio_clip(clip_path: str) -> list[tuple[int, int]]:
    """Fingerprint a short audio clip."""
    return fingerprint_audio(clip_path)
