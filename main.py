#!/usr/bin/env python3
"""
Shazam for Movies - Visual Fingerprinting Engine

Usage:
    python main.py index --file movie.mp4 --title "Movie Name"
    python main.py query --file clip.mp4
    python main.py test  --file movie.mp4 --title "Movie Name" [--start SEC] [--duration SEC]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import random
import subprocess
import tempfile
import sys

from engine.indexer import index_movie, get_duration
from engine.matcher import match_clip
from engine.db import Database
from engine.visual_fingerprint import _get_clip


def _warmup():
    """Load CLIP model before FAISS index to avoid OMP segfault on macOS."""
    _get_clip()


def cmd_index(args):
    _warmup()
    db = Database()
    try:
        movie_id = index_movie(args.file, args.title, db=db)
        print(f"Done. Movie ID: {movie_id}")
    finally:
        db.close()


def cmd_reset(args):
    """Reset all indexes and fingerprint data. Movies table is preserved."""
    db = Database()
    try:
        db.reset_indexes()
        db.save()
        print("Reset complete. Re-index your movies with: python main.py index --file ... --title ...")
    finally:
        db.close()


def cmd_query(args):
    _warmup()
    db = Database()
    try:
        result = match_clip(args.file, db=db, detect_screen=args.screen)
        if result is None:
            print("No confident match found.")
            sys.exit(1)
    finally:
        db.close()


def cmd_test(args):
    _warmup()
    db = Database()
    try:
        movie_id = index_movie(args.file, args.title, db=db)

        duration = get_duration(args.file)
        if args.start is not None:
            start = args.start
        else:
            margin = 60
            max_start = duration - margin - args.duration
            start = random.uniform(margin, max(margin + 1, max_start))

        clip_duration = args.duration
        print(f"\n{'='*50}")
        print(f"TEST: Extracting {clip_duration}s clip starting at {start:.1f}s")
        print(f"  Expected: {start:.1f}s "
              f"({int(start//3600):02d}:{int(start%3600//60):02d}:{start%60:05.2f})")
        print(f"{'='*50}")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            clip_path = tmp.name

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", args.file,
                "-t", str(clip_duration),
                "-c", "copy",
                "-loglevel", "error",
                clip_path,
            ], check=True)

            result = match_clip(clip_path, db=db)

            if result is None:
                print("\nTEST FAILED: No match found")
                sys.exit(1)

            delta = abs(result.timestamp_sec - start)
            print(f"TEST EVALUATION:")
            print(f"  Expected:   {start:.1f}s")
            print(f"  Got:        {result.timestamp_sec:.1f}s")
            print(f"  Delta:      {delta:.1f}s")
            print(f"  Movie:      {result.movie_title}")
            print(f"  Confidence: {result.confidence:.3f}")

            if delta < 5.0 and result.movie_id == movie_id:
                print(f"\n  PASS (delta={delta:.1f}s)")
            elif result.movie_id == movie_id:
                print(f"\n  PARTIAL (correct movie, delta={delta:.1f}s)")
            else:
                print(f"\n  FAIL (wrong movie or large delta)")
                sys.exit(1)

        finally:
            if os.path.exists(clip_path):
                os.unlink(clip_path)

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Shazam for Movies - Visual Fingerprinting Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index a movie")
    p_index.add_argument("--file", required=True)
    p_index.add_argument("--title", required=True)

    p_query = sub.add_parser("query", help="Query a clip")
    p_query.add_argument("--file", required=True)
    p_query.add_argument("--screen", action="store_true",
                         help="Enable screen detection (for laptop/monitor captures)")

    p_test = sub.add_parser("test", help="End-to-end roundtrip test")
    p_test.add_argument("--file", required=True)
    p_test.add_argument("--title", required=True)
    p_test.add_argument("--start", type=float, default=None)
    p_test.add_argument("--duration", type=float, default=10.0)

    sub.add_parser("reset", help="Reset all indexes (re-index movies after)")

    args = parser.parse_args()
    {"index": cmd_index, "query": cmd_query, "test": cmd_test, "reset": cmd_reset}[args.command](args)


if __name__ == "__main__":
    main()
