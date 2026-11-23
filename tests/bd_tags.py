#!/usr/bin/env python3
"""BD-rate of one of our configurations against another (both run by tests/bench_all.py / compare.py), per clip.

Changes that move bits between frame types (key frames, ALTREFs) also move our curve along SVT's, and a 4-point BD
against SVT then mixes the change with where the curves overlap. Comparing our two curves directly, over their own
shared quality range, isolates the change.

Run: python3 tests/bd_tags.py BASE_TAG NEW_TAG [--clips 1,2,3]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--clips", default="")
    args = ap.parse_args()
    clips = sorted(glob.glob("benckmark/video_*.*"), key=lambda p: int(re.search(r"video_(\d+)", p).group(1)))
    if args.clips:
        keep = set(args.clips.split(","))
        clips = [c for c in clips if re.search(r"video_(\d+)", c).group(1) in keep]
    total, n = 0.0, 0
    for clip in clips:
        stem = os.path.splitext(os.path.basename(clip))[0]
        out = "build-cuda/compare" if stem == "video_1" else os.path.join("build-cuda/compare", stem)
        try:
            a = json.load(open(os.path.join(out, args.base + ".json")))[args.base]
            b = json.load(open(os.path.join(out, args.new + ".json")))[args.new]
        except (OSError, KeyError):
            print(f"{stem}: missing results")
            continue
        bd = compare.bd_rate([r["kbps"] for r in a], [r["psnr"] for r in a], [r["kbps"] for r in b], [r["psnr"] for r in b])
        lo = max(min(r["psnr"] for r in a), min(r["psnr"] for r in b))
        hi = min(max(r["psnr"] for r in a), max(r["psnr"] for r in b))
        print(f"{stem:8s} {args.new} vs {args.base}: {bd:+.2f}%  (shared range {lo:.2f}-{hi:.2f} dB)")
        total += bd
        n += 1
    if n:
        print(f"mean: {total / n:+.2f}% over {n} clips")


if __name__ == "__main__":
    main()
