#!/usr/bin/env python3
"""BD-rate of one encoder configuration on every clip of benckmark/ (first N frames), against SVT-AV1 preset 8 and
libaom cpu-used 8 (reference encodes are cached by tests/compare.py).

Run: python3 tests/bench_all.py BINARY --tag TAG [--ours "ARGS"] [--clips 1,2,3] [--frames 120]
"""
import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ours", default="--pframes --persistent-cdf --qindex {q}")
    ap.add_argument("--qs", default="20,50,90,130")
    ap.add_argument("--refs", default="svt,aom")
    ap.add_argument("--clips", default="")
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()
    clips = sorted(glob.glob("benckmark/video_*.*"), key=lambda p: int(re.search(r"video_(\d+)", p).group(1)))
    if args.clips:
        keep = set(args.clips.split(","))
        clips = [c for c in clips if re.search(r"video_(\d+)", c).group(1) in keep]
    rows = []
    for clip in clips:
        out = subprocess.run([sys.executable, "tests/compare.py", args.binary, "--src", clip, "--tag", args.tag,
                              "--qs", args.qs, "--refs", args.refs, "--ours", args.ours,
                              "--frames", str(args.frames)], capture_output=True, text=True)
        if out.returncode:
            print(f"{clip}: compare.py failed\n{out.stderr[-2000:]}", file=sys.stderr)
            continue
        bd = dict(re.findall(r"BD-rate \S+ vs (\w+): ([+-]?[\d.]+|nan)%", out.stdout))
        fps = [json.loads(l)["fps"] for l in out.stdout.splitlines()
               if l.startswith("{") and json.loads(l)["name"].startswith(args.tag + "_q")]
        rows.append((os.path.basename(clip), bd, statistics.median(fps) if fps else float("nan")))
        print(f"{rows[-1][0]:14s} " + " ".join(f"vs {r} {bd.get(r, 'nan'):>6s}%" for r in args.refs.split(",")) +
              f"  median fps {rows[-1][2]:.1f}", flush=True)
    for r in args.refs.split(","):
        vals = [float(b[r]) for _, b, _ in rows if r in b and b[r] != "nan"]
        if vals:
            print(f"mean BD-rate vs {r}: {sum(vals) / len(vals):+.1f}% over {len(vals)} clips")


if __name__ == "__main__":
    main()
