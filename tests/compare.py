"""Rate/quality comparison of av1gpu-video against SVT-AV1 and libaom.

Run: python3 tests/compare.py BINARY [--src FILE] [--frames N] [--ours "ARGS"] [--refs svt,aom]

Encodes the first N frames (default 120) of --src (default benckmark/video_1.mp4), decodes every
output with dav1d, and reports bytes, kbit/s, PSNR-YUV (6:1:1), encode fps, and
BD-rate of av1gpu relative to each reference. Outputs go to build-cuda/compare/ (video_1) or
build-cuda/compare/<source name>/ (other sources).
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np

W, H, FPS = 1080, 1920, 30
FPSN, FPSD = 30, 1
SRC = "benckmark/video_1.mp4"
OUT = pathlib.Path("build-cuda/compare")
RANGE = "scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709"


def raw_source(frames):
    path = OUT / f"source_{frames}.yuv"
    if not path.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-map", "0:v:0", "-frames:v",
                        str(frames), "-vf", RANGE, "-f", "rawvideo", "-pix_fmt", "yuv420p", path],
                       check=True)
    return path


def decode(path):
    return subprocess.check_output(["ffmpeg", "-v", "error", "-xerror", "-c:v", "libdav1d", "-i",
                                    path, "-f", "rawvideo", "-"])


def psnr(ref, dist):
    size = W * H * 3 // 2
    assert len(ref) == len(dist), (len(ref), len(dist))
    a = np.frombuffer(ref, np.uint8).reshape(-1, size).astype(np.float64)
    b = np.frombuffer(dist, np.uint8).reshape(-1, size).astype(np.float64)
    y = W * H
    def p(sl):
        mse = np.mean((a[:, sl] - b[:, sl]) ** 2, axis=1)
        return np.mean(10 * np.log10(255 * 255 / np.maximum(mse, 1e-10)))
    return (6 * p(slice(0, y)) + p(slice(y, y + y // 4)) + p(slice(y + y // 4, size))) / 8


def bd_rate(r1, q1, r2, q2):
    """Bjontegaard delta rate of curve 2 vs curve 1 (percent, cubic fit)."""
    if len(r1) < 4 or len(r2) < 4:
        return float("nan")
    p1 = np.polyfit(q1, np.log(r1), 3)
    p2 = np.polyfit(q2, np.log(r2), 3)
    lo, hi = max(min(q1), min(q2)), min(max(q1), max(q2))
    if lo >= hi:
        return float("nan")
    i1 = np.polyval(np.polyint(p1), hi) - np.polyval(np.polyint(p1), lo)
    i2 = np.polyval(np.polyint(p2), hi) - np.polyval(np.polyint(p2), lo)
    return (math.exp((i2 - i1) / (hi - lo)) - 1) * 100


def run(name, cmd, source, frames, stdin=True, cache=False):
    out = OUT / f"{name}.ivf"
    cached = OUT / f"{name}_{frames}.row"
    if cache and cached.exists() and out.exists():
        row = json.loads(cached.read_text())
        print(json.dumps(row), flush=True)
        return row
    start = time.time()
    with open(source, "rb") as f:
        subprocess.run(cmd(out), stdin=f if stdin else None, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    seconds = time.time() - start
    ref = source.read_bytes()
    q = psnr(ref, decode(out))
    size = out.stat().st_size
    kbps = size * 8 / (frames / FPS) / 1000
    row = {"name": name, "bytes": size, "kbps": round(kbps, 1), "psnr": round(q, 3),
           "fps": round(frames / seconds, 2)}
    print(json.dumps(row), flush=True)
    cached.write_text(json.dumps(row))
    return row


def main():
    global W, H, FPS, FPSN, FPSD, SRC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--src", default="benckmark/video_1.mp4")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--ours", default="--fast --inter-qindex {q} --qindex {q} --rolling-reference")
    ap.add_argument("--qs", default="60,90,120,160")
    ap.add_argument("--refs", default="svt")
    ap.add_argument("--tag", default="ours")
    args = ap.parse_args()
    if args.src != "benckmark/video_1.mp4":
        info = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height,r_frame_rate", "-of", "csv=p=0", args.src], text=True)
        w, h, rate = info.strip().split(",")
        num, den = rate.split("/")
        W, H, FPS = int(w), int(h), float(num) / float(den)
        FPSN, FPSD = int(num), int(den)
        SRC = args.src
        OUT = OUT / pathlib.Path(args.src).stem
    OUT.mkdir(parents=True, exist_ok=True)
    src = raw_source(args.frames)
    size = f"{W}x{H}"
    rawin = ["-f", "rawvideo", "-color_range", "pc", "-pix_fmt", "yuv420p", "-s", size, "-r", f"{FPSN}/{FPSD}", "-i", str(src)]
    curves = {}
    for q in map(int, args.qs.split(",")):
        extra = args.ours.format(q=q).split()
        curves.setdefault(args.tag, []).append(run(
            f"{args.tag}_q{q}", lambda o: [args.binary, str(W), str(H), str(FPSN), str(FPSD), str(o), *extra],
            src, args.frames))
    refs = [r for r in args.refs.split(",") if r]
    for crf in (24, 32, 40, 48):
        if "svt" in refs:
            curves.setdefault("svt", []).append(run(
                f"svt_crf{crf}", lambda o: ["ffmpeg", "-v", "error", "-y", *rawin, "-c:v", "libsvtav1",
                                            "-preset", "8", "-crf", str(crf), "-color_range", "pc", str(o)],
                src, args.frames, stdin=False, cache=True))
        if "aom" in refs:
            curves.setdefault("aom", []).append(run(
                f"aom_crf{crf}", lambda o: ["ffmpeg", "-v", "error", "-y", *rawin, "-c:v", "libaom-av1",
                                            "-cpu-used", "8", "-row-mt", "1", "-crf", str(crf), "-b:v", "0",
                                            "-color_range", "pc", str(o)],
                src, args.frames, stdin=False, cache=True))
    ours = curves[args.tag]
    for ref in refs:
        c = curves[ref]
        bd = bd_rate([r["kbps"] for r in c], [r["psnr"] for r in c],
                     [r["kbps"] for r in ours], [r["psnr"] for r in ours])
        print(f"BD-rate {args.tag} vs {ref}: {bd:+.1f}%")
    (OUT / f"{args.tag}.json").write_text(json.dumps(curves, indent=1))


if __name__ == "__main__":
    sys.exit(main())
