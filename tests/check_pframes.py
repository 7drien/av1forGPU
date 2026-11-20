"""Run: python3 tests/check_pframes.py path/to/av1gpu-video [W H FRAMES]

Encodes moving synthetic content with --pframes, decodes it with dav1d, and requires the decoded
pictures to equal the encoder's own reconstruction (AV1GPU_DUMP_RECON) byte for byte.
"""
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np


def synth(w, h, frames, seed=5):
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 256, (h + 32 + frames, w + 32 + frames)).astype(np.float32)
    # smooth texture with some sharp detail
    k = np.ones(5) / 5
    big = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, big)
    big = np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, big)
    big = (big - big.min()) / (big.max() - big.min()) * 220 + 16
    out = bytearray()
    for f in range(frames):
        y = big[16 + f // 2: 16 + f // 2 + h, 16 + f: 16 + f + w].copy()
        # a block moving the other way
        bx, by = (w // 3 + 3 * f) % (w - 32), h // 3
        y[by:by + 32, bx:bx + 32] = 255 - y[by:by + 32, bx:bx + 32]
        y = y.clip(0, 255).astype(np.uint8)
        u = (y[::2, ::2].astype(np.int32) // 2 + 64).astype(np.uint8)
        v = (255 - y[1::2, 1::2]).astype(np.uint8)
        out += y.tobytes() + u.tobytes() + v.tobytes()
    return bytes(out)


def run(binary, w, h, frames, extra=()):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        raw = synth(w, h, frames)
        ivf, recon = td / "pf.ivf", td / "recon.yuv"
        env = dict(os.environ, AV1GPU_DUMP_RECON=str(recon))
        subprocess.run([binary, str(w), str(h), "30", "1", str(ivf), "--pframes", *extra], input=raw,
                       check=True, env=env)
        dec = subprocess.check_output(["ffmpeg", "-v", "error", "-xerror", "-c:v", "libdav1d", "-i", ivf,
                                       "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
        rec = recon.read_bytes()
        fs = w * h * 3 // 2
        assert len(dec) == frames * fs, (len(dec), frames * fs)
        assert len(rec) == len(dec), (len(rec), len(dec))
        for i in range(frames):
            a = np.frombuffer(dec[i * fs:(i + 1) * fs], np.uint8)
            b = np.frombuffer(rec[i * fs:(i + 1) * fs], np.uint8)
            if not np.array_equal(a, b):
                bad = np.nonzero(a != b)[0]
                raise SystemExit(f"frame {i}: {len(bad)} samples differ, first at {bad[0]} "
                                 f"(plane offset {bad[0] - (0 if bad[0] < w*h else w*h)})")
        src = np.frombuffer(raw, np.uint8).astype(np.float64)
        mse = np.mean((src - np.frombuffer(dec, np.uint8)) ** 2)
        print(f"{w}x{h} {frames} frames {extra}: decoder matches encoder, "
              f"{ivf.stat().st_size} bytes, PSNR {10 * np.log10(255 * 255 / mse):.2f} dB")


if __name__ == "__main__":
    b = sys.argv[1]
    if len(sys.argv) > 4:
        run(b, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        run(b, 256, 128, 6, ["--pf-gop", "1"])
        run(b, 200, 136, 6, ["--qindex", "60", "--pf-gop", "1"])
        run(b, 448, 256, 8, ["--persistent-cdf", "--pf-gop", "1"])
        run(b, 256, 128, 12)
        run(b, 200, 136, 11, ["--qindex", "60", "--pf-gop", "4"])
        run(b, 448, 256, 20, ["--persistent-cdf", "--pf-gop", "8", "--keyint", "9"])
        run(b, 320, 192, 18, ["--persistent-cdf", "--pf-gop", "16", "--qindex", "140"])
        run(b, 320, 192, 40, ["--persistent-cdf", "--bitrate", "400"])
        run(b, 448, 256, 12, ["--persistent-cdf", "--pf-tiles", "1", "1"])
        run(b, 712, 392, 18, ["--persistent-cdf", "--pf-tiles", "2", "2", "--qindex", "70"])
        run(b, 712, 392, 10, ["--pf-tiles", "3", "0", "--pf-gop", "1"])
        run(b, 712, 392, 18, ["--persistent-cdf", "--pf-gpu-entropy", "--qindex", "70"])
        run(b, 448, 256, 12, ["--persistent-cdf", "--pf-switchable", "--qindex", "70"])
        run(b, 448, 256, 12, ["--persistent-cdf", "--pf-switchable", "--pf-cpu-entropy"])
        run(b, 448, 256, 12, ["--persistent-cdf", "--pf-tx-types-reduced"])
        # 56 visible rows in the last superblock row: blocks and transforms cross the bottom edge
        run(b, 448, 184, 16, ["--persistent-cdf", "--qindex", "50"])
        run(b, 448, 184, 12, ["--pf-no-filters", "--pf-gop", "1"])
        # Final-partition re-evaluation also uses the unsplit evaluation kernel.
        run(b, 448, 184, 12, ["--persistent-cdf", "--qindex", "50", "--pf-no-tx-split", "--pf-no-rect"])
        run(b, 448, 256, 20, ["--persistent-cdf", "--pf-refs", "3", "--pf-gop", "16"])
        run(b, 712, 392, 18, ["--persistent-cdf", "--pf-lr", "--qindex", "120"])
        run(b, 448, 256, 12, ["--persistent-cdf", "--pf-lr", "--pf-cpu-entropy", "--pf-tiles", "1", "1"])
        run(b, 448, 256, 12, ["--persistent-cdf", "--pf-lr", "--pf-gpu-entropy"])
        run(b, 448, 256, 12, ["--persistent-cdf", "--pf-cpu-entropy"])
        run(b, 712, 392, 18, ["--persistent-cdf", "--pf-cpu-entropy", "--pf-tiles", "1", "1"])
        run(b, 256, 128, 70, ["--persistent-cdf", "--pf-gop", "32", "--keyint", "200"])
