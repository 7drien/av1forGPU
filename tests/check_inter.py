"""Run: python3 tests/check_inter.py path/to/av1gpu-video (requires FFmpeg and NumPy)."""
import pathlib
import subprocess
import sys
import tempfile

import numpy as np


def encode(binary, output, raw, mode, diagonal=True):
    motion = ["--diagonal-motion"] if diagonal else []
    subprocess.run([binary, "448", "128", "30", "1", output, "--fast", *motion, *mode],
                   input=raw, check=True)


def decode(path, filters=()):
    return subprocess.check_output(["ffmpeg", "-v", "error", "-xerror", "-c:v", "libdav1d",
                                    "-i", path, *filters, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])


def translate(reference, step=2, modes=7, block_size=64):
    shifted = bytearray()
    offset = 0
    for w, h, block in ((448, 128, block_size), (224, 64, block_size//2), (224, 64, block_size//2)):
        plane = np.frombuffer(reference, np.uint8, w*h, offset).reshape(h, w)
        offset += w*h
        y, x = np.indices(plane.shape)
        mode = (x//block + y//block) % modes
        dx = (((mode == 1) | (mode == 5)).astype(int) - ((mode == 2) | (mode == 6))) * (step if w == 448 else step//2)
        dy = (((mode == 3) | (mode == 5)).astype(int) - ((mode == 4) | (mode == 6))) * (step if w == 448 else step//2)
        shifted.extend(plane[(y+dy).clip(0,h-1), (x+dx).clip(0,w-1)].tobytes())
    return bytes(shifted)


def main(binary):
    with tempfile.TemporaryDirectory() as temp:
        directory = pathlib.Path(temp)
        key = directory / "key.ivf"
        source = np.random.default_rng(1234).integers(256, size=448*128*3//2, dtype=np.uint8).tobytes()
        encode(binary, key, source, [])
        reference = decode(key)
        # Mixed static blocks and six integer translations from the reconstructed keyframe.
        shifted = translate(reference)
        raw = source + shifted*4
        expected = reference + shifted*4
        video = directory / "inter.ivf"
        encode(binary, video, raw, ["--inter"])
        assert decode(video) == expected
        twice_shifted = translate(shifted)
        encode(binary, video, source + shifted + twice_shifted, ["--rolling-reference"])
        assert decode(video) == reference + shifted + twice_shifted
        types = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "frame=pict_type",
                                         "-of", "csv=p=0", video]).split()
        assert types == [b"I", b"P", b"P"], types
        cardinal = translate(reference, modes=5)
        encode(binary, video, source + cardinal + reference,
               ["--rolling-reference", "--anchor-reference"], diagonal=False)
        assert decode(video) == reference + cardinal + reference
        types = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "frame=pict_type",
                                         "-of", "csv=p=0", video]).split()
        assert types == [b"I", b"P", b"P"], types
        # At tick 2 the anchor becomes the decoded second P picture. Return
        # to that picture after displaying unrelated content in the rolling slot.
        second = translate(cardinal, modes=5)
        unrelated = bytes(value ^ 255 for value in second)
        encode(binary, video, source + cardinal + second + unrelated + second,
               ["--rolling-reference", "--anchor-interval", "2", "--lossless-inter-residual"],
               diagonal=False)
        assert decode(video) == reference + cardinal + second + unrelated + second
        for step in (4, 8, 16):
            wide = translate(reference, step)
            encode(binary, video, source + wide, ["--rolling-reference", "--motion-step", str(step)])
            assert decode(video) == reference + wide
        wide = translate(reference, step=16)
        narrow = translate(wide, step=4)
        encode(binary, video, source + wide + narrow, ["--rolling-reference", "--auto-motion-step"])
        assert decode(video) == reference + wide + narrow
        types = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "frame=pict_type",
                                         "-of", "csv=p=0", video]).split()
        assert types == [b"I", b"P", b"P"], types
        # Force nonzero coefficients in all three planes of the two inter frames.
        changed = bytearray(shifted)
        for i in range(0, len(changed), 113):
            changed[i] ^= 31
        encode(binary, video, source + changed + twice_shifted, ["--lossless-inter-residual"])
        assert decode(video) == reference + changed + twice_shifted
        types = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "frame=pict_type",
                                         "-of", "csv=p=0", video]).split()
        assert types == [b"I", b"P", b"P"], types
        # Single-sample corrections fall below the coefficient threshold.
        sparse_source = bytearray(shifted)
        sparse_source[9] ^= 1
        encode(binary, video, source + sparse_source, ["--inter-residual-threshold", "16"])
        assert decode(video) == reference + shifted
        # A DC-only correction survives thresholding and enters the next reference.
        flat = bytes([96]) * len(source)
        brighter = bytes([104]) * len(source)
        brighter_again = bytes([112]) * len(source)
        encode(binary, video, flat + brighter + brighter_again,
               ["--inter-residual-threshold", "32", "--rolling-reference"])
        assert decode(video) == flat + brighter + brighter_again
        encode(binary, video, flat + brighter + brighter_again,
               ["--inter-qindex", "32", "--rolling-reference"])
        # Q32 rounds the first DC correction to nine sample levels.
        assert decode(video) == flat + bytes([105])*len(source) + brighter_again
        encode(binary, video, source + changed + twice_shifted,
               ["--inter-qindex", "24", "--rolling-reference"])
        decoded = decode(video)
        expected_dct = reference + changed + twice_shifted
        assert len(decoded) == len(expected_dct)
        diff = (np.frombuffer(decoded, np.uint8).astype(np.float32) -
                np.frombuffer(expected_dct, np.uint8))
        assert np.mean(diff * diff) < 4, np.mean(diff * diff)
        # Independently moving 16x16 areas need locally selected references.
        local = translate(reference, modes=5, block_size=16)
        coarse = directory / "coarse.ivf"
        encode(binary, coarse, source + local,
               ["--inter-qindex", "24", "--rolling-reference"], diagonal=False)
        encode(binary, video, source + local,
               ["--inter-qindex", "24", "--rolling-reference", "--motion-block", "16"], diagonal=False)
        assert decode(video) == reference + local
        assert video.stat().st_size < coarse.stat().st_size * 0.75
        encode(binary, video, source + changed + twice_shifted,
               ["--inter-qindex", "24", "--inter-tx8", "--rolling-reference"])
        decoded = decode(video)
        assert len(decoded) == len(expected_dct)
        diff = (np.frombuffer(decoded, np.uint8).astype(np.float32) -
                np.frombuffer(expected_dct, np.uint8))
        assert np.mean(diff * diff) < 4, np.mean(diff * diff)
        encode(binary, video, source + changed + twice_shifted,
               ["--inter-qindex", "24", "--inter-tx-adaptive", "--rolling-reference"])
        decoded = decode(video)
        assert len(decoded) == len(expected_dct)
        diff = (np.frombuffer(decoded, np.uint8).astype(np.float32) -
                np.frombuffer(expected_dct, np.uint8))
        assert np.mean(diff * diff) < 4, np.mean(diff * diff)
        encode(binary, video, source + changed + twice_shifted,
               ["--inter-qindex", "24", "--inter-tx-adaptive", "--inter-rdoq", "--rolling-reference"])
        decoded = decode(video)
        assert len(decoded) == len(expected_dct)
        diff = (np.frombuffer(decoded, np.uint8).astype(np.float32) -
                np.frombuffer(expected_dct, np.uint8))
        assert np.mean(diff * diff) < 4, np.mean(diff * diff)
        # Persistent CDFs change only entropy coding: several chained P frames
        # must decode to exactly the same pictures as independent CDFs.
        chain = [cardinal]
        for _ in range(4):
            chain.append(translate(chain[-1], modes=5))
        decoded = []
        for extra in ([], ["--persistent-cdf"]):
            encode(binary, video, source + b"".join(chain),
                   ["--inter-qindex", "24", "--rolling-reference", "--inter-tx-adaptive",
                    "--inter-rdoq", "--anchor-reference", *extra], diagonal=False)
            decoded.append(decode(video))
        assert decoded[0] == decoded[1]
        cut = bytes([180]) * len(source)
        encode(binary, video, flat + flat + cut + cut,
               ["--inter-qindex", "32", "--rolling-reference", "--scene-cut-rms", "20"])
        types = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "frame=pict_type",
                                         "-of", "csv=p=0", video]).split()
        assert types == [b"I", b"P", b"I", b"P"], types
        assert decode(video) == flat + flat + cut + cut
        encode(binary, video, raw, ["--adaptive-fps"])
        pts = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "packet=pts",
                                       "-of", "csv=p=0", video]).split()
        assert pts == [b"0", b"1", b"4"], pts
        mp4 = directory / "held.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-c", "copy", mp4], check=True)
        assert decode(mp4, ["-vf", "fps=30"]) == expected


if __name__ == "__main__":
    main(sys.argv[1])
