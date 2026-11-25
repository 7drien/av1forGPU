# AV1 for RTX 30

An experimental AV1 **software encoder** running on CUDA, for NVIDIA GeForce RTX 30-series GPUs. These GPUs have no
AV1 encode hardware (NVENC AV1), so every coding decision is computed in CUDA kernels. The host CPU manages codec state,
syntax and the bitstream.

## Results at a glance

First 120 frames of five test clips, RTX 3060 Ti, clean GPU (2026-11-25). **Size** is BD-rate against SVT-AV1 preset 8:
bits needed for the same quality (negative = smaller files). **Speed** is our median fps relative to SVT-AV1 preset 8,
wall time including process start ([details](docs/benchmark.md)).

| Clip | Size vs SVT-AV1 p8 | Speed vs SVT-AV1 p8 | fps (ours / SVT) |
| --- | ---: | ---: | ---: |
| video_1, 1080×1920 static | +1.9% | −11% | 32.7 / 36.7 |
| video_2, 1080p concert | +4.6% | −30% | 23.3 / 33.1 |
| video_3, 4K street | **−16.5%** | −55% | 7.9 / 17.6 |
| video_4, 4K advert | **−29.8%** | −49% | 8.2 / 16.2 |
| video_5, 1080p animation | **−10.9%** | −41% | 24.8 / 41.9 |
| Mean | **−10.1%** | | |

Files are smaller than SVT-AV1 preset 8's on three of five clips (and within 5% on the other two); encoding is 11 to 55%
slower. Every stream decodes
with dav1d to exactly the encoder's reconstruction.

## Build

Requirements: NVIDIA driver, CUDA Toolkit 12+, CMake 3.24+, a C++17 compiler, JPEG / PNG / WebP development libraries,
and FFmpeg for input decoding and tests.

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real   # RTX 30: sm_86
cmake --build build-cuda -j 8 --target av1gpu-video
```

## Encode a video

The encoder reads full-range BT.709 raw YUV420 on standard input and writes IVF. Its arguments are width, height,
frame-rate numerator and denominator, and the output path.

```sh
ffmpeg -hide_banner -loglevel error -i benckmark/video_1.mp4 -map 0:v:0 \
  -vf scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709 -f rawvideo -pix_fmt yuv420p - | \
  build-cuda/third_party/av1gpu/av1gpu-video 1080 1920 30 1 out.ivf --pframes --persistent-cdf --qindex 100
```

`--pframes` is the recommended mode. Without it the tool codes independent key frames, in batches.

### What `--pframes` does

- **GOP structure:** key frames (scene cuts, or every 120 frames by default), then mini-GOPs of up to 32 frames
  (`--pf-gop`) coded as a hierarchical pyramid. Fast motion shortens them to 16 or 8 frames.
  - The last frame of each mini-GOP is coded first as a hidden ALTREF. On noisy content it is temporally filtered over
    ±8 frames.
  - The other frames predict from LAST, ALTREF, or their compound average.
- **Motion search (CUDA):** a hierarchical coarse search (1/16 and 1/4 resolution), then per-block full-pel and
  quarter-pel refinement on exact AV1 interpolation phase planes. Every AV1 partition shape from 64×64 to 8×8 is searched
  except those with a 4-pixel side.
- **Mode decision (CUDA):** for each block, by rate-distortion cost:
  - partition and skip
  - transform size and split (TX_MODE_SELECT) and transform type
  - interpolation filter (switchable)
  - then six refinement passes re-check vectors against the decoder's candidate lists
- **Intra blocks in inter frames:** blocks of every shape may switch to intra. They choose among 10 luma modes, a chroma mode,
  intra transform types and splits, and are reconstructed in decode order by a superblock wavefront.
- **In-loop filters (CUDA):** deblocking, CDEF (per-superblock presets) and luma Wiener loop restoration, all chosen by
  measured error.
- **Entropy coding:** CUDA builds the mode info and coefficient contexts and emits symbol records. A writer thread
  range-codes them, overlapped with the GPU's work on later frames.

### Main options

| Option | Effect |
| --- | --- |
| `--qindex N` | key-frame quantizer (1–255, default 100); P frames use N + 90 (`--pf-qoffset`), lowered toward the pyramid base |
| `--bitrate KBPS` | rate control, one base quantizer per key frame or mini-GOP (`--rc-buffer` seconds, default 2) |
| `--persistent-cdf` | carry adapted probabilities between frames (recommended) |
| `--kf-preset NAME` | key-frame search preset (default `fast`; `slow` is 9–22% slower for 0.2–1.1% better BD-rate) |
| `--pf-tf-frames K` | ALTREF temporal filter range (default 8, 0 off; noisy content only) |
| `--pf-tiles C R` | log2 tile columns and rows |
| `--pf-no-intra`, `--pf-no-rect`, `--pf-no-tx-split`, `--pf-no-filters` | turn off individual tools |
| `--pf-cpu-entropy`, `--pf-gpu-entropy` | where the range coder runs (identical streams; the default is the fastest) |

Environment variables named `AV1GPU_PF_*` expose experiment knobs. They are documented where they are measured, in
[docs/svt_gap_plan.md](docs/svt_gap_plan.md).

## Test

```sh
python3 tests/check_pframes.py build-cuda/third_party/av1gpu/av1gpu-video   # dav1d decodes exactly the reconstruction
python3 tests/bench_all.py build-cuda/third_party/av1gpu/av1gpu-video --tag mytag --refs svt   # BD-rate, all clips
python3 tests/check_inter.py build-cuda/third_party/av1gpu/av1gpu-video     # older --inter modes
```

## Repository layout

- `third_party/av1gpu/`: the CUDA AV1 encoder. It is based on [maddog107/av1gpu](https://github.com/maddog107/av1gpu)
  (BSD-2-Clause) and extended here with the inter-frame encoder (`encoder/apps/pframe*.cu/.h`, `av1gpu_video.cpp`).
- `src/`, `include/`: the original analysis prototype and header writers (`make` builds them).
- `tests/`: bitstream checks and benchmark scripts.
- `docs/`:
  - [benchmark.md](docs/benchmark.md): results
  - [svt_gap_plan.md](docs/svt_gap_plan.md): measurement log
  - [architecture.md](docs/architecture.md): design notes

The older experimental temporal modes (`--inter`, `--rolling-reference`, `--inter-qindex`, `--adaptive-fps`, ...) are
still available. `av1gpu-video --help` lists them, and [benckmark/inter_sweep](benckmark/inter_sweep/README.md) has
their results.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md). The project is released under the [BSD 2-Clause License](LICENSE.md).
