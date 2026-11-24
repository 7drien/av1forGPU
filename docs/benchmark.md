# Benchmarks

All results: NVIDIA GeForce RTX 3060 Ti (8 GB, `sm_86`), driver 595.99.02, CUDA 12.4.

## Current results (`--pframes`, 2026-11-24, with rectangular intra blocks)

First 120 frames of every clip in `benckmark/`, encoded with
`--pframes --persistent-cdf --qindex {20,50,90,130}`, decoded with dav1d. BD-rate uses YUV-weighted PSNR (6:1:1).
**Negative BD-rate means fewer bits than the reference at the same quality.**

| Clip | Content | BD-rate vs SVT-AV1 preset 8 | BD-rate vs libaom cpu-used 8 | fps (ours) | fps (SVT-AV1 p8) |
| --- | --- | ---: | ---: | ---: | ---: |
| video_1 | 1080×1920, 30 fps, static | +0.2% | | 34.9 | 36.5 |
| video_2 | 1920×1080, 30 fps, concert, cuts | +6.7% | | 27.0 | 34.4 |
| video_3 | 3840×2160, 24 fps, handheld street | **−18.2%** | | 8.7 | 17.3 |
| video_4 | 3840×2160, 24 fps, advert | **−32.1%** | | 9.4 | 16.4 |
| video_5 | 1920×1080, animation | **−10.9%** | | 26.3 | 41.2 |
| **Mean** | | **−10.9%** | | | |

- **Compression:** better than SVT-AV1 preset 8 on three of five clips, at parity on video_1, behind on video_2.
  libaom cpu-used 8 (before rectangular intra: +25.4%, +15.7%, +43.4%, +9.6%, +4.9%, mean +19.8%) is not re-measured.
- **Speed:** 0.5× to 0.95× of SVT-AV1 preset 8. fps is the median of the four encodes, wall time including process
  startup.
- SVT and libaom figures come from cached reference encodes (`tests/compare.py`).
- The machine also ran other CPU jobs (load average 8 to 19). The latest run shared the GPU with another job, so its
  fps (not shown) was about half; the fps above come from the previous clean run (identical output).

### Progress during the day (BD-rate vs SVT-AV1 preset 8)

| Clip | Morning (`b40`) | Now | fps morning → now |
| --- | ---: | ---: | ---: |
| video_1 | +5.3% | +0.2% | 24.5 → 34.9 |
| video_2 | +18.4% | +6.7% | 19.5 → 27.0 |
| video_3 | −13.2% | −18.2% | 7.8 → 8.7 |
| video_4 | −25.6% | −32.1% | 7.9 → 9.4 |
| video_5 | −10.8% | −10.9% | 21.0 → 26.3 |
| Mean | −5.2% | −10.9% | |

Main changes:

- **Intra blocks in inter frames.** About −10% on video_2, −5% on video_1, −5 to −7% on the 4K clips. Blocks of every
  shape can switch to intra using:
  - 10 luma modes and a chroma mode, both chosen by rate-distortion
  - intra transform types
  - luma transform splits

  The GPU decides intra in parallel, then a superblock wavefront reconstructs the chosen blocks in decode order.
- **Fast key-frame preset by default.** 9 to 22% faster for +0.2 to +1.1% BD-rate (`--kf-preset slow` restores the old
  preset).
- **Startup and I/O.** Input is read on a thread, filter uploads are deferred, and the default
  `CUDA_DEVICE_MAX_CONNECTIONS` is kept. Together these give about 15% more fps, with identical output.
- **Kernel cuts with identical output:** 64-point inverse transform, transform-type search, early exits in the intra
  decision.

See [svt_gap_plan.md](svt_gap_plan.md) for every measurement, including the experiments that did not help.

## Reproduce

```sh
# all clips against SVT-AV1 preset 8 (and libaom with --refs svt,aom)
python3 tests/bench_all.py build-cuda/third_party/av1gpu/av1gpu-video --tag mytag --refs svt
# BD-rate between two of our own runs
python3 tests/bd_tags.py baseline_tag mytag
# bitstream check: dav1d must decode every configuration to the encoder's reconstruction
python3 tests/check_pframes.py build-cuda/third_party/av1gpu/av1gpu-video
```

Encoder outputs and rows are written to `build-cuda/compare/`.

## Earlier milestones (video_1, 1,207 frames)

| Milestone | Speed | Notes |
| --- | ---: | --- |
| CUDA block-statistics prototype | 329 fps | analysis only, no bitstream |
| First all-intra AV1 encode | 4.45 fps | every frame a key frame, 34.1 MB |
| Batched all-intra (default / `--fast`) | 12.1 / 31.3 fps | identical output / turbo preset |
| First `--pframes` inter encoder | 41 fps | +45.7% vs SVT-AV1 p8, −25.4% vs SVT-AV1 p12 |

The `--pframes` streams decode byte-identically with dav1d, libaom and NVDEC (`av1_cuvid`). The three entropy paths
(`--pf-cpu-entropy`, the default GPU modeling with a host range coder, and `--pf-gpu-entropy`) produce identical
streams. The default is the fastest.
