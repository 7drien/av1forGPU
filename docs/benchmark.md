# Benchmarks

All results: NVIDIA GeForce RTX 3060 Ti (8 GB, `sm_86`), driver 595.99.02, CUDA 12.4.

## Current results (`--pframes`, 2026-11-25, clean GPU)

First 120 frames of every clip in `benckmark/`, encoded with
`--pframes --persistent-cdf --qindex {20,50,90,130}`, decoded with dav1d. **Size** is BD-rate with YUV-weighted PSNR
(6:1:1) against SVT-AV1 preset 8 at CRF 24/32/40/48: the bits needed for the same quality (negative = smaller).
**Speed** is the ratio of median fps (four encodes each, wall time including process start).

| Clip | Content | Size vs SVT-AV1 p8 | Speed vs SVT-AV1 p8 | fps (ours) | fps (SVT-AV1 p8) |
| --- | --- | ---: | ---: | ---: | ---: |
| video_1 | 1080×1920, 30 fps, static | **−0.6%** | −27% (0.73×) | 26.7 | 36.7 |
| video_2 | 1920×1080, 30 fps, concert, cuts | +3.2% | −38% (0.62×) | 20.4 | 33.1 |
| video_3 | 3840×2160, 24 fps, handheld street | **−18.7%** | −65% (0.35×) | 6.1 | 17.6 |
| video_4 | 3840×2160, 24 fps, advert | **−33.5%** | −62% (0.38×) | 6.2 | 16.2 |
| video_5 | 1920×1080, animation | **−11.0%** | −50% (0.50×) | 21.1 | 41.9 |
| **Mean** | | **−12.1%** | | | |

- **Size:** smaller than SVT-AV1 preset 8 on four of five clips, 3.2% larger on video_2.
- **Speed:** 0.35× to 0.73× of SVT-AV1 preset 8. The exhaustive intra search (every shape, every mode checked by
  rate-distortion, CfL) is a large part of it: video_2 takes 4.4 s for 120 frames with `--pf-no-intra` against about 6 s
  by default (and about +10% size).
- No other GPU process ran during this benchmark (load average below 1). SVT figures come from cached reference
  encodes (`tests/compare.py`).

### Progress during the day (BD-rate vs SVT-AV1 preset 8)

| Clip | Morning (`b40`) | Now | fps morning → now |
| --- | ---: | ---: | ---: |
| video_1 | +5.3% | −0.6% | 24.5 → 26.7 |
| video_2 | +18.4% | +3.2% | 19.5 → 20.4 |
| video_3 | −13.2% | −18.7% | 7.8 → 6.1 |
| video_4 | −25.6% | −33.5% | 7.9 → 6.2 |
| video_5 | −10.8% | −11.0% | 21.0 → 21.1 |
| Mean | −5.2% | −12.1% | |

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
