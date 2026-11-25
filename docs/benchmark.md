# Benchmarks

All results: NVIDIA GeForce RTX 3060 Ti (8 GB, `sm_86`), driver 595.99.02, CUDA 12.4.

## Current results (`--pframes`, 2026-11-25, clean GPU)

First 120 frames of every clip in `benckmark/`, encoded with
`--pframes --persistent-cdf --qindex {20,50,90,130}`, decoded with dav1d. **Size** is BD-rate with YUV-weighted PSNR
(6:1:1) against SVT-AV1 preset 8 at CRF 24/32/40/48: the bits needed for the same quality (negative = smaller).
**Speed** is the ratio of median fps (four encodes each, wall time including process start).

| Clip | Content | Size vs SVT-AV1 p8 | Speed vs SVT-AV1 p8 | fps (ours) | fps (SVT-AV1 p8) |
| --- | --- | ---: | ---: | ---: | ---: |
| video_1 | 1080×1920, 30 fps, static | +1.9% | −11% (0.89×) | 32.7 | 36.7 |
| video_2 | 1920×1080, 30 fps, concert, cuts | +4.6% | −30% (0.70×) | 23.3 | 33.1 |
| video_3 | 3840×2160, 24 fps, handheld street | **−16.5%** | −55% (0.45×) | 7.9 | 17.6 |
| video_4 | 3840×2160, 24 fps, advert | **−29.8%** | −49% (0.51×) | 8.2 | 16.2 |
| video_5 | 1920×1080, animation | **−10.9%** | −41% (0.59×) | 24.8 | 41.9 |
| **Mean** | | **−10.1%** | | | |

- **Size:** smaller than SVT-AV1 preset 8 on three of five clips, 1.9% / 4.6% larger on video_1 / video_2.
- **Speed:** 0.45× to 0.89× of SVT-AV1 preset 8. Rectangular intra blocks are the costly part of the intra search
  (22-36% of the encode with every rectangle): they are now off at 4 MP and above and limited to 64x16 and larger
  rectangles below (`--pf-intra-rect` / `--pf-no-intra-rect`, `AV1GPU_PF_INTRA_RECTMIN`; all rectangles at 1080p:
  video_1 -0.6% / 26.7 fps, video_2 +3.2% / 20.4 fps).
- No other GPU process ran during this benchmark (load average below 1). SVT figures come from cached reference
  encodes (`tests/compare.py`).

### Progress during the day (BD-rate vs SVT-AV1 preset 8)

| Clip | Morning (`b40`) | Now | fps morning → now |
| --- | ---: | ---: | ---: |
| video_1 | +5.3% | +1.9% | 24.5 → 32.7 |
| video_2 | +18.4% | +4.6% | 19.5 → 23.3 |
| video_3 | −13.2% | −16.5% | 7.8 → 7.9 |
| video_4 | −25.6% | −29.8% | 7.9 → 8.2 |
| video_5 | −10.8% | −10.9% | 21.0 → 24.8 |
| Mean | −5.2% | −10.1% | |

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
