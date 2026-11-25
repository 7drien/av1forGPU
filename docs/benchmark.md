# Benchmarks

All results: NVIDIA GeForce RTX 3060 Ti (8 GB, `sm_86`), driver 595.99.02, CUDA 12.4.

## Current results (`--pframes`, 2026-11-25)

First 120 frames of every clip in `benckmark/`, encoded with
`--pframes --persistent-cdf --qindex {20,50,90,130}`, decoded with dav1d. BD-rate uses YUV-weighted PSNR (6:1:1).
**Negative BD-rate means fewer bits than the reference at the same quality.**

| Clip | Content | BD-rate vs SVT-AV1 preset 8 | BD-rate vs libaom cpu-used 8 | fps (ours) | fps (SVT-AV1 p8) |
| --- | --- | ---: | ---: | ---: | ---: |
| video_1 | 1080×1920, 30 fps, static | **−0.9%** | | not re-measured | 36.5 |
| video_2 | 1920×1080, 30 fps, concert, cuts | +3.3% | | about 20 | 34.4 |
| video_3 | 3840×2160, 24 fps, handheld street | **−18.7%** | | not re-measured | 17.3 |
| video_4 | 3840×2160, 24 fps, advert | **−33.5%** | | not re-measured | 16.4 |
| video_5 | 1920×1080, animation | **−10.9%** | | 26.3 | 41.2 |
| **Mean** | | **−12.2%** | | | |

- **Compression:** better than SVT-AV1 preset 8 on four of five clips, behind on video_2.
  libaom cpu-used 8 (before rectangular intra: +25.4%, +15.7%, +43.4%, +9.6%, +4.9%, mean +19.8%) is not re-measured.
- **Speed:** about half of SVT-AV1 preset 8. The exhaustive intra search (every shape, every mode checked by
  rate-distortion, CfL) costs time: on a clean GPU, 120 frames of video_2 take about 6.1 s, against about 3.5 s for SVT
  (4.4 s with `--pf-no-intra`, at about +10% BD-rate). The other clips have not been re-timed on a clean GPU since; before
  the intra searches they ran at 0.5× to 0.95× of SVT.
- SVT and libaom figures come from cached reference encodes (`tests/compare.py`).
- Another job used the GPU intermittently while these were measured; speed was timed only in windows where the GPU
  had no other process.

### Progress during the day (BD-rate vs SVT-AV1 preset 8)

| Clip | Morning (`b40`) | Now | fps morning → now |
| --- | ---: | ---: | ---: |
| video_1 | +5.3% | −0.9% | 24.5 → not re-measured |
| video_2 | +18.4% | +3.3% | 19.5 → about 20 |
| video_3 | −13.2% | −18.7% | 7.8 → not re-measured |
| video_4 | −25.6% | −33.5% | 7.9 → not re-measured |
| video_5 | −10.8% | −10.9% | 21.0 → not re-measured |
| Mean | −5.2% | −12.2% | |

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
