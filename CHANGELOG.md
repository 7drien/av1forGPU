# Changelog

Notable changes to the `--pframes` inter encoder. BD-rate is against SVT-AV1 preset 8; negative is better.
Full measurements: [docs/svt_gap_plan.md](docs/svt_gap_plan.md).

## 2026-11-24

- **Rectangular intra blocks** (HORZ / VERT, HORZ_4 / VERT_4, three-way halves): video_2 −3.8%, video_1 −2.7%, video_3
  −2.2%, video_4 −4.2%. Standing: video_1 +0.2%, video_2 +6.7%, video_3 −18.2%, video_4 −32.1%, video_5 −10.9% (mean −10.9%).

- **Intra blocks in inter frames.** BD-rate: video_2 −6.4%, video_1 −2.3%, video_3 / video_4 about −3%. The parts:
  - 10 luma modes, transform types and luma transform splits
  - chroma modes chosen by rate-distortion
  - a decode-order GPU wavefront that reconstructs the intra blocks
- **Speed:**
  - key frames use the `fast` preset by default: +9 to 22% fps for +0.2 to 1.1% BD-rate
  - startup without 32 CUDA connections, an input reader thread and deferred filter uploads
  - identical-output kernel work: 64-point inverse, transform-type search, intra decision early exits
- Still-content key-frame quantizer offset 55 (was 40).
- Standing: video_1 +2.9%, video_2 +10.9%, video_3 −16.2%, video_4 −29.0%, video_5 −10.8% (mean −8.4%).

## Earlier

- **Hierarchical coarse motion search:** video_3 −30%.
- **Pyramid and references:** ALTREF boost tuning, temporal filter, three references, TMVP, skip_mode.
- **Tools and entropy:** per-superblock CDEF, loop restoration, GPU entropy paths.
- **First `--pframes` encoder:** +45.7% at the first measurement, about 41 fps on video_1.
