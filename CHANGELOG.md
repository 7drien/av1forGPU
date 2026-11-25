# Changelog

Notable changes to the `--pframes` inter encoder. BD-rate is against SVT-AV1 preset 8; negative is better.
Full measurements: [docs/svt_gap_plan.md](docs/svt_gap_plan.md).

## 2026-11-25

- **Speed:** rectangular intra blocks off at 4 MP and above and limited to 64x16 and larger below: +15 to 32% fps
  (video_1 0.89×, video_2 0.70×, video_3 0.45×, video_4 0.51×, video_5 0.59× of SVT-AV1 preset 8) for 1 to 4 points of
  size (video_1 +1.9%, video_2 +4.6%, video_3 −16.5%, video_4 −29.8%, video_5 −10.9%).

- **Intra mode search by rate-distortion:** every luma mode (`AV1GPU_PF_INTRA_TOPK`), every chroma mode, CfL for blocks up
  to 32x32: video_2 about −3%, video_1 about −1%. It costs speed: on a clean GPU, video_2 takes about 6.1 s for 120 frames
  against about 3.5 s for SVT-AV1 preset 8.
- **Speed:**
  - intra early exit at 1.5× the inter cost (−3.8% time for +0.11% BD-rate)
  - decision buffers shared between luma and chroma (identical output)
  - thread-team transforms for the 64x64 decision (identical output)
- Standing against SVT-AV1 preset 8 (clean GPU): size video_1 −0.6%, video_2 +3.2%, video_3 −18.7%, video_4 −33.5%,
  video_5 −11.0% (mean −12.1%); speed −27%, −38%, −65%, −62%, −50%.

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
