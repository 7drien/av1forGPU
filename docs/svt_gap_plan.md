# Plan: SVT-AV1 tools missing from the CUDA inter encoder

Baseline (120 frames of `video_1.mp4`, matched bitrate): about 24 fps, BD-rate about +38% against SVT-AV1 preset 8
with fixed quantizers. Each feature below must keep the rule that dav1d decodes exactly the encoder's
reconstruction (`tests/check_pframes.py`), and its gain must be measured with `tests/compare.py`.

The gains are rough figures from the published SVT-AV1 and libaom tool ablations, not measurements on this encoder.

## Feature list

| # | Tool (SVT-AV1 has it, we do not) | Rough gain | GPU work | Touches |
|---|---|---|---|---|
| T1 | Inter transform types (full `TX_SET_INTER_1/2/3`: ADST, FLIPADST, IDTX, 1-D) | 3–5% | per-type fwd/quant/rate in parallel, one warp per (block, type) | evaluate, commit, tile coder, header (`reduced_tx_set = 0`) |
| T2 | Intra blocks in inter frames (DC/directional/smooth/paeth, CFL) | 2–5%, large on scene cuts | wavefront over superblock diagonals | evaluate, decide, commit, tile coder |
| T3 | More references: LAST2/LAST3/GOLDEN/BWDREF/ALTREF2, single and compound | 3–6% | search per reference in parallel streams | search, chooseInter, MV stack, ref contexts |
| T4 | Order hints + temporal MV projection + skip mode + distance-weighted compound | 2–4% | motion-field projection kernel per 8x8 | header, MV stack, writer |
| T5 | Switchable interpolation filters (regular / smooth / sharp, dual filter) | 1–3% | phase planes per filter, choice in refine | phase planes, evaluate, tile coder |
| T6 | Blocks with a 4-pixel side (4x4, 4x8, 8x4, 16x4, 4x16) | 1–3% | 4x4 decision grid, chroma from sub-block vectors | grid, decide, commit, tile coder |
| T7 | Loop restoration (Wiener, self-guided) | 2–5% | per-unit filter search, one CTA per 64x64 unit | filters, tile coder (LR syntax per SB), header |
| T8 | Per-superblock CDEF strengths (`cdef_bits` > 0) | 0.5–1.5% | already per-SB SSE, add selection | filters, tile coder, header |
| T9 | OBMC (overlapped block motion) | 1–2% | neighbor-blended prediction per block | evaluate, tile coder (`motion_mode`) |
| T10 | Warped and global motion | 0.5–2% | per-block affine fit from neighbors, frame-level global search | evaluate, tile coder, header |
| T11 | Inter-intra and wedge / difference-weighted compound | 1–2% | masks per block | evaluate, tile coder |
| T12 | Adaptive quantization (segmentation, `delta_q`, `delta_lf`) | 1–3% perceptual | per-SB variance kernel | analyze, tile coder, header |
| T13 | Better rate control (lookahead, per-frame QP from complexity, two-pass) | bitrate accuracy | reuse coarse search costs | host |
| T14 | High-precision MVs (1/8 sample) at low QP | 0.5–1% | eighth-sample planes or on-the-fly filter | refine, tile coder |
| T15 | Palette and intra block copy (screen content) | large on screen content only | hash search | intra path |
| T16 | Super-resolution, reference scaling, frame-size changes | low-rate only | resampling kernel | header, pipeline |

## Dependencies

- T2 is needed before T11 (inter-intra) and makes T6 cheaper (intra 4x4 shares the grid).
- T4 needs T3 (several references with order hints).
- T6 changes the decision grid that T2, T9, T10 also write. Do it before them or design the grid for 4x4 first.
- T1, T5, T7, T8, T12, T13, T14 have no dependencies.

## Tasks that can run in parallel (separate people or branches)

Independent, can start now and merge in any order:
1. **T1** inter transform types (evaluate + tile coder).
2. **T5** switchable interpolation filters (phase planes + tile coder).
3. **T7 + T8** loop restoration and per-SB CDEF (filter stage only).
4. **T12** adaptive quantization.
5. **T13** rate control (host only).
6. **T14** high-precision MVs.

Second wave, after the grid redesign:
7. **T6** 4x4 grid, then in parallel **T2** intra-in-inter, **T9** OBMC, **T10** warped motion.
8. **T3** more references, then **T4** temporal MVs and skip mode.
9. **T11** after T2.

Within each feature the GPU work is data-parallel: blocks of one size are evaluated together, one warp per
(block, candidate). Only the range coder and the T2 wavefront have serial dependencies.

## Order chosen for this repository

T1 first (largest expected gain for the least new syntax, and the transform and scan tables already exist for the
intra path), then T5, T7, T3, T2, T6.

## Status

- T1 stage 1 done: each luma transform up to 32x32 chooses IDTX or DCT_DCT (`TX_SET_INTER_3`, `reduced_tx_set = 1`),
  chroma inherits the top-left luma type. BD-rate against SVT-AV1 preset 8: +38.2% → +32.5%, about 25 fps (was 26).
  `--pf-no-tx-types` disables it. Types travel in `BlockInfo::txTypes` (one nibble per 4x4).
- T1 stage 2 done: `reduced_tx_set = 0`. Luma transforms search every type of their inter set (16 for 4x4 / 8x8
  sizes, 12 for 16x16, IDTX / DCT_DCT for 32-point sizes), priced by the quantizer's distortion estimate with the
  default inter_tx_type CDFs; the winner is reconstructed exactly (flipped types included). Chroma inherits the luma
  type when its own set allows it. BD-rate against SVT-AV1 preset 8: +32.5% → +25.3%, about 22 fps.
  `--pf-tx-types-fast` (7 types: +26.0%), `--pf-tx-types-reduced` (stage 1), `--pf-no-tx-types`.
- T5 done, off by default (`--pf-switchable`): `is_filter_switchable = 1`, one interp_filter per block chosen by luma
  prediction SSE plus bits, coded with the spec contexts; GLOBALMV blocks code none. Bit-exact with dav1d. On the
  120-frame test it does not pay for its per-block symbol: +25.7% against +25.3% with the fixed filter (best bias,
  `AV1GPU_PF_FBIAS` = 16). A gain needs a joint motion and filter search or a full RD choice per filter.

## Speed track

- Per-kernel timing (`AV1GPU_PF_PROFILE2=1`): rectangular partitions were about half of the analysis time (refine
  about 7 ms and evaluate about 12 ms per 1080x1920 frame). The filter stage costs about 6% (23.6 against 25.0 fps
  without it): its 33 ms of host time was mostly waiting for the analysis kernels.
- A1 done: rectangular searches run after the square evaluation and skip nodes whose square evaluation prefers skip
  by a factor 2 or more (`--pf-no-prune` disables it). 21.6 → about 30 fps for +25.3% → +25.7% BD-rate.
- `--pf-fast`: split transforms code DCT_DCT only (no type search): +28.1% at about 35 fps (default +25.7% at about
  30 fps; a seven-type subset on splits, `AV1GPU_PF_SPLITTX=1`, gives +26.1% at about 31 fps).

