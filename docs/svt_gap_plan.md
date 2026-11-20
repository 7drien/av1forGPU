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
- T3 stage 1, off by default (`--pf-refs 3`): pyramid frames also predict from the next anchor after their interval
  (ALTREF2, single prediction). Bit-exact; +25.7% → +25.3% for about 9% less speed, below the speed budget. A far
  past reference (a pinned golden frame) and compound pairs with it are the next things to try.
- T7 stage 1, off by default (`--pf-lr`): luma Wiener loop restoration in 64-sample units. The taps come from
  alternating 3x3 least squares per unit on the GPU, then the exact spec filter (stripes, deblocked rows outside the
  stripe) decides per unit; `read_lr` symbols use the recentred subexponential code. Video sequence headers now set
  `enable_restoration` (key frames code `lr_type` NONE). Bit-exact with dav1d in all three entropy paths. On the
  120-frame test (46 to 57 dB PSNR) it gains 0.1% (+25.7% → +25.6%) for 8% speed; lower thresholds lose. Worth
  re-measuring at lower qualities and with chroma / self-guided filters.

## Quality allocation

- SVT-AV1 spends 36 to 42% of its bits on the key frame of the test; we spent 6%. Coding inter frames further
  from the key frame (`--pf-qoffset` 60 → 130) and boosting the hidden ALTREF frames more (`--pf-altref-boost`
  100 → 128) moves BD-rate from +25.7% to +19.6% with no speed cost (offset 90: +23.0%, 100: +21.7%,
  160: +23.2%; boost 70: +22.5%, 160: +25.9%). These are the new defaults.
- Rate control now steers the inter-frame quantizer (key frames follow at minus the offset); before, the key-frame
  quantizer floor of 20 kept inter frames at 150 or more and capped the bitrate.
- Matched-bitrate benchmark (120 frames, `-b:v` targets 600 to 4000 kbit/s, 5 points, same session): BD-rate
  against SVT-AV1 preset 8 in its bitrate mode is -12.4% (51.4 to 56.5 dB overlap), at 21 to 28 fps against SVT's
  26 to 31 fps on this machine. Against SVT in CRF mode (`tests/compare.py`) it is still +19.6%; against libaom
  `cpu-used 8` (bitrate mode) +35.8%. Our short-clip rate control overshoots its target by 11 to 18%.
- Key frames alone (first frame) needed 18.4% more bits than SVT-AV1 preset 8 (libaom: 31.0%). With `--pframes`
  key frames now use the base encoder's `slow` preset (`--kf-preset NAME` overrides): key frames +10.7%, full
  120-frame sequence +19.6% → +18.6%, about 8% slower at the default 120-frame key interval.
- Tuning checked and left unchanged: `--pf-lambda` 0.5 (+19.9%) and 1.0 (+22.1%), `--pf-leaf-offset 10` (+20.0%),
  `--pf-gop 16` (+20.5%), ALTREF temporal filtering 2 / 4 frames (+20.4% / +20.6%).
- The test clip has no scene cuts (largest inter frame 3 KB against a 35 KB key frame), so intra blocks in inter
  frames (T2) would gain little on it; it needs a clip with cuts and uncovered content to be measured.
- Inter-frame bits by syntax group (`AV1GPU_PF_BITSTATS=1` with the default entropy path): coefficients 64 to 74%,
  partition / modes / references 13 to 22%, motion vectors 8 to 10%, transform types about 4%. The key frame is
  about 11% of the sequence. Residual coding and prediction quality are where the remaining gap is.
- Quantizer rounding 0.33 → 0.40 (+18.6% → +18.3%); other rounding / deadzone values were within noise.

## Second clip: video_2.mp4 (1920x1080, 29.92 fps, camera motion and cuts)

`tests/compare.py --src benckmark/video_2.mp4` (outputs in `build-cuda/compare/video_2/`), first 120 frames:

| Encoder state | BD-rate vs SVT-AV1 preset 8 | fps |
| --- | ---: | ---: |
| previous defaults (qoffset 60, boost 100, base key frames) | +79.2% | 13 to 21 |
| current defaults without scene cuts | +85.4% | 13.5 to 19 |
| + scene-cut key frames | +80.5% | 12 to 17 |

SVT-AV1 preset 8 runs at 32 to 34 fps on this clip; libaom `cpu-used 8` is at +90.8% against our first run.

- Loop restoration had stayed on by default after `a93dc01` (an edit that turned it off did not apply). Measured on
  both clips it helps (video_2: +90.0% → +85.4%; video_1: +18.7% → +18.3%), so it is now the documented default;
  `--pf-no-lr` turns it off.
- Scene cuts (`--pf-scene-cut T`, default 15, 0 disables): the mean difference of 16x16 luma block means between
  consecutive frames is above T and more than twice the next frame's (sustained motion is not a cut). The cut frame
  becomes a key frame at the inter quantizer minus `--pf-cut-qoffset` (default 60) and the mini-GOP ends before it.
  Four cuts in the first 120 frames of video_2, none in video_1.
- On video_2, motion vectors take 21% of inter bits (video_1: 8 to 10%) and evaluation is twice as slow (fewer
  skipped blocks). Temporal motion vector candidates (T4, order hints) and a finer motion search are the main
  missing pieces for this kind of content.

### Content-adaptive quality allocation (video_2 as the main benchmark)

- Sweeps on video_2: ALTREF boost 128 → 80 with key offset 90 gave +61.7%, boost 60 +61.3%, boost 40 +63.6%;
  λ 0.5 +79.3%, GOP 16 +80.3%. The same fixed setting cost video_1 5 points (+23.3%): static content wants a strong
  ALTREF boost and a much better key frame, moving content does not.
- `--pf-adaptive` (default; `--pf-no-adaptive` disables it): motion = mean block-mean difference of consecutive
  frames over the mini-GOP (video_1 0.05 to 0.16, video_2 2.2 to 5.6). The ALTREF boost goes from 128 (motion
  <= 0.3) to 60 (motion >= 1.5); periodic key frames use the inter quantizer minus 130 (static) to 90 (moving), from a
  one-mini-GOP lookahead. The base `--pf-qoffset` is now 90.
- Result: video_2 +80.5% → +59.1%, video_1 +18.6% (unchanged within noise). An end-point difference was tried
  first and failed on video_2, whose cuts make short mini-GOPs (+77.3%).

### Motion search range (step 2 diagnostics)

- video_2 BD-rate by coarse search radius: 32 px +65.1%, 64 +59.1%, 96 +57.6%, 128 +56.5%, 192 +55.4%. Fast motion
  ran past the old ±64 / ±96 window. The quarter-resolution search now allows R up to 48 (±192 px), the new
  default (`AV1GPU_PF_RANGE` overrides it). Cost: coarse search 0.6 → 4.8 ms per 1080p frame. video_1: +18.9%
  (+18.6% before, within noise).
- Motion-vector bits in the evaluation's RD cost were weighted 0.25; 0.5 is the new default (video_2 +55.4% → +54.5%,
  1.0: +56.3%; video_1 +18.8%).

### T4: order hints and temporal motion vector prediction

- Stage A: `--pframes` streams set `enable_order_hint` (7 bits) and `enable_ref_frame_mvs`; key frames carry
  `order_hint` (base encoder, `av1g_set_video_order_hint`), inter frames `order_hint`, `ref_order_hint[]` (error
  resilient mode), `frame_refs_short_signaling`, `use_ref_frame_mvs` and `skip_mode_present` when skip mode is
  allowed. Extra-search candidates from references in the other direction are negated (RefFrameSignBias).
- Stage B (`--persistent-cdf`; `--pf-no-tmvp` disables it): each referenced frame's motion field is stored per slot
  from the decisions grid (7.19), the host projects it for the next frame (7.9, sequential: last writer wins), the
  tile coder adds the temporal candidates (7.10.2.5 / 7.10.2.6) in all three entropy paths, and the GPU motion search
  uses the block's projected vector as a start point and a second predictor. Bit-exact with dav1d and libaom.
- Gain on video_2: +54.1% → +53.7%. 79% of blocks still code NEWMV: the search returns the SAD-optimal vector, which
  rarely equals a candidate. A snap-to-neighbour pass (same-shape neighbours) did not help (+1.5% bytes); a mode
  refinement on the final partition with the real candidate lists is the next step for vector bits.

### Edge bug fixed (present since before this plan)

- Frames whose last superblock row or column is partly outside the frame by more than the forced split size
  (1080 rows: 56 visible rows; 64x16 / 64x32 blocks crossing the bottom) lost coefficient rows in `commit`: it looked up
  the decisions grid at cells outside the frame, which are empty, and zeroed those levels. The stream decoded, but the
  decoder's pictures drifted from the encoder's references. Now out-of-frame samples use the block of the last inside
  cell. `tests/check_pframes.py` has 448x184 cases (56 visible rows) that fail without the fix.
- `--pf-no-filters` streams did not decode since slow-preset key frames (enable_cdef in the sequence header without
  CDEF parameters in inter frames); fixed.
- After the fixes: video_2 +53.6% against SVT-AV1 preset 8 (+60.0% against libaom), video_1 +18.4% (+44.2%).

### Speed on video_2 (main benchmark)

- Evaluation dominates (57 to 62 ms of about 75 ms per 1080p frame). Transform-type search: 28 ms of it.
- Type screening by forward-transform magnitude, quantizer estimate for the best 3 only: 61.6 → 57.3 ms, BD-rate
  unchanged (+53.8%). Deciding the depth first and searching types on that depth only did not help (the chosen depth
  carries most of the search) and was reverted.
- Scene-cut key frames use a second key-frame encoder with base-preset speed knobs (and the main encoder's
  sequence-level tools): +53.8% → +54.1%, 9 to 10% faster (`--pf-slow-cuts` keeps the slow preset).
- `--pf-fast`: +55.1%, 12 to 17 fps. SVT-AV1 preset 8: 32 to 34 fps on this clip.
- Next: restructure evaluation (several warps per block or batched same-size transforms; one warp per 64x64 block
  gives only about 500 warps per frame).
- Split-external evaluation of square blocks (one warp per split transform, then a finalize kernel): same decisions,
  evaluation 61.0 → 51.4 ms per frame.
- Rectangular-partition pruning margin by motion (`--pf-adaptive`): 2.0 for static content, 0.5 for moving content
  (fixed 0.5: video_2 +53.7% at 17.7 fps but video_1 +22.4%; 1.0: video_2 +53.5%, video_1 +19.9%). Result: video_2
  +53.3% at about 15.8 fps (q50), video_1 +18.5%.
- Split-external evaluation for rectangular blocks as well: evaluation 47.4 → 33.3 ms per frame, same decisions;
  video_2 15.8 to 20.6 fps at +53.3%.
- Scene-cut and motion measures memoize their frame differences (+5% fps, same output).
- `--pf-cut-preset NAME` sets the scene-cut key-frame preset (default base; turbo: +54.0%, about 7% faster).
- Remaining per-frame cost on video_2 (60 frames): GPU analysis ~33 ms per inter frame (coarse 5, refine 7, evaluation
  24), filters 5 ms, key frames ~200 ms each (28% of the time with 5 key frames in 60 frames).
- Depth-2 split search (16 transforms per square block) runs only for blocks whose depth-1 split beat the whole
  transform (`AV1GPU_PF_D2MARGIN`, default 1.0): evaluation about 33 → 31 ms, same BD-rate (+53.3%).
- Tried and reverted: a two-level coarse search (eighth resolution then quarter resolution) cut the coarse stage from
  5.3 to 0.8 ms but cost 1.4 to 1.6 points of BD-rate on video_2.
- Per inter frame on video_2: GPU work about 38 ms (analysis 30, filters 5, entropy modeling 3) against about 49 ms of
  wall time; key frames about 200 ms each. The remaining speed is in CPU / GPU serialization and key frames.
- Filter searches read their candidate errors back once per search (was one synchronous copy per candidate): ~3.5%
  faster, same decisions.
- Determinism bug fixed: the frame-difference memo (scene cuts, motion) was keyed by buffer address, and freed frame
  buffers could come back at the same address, so decisions occasionally depended on allocation timing (two runs of the
  same command gave 1,601,151 and 1,629,940 bytes). The memo is cleared whenever frames leave the queue; repeated runs
  are now identical. Measurements between the memo commit and this fix carry about ±1 point of noise.
- Clean measurement after the fix (120 frames): video_2 +53.4% against SVT-AV1 preset 8 (+59.5% against libaom) at
  16.3 to 22.0 fps (SVT: 32 to 34 fps); video_1 +18.6% (+44.4%) at 21 to 25.5 fps (SVT: 36 to 37 fps).

### Asynchronous host pipeline

- Stage one padded transfer per source in two pinned buffers on a copy stream. Upload the ALTREF first, then send
  later pictures in coding order during analysis. Inter analysis and filters use an explicit compute stream. The
  host projects TMVP during coarse search, GPU entropy reuses a snapshot of that field, and the CPU entropy writer
  waits for its own downloads. Deblocking and CDEF choose their candidates on the device. The host receives a
  single filter result per frame. The candidate workspace adds about 23.7 MiB at 1080p.
- Byte-identical to `d437504` on both clips at qindex 20, 50, 90, and 130, including repeated default runs at 20,
  50, and 90. The GPU and CPU entropy modes also matched in repeated runs at 20 and 90. Both decoder suites passed.
  BD-rate vs SVT-AV1 preset 8 stayed at +53.4% for video_2 and +18.6% for video_1.
- Wall fps on the first 120 frames at qindex 50, median of five alternating runs per build with no other GPU
  compute observed: video_2 18.25 → 19.11 (+4.7%), video_1 22.46 → 24.16 (+7.5%). Video_2 at qindex 20,
  three runs per build: 16.71 → 17.38 (+4.0%). The first video_1 timing sweep and the first video_2 qindex 90
  sweep had uneven GPU sharing. A single quieter video_2 qindex 90 pair ran at 20.29 → 21.29 fps.
- Profile medians at qindex 50 (three alternating runs, first 60 frames): subtracting measured keyframe GPU time
  left 45.8 → 43.1 ms per inter frame on video_2 and 34.4 → 31.0 ms on video_1. Evaluation stayed at about 21.8 ms
  and 12.5 ms respectively. The filter result still gates entropy submission.
