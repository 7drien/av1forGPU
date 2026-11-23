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

### Motion refinement on the final partition

- Once the tile partition is fixed, one parallel pass builds the tile coder's spatial and temporal MV stacks
  from the searched vectors. It prices NEAREST, NEAR at the available DRL indices, GLOBAL and compound modes
  with the default inter CDFs. Mixed compound modes may keep one searched vector. Luma SATD shortlists a cheaper
  candidate for each block. The regular evaluator recomputes exact prediction, chroma, transform type and depth,
  skip, levels and reconstruction before deciding whether to take it. Entropy coding rebuilds the stacks from
  the accepted vectors, so neighbouring changes can affect the final mode. `--pf-no-mode-refine` disables the pass.
- The first implementation looked up trial flags in all shape grids. On video_2 at q50 it spent 5.84 ms per inter
  frame on refinement, including about 4.72 ms on trial evaluation and 0.98 ms preparing and screening candidates.
  The follow-up prices the stacks one thread per block, screens one warp per block and compacts trial indices
  by shape before transform evaluation.
- On 120-frame sweeps against SVT-AV1 preset 8 (q20/50/90/130), video_2 improved from +53.4% to +47.7% BD-rate
  and video_1 from +18.6% to +13.1%. The compacted pass produced the same IVF at all eight clip/quantizer points
  as the first implementation.
- At q50, first-list NEWMV went from 79.6% to 56.0% of inter blocks on video_2 and from 23.2% to 17.6% on video_1
  (CPU entropy counts). The adaptive MV share of inter bits went from 24.2% to 17.6% and from 8.7% to 6.8%,
  respectively (`AV1GPU_PF_BITSTATS=1`).
- The compacted pass took 5.14 ms per video_2 inter frame at q50, including 4.16 ms for trial evaluation and
  0.83 ms to prepare and screen candidates (`AV1GPU_PF_PROFILE=1 AV1GPU_PF_PROFILE2=1`).
- Median wall throughput across three alternating unprofiled q50 pairs was 19.50 → 17.56 fps on video_2
  (−9.9%) and 24.33 → 22.98 fps on video_1 (−5.6%). GPU contention varied during the four-point sweeps.
- The 120-frame video_2 reconstruction at q50 matched dav1d across all 373,248,000 bytes. Disabled refinement
  reproduced the starting revision's IVF. The P-frame and legacy inter suites passed.

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

### Integration of the final-partition refinement and the asynchronous pipeline

Both were developed in parallel (separate worktrees) and merged in `7638583`. Verified on the merged build: byte
identity with the refinement branch by default and with the previous main under `--pf-no-mode-refine`, in 8
configurations (all three entropy paths, no TMVP, tiles + three references, temporal filter + overlays across key
frames, no filters); both test suites; repeated runs identical; a full 120-frame video_2 stream decoded by dav1d and
libaom equal to the encoder reconstruction.

| Clip (120 frames) | BD-rate vs SVT-AV1 preset 8 | vs libaom cpu-used 8 | median fps at q50 (previous main) |
| --- | ---: | ---: | ---: |
| video_2 | +47.7% (was +53.4%) | +53.7% (was +59.5%) | 19.2 (20.3) |
| video_1 | +13.1% (was +18.6%) | +37.7% (was +44.4%) | 25.5 (25.0) |

### Repeated final-partition refinement (`--pf-refine-passes N`, default 6)

- The refinement prices every block against a snapshot of its neighbours' vectors (Jacobi), so a block that
  adopts a neighbour's vector often loses it when that neighbour switches too, and a uniform-motion area can only
  converge one block further per pass. The pass is now repeated: each pass rebuilds the stacks from the accepted
  vectors, compares against the block's current cost (the accepted trial's distortion and coefficient rate for
  blocks changed by an earlier pass), skips a block whose priced menu equals the previous pass's (its screen and
  trial would repeat), and does not re-evaluate a proposal that was rejected at the same rate saving. Each pass
  commits only its own accepted blocks. `--pf-refine-passes 1` reproduces the previous output byte for byte.
- Trials per pass roughly halve (video_2 q50, about 1200 → 660 → 360 → 200 → 110 → 60 per frame).
- The trial evaluation was latency bound, not work bound: per pass the 14 shape evaluations ran as a serial chain of
  48 kernels, each as long as its slowest block (one warp evaluating a 64x64 block takes about 0.3 ms), so a pass
  with 60 trials cost almost as much as one with 1200. The shapes write disjoint buffers, so their trials now run
  concurrently on one stream per shape (fork / join events). Pass spans on video_2 q50 (nsys): 5.0, 4.2, 3.9, 3.6,
  3.5, 2.8 ms → 3.7, 2.7, 2.1, 1.7, 1.5, 1.4 ms. Output unchanged.
- The candidate screen then took 0.45-0.7 ms per pass, again bound by single warps scanning 64x64 blocks: it now uses
  one CTA of four warps per block. video_2 q50 at 6 passes 17.0 → 17.4 fps, output unchanged. The remaining floor
  of a pass is the 64x64 trial chain (one warp per block in `evaluate`, about 0.3 ms even for a few blocks; 2.4 ms
  in the first pass).

| video_2, 120 frames, passes | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BD-rate vs SVT-AV1 preset 8 | +47.7% | +42.3% | +38.9% | +37.0% | +35.0% | +34.3% |
| fps at q50 (direct runs, before the CTA screen) | 19.7 | 18.9 | 18.2 | 17.7 | 17.0 | 16.5 |

- A block whose trial is rejected keeps the same menu, so later passes skipped it and its second-best candidate was
  never tried. Rejected proposals are now remembered (up to four, with their rate saving) and excluded from the
  screen, and a block rejected in the previous pass is screened again. video_2 +35.0% → +34.6%, fps −1.4%.
  `--pf-rdoq` on top: +34.8% (no gain, slower).
- video_1: +13.1% (1 pass) → +9.5% (3) → +8.2% (6); vs libaom +37.7% → +31.6%. video_2 vs libaom (3 passes):
  +44.0%.
- Verified: both suites (plus a single-pass case), repeated runs identical, a full 120-frame video_2 stream decoded
  by dav1d equal to the encoder reconstruction.

### Where the remaining video_2 gap is (after 6 refinement passes)

- Key frames are not the problem: on seven scene frames of video_2 coded all-intra, the base intra encoder is +1.3%
  BD-rate against SVT-AV1 preset 8 forced to `keyint=1`. Key frames are 15% of our q50 stream (cuts at 5, 10, 17, 23).
- Per segment against SVT CRF 32 (similar PSNR): the cut-heavy frames 0-22 are close (31% more bytes at +0.8 dB);
  the long handheld concert shot 23-119 (dark, noisy) takes 15% more bytes at 0.5 dB lower luma PSNR. SVT codes it
  with 16-frame mini-GOPs although configured for 32.
- Bit split at q50: coefficients 77.5%, MV 9.2%, modes / partition 11.5%; NEWMV 29% of inter blocks (56% with one
  pass).
- Rechecked on video_2 (all tuned on video_1 before): ALTREF temporal filter 2 / 4 frames +37.2% / +37.8% (worse than
  +34.6%), `--pf-rdoq` +34.8%, `--pf-lambda` 0.5 +33.8% and 1.0 +37.6%, `--pf-altref-boost 80` +35.8%,
  `--pf-switchable` +33.6% (video_1 +8.0% against +7.9%; costs 7% fps, so still off), `--pf-gop 16` +34.2%,
  `--pf-gop 8` +33.5%.

### Motion-adaptive mini-GOP length

- With `--pf-adaptive` (default), a mini-GOP is cut to 16 frames when the block-mean motion of the next eight frames
  from the anchor is at least 0.7, and to 8 frames from 1.0 (`AV1GPU_PF_GOP16` / `AV1GPU_PF_GOP8` override the
  thresholds). video_1 measures about 0.05 (rising to 0.4-0.56 late in the clip, where 0.3 / 0.8 thresholds made it
  +9.8%), video_2's long shot about 1.0-1.6.
- video_2 +34.6% → +33.6% against SVT-AV1 preset 8 (+37.9% against libaom), video_1 unchanged at +7.9%; fps unchanged.

## Five-clip benchmark (`tests/bench_all.py`)

`python3 tests/bench_all.py BINARY --tag TAG [--ours ARGS]` runs `tests/compare.py` on every `benckmark/video_*` clip
(first 120 frames, q 20/50/90/130) and prints BD-rate against SVT-AV1 preset 8 and libaom cpu-used 8 with the median
fps. `compare.py` now computes PSNR one frame at a time (4K clips no longer exhaust memory; same values) and keeps
full-range sources (video_3) in full range.

| Clip | Content | vs SVT | vs libaom | fps (SVT / libaom) |
| --- | --- | ---: | ---: | ---: |
| video_1 | 1080x1920, static | +7.9% | +31.3% | 23.3 (36.7 / 10.6) |
| video_2 | 1080p concert, cuts | +33.6% | +37.9% | 16.3 (33.1 / 4.9) |
| video_3 | 4K handheld street, dark | +69.3% | +192.8% | 6.0 (17.6 / 3.2) |
| video_4 | 4K advert, logos, cuts | -11.3% | +36.5% | 5.9 (16.2 / 1.3) |
| video_5 | 1080p animation | -3.8% | +12.9% | 15.0 (41.9 / 7.9) |
| mean | | +19.1% | +62.3% | |

### video_3 investigation (not solved)

- Not noise (estimated sigma 0.5), not range clipping (luma 20..137), not the key frames (ours are as good as
  SVT's / libaom's and cheaper), not the in-loop filters, not 1/8-pel vectors (best 64x64 prediction 3.75 → 3.68),
  not global / warped motion or OBMC (disabling them in libaom changes nothing on this clip).
- The loss is in the inter frames, in textured areas (stone wall): about 0.7 dB at the same GOP structure and frame
  sizes as SVT. Our predicted frames are 98% skipped blocks, 84% of the area in 64x64 blocks; the best 64x64
  translational prediction of the wall has MSE 3.75 against 2.08 for 16x16 blocks (source to source).
- Tried on video_3: ALTREF temporal filter (rewritten: libaom-style non-local-means weights from the estimated noise
  and the inter quantizer, quarter-pel alignment; the old weights used the ALTREF's own quantizer, near 1 after the
  boost, so they filtered almost nothing on any clip) +73.7% / +81.0% with 2 / 4 frames (off by default);
  ALTREF boost 40 / 80 +75.3% / +68.4%; fixed 32-frame GOP +71.7%; `--pf-switchable` +65.0%; `--pf-lambda` 0.5
  +63.3% (0.35: +67.1%); 3 references +65.1%; geometric pyramid quantizers (boost halved per level) worse on every
  clip.

## Speed: profile and parallelism (all clips)

GPU busy 91-93% of the wall time (nsys, video_2 1080p and video_3 4K at q50), so the host pipeline is not the limit.
GPU time: evaluation 48-55%, motion search 25%, key-frame encoder 20-25%, filters + entropy + refinement about 15%.

- Key frames used one tile (`slow` preset): the intra reconstruction wavefront runs at most 15 (1080p) / 30 (4K)
  CTAs, and the key-frame entropy coder one CTA (255 ms per 4K key frame). `--kf-tiles MIN_MP,C,R` (default 2,2,2:
  4x4 tiles from 2 MP; 0 = one tile): one key frame 470 → 237 ms at 1080p, 1170 → 587 ms at 4K. Five clips: mean
  BD-rate +19.1% → +19.4%, fps +4% (video_1, video_2), +2% (video_3), 0 (video_4), +11% (video_5).
- Coarse search: four candidates per lane with packed byte SADs (`__vsadu4`, funnel shifts on word-aligned rows)
  instead of 64 byte loads and differences per candidate. Same output.
- Evaluation: the four square sizes run concurrently (one stream each); each rectangle's pruning, refinement, choice
  and evaluation follow its square size on its own stream; all join before the decision. Same output.
- Both together, byte-identical on video_1 / 2 / 3 / 4: video_2 q20 14.3 → 17.2 fps, q90 17.1 → 20.7, video_1 q50
  22.4 → 27.0 (+21%); 4K (video_3, video_4 q50) 5.6 → 6.1 (+9%: 4K kernels already fill the GPU).

### Square-children pruning on moving content

- Rectangles were already skipped when their square node clearly prefers skip (`pruneRect`). The same rule now
  applies to square children: sizes are evaluated 64 -> 32 -> 16 -> 8 and a child is searched only when its parent
  codes a residual and its coded cost is below the skip cost times the margin (a pruned parent prunes its children;
  a parent crossing the frame edge, which must split, never does).
- Always on (margin 1): five-clip mean +19.4% → +20.3%, but video_1 +8.6% → +17.9% (static content loses 0.3 to
  0.75 dB: detail is refined in small blocks over frames; a limit on the parent's skip distortion did not help, even
  0.002 squared quantizer steps per sample gave +11.1%), while the moving clips improved (video_3 +69.5% → +66.3%,
  video_4 -11.3% → -12.3%, video_5 -3.6% → -4.3%) and ran 11 to 31% faster. Margin 2: similar (+20.6%).
- Enabled only for mini-GOPs whose motion (block-mean difference, as `--pf-adaptive`) is at least 0.6 (video_1 at most
  0.3; video_5 0.72; video_2 / 3 / 4 up to 6-10). `--pf-square-prune M` forces a margin (0: off).
- Five clips: video_1 +8.6% (unchanged), video_2 +33.8%, video_3 +66.3%, video_4 -13.2%, video_5 -4.3%; mean
  +19.4% → +18.2%. fps: video_2 20.4, video_3 7.7, video_4 6.7, video_5 21.0 (the machine was rebooted between
  runs with a new kernel and driver, and identical encodes now run about 15% faster, so the speed gain is measured
  only within one session: margin-1 pruning +11% to +31%).
- The gains on moving clips mean the decision favours splitting too much there: children win the cost comparison
  but lose in BD-rate, so their side-information cost is probably underestimated.

### Per-block mode cost follows motion

- The evaluation charged every block 2 bits of mode information besides its vectors. Raising it (`AV1GPU_PF_MODEBITS`)
  helps moving content and hurts static content: 4 bits video_1 +0.3, video_2 -1.0, video_3 -3.1, video_5 -0.4 points;
  8 bits video_1 +1.8, video_2 -1.6, video_3 -5.4, video_5 -0.4. Full-weight vector bits (`AV1GPU_PF_MVW` 1.0
  instead of 0.5) changed nothing overall (video_3 -1.0, video_1 +0.7, video_2 +0.3).
- With `--pf-adaptive` the charge now ramps from 2 bits (motion <= 0.3) to 12 bits (motion >= 1.5;
  `AV1GPU_PF_MODEBITS_MOVING`, `AV1GPU_PF_MODEBITS_HI` for tuning). Maximum 8: mean +16.4%; 12: +16.2%; a ramp ending at
  0.8 moved video_3 -0.7 and video_5 +0.3 (kept 1.5).
- Five clips: video_1 +8.5%, video_2 +32.4%, video_3 +61.3%, video_4 -16.5%, video_5 -4.7%; mean +18.2% → +16.2%.
  Same speed.

### Defaults rechecked on five clips: lambda 0.5 and switchable filters

BD-rate against SVT-AV1 preset 8 (mean of five clips; fps of video_2 / video_3), from the defaults above (+16.2%):

| Change | mean | video_1 | video_2 | video_3 | video_4 | video_5 | fps v2 / v3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | +16.2% | +8.5% | +32.4% | +61.3% | -16.5% | -4.7% | 20.5 / 7.8 |
| `--pf-tx-types-fast` | +16.5% | +8.8% | +32.8% | +61.5% | -16.4% | -4.2% | 20.9 / 8.1 |
| `--pf-refine-passes 3` | +18.1% | +9.9% | +35.3% | +63.8% | -14.1% | -4.4% | 21.5 / 8.5 |
| `--pf-fast` | +17.4% | +10.1% | +33.6% | +62.8% | -15.8% | -3.9% | 22.2 / 8.6 |
| `--pf-lambda 0.6` | +15.2% | +8.3% | +31.2% | +58.7% | -17.6% | -4.4% | 20.4 / 7.7 |
| `--pf-lambda 0.5` | +14.6% | +8.2% | +30.4% | +56.5% | -18.4% | -3.7% | 19.1 / 7.7 |
| `--pf-switchable` | +14.8% | +8.6% | +31.4% | +58.2% | -17.1% | -7.1% | 18.5 / 7.2 |
| lambda 0.5 + switchable | **+13.1%** | +8.3% | +29.4% | +53.0% | -18.9% | -6.5% | 18.6 / 7.1 |
| ... + split pruning | +13.4% | +8.3% | +30.2% | +53.1% | -18.8% | -6.0% | 19.1 / 7.3 |

- New defaults: lambda multiplier 0.5 (was 0.7; with the motion-dependent mode cost a lower lambda now helps every clip
  but video_5) and switchable interpolation filters (`--pf-no-switchable` turns them off). Mean +16.2% → +13.1%, about
  8% slower.
- Split pruning (moving content: no split-transform search when the whole-block transform quantizes to nothing,
  `AV1GPU_PF_SPLITPRUNE=1`) is implemented but off: +0.3 points for 2 to 3% speed.

### Motion search: four warps per large block; kernel experiments without counters

- Per-stage timing (`AV1GPU_PF_PROFILE2=1`, video_2 q50): `refine` of 64x64 blocks was the largest single item (5.5 ms per
  inter frame of 9.6 ms of refinement): one warp per block, 50-80 candidate SADs of 4096 samples each. 2048- and
  4096-sample blocks now use four warps (one block per CTA, SAD sums through shared memory). Same output; video_2
  17.1 → 17.6 fps (q20) and 20.8 → 21.5 (q90), video_3 6.83 → 6.90. 256 lanes for 64x64 or 64 lanes for 1024-sample
  blocks were no better.
- No gain (reverted, outputs identical): packed 4-byte SADs (`__vsadu4` on funnel-shifted loads) in the refinement
  (memory-latency bound, not arithmetic; the register-cached variant was 12% slower), a parallel screen of the 16
  types of 4x4 transforms (one type per lane, bit-exact with the sequential screen: +0.5%; the candidates' quantizer
  and rate estimates dominate), a 128-register cap on `evalSplit` for more resident warps (-0.5%).
- GPU idle time is 9% of an encode: 180 ms of startup in the first 213 ms, and 8 to 16 ms at each mini-GOP boundary.
  Process start plus CUDA context cost about 0.35 s (lazy module loading is already the default).
- Without performance counters (admin only on this machine) the kernels' limiters are guessed from such experiments.

### Value of the rectangular shapes (five clips, defaults otherwise)

| Change | mean vs SVT | video_1 | video_2 | video_3 | video_4 | video_5 | speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | +13.1% | +8.3% | +29.4% | +53.0% | -18.9% | -6.5% | |
| `--pf-no-rect4` (no 32x8 .. 16x64) | +14.0% | +10.0% | +30.8% | +53.6% | (see log) | -6.3% | +6 to 9% |
| `--pf-no-rect` | +17.5% | +12.7% | +35.1% | +57.5% | -14.7% | -3.3% | +15 to 24% |

- Every pruning measured so far trades about 0.2 to 0.3 BD-rate points per 1% of speed (refinement passes, transform
  types, square / split pruning, rectangles), so pruning alone cannot double the speed; the kernels' efficiency must.
- Peak GPU memory: 2.1 GB at 1080p, 6.6 GB at 4K (of 8 GB): a second analyzer for frame-level parallelism (independent
  leaf frames of the pyramid) fits at 1080p only.
- Frame-level parallelism tried (reverted): a second analyzer coded the second leaf of each independent pair of
  pyramid leaves (lo + 1 and lo + 3 of each interval of four), its analysis queued before the first leaf's filters,
  frames finishing in coding order. Deterministic and correct, but video_2 19.11 → 19.21 fps: nsys shows the second
  analyzer's kernels overlapping the first's for only 3% of their time. With the 14 shape streams of one frame the
  SMs' registers and shared memory are already taken, so another frame's kernels cannot become resident: the GPU is
  resource-saturated, and only less work or kernels that need fewer resources per unit of work can make it faster.

### skip_mode (T4 item)

- Frames between a past and a future reference whose `skip_mode_params()` pair is our compound pair (LAST, ALTREF) now
  code `skip_mode_present = 1` (`--pf-no-skip-mode` turns it off). The tile coder codes `skip_mode` (spec CDF, context
  above + left) for every block of at least 8x8 and nothing else for a skip_mode block; a block uses it when it already
  decodes identically: skip, compound LAST / ALTREF, EIGHTTAP and the vectors of the stack's NEAREST pair at index 0
  (decided in the parallel mode choice, phase 2, where the stack depends only on the final grid).
- Alone this saves nothing (video_2 q50 +0.05%: few blocks match exactly). The final-partition refinement now also
  proposes skip_mode for every block (compound NEAREST pair of the LAST / ALTREF stack, rate = the skip_mode symbol,
  EIGHTTAP forced, accepted only without residual). Extra credit for the references / skip flag / filter it saves:
  4.5 bits video_2 +2.3, video_3 -0.8, video_5 -1.3 points; 0 bits video_2 -1.0, video_3 -2.9, video_5 -0.8; -2 bits
  slightly worse than 0 (kept 0).
- Five clips: video_1 +8.4%, video_2 +28.4%, video_3 +50.1%, video_4 -19.8%, video_5 -7.3%; mean +13.1% → +12.0%
  (libaom: +51.4%). Same speed.

### Per-superblock CDEF presets (T8)

- The CDEF search already renders the frame at each of the 9 candidate strengths. It now also sums each coded 64x64
  superblock's luma and chroma SSE per candidate, picks up to 8 presets (luma candidate, chroma candidate) greedily on
  one CTA (exact integer sums), keeps 1, 2, 4 or 8 by SSE + lambda x cdef_bits per coded superblock, gives every
  superblock its best preset, and gathers each superblock from its candidate (CDEF reads only the deblocked frame, so
  this equals filtering it with its preset). The header codes cdef_bits and the presets; the tile coder codes
  cdef_idx at the first non-skip block of each superblock (read_cdef), on both GPU and CPU paths.
- Five clips: video_1 +8.4% → +6.7%, video_2 +28.4% → +28.3%, video_3 +50.1% → +48.7%, video_4 -19.8% → -20.1%,
  video_5 -7.3% → -7.7%; mean +12.0% → +11.2%. Same speed. `--pf-no-cdef-sb` restores one strength per frame.
- Decision constants rechecked (video_1/2/3/5): skip flag 0.3/2.5 or 1/1 bits instead of 0.5/1.5, partition split
  bits x2 or x0.5: all within 0.3 points (left unchanged).

