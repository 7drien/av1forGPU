# SVT-AV1 feature map for native CUDA

SVT-AV1 is the CPU comparison encoder. The native encoder uses its own CUDA analysis and AV1 bitstream path. This inventory covers the public switches listed in SVT-AV1's parameter appendix and groups the major internal tools below. Availability varies with version and build. [Parameter appendix](https://github.com/AOMediaCodec/SVT-AV1/blob/main/Docs/Parameters.md) · [Encoder design and algorithm appendices](https://github.com/AOMediaCodec/SVT-AV1/blob/main/Docs/svt-av1-encoder-design.md)

| Feature family | Native CUDA status and GPU implementation path |
| --- | --- |
| Intra prediction, partition search, transform selection and coefficient coding | CUDA backend already implements these for keyframes. Reuse its transforms, quantization and tile entropy path when adding inter residuals. |
| Inter motion estimation, motion vectors, global and compound prediction, motion field | CUDA searches whole-pixel translations on selectable 8–64 pixel regions against rolling and optional anchor references, with a GPU prescan to choose displacement per frame. The AV1 packet currently signals these as global translations and forced reference segments. Add GPU image pyramids, shortlist candidates, and refine at subpixel precision; true block motion vectors and compound prediction still require bitstream syntax and decoder-identical interpolation. |
| Rate-distortion optimization, adaptive quantization, variance boost, ROI and quantization matrices | Fixed Q in the video CLI. Some intra tools exist in the backend. Evaluate rate and distortion per block on CUDA, then use reductions for frame-level budgets and spatial Q maps. |
| CQP, CRF, VBR, CBR, capped bitrate, multipass, buffer control and recode | Video CLI exposes fixed Q. Use GPU frame statistics and host-side control for bitrate and buffer decisions. Recode only when the estimated benefit warrants a second GPU pass. |
| Low delay and random access prediction structures, hierarchical GOPs, scene cuts, startup controls, switch frames, overlays and lookahead | Error-driven keyframes and a configurable maximum interval are implemented. Add parallel lookahead and dependency-aware reference scheduling after inter residuals are working. |
| Temporal filtering, denoising, film grain modeling and grain synthesis | Decoder-side luma grain takes a manual strength. An automatic model needs GPU noise estimation and motion-compensated denoising. |
| Deblocking, CDEF, loop restoration, super-resolution and resizing | Deblocking and CDEF exist in the intra backend. Restoration and scaling need decoder-identical reconstruction before filtered frames can serve as references. |
| Screen content, intra block copy, palette and content detection | Intra palette support exists. Parallel block hashes and activity statistics can shortlist screen-content modes before rate-distortion evaluation. |
| Tiles, presets, tuning, fast-decode, pipeline parallelism and CPU affinity/SIMD | Intra batches and GPU tile entropy already exist. Inter tiles are currently single-tile with host entropy coding. Use CUDA streams and events to overlap transfers and analysis, and parallelize independent tiles where the bitrate cost is acceptable. CPU affinity and SIMD switches have no direct CUDA counterpart. |
| Bit depth, chroma, profiles, levels, HDR/color metadata, AVIF, IVF/OBU, reporting and input controls | Video CLI handles 8-bit 4:2:0 IVF. Extend format coverage once the prediction path is stable. Container and metadata handling belong on the host. |

## Documented command-line option index

`--ac-bias`, `--adaptive-film-grain`, `--aq-mode`, `--asm`, `--avif`, `--buf-initial-sz`, `--buf-optimal-sz`, `--buf-sz`, `--chroma-qindex-offsets`, `--chroma-qm-max`, `--chroma-qm-min`, `--chroma-sample-position`

`--chroma-u-ac-qindex-offset`, `--chroma-u-dc-qindex-offset`, `--chroma-v-ac-qindex-offset`, `--chroma-v-dc-qindex-offset`, `--color-format`, `--color-help`, `--color-primaries`, `--color-range`, `--content-light`, `--cqp`, `--crf`, `--enable-cdef`

`--enable-dg`, `--enable-dlf`, `--enable-intrabc`, `--enable-kf-tf`, `--enable-mfmv`, `--enable-overlays`, `--enable-qm`, `--enable-restoration`, `--enable-stat-report`, `--enable-tf`, `--enable-variance-boost`, `--errlog`

`--fast-decode`, `--fgs-table`, `--film-grain`, `--film-grain-denoise`, `--force-key-frames`, `--forced-max-frame-height`, `--forced-max-frame-width`, `--fps-denom`, `--fps-num`, `--frame-resz-denoms`, `--frame-resz-events`, `--frame-resz-kf-denoms`

`--gop-constraint-rc`, `--hbd-mds`, `--help`, `--hierarchical-levels`, `--inj`, `--inj-frm-rt`, `--input-depth`, `--irefresh-type`, `--ivf`, `--key-frame-chroma-qindex-offset`, `--key-frame-qindex-offset`, `--keyint`

`--lambda-scale-factors`, `--level`, `--lookahead`, `--lossless`, `--lp`, `--luma-y-dc-qindex-offset`, `--luminance-qp-bias`, `--mastering-display`, `--matrix-coefficients`, `--max-inter-bitrate-pct`, `--max-intra-bitrate-pct`, `--max-qp`

`--max-tx-size`, `--maxsection-pct`, `--mbr`, `--mbr-overshoot-pct`, `--min-qp`, `--minsection-pct`, `--nb`, `--no-progress`, `--obu`, `--overshoot-pct`, `--pass`, `--passes`

`--pred-struct`, `--preset`, `--profile`, `--progress`, `--qindex-offsets`, `--qm-max`, `--qm-min`, `--qp`, `--qp-file`, `--qp-scale-compress-strength`, `--qpfile`, `--rc`

`--recode-loop`, `--resize-denom`, `--resize-kf-denom`, `--resize-mode`, `--roi-map-block-size`, `--roi-map-file`, `--rtc`, `--scd`, `--scm`, `--sframe-dist`, `--sframe-mode`, `--sframe-posi`

`--sframe-qp`, `--sframe-qp-offset`, `--sharpness`, `--skip`, `--startup-mg-size`, `--startup-qp-offset`, `--stat-file`, `--stats`, `--superres-denom`, `--superres-kf-denom`, `--superres-kf-qthres`, `--superres-mode`

`--superres-qthres`, `--svtav1-params`, `--tbr`, `--tf-strength`, `--tier`, `--tile-columns`, `--tile-rows`, `--transfer-characteristics`, `--tune`, `--undershoot-pct`, `--use-fixed-qindex-offsets`, `--use-q-file`

`--variance-boost-curve`, `--variance-boost-strength`, `--variance-octile`, `--version`

## CUDA benchmark

On the RTX 3060 Ti at Q156, the five-candidate result measured 16.326 MB, 26.92 fps, 40.840 dB PSNR and 0.98056 SSIM. The seven-candidate run measured 16.337 MB, 26.41 fps, 40.840 dB and 0.98056 SSIM. Each run produced 478 inter frames and 729 keyframes. The extra candidates had negligible impact on this clip, so they are opt-in. Timings are single end-to-end runs. [Seven-candidate sample](../benckmark/inter_sweep/cuda156_diagonal.mp4)
