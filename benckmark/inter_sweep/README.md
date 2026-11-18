# Native CUDA inter-frame sweep

All four runs used the RTX 3060 Ti, the complete 1080×1920 30 fps input, the same full-range conversion and original AAC track, and `--fast --inter`. Each MP4 decoded to 1,207 frames with the expected timestamps. Wall time includes FFmpeg decode, range conversion, encoding, and MP4 muxing. PSNR and SSIM compare decoded video with the same full-range source. Each speed is from a single run.

| Output | MB | Seconds | fps | PSNR (dB) | SSIM | Inter frames |
|---|---:|---:|---:|---:|---:|---:|
| [cuda156](cuda156.mp4) | 16.33 | 44.84 | 26.92 | 40.84 | 0.98056 | 478 |
| [cuda160](cuda160.mp4) | 15.58 | 43.83 | 27.54 | 40.49 | 0.97977 | 484 |
| [cuda190](cuda190.mp4) | 10.60 | 40.04 | 30.14 | 37.50 | 0.96971 | 517 |
| [cuda220](cuda220.mp4) | 11.10 | 46.22 | 26.11 | 34.61 | 0.95806 | 118 |

Against the earlier CUDA all-intra Q160 result (25.85 MB, 31.91 fps, 40.89 dB, SSIM 0.97987), inter Q156 saves 36.8% at 40.84 dB. Q220 produced more keyframes than Q190. Its coarse reconstructed references frequently exceeded the local error limit, so it was both larger and lower quality than Q190. The encoder uses a single reference and falls back to a full keyframe for changes that its five block translations cannot represent.

The [detail crops](detail-crops.png) show the source, all-intra Q160, and inter Q156, Q190 and Q220 from top to bottom. The blue grid at 13.97 s retains visible mottling even at Q156. Q220 also deforms the edges of the digits at 7.97 s. The small chart text at 32.30 s stays readable in the Q156 crop.

The four-second [grain example](sample_grain.mp4) uses manually selected luma grain strength 12. The separate adaptive156.mp4 uses variable frame timing. The fixed-rate sweep above uses neither option.

Adaptive Q156 stored 771 pictures for the original 1,207 display ticks and preserved 40.233333 seconds of video. Scored at the original 30 fps display cadence, it measured 40.836 dB and SSIM 0.980540.

## Sparse inter residuals

Full 1,207-frame encodes on RTX 3060 Ti used `--fast --qindex 156 --inter-residual-threshold N --rolling-reference`. The keyframes used Q156 and the inter frames used pruned 4×4 Walsh–Hadamard coefficients. All MP4s retain the original audio. Decode checks verified the frame count, dimensions and timing. Speed is end-to-end wall throughput from one run per setting.

| Threshold | MP4 MB | fps | PSNR (dB) | SSIM | Keyframes |
|---|---:|---:|---:|---:|---:|
| 8 | 33.82 | 75.72 | 46.13 | 0.98510 | 11 |
| 24 | 19.54 | 92.00 | 40.57 | 0.97199 | 11 |
| 32 | 16.36 | 73.15 | 39.04 | 0.96844 | 11 |

At threshold 32, fixed-keyframe reference output measured 161.82 MB and 39.24 dB. Refreshing the reference reduced the output to 16.36 MB with a 0.20 dB lower PSNR. The fixed-reference run preceded the final scratch-buffer and residual bounds changes, so its timing is omitted here. SVT-AV1 CRF32 measured 5.45 MB, 40.53 fps, 50.87 dB and 0.99654 SSIM on the same input. Sparse residuals narrowed the throughput gap but still trail its compression and quality.

### Adaptive residual entropy

Residual frames now adapt their AV1 symbol probabilities within each frame. The runs used the same Q156, rolling reference mode, source, and original audio as the fixed-probability runs above.

| Threshold | Fixed MP4 MB | Adaptive MP4 MB | Adaptive fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|---:|
| 8 | 33.82 | 21.27 | 79.62 | 46.13 | 0.98510 |
| 24 | 19.54 | 11.99 | 93.35 | 40.57 | 0.97199 |

The 37–39% size reduction comes from coding the same coefficients more efficiently. The host still emits residual entropy symbols. The CPU comparison at CRF32 remains smaller and higher quality at 5.45 MB and 50.87 dB.

### Motion displacement sweep

All runs use threshold 8, Q156 for keyframes, rolling reference, and the original audio. Each search uses zero motion and the four cardinal candidates at the selected displacement. All outputs decoded to 1,207 frames at the source dimensions and timestamps. Timings are single end-to-end measurements.

| Step (pixels) | MP4 MB | fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| 2 (default) | 21.27 | 79.62 | 46.13 | 0.98510 |
| 4 | 21.25 | 46.35 | 46.17 | 0.98531 |
| 8 | 21.17 | 82.45 | 46.27 | 0.98580 |
| 16 | 21.06 | 71.04 | 46.38 | 0.98631 |

The wider distances helped slightly on this clip. Each run still searches only one nonzero displacement, so the default remains two pixels.

### Quantized DCT inter residuals

These runs use `--inter-qindex N --rolling-reference --fast --qindex 156`. CUDA quantizes and reconstructs 4×4 DCT residuals. The encoder uses 8×8 coding blocks. All 1,207 frames decoded at the expected resolution and timestamps. Original audio is included.

| Inter Q index | MP4 MB | fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| 64 | 31.81 | 56.77 | 48.05 | 0.99009 |
| 96 | 20.56 | 64.55 | 45.79 | 0.98506 |
| 128 | 10.39 | 67.96 | 43.08 | 0.97681 |

The Q128 DCT output is smaller and higher quality than the adaptive WHT threshold-24 output (11.99 MB, 40.57 dB, 0.97199 SSIM). It is slower than that WHT run. The CPU comparison still has a substantial compression advantage.

### 8×8 luma DCT

Same settings as the 4×4 DCT sweep, with `--inter-tx8`. Chroma remains 4×4. All 1,207 frames decoded at the original resolution and timestamps. Measurements are single end-to-end runs.

| Inter Q index | MP4 MB | fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| 64 | 31.02 | 48.36 | 48.08 | 0.99027 |
| 96 | 19.69 | 58.65 | 45.84 | 0.98545 |
| 128 | 12.54 | 62.44 | 42.65 | 0.97496 |

At Q96, 8×8 was 4.2% smaller and 0.05 dB higher in PSNR. At Q128, fixed 8×8 lost on both size and quality. The result supports offering both sizes to a block-level rate-distortion decision.

### Adaptive luma transform selection

These runs replace the fixed transform flag with `--inter-tx-adaptive`. CUDA compares the reconstructed pixel error and estimated coefficient cost for 4×4 and 8×8 on each coding block. The selected reconstruction feeds the rolling reference.

| Inter Q index | MP4 MB | fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| 96 | 16.57 | 42.76 | 46.11 | 0.98613 |
| 128 | 10.64 | 58.51 | 42.94 | 0.97617 |

Adaptive Q96 saved 15.8% over fixed 8×8 and gained 0.26 dB. Adaptive Q128 trailed fixed 4×4 slightly on both size and quality. Both full outputs passed independent decode and timing checks. Compared with fixed transforms, this implementation computes and reconstructs both candidates, which lowers throughput.

### Rolling plus keyframe reference

Adding `--anchor-reference` to adaptive Q96 produced 16.51 MB at 45.77 fps, 46.32 dB PSNR and 0.98721 SSIM. The preceding single-reference run measured 16.57 MB at 42.76 fps, 46.11 dB and 0.98613 SSIM. All 1,207 pictures and their timestamps were verified. These are single end-to-end runs. The timing difference does not establish a repeatable speed gain.

### Anchor refresh interval

Adaptive Q96 and the same source and muxing settings as above. The rolling slot refreshes on each inter frame. A periodic anchor copies the decoded picture every N frames in addition to resetting on keyframes. All outputs preserved the original 1,207 timestamps.

| Anchor policy | MP4 MB | fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| Keyframes only | 16.51 | 45.77 | 46.32 | 0.98721 |
| Refresh every 8 frames | 16.70 | 56.72 | 46.15 | 0.98636 |
| Refresh every 16 frames | 16.68 | 72.62 | 46.17 | 0.98648 |

The periodic settings slightly worsened both bitrate and distortion on this clip. Throughput varied substantially across these single runs. The fixed anchor remains the default when a secondary reference is enabled.

### Scene cut keyframes

At adaptive Q96 with rolling and anchor references, the scene-cut RMS settings 6, 12 and 24 all produced MP4s byte-identical to the 16.51 MB anchor Q96 output. No additional keyframes were triggered on this clip. The synthetic abrupt-cut regression produced I/P/I/P as expected and decoded pixel for pixel. The option is disabled by default.

### Rate-aware inter quantization

These runs add `--inter-rdoq` to the rolling plus keyframe reference benchmark with adaptive luma transforms. CUDA compares candidate coefficient levels using squared transform error and an estimated symbol cost. All 1,207 frames decoded at their original resolution and timestamps.

| Inter configuration | MP4 MB | fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|
| Nearest Q96 | 16.51 | 45.77 | 46.32 | 0.98721 |
| Rate-aware Q80 | 12.54 | 84.64 | 46.89 | 0.98867 |
| Rate-aware Q96 | 11.31 | 83.30 | 45.75 | 0.98594 |

Rate-aware Q80 is 24.1% smaller than nearest Q96 and gains 0.57 dB. The timings are single end-to-end runs and need a controlled repeat before attributing the full speed difference to quantization. SVT-AV1 CRF32 still compresses to 5.45 MB at 50.87 dB.

### Automatic motion displacement

Adding `--auto-motion-step` to rate-aware Q80 produced 12.27 MB, 47.14 dB PSNR and 0.98967 SSIM. The fixed two-pixel run produced 12.54 MB, 46.89 dB and 0.98867 SSIM. The prescan selected 2, 4, 8 and 16 pixels on 497, 62, 66 and 571 frames, respectively. The full video passed decode and timing checks.

A same-build repeat reproduced both IVF files byte for byte. Decode, range conversion, and encode to IVF took 14.30 seconds for fixed motion and 14.64 seconds for automatic motion, or 84.41 and 82.45 fps respectively. Those measurements exclude MP4 muxing.

### Smaller GPU motion regions

`--motion-block 8|16|32|64` controls the CUDA motion-search region for quantized inter frames. The default remains 64. These runs use the automatic motion step, adaptive luma transforms, rolling and anchor references, RDO quantization, inter Q80, and keyframe Q156. The Q88 run changes only inter quantization. MP4s in this directory include the original audio. All outputs decoded to 1,207 frames with matching dimensions and timestamps; throughput includes input conversion and muxing.

| Motion region | Inter Q | MP4 MB | fps | PSNR (dB) | SSIM |
|---|---:|---:|---:|---:|---:|
| 64×64 | 80 | 12.27 | 82.45* | 47.14 | 0.98967 |
| 32×32 | 80 | 12.88 | 77.37 | 47.43 | 0.99071 |
| 16×16 | 80 | 13.71 | 67.73 | 47.78 | 0.99188 |
| 32×32 | 88 | 12.30 | 75.11 | 46.92 | 0.98984 |

\*The 64×64 speed is the earlier IVF-only repeat and excludes muxing. Its size and quality are from the muxed MP4. The smaller regions are optional quality/size choices; they do not yet match the CPU SVT-AV1 size at its quality. A synthetic clip with independently translated 16×16 regions decodes exactly at inter Q24 and falls from 113 KB with 64×64 search to 49 KB with 16×16 search.
