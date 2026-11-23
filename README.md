# AV1 for RTX 30

An experimental AV1 **software encoder** that uses CUDA compute on NVIDIA GeForce RTX 30-series GPUs. The encoder must generate AV1 bitstreams in software; it must not depend on NVENC AV1, which RTX 30-series cards do not provide.

## Initial target

Start with a correctness-first, offline encoder for raw 8-bit 4:2:0 video. Keep the first implementation deliberately narrow: one frame at a time, a small set of intra prediction modes, a simple quantizer, and a single tile. The first milestone is a standards-valid AV1 elementary stream that an independent decoder can read. Performance and broader codec tools come after that milestone.

The input interface will initially be raw YUV420 frames, and the output will be an AV1 OBU bitstream. Container muxing, audio, real-time guarantees, 10-bit/HDR, and competitive compression efficiency are outside the first milestone.

## Design principles

- CUDA performs the parallel pixel and block work; CPU code manages sequence/frame state and bitstream assembly.
- Keep the bitstream writer and syntax decisions explicit, so conformance failures can be localized.
- Implement features incrementally and validate every bitstream with an independent AV1 decoder.
- Do not route encode work through NVENC. RTX 30's AV1 decode support does not provide AV1 encode hardware.

## Roadmap

1. Define project structure, supported profile/tools, and build requirements.
2. Add input frame handling and a minimal AV1 OBU/sequence-header writer.
3. Produce a decodable intra-only stream with a deliberately simple coding path.
4. Move block transforms, quantization, and candidate evaluation to CUDA.
5. Add inter prediction/motion search, rate control, and more AV1 tools in measured steps.
6. Profile on RTX 30 hardware and optimize GPU/CPU synchronization and tile scheduling.

See [docs/architecture.md](docs/architecture.md) for the initial implementation boundaries.

## Build and current status

The CUDA video encoder reads raw 8-bit YUV420 and writes IVF. Its default mode batches independent keyframes. `--pframes` enables the CUDA inter encoder described below: per-block motion vectors, a hidden-ALTREF pyramid with compound prediction, in-loop filters and optional rate control. The older `--inter` modes remain for experiments.

Requirements: NVIDIA driver, CUDA Toolkit 12+, CMake 3.24+, a C++17 compiler, and JPEG/PNG/WebP development libraries. Build for an RTX 3060 Ti with:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda -j 1
build-cuda/third_party/av1gpu/av1gpu --help
```

Encode the sample clip at 30 fps:

```sh
ffmpeg -hide_banner -loglevel error -i benckmark/video_1.mp4 -map 0:v:0 \
  -vf scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709 -f rawvideo -pix_fmt yuv420p - | \
  build-cuda/third_party/av1gpu/av1gpu-video 1080 1920 30 1 build-cuda/video_1.ivf
```

The video tool expects full-range BT.709 raw YUV420 on standard input and takes width, height, frame-rate numerator/denominator, and output IVF path. It batches up to eight independent frames, targeting 16 million padded luma pixels per batch, and keeps coding decisions in GPU memory through entropy coding. Append `--fast` after the output path to use the turbo3 search preset and enable parallel tiles on large frames. This changes compression efficiency. Omit the flag to retain the original search settings. Use `--qindex N` (1–255, default 100) to adjust compression. Higher values discard more detail and generally produce smaller files. The resulting bitrate depends on the clip. Convert limited-range sources before piping them in, as shown above.

The standalone analysis prototype can still be built with `make` (default CUDA target `sm_86`).

```sh
make
ffmpeg -i benckmark/video_1.mp4 -frames:v 1 -pix_fmt yuv420p -f rawvideo build/frame.yuv
build/av1gpu-analyze build/frame.yuv 1080 1920
build/av1gpu-analyze-stream 1080 1920 < build/video.yuv
build/av1gpu-seqhdr still 1080 1920 8 build/still_sequence_header.obu
build/av1gpu-seqhdr video 1080 1920 8 build/video_sequence_header.obu
build/av1gpu-framehdr 1080 1920 build/frame_header.obu
```

The single-frame analysis tool accepts exactly one tightly packed, 8-bit planar YUV420 frame; the stream tool reads frames from standard input. The sequence-header tool takes an explicit level index (8 is AV1 level 4.0). The frame-header tool emits only a header OBU; it must be paired with matching sequence and tile data and is not a decodable frame by itself. See [docs/benchmark.md](docs/benchmark.md) for measured prototype and encoder-foundation results.

### CUDA inter encoder (`--pframes`)

`--pframes` is the recommended video mode. After each keyframe it codes mini-GOPs of up to 32 frames (`--pf-gop N`) as a hierarchical pyramid. The last frame is coded first as a hidden ALTREF. Each middle frame predicts from its nearest coded past frame (LAST) and future frame (ALTREF). Hidden frames are displayed later with `show_existing_frame`.

Per frame, CUDA does the following:

- **Motion search:** for each reference, an exhaustive quarter-resolution search, then a per-block full-pel and quarter-sample search on exact AV1 interpolation phase planes. Blocks from 64×64 down to 8×8 in every AV1 partition shape except those with a 4-pixel side (square, HORZ/VERT halves, HORZ_4/VERT_4 quarters, three-way HORZ_A/B and VERT_A/B) each carry their own vector.
- **Prediction:** each block uses LAST, ALTREF, or their compound average.
- **Residuals:** a quantized DCT the size of the block (up to 64×64, rectangular for rectangular blocks), or a TX_MODE_SELECT transform split: one or two levels for square blocks, one level (longer side halved) for rectangular blocks. Partition, skip and transform choices are made by rate-distortion cost.
- **Loop filtering:** deblocking and CDEF with levels chosen by measured error.
- **Entropy modeling:** parallel kernels build the mode info and coefficient contexts, choose NEAREST/NEAR/NEW modes from the AV1 motion-vector candidate list, and emit compact symbol records with exact contexts.
- **Reconstruction:** decoder-exact, so it can serve as a reference picture.

A writer thread range-codes the GPU's symbol records and writes the stream, overlapped with the GPU's work on later frames. `--pf-gpu-entropy` also runs the range coder on the GPU (one warp per tile, on a low-priority stream). `--pf-cpu-entropy` runs every entropy phase on the CPU. All three produce identical bitstreams; the default is the fastest at every bitrate tested. `--pf-tiles C R` sets log2 tile columns and rows.

```sh
ffmpeg -hide_banner -loglevel error -i benckmark/video_1.mp4 -map 0:v:0 \
  -vf scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709 -f rawvideo -pix_fmt yuv420p - | \
  build-cuda/third_party/av1gpu/av1gpu-video 1080 1920 30 1 out.ivf --pframes --persistent-cdf --qindex 100
```

Options:

- `--qindex N` sets the keyframe quantizer. P frames use N + 60 (`--pf-qoffset`), lowered by up to `--pf-altref-boost` (default 100) toward the base of the pyramid.
- `--bitrate KBPS` enables rate control: one base quantizer per keyframe or mini-GOP period, with a buffer of `--rc-buffer` seconds (default 2).
- `--persistent-cdf` carries adapted probabilities between frames through each reference slot.
- `--pf-tf-frames K` (default 4, 0 off): GPU motion-compensated temporal filter of ALTREF sources over ±K frames, with overlay frames for display; key frames are filtered over the next 8 frames. Both only on noisy content (estimated luma noise ≥ 0.25).
- Switchable interpolation filters are on by default (`--pf-no-switchable`); the lambda multiplier defaults to 0.5.
- `--pf-rdoq` enables greedy rate-distortion optimized quantization. Off by default: no gain on the sample clip.
- `--pf-no-mode-refine` disables the final-partition motion refinement that tests the decoder's spatial and temporal vectors.
- `--pf-no-filters` turns off in-loop filtering. `--pf-no-tx-split` and `--pf-no-rect` turn off transform split and rectangular partitions.

On the 1,207-frame sample clip this path runs at about 41 fps on an RTX 3060 Ti, including FFmpeg's H.264 decode. Its output decodes byte-identically with dav1d, libaom and the GPU's NVDEC AV1 decoder. See [docs/benchmark.md](docs/benchmark.md) for rate-quality results.

### Native temporal prediction (earlier experimental modes)

Append `--inter` to the existing av1gpu-video command. CUDA evaluates five integer translations by default, or seven with `--diagonal-motion` per 64×64 region against the most recent reconstructed keyframe, including chroma. AV1 segmentation selects each block's translation. Use `--motion-step N` to set the displacement to an even number of pixels from 2 to 16. Each search evaluates zero and the enabled directions at that displacement. Use `--auto-motion-step` with fixed frame timing to choose among 2, 4, 8 and 16 pixels from a sampled CUDA luma search before the full luma and chroma search. The encoder writes an inter frame when local and frame-average prediction errors fit the quantizer-dependent limits. Larger changes trigger a new keyframe, with a maximum spacing of 120 input ticks by default, configurable with `--keyint`. Use `--scene-cut-rms N` to enable early keyframes even in residual modes. It requires frame-wide prediction RMS above N and at least one third of samples in regions above that threshold. The measurement includes luma and chroma. This implementation uses skipped residuals on predicted pictures. Its single-tile inter path supports even dimensions up to 4096 pixels wide and 9,437,184 padded luma samples. Other sizes use the existing keyframe path.

Use `--lossless-inter-residual` to exercise the experimental inter coefficient path. CUDA computes 4×4 lossless residual transforms against the reconstructed keyframe, and the tile writer emits their coefficients. Predicted frames reconstruct the input exactly. Keyframes still use the configured quantizer, so the overall stream is lossy. The mode requires dimensions divisible by eight within the single-tile limits and fixed frame timing. It uses host entropy coding for inter residuals and is intended for decoder verification.

Use `--inter-residual-threshold N` to try sparse residuals (1–255). CUDA drops small Walsh–Hadamard coefficients, and the tile writer skips coding blocks with no remaining corrections. This uses the same bitstream transform and input restrictions as exact residual mode. Higher thresholds discard more detail.

Use `--inter-qindex N` (1–255) for experimental quantized DCT inter residuals. It uses 8×8 coding blocks with 4×4 transforms and supports rolling references. Add `--inter-tx8` for 8×8 luma transforms. Chroma remains 4×4. With rolling references, `--inter-tx-adaptive` compares 4×4 and 8×8 luma candidates on CUDA using reconstructed error and an estimated coefficient cost. Add `--inter-rdoq` to let CUDA compare the nearest coefficient level, the next lower level, and zero using an estimated coding cost. The mode applies a smaller rate penalty to DC than to AC coefficients. Add `--persistent-cdf` (requires `--inter-qindex` and `--rolling-reference`, no `--adaptive-fps`) to carry adapted entropy-coding probabilities and global-motion parameters from one P frame to the next through `primary_ref_frame`. Pictures are identical to the default mode. On the sample clip at inter Q80, 32×32 motion blocks, it reduced the IVF from 21,115,367 to 20,946,208 bytes (−0.8%). `--qindex` still sets the keyframe quantizer. The mode has the same dimension and timing limits as the other residual paths.

Use `--rolling-reference` to search against the latest reconstructed picture. CUDA writes each accepted inter prediction to a second reference buffer and swaps buffers for the next frame. The decoder refreshes the same reference slot. With residual coding, the GPU adds the same decoded corrections before refreshing the slot. This experimental mode requires fixed frame timing and dimensions divisible by eight.

Add `--anchor-reference` to rolling reference mode to retain the most recent reconstructed keyframe in an additional AV1 reference slot. CUDA compares a zero-motion prediction from that keyframe against the rolling-reference candidates for each 64×64 region. The option uses cardinal motion candidates and cannot be combined with `--diagonal-motion`. Use `--anchor-interval N` (2–10000 frames) to enable the anchor and refresh it periodically from the decoded inter picture. A keyframe resets both slots and restarts the interval.

Use `--adaptive-fps` to enable prediction and hold a displayed picture when the extra prediction error is small. IVF presentation timestamps use the original input time base. Remux with FFmpeg stream copy to preserve variable sample durations in MP4. Input is still a fixed-rate raw stream. Add `--grain 12` for optional decoder-synthesized luma grain (strength 0–32, default 0). Grain uses a new seed per picture, so enabling grain retains every input tick. The model uses a manually selected amplitude.

Run `python3 tests/check_pframes.py build-cuda/third_party/av1gpu/av1gpu-video` to check that dav1d decodes `--pframes` streams to exactly the encoder's reconstruction. It covers pyramids, keyframe boundaries, tiles, both entropy coders and rate control. Run `python3 tests/compare.py build-cuda/third_party/av1gpu/av1gpu-video --refs svt,aom` for BD-rate against SVT-AV1 and libaom on the first 120 frames of the sample clip. Run `python3 tests/check_inter.py build-cuda/third_party/av1gpu/av1gpu-video` to check reconstructed motion blocks and variable timing with dav1d. Full sample results are in [the native inter sweep](benckmark/inter_sweep/README.md).
