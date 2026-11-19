# Initial architecture and milestone

## Goal

Build an AV1 encoder in software whose compute-heavy stages run on CUDA-capable RTX 30 GPUs. The target is not to reproduce NVENC; AV1 encoding is performed by CUDA kernels plus host-side codec control and bitstream generation.

## First supported stream

- AV1 Main profile, 8-bit, 4:2:0
- Raw planar YUV input
- Offline, intra-only, one tile
- A deliberately small, documented coding-tool subset
- AV1 elementary stream in OBUs; no container or audio

This is a staged target, not a claim that a subset will be accepted by every decoder. Each enabled tool must be represented correctly in the AV1 syntax and tested with an independent decoder.

## Pipeline split

**Host (C++):** input and frame lifecycle, sequence/frame syntax, rate-control state, probability/CDF state, tile scheduling, OBU assembly, and diagnostics.

**Device (CUDA):** block statistics and candidate evaluation first; then prediction, transforms, quantization, reconstruction, and filtering as those tools are added. Keep data resident on the GPU across stages where practical.

Entropy coding and syntax packing begin on the host for easier debugging. Tile-level parallel bitstream generation can be investigated later, once the serial path is correct and profiling shows it is a bottleneck.

## Current implementation: all-intra AV1 video

`third_party/av1gpu` supplies a CUDA software encoder for AVIF still images. The added `av1gpu-video` front end accepts one full-range BT.709 planar 8-bit YUV420 frame at a time, uploads padded planes, runs the backend's mode decisions, reconstruction, filtering, and entropy coding on CUDA, and writes a low-overhead AV1 stream in an IVF container. Convert limited-range input before encoding. Every picture is a shown keyframe; no inter prediction or rate control is implemented. The older `av1gpu-analyze` tools remain useful for isolated block-statistics experiments.

Build with `make` (CUDA Toolkit `nvcc` required). The default target is `sm_86`, matching the development GeForce RTX 3060 Ti.

The `av1gpu-seqhdr` utility writes a standalone sequence-header OBU in `still` or `video` mode. It takes dimensions and an explicit AV1 level index; it does not infer a level or emit frame data. The video header disables optional tools to keep the syntax subset narrow.

The `av1gpu-framehdr` utility writes a reduced still-picture key-frame header OBU with one tile and base quantizer zero. It must match a reduced still-picture sequence header with identical dimensions. This header does not include the tile group or entropy-coded block data.

## CUDA encoder foundation

`third_party/av1gpu` supplies a working CUDA software encoder backend for still images. It performs block decisions, reconstruction, post-filtering, and tile entropy coding on CUDA and writes AVIF. A locally built `sm_86` binary encoded a 1080×1920 frame extracted from the sample video in 200.54 ms of GPU pipeline time, and FFmpeg decoded its output. See [benchmark.md](benchmark.md) for the exact run and limitations. Its BSD-2-Clause license and third-party notices are retained with the source.

## CUDA inter frames (`--pframes`)

The supported inter subset:

- **Frames:** single-tile by default (uniform tiles optional), 64×64 superblocks, `enable_order_hint = 0`. Each mini-GOP is a pyramid of hidden ALTREF and middle frames, displayed with `show_existing_frame`. A filtered ALTREF is displayed through an overlay frame instead.
- **Blocks:** NONE, SPLIT, HORZ, VERT, HORZ_A/B, VERT_A/B and (for 32×32 and 64×64 nodes) HORZ_4/VERT_4 partitions. Blocks are at least 8×8. Each block is single-reference (LAST or ALTREF) or a COMPOUND_AVERAGE of both. Modes are NEAREST, NEAR, GLOBAL and NEW, with their compound forms. Vectors use quarter-sample precision and the regular eight-tap filter.
- **Residuals:** TX_MODE_SELECT. DCT_DCT transforms equal to the block (square or 2:1, up to 64×64), or a luma transform split: up to depth 2 (MAX_VARTX_DEPTH) for square blocks, depth 1 for rectangular blocks, with `reduced_tx_set`.
- **Filters:** deblocking and CDEF, with one CDEF strength per frame.
- **CDFs:** frame-end CDFs are kept per reference slot for `primary_ref_frame`.

Pipeline (`encoder/apps/pframe.cu`, `pframe_tile.cuh`, `pframe_writer.h`):

1. **Phase planes (GPU):** each reference is expanded into its 16 exact quarter-sample planes.
2. **Search (GPU):** a quarter-resolution coarse search, then per-size refinement on the phase planes, then the LAST / ALTREF / compound choice.
3. **Evaluation (GPU):** exact prediction, forward DCT, quantization, an AV1-context rate estimate, and an exact inverse transform. Bottom-up partition decisions, then commit of the reconstruction and levels.
4. **Filters (GPU):** deblocking and CDEF searches on the reconstruction.
5. **Entropy coding (GPU + host):** on a separate stream, one thread per superblock builds decode-order block lists. One thread per block then fills mode info and coefficient-context maps, chooses inter modes from the motion-vector candidate list (costs from the default CDFs, so blocks are independent), and emits symbol records with exact contexts. A host writer thread range-codes the records, which is the one inherently serial step. The all-GPU option range-codes on the GPU instead. The host writes the frame header and OBUs.
6. **Rate control (host):** one quantizer per period, adjusted by the measured bits.

Every feature is validated by decoding with dav1d and comparing to the encoder reconstruction. Full-clip streams are also cross-checked with libaom and NVDEC.

## Next milestone: inter-frame coding and quality

Done: CUDA motion search with per-block vectors, a hidden-ALTREF pyramid with compound prediction, rate control, in-loop filters, GPU and overlapped CPU entropy coding, and BD-rate tooling (`tests/compare.py`). Remaining gaps relative to SVT-AV1 preset 8:

1. Blocks with a 4-pixel side (4×4, 4×8, 8×4, 16×4, 4×16). Their chroma is predicted from several luma blocks' vectors.
2. Intra blocks in inter frames. These need a wavefront for neighbor reconstruction.
3. Switchable interpolation filters, inter transform types other than DCT_DCT, OBMC and warped motion.
4. Broader input formats and audio/container support, once the elementary video path is stable.

The all-intra milestone requires independent decoding of the full output with the expected frame count and timestamps. The sample video benchmark is recorded in [benchmark.md](benchmark.md).

## Key risks

- AV1 has extensive syntax and tool interactions; a narrow profile/tool subset is essential for a first conforming stream.
- GPU parallelism is abundant in pixel/block calculations, while entropy coding and adaptive context updates create dependencies. Tile boundaries offer a path to parallelism but add complexity.
- RTX 30 cards vary in compute capability and memory bandwidth. Select a baseline GPU after gathering the target model(s).
- A practical encoder requires an independent decoder in the development loop and a CUDA-capable machine for compilation and profiling.
