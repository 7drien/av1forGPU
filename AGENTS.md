# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains implementation code; CUDA kernels live in `src/cuda/`, and bitstream code in `src/bitstream/`.
- `include/` mirrors those modules and contains public headers.
- `docs/architecture.md` describes the encoder stages and current scope.
- `benckmark/` contains sample media. The current executable accepts one raw, tightly packed 8-bit YUV420 frame; it does not decode MP4 or emit a complete AV1 stream yet.
- `build/av1gpu-analyze-stream` reads multiple raw YUV420 frames from standard input and reports CUDA block-analysis throughput.
- There is no test directory or test framework currently.

## Build & Local Development

- `make` builds `build/av1gpu-analyze` with `nvcc` for `sm_86` (RTX 3060 Ti). Override the GPU target with `make CUDA_ARCH=sm_XX` when appropriate.
- `make clean` removes generated build files.
- Example: `ffmpeg -i benckmark/video_1.mp4 -frames:v 1 -pix_fmt yuv420p -f rawvideo build/frame.yuv`, then `build/av1gpu-analyze build/frame.yuv 1080 1920`.
- Stream analysis: `ffmpeg -i benckmark/video_1.mp4 -f rawvideo -pix_fmt yuv420p - | build/av1gpu-analyze-stream 1080 1920`.
- A working NVIDIA driver and CUDA Toolkit are required. Keep generated binaries and media under `build/`.

## Coding Style & Naming

- Use C++17 and CUDA C++; use four spaces for indentation and keep braces consistent with nearby code.
- Use `PascalCase` for types, `camelCase` for functions, and descriptive names for CUDA kernels (currently suffixed `Kernel`). Keep implementation in the matching module directory and expose declarations through `include/`.
- Check CUDA runtime calls and validate input dimensions, sizes, and bitstream bounds. Avoid introducing host-side codec substitutes for GPU encoding work.

## Testing & Validation

- Tests live in `tests/` (no framework; each script exits non-zero on failure). After building `build-cuda`:
  - `python3 tests/check_pframes.py build-cuda/third_party/av1gpu/av1gpu-video`: `--pframes` streams must decode with dav1d to exactly the encoder reconstruction.
  - `python3 tests/check_inter.py build-cuda/third_party/av1gpu/av1gpu-video`: the older `--inter` modes.
  - `python3 tests/compare.py build-cuda/third_party/av1gpu/av1gpu-video --refs svt,aom`: BD-rate against SVT-AV1 and libaom (outputs in `build-cuda/compare/`).
  - `tests/gen_inter_tables.py` regenerates `encoder/apps/inter_tables.h` from the AV1 specification source.
- For now, `make` checks compilation. For encoder work, add bitstream validation with an independent AV1 decoder as soon as the feature can emit a complete stream; document the input and expected result.

## Commits & Pull Requests

- No usable Git history is available to establish a repository-specific commit convention. Use short imperative subjects, optionally scoped, such as `cuda: add block activity kernel`.
- Pull requests should describe scope, list build/validation results, note GPU requirements, and link issues. Include decoder output or commands for bitstream changes.

## Architecture Notes

CUDA is intended for parallel pixel/block work; host code owns codec state, syntax, and OBU assembly. Keep these responsibilities separate and update `docs/architecture.md` when the supported AV1 subset or pipeline changes.
