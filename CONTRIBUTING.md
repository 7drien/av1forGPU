# Contributing

Thanks for your interest. This is an experimental encoder: changes are accepted when they are measured and the streams
stay decodable.

## Build

Requirements: NVIDIA GPU (developed on an RTX 3060 Ti, `sm_86`), driver, CUDA Toolkit 12+, CMake 3.24+, a C++17
compiler, FFmpeg with libdav1d (tests), and JPEG / PNG / WebP development libraries.

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda -j 8 --target av1gpu-video
```

## Before opening a pull request

1. **Bitstream exactness** (required for any encoder change):
   `python3 tests/check_pframes.py build-cuda/third_party/av1gpu/av1gpu-video` must report that dav1d decodes every
   configuration to exactly the encoder's reconstruction.
2. **Output-neutral speed changes**: show that the output is byte-identical (`cmp` of `.ivf` files before / after) and
   give interleaved timings of both binaries.
3. **Compression changes**: give BD-rate with `tests/bench_all.py` and `tests/bd_tags.py` (see
   [docs/benchmark.md](docs/benchmark.md)), at least on the clips the change targets, and note the speed effect.
4. Note GPU and driver versions for timing results, and whether other GPU / CPU jobs were running.

## Style

- C++17 / CUDA C++, four-space indentation, braces as in the surrounding code.
- `PascalCase` types, `camelCase` functions, CUDA kernels named for what they do.
- CUDA does the parallel pixel / block work; host code owns codec state, syntax and OBU assembly.
- Check CUDA calls and validate dimensions, sizes and bitstream bounds.
- Commit subjects: short and imperative, optionally scoped (`video: ...`, `cuda: ...`, `docs: ...`).
- Record measurements (including the experiments that did not help) in [docs/svt_gap_plan.md](docs/svt_gap_plan.md).

## License

By contributing you agree that your contributions are licensed under the [BSD 2-Clause License](LICENSE.md).
