## Summary

<!-- What changes and why. -->

## Validation

- [ ] `python3 tests/check_pframes.py build-cuda/third_party/av1gpu/av1gpu-video`: all configurations decode exactly
- [ ] Output-neutral change: `.ivf` byte-identical before / after (clips and quantizers used: ...)
- [ ] Compression change: BD-rate (`tests/bench_all.py`, `tests/bd_tags.py`) on: ...
- [ ] Speed: interleaved timings (GPU, driver, machine load): ...

## Notes

<!-- GPU requirements, new knobs or environment variables, docs updated (docs/benchmark.md, docs/svt_gap_plan.md). -->
