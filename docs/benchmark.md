# Sample Video GPU Analysis Benchmark

This measures the current CUDA block-analysis stage, **not AV1 encoding**. The project does not yet produce frame or tile payloads.

## Input and environment

- Input: `benckmark/video_1.mp4`, H.264, 1080×1920, 30 fps, 1,207 frames (40.233 s), YUV420.
- GPU: GeForce RTX 3060 Ti, compute capability 8.6, 8 GiB.
- Driver: 595.99.02. CUDA compiler: nvcc 12.4.
- FFmpeg decodes the H.264 input to raw 8-bit YUV420 on the CPU. The stream tool uploads each frame's planes, runs 8×8 block statistics on CUDA, and copies the statistics back.

## Command and result

```sh
set -o pipefail
/usr/bin/time -p sh -c 'ffmpeg -hide_banner -loglevel error -i benckmark/video_1.mp4 -map 0:v:0 -f rawvideo -pix_fmt yuv420p - | build/av1gpu-analyze-stream 1080 1920'
```

Observed run: the analysis process handled all 1,207 frames in 3.668 s (329 frames/s, 682 Mpixels/s luma). Whole command wall time was 4.01 s, or about 301 frames/s including process startup and FFmpeg decode. Average block mean was 99.451 and variance was 63.616. These numbers describe this stage and machine only; they are not an AV1 encode speed or quality result.

## CUDA encoder feasibility probe

The included upstream CUDA encoder (`third_party/av1gpu`, BSD-2-Clause; upstream: [maddog107/av1gpu](https://github.com/maddog107/av1gpu)) was built for `sm_86` from the repository's top-level CMake project. It encodes still images to AVIF, so this probe checks that the software coding pipeline runs on the RTX 3060 Ti; it is not the requested video benchmark.

Frame 0 was decoded from the supplied MP4 to PNG and encoded with `--gpu-entropy`. FFmpeg then identified and decoded the resulting AVIF successfully.

```sh
ffmpeg -hide_banner -loglevel error -i benckmark/video_1.mp4 -frames:v 1 -y /tmp/video_1_frame.png
build-cuda/third_party/av1gpu/av1gpu /tmp/video_1_frame.png /tmp/video_1_frame.avif --stats --gpu-entropy
ffprobe -v error -show_entries stream=codec_name,width,height,pix_fmt -of default=noprint_wrappers=1 /tmp/video_1_frame.avif
```

Observed result: 1080×1920 AV1, YUV420p, 22,839 bytes; GPU encode pipeline 153.97 ms, wall encode 155.26 ms, entropy coding 8.69 ms, 37.9 MiB device memory. PNG load took 103.78 ms. This remains a still-image diagnostic; the full video benchmark below measures the all-intra IVF path.

## Full all-intra video encode

The first complete encode of the supplied video used the CUDA encoder above, with FFmpeg decoding H.264 to raw YUV420 on the CPU. The AV1 encoder encoded every picture as a keyframe on the RTX 3060 Ti. Output: [`build-cuda/video_1_allintra.ivf`](../build-cuda/video_1_allintra.ivf) (30 MiB; local build artifact).

```sh
/usr/bin/time -p bash -o pipefail -c 'ffmpeg -hide_banner -loglevel error -i benckmark/video_1.mp4 -map 0:v:0 -vf scale=in_range=tv:out_range=pc:in_color_matrix=bt709:out_color_matrix=bt709 -f rawvideo -pix_fmt yuv420p - | build-cuda/third_party/av1gpu/av1gpu-video 1080 1920 30 1 build-cuda/video_1_allintra.ivf'
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,nb_read_frames -of default=noprint_wrappers=1 build-cuda/video_1_allintra.ivf
ffmpeg -hide_banner -v error -i build-cuda/video_1_allintra.ivf -f null -
```

Observed final run: 1,207 frames, 262.79 s reported GPU pipeline time, 270.52 s encoder wall time, and 270.95 s whole command time including FFmpeg decode and range conversion. That is 4.45 frames/s or 0.148× the source's 30 fps rate. The IVF is 34,077,606 bytes (about 6.78 Mbit/s over 40.233 s). FFmpeg decoded all 1,207 frames; output metadata is AV1, full-range BT.709, with timestamps from 0.000 to 40.200 seconds at 30 fps. This is a functional all-intra baseline, not an efficiency or quality comparison: every frame is a keyframe, and no inter-frame coding or rate control is implemented. The earlier pass without range conversion is superseded and should not be used for color or bitrate comparisons.

## Batched CUDA encoder

On the same RTX 3060 Ti and 1,207-frame clip, rerun the full encode command above with a new output filename. Append `--fast` after the filename for the fast preset. These are single-run wall times including decode, range conversion, and IVF output. PSNR compares all decoded YUV planes against the same range-converted source.

| Mode | Seconds | Frames/s | IVF bytes | Average PSNR |
| --- | ---: | ---: | ---: | ---: |
| Previous encoder | 270.95 | 4.45 | 34,077,606 | 48.07 dB |
| Batched default | 100.00 | 12.07 | 34,077,606 | 48.07 dB |
| Batched fast | 38.51 | 31.34 | 50,545,035 | 46.81 dB |

The default output matched the previous IVF byte for byte. Both new modes processed seven frames per full batch on this clip. Fast mode also used 4×4 tiles and the turbo3 search preset. Batch buffering adds latency even when throughput exceeds the playback rate.

## CUDA inter encoder (`--pframes`)

Same RTX 3060 Ti and clip. The rate-quality points come from `python3 tests/compare.py build-cuda/third_party/av1gpu/av1gpu-video --refs svt,aom --ours "--pframes --persistent-cdf --qindex {q}" --qs 40,80,120,160`. That covers the first 120 frames, range-converted to full-range BT.709, decoded with dav1d. PSNR is YUV-weighted 6:1:1. References are SVT-AV1 at CRF 24/32/40/48 (preset 8, plus preset 12) and libaom with `cpu-used 8`.

| qindex | kbit/s | PSNR (dB) |
| ---: | ---: | ---: |
| 40 | 2506.0 | 56.06 |
| 80 | 785.3 | 51.32 |
| 120 | 454.2 | 48.73 |
| 160 | 260.8 | 45.79 |

BD-rate of `--pframes` (negative is better):

| Against | BD-rate |
| --- | ---: |
| This encoder at the start of the inter work (`--inter-qindex`, rolling reference) | −81.8% |
| SVT-AV1 preset 12 | −25.4% |
| SVT-AV1 preset 8 | +45.7% |
| libaom `cpu-used 8` | +79.4% |

Contributions measured on the way (BD-rate against SVT-AV1 preset 8): transform split +72.5% → +54.1%, rectangular partitions → +45.5%, HORZ_4/VERT_4 → +41.8%, three-way partitions → +40.4%, depth-2 transform split → +38.8%, rectangular transform split → +38.2% (about 27 fps on the 120-frame test).

Full 1,207-frame clip, piped from FFmpeg as in the commands above:

| Mode | Wall time | Frames/s | Output |
| --- | ---: | ---: | ---: |
| `--pframes --persistent-cdf --qindex 100` | 29.8 s | 41.1 | 3,322,332 bytes |
| `--pframes --persistent-cdf --bitrate 600` | 29.6 s | 41.3 | 3,017,131 bytes (599.9 kbit/s) |

dav1d, libaom and NVDEC (`av1_cuvid`) decoded both streams to byte-identical output (1,207 frames).

Entropy paths on the first 120 frames (identical output in every case):

| qindex | `--pf-cpu-entropy` | default (GPU modeling, host range coder) | `--pf-gpu-entropy` |
| ---: | ---: | ---: | ---: |
| 100 | 45.0 fps | 45.4 fps | 42.3 fps |
| 40 | 43.1 fps | 44.2 fps | 26.8 fps |
| 1 | 38.7 fps | 41.5 fps | 19.4 fps |

The all-GPU range coder is a single serial thread per tile, so it trails at high bitrates. Symbol modeling, the parallel part, runs on the GPU in every mode except `--pf-cpu-entropy`.

## All benchmark clips against SVT-AV1 preset 8 (commit `aca6329`, 2026-11-24)

`python3 tests/bench_all.py build-cuda/third_party/av1gpu/av1gpu-video --tag intra6 --refs svt`: the first 120 frames of
every clip in `benckmark/`, `--pframes --persistent-cdf --qindex {20,50,90,130}`, decoded with dav1d, YUV-weighted PSNR,
BD-rate against SVT-AV1 preset 8 at CRF 24/32/40/48 (negative: fewer bits than SVT for the same quality). The fps columns
are the medians of the four encodes, wall time including startup. The SVT figures come from cached reference encodes, and
the machine had other CPU jobs running (load average 8 to 19), so treat the speed numbers as approximate.

| Clip | Content | BD-rate vs SVT (previous `b40`) | BD-rate vs SVT (intra blocks) | fps ours (previous) | fps SVT |
| --- | --- | ---: | ---: | ---: | ---: |
| video_1 | 1080×1920, 30 fps | +5.3% | +4.1% | 26.2 (24.5) | 36.7 |
| video_2 | 1920×1080, 29.9 fps | +18.4% | +11.4% | 20.9 (19.5) | 33.1 |
| video_3 | 3840×2160, 23.98 fps | −13.2% | −15.7% | 7.7 (7.8) | 17.6 |
| video_4 | 3840×2160, 23.98 fps | −25.6% | −28.2% | 7.7 (7.9) | 16.2 |
| video_5 | 1080p animation | −10.8% | −10.8% | 21.1 (21.0) | 41.9 |
| Mean | | −5.2% | −7.8% | | |

What changed since `b40`: square blocks of inter frames may now be intra-coded (DC / V / H / SMOOTH / SMOOTH_V / SMOOTH_H /
PAETH luma, UV_DC chroma, one DCT_DCT transform per plane). A parallel GPU pass decides intra against the block's inter
cost, then a single-launch superblock wavefront reconstructs the chosen blocks in decode order. `--pf-no-intra` turns it
off. Gain over `b40`: video_1 −1.2%, video_2 −5.9%, video_3 −3.2%, video_4 −3.2%, video_5 0.0% (mean −2.7%). The streams
decode with dav1d to exactly the encoder's reconstruction (`tests/check_pframes.py`, all configurations). The extra mode
bits charged to an intra block (`AV1GPU_PF_INTRABITS`, default 6) barely matter: 2 and 12 moved BD-rate by at most 0.3%.

Status against the goal: more efficient than SVT preset 8 on video_3, video_4 and video_5, still behind on video_1
(+4.1%) and video_2 (+11.4%); slower than SVT on every clip (about 0.45 to 0.7× its speed).
