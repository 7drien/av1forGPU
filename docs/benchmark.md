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
| 40 | 2637.6 | 55.50 |
| 80 | 824.4 | 50.77 |
| 120 | 470.0 | 47.96 |
| 160 | 261.5 | 45.26 |

BD-rate of `--pframes` (negative is better):

| Against | BD-rate |
| --- | ---: |
| This encoder at the start of the inter work (`--inter-qindex`, rolling reference) | −79.0% |
| SVT-AV1 preset 12 | −11.2% |
| SVT-AV1 preset 8 | +72.5% |
| libaom `cpu-used 8` | +112.9% |

Full 1,207-frame clip, piped from FFmpeg as in the commands above:

| Mode | Wall time | Frames/s | Output |
| --- | ---: | ---: | ---: |
| `--pframes --persistent-cdf --qindex 100` | 20.5 s | 59.9 | 3,477,442 bytes |
| `--pframes --persistent-cdf --bitrate 600` | 20.0 s | 61.4 | 3,007,951 bytes (598.1 kbit/s) |

dav1d, libaom and NVDEC (`av1_cuvid`) decoded both streams to byte-identical output (1,207 frames). P-frame GPU time is about 8 ms per frame: refinement 5.3 ms, evaluation 2.0 ms, coarse search 0.6 ms. Filters add about 2 ms. Entropy coding runs concurrently on the CPU.
