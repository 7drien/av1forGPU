// Experimental CUDA AV1 video encoder. Input is raw 8-bit planar YUV420;
// all AV1 decisions, reconstruction, filters, and entropy coding use the CUDA backend.
#include "presets.h"
#include "inter_packet.h"
#include "pframe.h"
#include "pframe_writer.h"
#include "av1g/av1g.h"
#include "pframe_mf.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <cmath>
#include <functional>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void put_le(FILE *f, uint64_t v, int n) {
    for (int i = 0; i < n; ++i) fputc((int)(v >> (8 * i)) & 255, f);
}

static bool write_ivf_header(FILE *f, int w, int h, int fpsNum, int fpsDen, uint32_t frames) {
    return fwrite("DKIF", 1, 4, f) == 4 && (put_le(f, 0, 2), true) && (put_le(f, 32, 2), true) &&
           fwrite("AV01", 1, 4, f) == 4 && (put_le(f, w, 2), true) && (put_le(f, h, 2), true) &&
           (put_le(f, fpsNum, 4), true) && (put_le(f, fpsDen, 4), true) && (put_le(f, frames, 4), true) &&
           (put_le(f, 0, 4), true);
}

static bool write_ivf_frame(FILE *f, const std::vector<uint8_t> &obu, uint64_t timestamp) {
    if (obu.size() > UINT32_MAX) return false;
    put_le(f, obu.size(), 4);
    put_le(f, timestamp, 8);
    return fwrite(obu.data(), 1, obu.size(), f) == obu.size();
}

static bool read_frame(FILE *f, int w, int h, std::vector<uint8_t> &src) {
    const size_t cw = (w + 1) / 2, ch = (h + 1) / 2;
    const size_t yBytes = (size_t)w * h, cBytes = cw * ch;
    src.resize(yBytes + 2 * cBytes);
    size_t n = fread(src.data(), 1, src.size(), f);
    if (n == 0 && feof(f)) return false;
    if (n != src.size()) {
        fprintf(stderr, "incomplete raw frame: got %zu of %zu bytes\n", n, src.size());
        exit(2);
    }
    return true;
}

// Sums of the subsampled samples (every second row and column) of each 16x16 luma block, raster order: the block
// statistics of the scene-cut and motion measures (blockMeanDiff / blockMeanCorr).
static std::vector<int> block_sums16(const std::vector<uint8_t> &f, int w, int h) {
    std::vector<int> out;
    out.reserve(size_t(w / 16) * (h / 16));
    for (int by = 0; by + 16 <= h; by += 16)
        for (int bx = 0; bx + 16 <= w; bx += 16) {
            int s = 0;
            for (int y = 0; y < 16; y += 2)
                for (int x = 0; x < 16; x += 2) s += f[size_t(by + y) * w + bx + x];
            out.push_back(s);
        }
    return out;
}

// Raw input frames read ahead on a thread, with their block sums, so reading and the frame statistics overlap the GPU
// work. The state is shared with the thread, which is detached at destruction (it may be blocked reading stdin).
class FrameReader {
public:
    struct Frame { std::vector<uint8_t> data; std::vector<int> sums; };
    FrameReader(int w, int h, size_t ahead) : st_(std::make_shared<State>()) {
        auto st = st_;
        std::thread([st, w, h, ahead] {
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(st->mu);
                    st->cv.wait(lock, [&] { return st->stop || st->q.size() < ahead; });
                    if (st->stop) return;
                }
                Frame f;
                const bool ok = read_frame(stdin, w, h, f.data);
                if (ok) f.sums = block_sums16(f.data, w, h);
                std::lock_guard<std::mutex> lock(st->mu);
                if (ok) st->q.push_back(std::move(f));
                else st->eof = true;
                st->cv.notify_all();
                if (!ok) return;
            }
        }).detach();
    }
    ~FrameReader() {
        std::lock_guard<std::mutex> lock(st_->mu);
        st_->stop = true;
        st_->cv.notify_all();
    }
    // the next frame, or false at the end of the input
    bool next(Frame &f) {
        std::unique_lock<std::mutex> lock(st_->mu);
        st_->cv.wait(lock, [&] { return !st_->q.empty() || st_->eof; });
        if (st_->q.empty()) return false;
        f = std::move(st_->q.front());
        st_->q.pop_front();
        st_->cv.notify_all();
        return true;
    }
private:
    struct State {
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Frame> q;
        bool eof = false, stop = false;
    };
    std::shared_ptr<State> st_;
};

static void pad_plane(const uint8_t *src, int w, int h, int pitch, int paddedW, int paddedH, uint8_t *dst) {
    for (int y = 0; y < paddedH; ++y) {
        const uint8_t *row = src + (size_t)std::min(y, h - 1) * pitch;
        uint8_t *out = dst + (size_t)y * paddedW;
        memcpy(out, row, w);
        memset(out + w, row[w - 1], paddedW - w);
    }
}

static void pad_plane(const uint8_t *src, int w, int h, int pitch, int paddedW, int paddedH,
                      std::vector<uint8_t> &dst) {
    dst.resize((size_t)paddedW * paddedH);
    pad_plane(src, w, h, pitch, paddedW, paddedH, dst.data());
}

class FrameUploads {
public:
    FrameUploads(int w, int h, int aw, int ah, size_t count)
        : w(w), h(h), aw(aw), ah(ah), yBytes(size_t(aw) * ah), frames(count) {}
    FrameUploads(const FrameUploads &) = delete;
    FrameUploads &operator=(const FrameUploads &) = delete;

    ~FrameUploads() {
        if (stream) cudaStreamSynchronize(stream);
        for (auto &frame : frames) {
            if (frame.ready) cudaEventDestroy(frame.ready);
            if (frame.planes[0]) cudaFree(frame.planes[0]);
        }
        for (auto &stage : staging) {
            if (stage.done) cudaEventDestroy(stage.done);
            if (stage.data) cudaFreeHost(stage.data);
        }
        if (consumed) cudaEventDestroy(consumed);
        if (stream) cudaStreamDestroy(stream);
    }

    bool init() {
        cudaError_t e = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming);
        for (auto &stage : staging) {
            if (e == cudaSuccess) e = cudaMallocHost(&stage.data, yBytes * 3 / 2);
            if (e == cudaSuccess) e = cudaEventCreateWithFlags(&stage.done, cudaEventDisableTiming);
        }
        for (auto &frame : frames) {
            if (e == cudaSuccess) e = cudaMalloc(&frame.planes[0], yBytes * 3 / 2);
            if (e == cudaSuccess) {
                frame.planes[1] = frame.planes[0] + yBytes;
                frame.planes[2] = frame.planes[1] + yBytes / 4;
                e = cudaEventCreateWithFlags(&frame.ready, cudaEventDisableTiming);
            }
        }
        return check(e);
    }

    bool upload(size_t index, const std::vector<uint8_t> &raw) {
        auto &stage = staging[nextStage];
        nextStage = (nextStage + 1) % staging.size();
        if (!check(cudaEventSynchronize(stage.done))) return false;
        const size_t y = size_t(w) * h;
        pad_plane(raw.data(), w, h, w, aw, ah, stage.data);
        for (int p = 0; p < 2; ++p)
            pad_plane(raw.data() + y + p * (y / 4), w / 2, h / 2, w / 2, aw / 2, ah / 2,
                      stage.data + yBytes + p * (yBytes / 4));
        auto &frame = frames[index];
        cudaError_t e = cudaMemcpyAsync(frame.planes[0], stage.data, yBytes * 3 / 2, cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess) e = cudaEventRecord(stage.done, stream);
        if (e == cudaSuccess) e = cudaEventRecord(frame.ready, stream);
        return check(e);
    }

    bool wait(size_t index, cudaStream_t consumer) const {
        return check(cudaStreamWaitEvent(consumer, frames[index].ready, 0));
    }

    // Input slots may be overwritten once the compute stream has consumed the current mini-GOP.
    bool finishBatch(cudaStream_t consumer) {
        cudaError_t e = cudaEventRecord(consumed, consumer);
        if (e == cudaSuccess) e = cudaStreamWaitEvent(stream, consumed, 0);
        return check(e);
    }

    const std::array<uint8_t *, 3> &operator[](size_t index) const { return frames[index].planes; }

private:
    static bool check(cudaError_t e) {
        if (e != cudaSuccess) fprintf(stderr, "P-frame input transfer: %s\n", cudaGetErrorString(e));
        return e == cudaSuccess;
    }
    struct Input {
        std::array<uint8_t *, 3> planes{};
        cudaEvent_t ready = nullptr;
    };
    struct Stage {
        uint8_t *data = nullptr;
        cudaEvent_t done = nullptr;
    };
    int w, h, aw, ah;
    const size_t yBytes;
    std::vector<Input> frames;
    std::array<Stage, 2> staging{};
    size_t nextStage = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t consumed = nullptr;
};

int main(int argc, char **argv) {
    // CUDA_DEVICE_MAX_CONNECTIONS stays at the driver default (8): 32 hardware queues let the fifteen analysis streams
    // overlap a little more, but cost about 0.25 s of context creation and 0.23 s of teardown per process (1080p, 120
    // frames: 6.26 -> 5.65 s; 4K: no difference)
    if (argc < 6) {
        fprintf(stderr, "usage: ffmpeg ... -f rawvideo -pix_fmt yuv420p - | av1gpu-video WIDTH HEIGHT FPS_NUM FPS_DEN OUT.ivf [--fast] [--inter] [--inter-tx8 | --inter-tx-adaptive] [--inter-rdoq] [--persistent-cdf] [--diagonal-motion] [--motion-step 2..16 | --auto-motion-step] [--motion-block 8|16|32|64] [--rolling-reference] [--anchor-reference [--anchor-interval N]] [--lossless-inter-residual | --inter-residual-threshold N | --inter-qindex N] [--keyint 1..10000] [--scene-cut-rms 1..255] [--adaptive-fps] [--grain 0..32] [--qindex 1..255] [--pframes [--bitrate KBPS [--rc-buffer S]] [--pf-gop 1..32] [--pf-qoffset N] [--pf-altref-boost N] [--pf-leaf-offset N] [--pf-qindex N] [--pf-lambda X] [--pf-tiles C R] [--pf-cpu-entropy | --pf-gpu-entropy] [--pf-tf-frames 0..8 [--pf-tf-strength X]] [--pf-rdoq] [--pf-no-switchable] [--pf-no-mode-refine] [--pf-refine-passes 0..8] [--pf-square-prune M] [--pf-no-filters] [--kf-tiles MIN_MP,C,R | 0]]\n");
        return 2;
    }
    const int w = atoi(argv[1]), h = atoi(argv[2]), fpsNum = atoi(argv[3]), fpsDen = atoi(argv[4]);
    if (w < 1 || h < 1 || w > 65536 || h > 65536 || fpsNum < 1 || fpsDen < 1) {
        fprintf(stderr, "invalid dimensions or frame rate\n");
        return 2;
    }
    bool temporal = false, adaptiveFps = false;
    int grain=0;
    int modeCount = 5, motionStep = 2, motionBlockSize = 64;
    bool rolling = false, useResiduals = false, luma8 = false, adaptiveTx = false;
    bool autoMotionStep = false, persistentCdf = false;
    bool interRdoq = false;
    bool usePframes = false;
    pf::Settings pfSettings;
    int pfQindex = 0, pfQoffset = 90, pfGop = 32, pfAltrefBoost = 128, pfLeafOffset = 0;
    double targetKbps = 0, rcBufferSeconds = 2.0;
    int pfTileCols = -1, pfTileRows = -1;
    bool pfLr = true;     // luma Wiener loop restoration (--pf-no-lr disables it)
    int pfLrFrames = 0;
    double gopMotion = 0.0;      // motion of the current mini-GOP (mean block-mean difference of consecutive frames)
    auto adaptiveMargin = [](double motion) { const double t = std::min(1.0, std::max(0.0, (motion - 0.3) / 1.2)); return 2.0 + (0.5 - 2.0) * t; };
    const char *pfCutPresetName = "base";   // preset of scene-cut key frames (--pf-cut-preset)
    bool pfCutPreset = true;    // base-preset scene-cut key frames (--pf-slow-cuts keeps the main preset)
    bool pfTmvp = true;          // temporal motion vector candidates (--pf-no-tmvp)
    pfm::MotionField mfield;
    constexpr int kOrderHintBits = 7;   // --pframes streams: enable_order_hint with 7-bit hints
    uint64_t frameOrder = 0;            // display index of the inter frame being coded
    int slotHint[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool pfSkipMode = true;       // --pf-no-skip-mode
    bool pfCdefSb = true;         // per-superblock CDEF presets (--pf-no-cdef-sb)
    float pfSquarePrune = -1.f;  // --pf-square-prune M: fixed square-children pruning margin (default: motion-adaptive)
    bool pfAdaptive = true;     // content-adaptive ALTREF boost and key-frame offset (--pf-no-adaptive)
    // --pf-adaptive: mini-GOP length 16 / 8 from this block-mean motion of the next eight frames
    const double gopMotion16 = getenv("AV1GPU_PF_GOP16") ? atof(getenv("AV1GPU_PF_GOP16")) : 0.7;
    const double gopMotion8 = getenv("AV1GPU_PF_GOP8") ? atof(getenv("AV1GPU_PF_GOP8")) : 1.0;
    uint64_t pfCuts = 0;
    double pfSceneCut = 15.0;
    int pfCutQoffset = 30;      // scene-cut key frames: inter quantizer minus this (--pf-cut-qoffset)   // scene cut: mean luma block-mean difference (0 disables, --pf-scene-cut)
    int pfRefs = 2;   // references per inter frame: 2 (LAST + ALTREF) or 3 (+ the next anchor as ALTREF2, --pf-refs 3)
    int pfEntropyMode = 1;   // 0: CPU, 1: GPU modeling + host range coder, 2: all GPU
    int pfTfFrames = 8;   // ALTREF temporal filter over +-8 frames (noisy content only; --pf-tf-frames 0 off)
    float pfTfStrength = 1.0f;
    bool pfFilters = true, pfCdef = false;
    int residualThreshold = 0, interQindex = 0;
    bool useAnchor = false;
    unsigned anchorInterval = 0;
    unsigned keyInterval = 120;
    unsigned sceneCutRms = 0;
    GpConfig cfg;
    cfg.qindex = 100;
    bool kfPresetSet = false;
    float kfTileMp = 2.f;
    int kfTileC = 2, kfTileR = 2;
    for (int i = 6; i < argc; ++i) {
        if (!strcmp(argv[i], "--kf-tiles") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "0")) kfTileMp = 0.f;
            else if (sscanf(v, "%f,%d,%d", &kfTileMp, &kfTileC, &kfTileR) != 3 || kfTileMp <= 0 || kfTileC < 0 ||
                     kfTileC > 6 || kfTileR < 0 || kfTileR > 6) {
                fprintf(stderr, "--kf-tiles takes MIN_MP,C,R (log2 tile columns / rows 0..6) or 0\n");
                return 2;
            }
            continue;
        }
        if (!strcmp(argv[i], "--kf-preset") && i + 1 < argc) {
            if (!apply_preset(cfg, argv[++i])) { fprintf(stderr, "unknown --kf-preset\n"); return 2; }
            kfPresetSet = true;
        } else if (!strcmp(argv[i], "--fast")) {
            apply_preset(cfg, "turbo3");
            kfPresetSet = true;
        } else if (!strcmp(argv[i], "--grain") && i+1<argc) {
            char *end;
            const char *arg = argv[++i];
            long value = strtol(arg, &end, 10);
            if (end==arg || *end || value<0 || value>32) {
                fprintf(stderr,"grain strength must be 0..32\n"); return 2;
            }
            grain=static_cast<int>(value);
        } else if (!strcmp(argv[i], "--adaptive-fps")) {
            adaptiveFps = temporal = true;
        } else if (!strcmp(argv[i], "--inter")) {
            temporal = true;
        } else if (!strcmp(argv[i], "--inter-tx-adaptive")) {
            adaptiveTx = true;
        } else if (!strcmp(argv[i], "--inter-tx8")) {
            luma8 = true;
        } else if (!strcmp(argv[i], "--inter-rdoq")) {
            interRdoq = true;
        } else if (!strcmp(argv[i], "--inter-qindex") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 1 || value > 255) {
                fprintf(stderr, "inter qindex must be an integer from 1 to 255\n");
                return 2;
            }
            interQindex = static_cast<int>(value);
            useResiduals = temporal = true;
        } else if (!strcmp(argv[i], "--inter-residual-threshold") && i + 1 < argc) {
            char *end;
            const char *arg = argv[++i];
            long value = strtol(arg, &end, 10);
            if (end == arg || *end || value < 1 || value > 255) {
                fprintf(stderr, "inter residual threshold must be an integer from 1 to 255\n");
                return 2;
            }
            residualThreshold = static_cast<int>(value);
            useResiduals = temporal = true;
        } else if (!strcmp(argv[i], "--lossless-inter-residual")) {
            useResiduals = temporal = true;
        } else if (!strcmp(argv[i], "--anchor-interval") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 2 || value > 10000) {
                fprintf(stderr, "anchor interval must be an integer from 2 to 10000\n");
                return 2;
            }
            anchorInterval = static_cast<unsigned>(value);
            useAnchor = true;
        } else if (!strcmp(argv[i], "--anchor-reference")) {
            useAnchor = true;
        } else if (!strcmp(argv[i], "--rolling-reference")) {
            rolling = temporal = true;
        } else if (!strcmp(argv[i], "--diagonal-motion")) {
            modeCount = MotionModes;
        } else if (!strcmp(argv[i], "--auto-motion-step")) {
            autoMotionStep = true;
        } else if (!strcmp(argv[i], "--pframes")) {
            usePframes = temporal = true;
        } else if (!strcmp(argv[i], "--pf-qindex") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 1 || value > 255) {
                fprintf(stderr, "P-frame qindex must be an integer from 1 to 255\n");
                return 2;
            }
            pfQindex = static_cast<int>(value);
        } else if (!strcmp(argv[i], "--pf-qoffset") && i + 1 < argc) {
            pfQoffset = atoi(argv[++i]);
            if (pfQoffset < -100 || pfQoffset > 200) {
                fprintf(stderr, "P-frame qindex offset must be in -100..200\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--pf-gop") && i + 1 < argc) {
            pfGop = atoi(argv[++i]);
            if (pfGop < 1 || pfGop > 32) { fprintf(stderr, "--pf-gop must be 1..32\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-altref-boost") && i + 1 < argc) {
            pfAltrefBoost = atoi(argv[++i]);
            if (pfAltrefBoost < 0 || pfAltrefBoost > 200) { fprintf(stderr, "--pf-altref-boost must be 0..200\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-leaf-offset") && i + 1 < argc) {
            pfLeafOffset = atoi(argv[++i]);
            if (pfLeafOffset < 0 || pfLeafOffset > 100) { fprintf(stderr, "--pf-leaf-offset must be 0..100\n"); return 2; }
        } else if (!strcmp(argv[i], "--bitrate") && i + 1 < argc) {
            targetKbps = atof(argv[++i]);
            if (!(targetKbps >= 1 && targetKbps <= 1e6)) { fprintf(stderr, "--bitrate must be 1..1000000 kbit/s\n"); return 2; }
        } else if (!strcmp(argv[i], "--rc-buffer") && i + 1 < argc) {
            rcBufferSeconds = atof(argv[++i]);
            if (!(rcBufferSeconds >= 0.1 && rcBufferSeconds <= 60)) { fprintf(stderr, "--rc-buffer must be 0.1..60 seconds\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-tiles") && i + 2 < argc) {
            pfTileCols = atoi(argv[++i]);
            pfTileRows = atoi(argv[++i]);
            if (pfTileCols < 0 || pfTileCols > 6 || pfTileRows < 0 || pfTileRows > 6) {
                fprintf(stderr, "--pf-tiles takes log2 tile columns and rows (0..6)\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--pf-tf-frames") && i + 1 < argc) {
            pfTfFrames = atoi(argv[++i]);
            if (pfTfFrames < 0 || pfTfFrames > 8) { fprintf(stderr, "--pf-tf-frames must be 0..8\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-tf-strength") && i + 1 < argc) {
            pfTfStrength = strtof(argv[++i], nullptr);
            if (!(pfTfStrength > 0.f && pfTfStrength <= 16.f)) { fprintf(stderr, "--pf-tf-strength must be in (0, 16]\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-no-rect")) {
            pfSettings.rectPartitions = false;
        } else if (!strcmp(argv[i], "--pf-refs") && i + 1 < argc) {
            pfRefs = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pf-lr")) {
            pfLr = true;
        } else if (!strcmp(argv[i], "--pf-scene-cut") && i + 1 < argc) {
            pfSceneCut = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--pf-cut-qoffset") && i + 1 < argc) {
            pfCutQoffset = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--pf-no-adaptive")) {
            pfAdaptive = false;
        } else if (!strcmp(argv[i], "--pf-no-tmvp")) {
            pfTmvp = false;
        } else if (!strcmp(argv[i], "--pf-no-cdef-sb")) {
            pfCdefSb = false;
        } else if (!strcmp(argv[i], "--pf-no-skip-mode")) {
            pfSkipMode = false;
        } else if (!strcmp(argv[i], "--pf-no-8x8")) {
            pfSettings.blocks8 = false;
        } else if (!strcmp(argv[i], "--pf-no-rect4")) {
            pfSettings.rect4 = false;
        } else if (!strcmp(argv[i], "--pf-no-intra")) {
            pfSettings.intra = false;
        } else if (!strcmp(argv[i], "--pf-no-mode-refine")) {
            pfSettings.modeRefine = false;
        } else if (!strcmp(argv[i], "--pf-square-prune") && i + 1 < argc) {
            pfSquarePrune = strtof(argv[++i], nullptr);
            if (!(pfSquarePrune >= 0.f && pfSquarePrune <= 16.f)) { fprintf(stderr, "--pf-square-prune must be 0..16\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-refine-passes") && i + 1 < argc) {
            pfSettings.modeRefinePasses = atoi(argv[++i]);
            if (pfSettings.modeRefinePasses < 0 || pfSettings.modeRefinePasses > 8) {
                fprintf(stderr, "--pf-refine-passes must be 0..8\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--pf-slow-cuts")) {
            pfCutPreset = false;
        } else if (!strcmp(argv[i], "--pf-cut-preset") && i + 1 < argc) {
            pfCutPresetName = argv[++i];
        } else if (!strcmp(argv[i], "--pf-no-lr")) {
            pfLr = false;
        } else if (!strcmp(argv[i], "--pf-fast")) {
            pfSettings.splitTypes = 2;
        } else if (!strcmp(argv[i], "--pf-no-prune")) {
            pfSettings.pruneRect = false;
        } else if (!strcmp(argv[i], "--pf-switchable")) {
            pfSettings.switchableFilter = true;
        } else if (!strcmp(argv[i], "--pf-no-switchable")) {
            pfSettings.switchableFilter = false;
        } else if (!strcmp(argv[i], "--pf-no-tx-types")) {
            pfSettings.txTypes = 0;
        } else if (!strcmp(argv[i], "--pf-tx-types-reduced")) {
            pfSettings.txTypes = 1;
        } else if (!strcmp(argv[i], "--pf-tx-types-fast")) {
            pfSettings.txTypes = 2;
        } else if (!strcmp(argv[i], "--pf-no-tx-split")) {
            pfSettings.txSplit = false;
        } else if (!strcmp(argv[i], "--pf-rdoq")) {
            pfSettings.rdoq = true;
        } else if (!strcmp(argv[i], "--pf-gpu-entropy")) {
            pfEntropyMode = 2;
        } else if (!strcmp(argv[i], "--pf-cpu-entropy")) {
            pfEntropyMode = 0;
        } else if (!strcmp(argv[i], "--pf-no-filters")) {
            pfFilters = false;
        } else if (!strcmp(argv[i], "--pf-lambda") && i + 1 < argc) {
            pfSettings.lambdaMul = strtof(argv[++i], nullptr);
            if (!(pfSettings.lambdaMul > 0.f && pfSettings.lambdaMul < 100.f)) {
                fprintf(stderr, "P-frame lambda multiplier must be in (0, 100)\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--persistent-cdf")) {
            persistentCdf = true;
        } else if (!strcmp(argv[i], "--motion-block") && i + 1 < argc) {
            const char *arg = argv[++i];
            if (strcmp(arg, "8") && strcmp(arg, "16") && strcmp(arg, "32") && strcmp(arg, "64")) {
                fprintf(stderr, "motion block must be 8, 16, 32, or 64\n"); return 2;
            }
            motionBlockSize = atoi(arg);
        } else if (!strcmp(argv[i], "--motion-step") && i + 1 < argc) {
            char *end;
            const char *arg = argv[++i];
            long value = strtol(arg, &end, 10);
            if (end == arg || *end || value < 2 || value > 16 || value % 2) {
                fprintf(stderr, "motion step must be an even integer from 2 to 16\n");
                return 2;
            }
            motionStep = static_cast<int>(value);
        } else if (!strcmp(argv[i], "--scene-cut-rms") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 1 || value > 255) {
                fprintf(stderr, "scene-cut RMS must be an integer from 1 to 255\n");
                return 2;
            }
            sceneCutRms = static_cast<unsigned>(value);
        } else if (!strcmp(argv[i], "--keyint") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 1 || value > 10000) {
                fprintf(stderr, "keyint must be an integer from 1 to 10000\n");
                return 2;
            }
            keyInterval = static_cast<unsigned>(value);
        } else if (!strcmp(argv[i], "--qindex") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (*end || value < 1 || value > 255) {
                fprintf(stderr, "qindex must be an integer from 1 to 255\n");
                return 2;
            }
            cfg.qindex = static_cast<int>(value);
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            return 2;
        }
    }
    const int aw = (w + 63) & ~63, ah = (h + 63) & ~63;
    // The inter packet uses a single tile.
    const bool canPredict = temporal && !(w&1) && !(h&1) && w<=4096 &&
        size_t(aw)*ah<=4096*2304;
    if (rolling && (!canPredict || w%8 || h%8 || adaptiveFps)) {
        fprintf(stderr, "rolling reference requires inter-compatible dimensions divisible by eight and fixed frame timing\n");
        return 2;
    }

    if (useResiduals && (!canPredict || w%8 || h%8 || adaptiveFps)) {
        fprintf(stderr, "inter residuals require inter dimensions divisible by eight and fixed frame timing\n");
        return 2;
    }
    if (interQindex && residualThreshold) {
        fprintf(stderr, "choose either DCT quantization or WHT thresholding\n");
        return 2;
    }
    if (luma8 && !interQindex) {
        fprintf(stderr, "--inter-tx8 requires --inter-qindex\n");
        return 2;
    }
    if (adaptiveTx && (!interQindex || !rolling || luma8)) {
        fprintf(stderr, "--inter-tx-adaptive requires --inter-qindex and --rolling-reference, and excludes --inter-tx8\n");
        return 2;
    }
    if (useAnchor && (!rolling || modeCount != 5)) {
        fprintf(stderr, "--anchor-reference requires rolling reference and cardinal motion\n");
        return 2;
    }
    if (useAnchor) modeCount = AnchorMode + 1;
    if (interRdoq && !interQindex) {
        fprintf(stderr, "--inter-rdoq requires --inter-qindex\n");
        return 2;
    }
    if (motionBlockSize != 64 && (!interQindex || !rolling)) {
        fprintf(stderr, "smaller motion blocks require --inter-qindex and --rolling-reference\n");
        return 2;
    }
    if (persistentCdf && !usePframes && (!interQindex || !rolling || adaptiveFps)) {
        fprintf(stderr, "--persistent-cdf requires quantized inter frames and a rolling reference\n");
        return 2;
    }
    if (targetKbps > 0 && !usePframes) {
        fprintf(stderr, "--bitrate requires --pframes\n");
        return 2;
    }
    if (usePframes && (w % 8 || h % 8 || w < 16 || h < 16 || adaptiveFps || useResiduals || rolling || useAnchor)) {
        fprintf(stderr, "--pframes requires dimensions divisible by eight, fixed frame timing, and no other inter mode\n");
        return 2;
    }
    if (autoMotionStep && (!canPredict || adaptiveFps)) {
        fprintf(stderr, "--auto-motion-step requires inter encoding with fixed frame timing\n");
        return 2;
    }
    FILE *outFile = fopen(argv[5], "wb+");
    if (!outFile) { perror(argv[5]); return 2; }
    if (!write_ivf_header(outFile, w, h, fpsNum, fpsDen, 0)) { fprintf(stderr, "cannot write IVF header\n"); return 2; }

    const int acw = aw / 2, ach = ah / 2, cw = (w + 1) / 2, ch = (h + 1) / 2;
    uint8_t *d[3] = {nullptr, nullptr, nullptr};
    const size_t sizes[3] = {(size_t)aw * ah, (size_t)acw * ach, (size_t)acw * ach};
    const int batchSize = std::max<size_t>(1, std::min<size_t>(8, 16000000 / sizes[0]));
    for (int p = 0; p < 3; ++p) {
        cudaError_t e = cudaMalloc(&d[p], sizes[p] * (temporal ? 1 : batchSize));
        if (e != cudaSuccess) { fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(e)); return 1; }
    }

    const size_t ySize = size_t(w)*h;
    uint8_t *reference = nullptr;
    uint8_t *anchor = nullptr;
    uint8_t *nextReference = nullptr;
    MotionBlock *deviceBlocks = nullptr;
    std::vector<MotionBlock> blocks(canPredict ? size_t((w+motionBlockSize-1)/motionBlockSize)*((h+motionBlockSize-1)/motionBlockSize) : 0);
    if (canPredict && (cudaMalloc(&reference, ySize*3/2)!=cudaSuccess ||
                       cudaMalloc(&deviceBlocks, blocks.size()*sizeof(MotionBlock))!=cudaSuccess)) {
        fprintf(stderr,"cannot allocate CUDA reference buffers\n"); return 1;
    }
    if (rolling && cudaMalloc(&nextReference, ySize*3/2) != cudaSuccess) {
        fprintf(stderr, "cannot allocate rolling reference\n");
        return 1;
    }
    FILE *reconDump = getenv("AV1GPU_DUMP_RECON") ? fopen(getenv("AV1GPU_DUMP_RECON"), "wb") : nullptr;
    pf::Analyzer pfAnalyzer;
    double pfBits[4] = {};
    double pfMs = 0, pfFilterMs = 0, pfWriteMs = 0, kfEncodeMs = 0, kfEntropyMs = 0;
    int kfCount = 0;
    uint64_t rcPeriods = 0;
    uint64_t pfModes[4] = {}, pfCompound = 0;
    if (usePframes) {
        if (!reference && cudaMalloc(&reference, ySize*3/2) != cudaSuccess) { fprintf(stderr, "reference allocation failed\n"); return 1; }
        if (!nextReference && cudaMalloc(&nextReference, ySize*3/2) != cudaSuccess) { fprintf(stderr, "reference allocation failed\n"); return 1; }
    }
    if (useAnchor && cudaMalloc(&anchor, ySize*3/2) != cudaSuccess) {
        fprintf(stderr, "anchor reference allocation failed\n");
        return 1;
    }
    int16_t *deviceLevels = nullptr;
    std::vector<int16_t> residualLevels;
    if (useResiduals) {
        residualLevels.resize(ySize*3/2);
        auto status = cudaMalloc(&deviceLevels, residualLevels.size()*sizeof(int16_t));
        if (status != cudaSuccess) {
            fprintf(stderr, "residual allocation failed: %s\n", cudaGetErrorString(status));
            return 1;
        }
    }
    int16_t *alternateLevels = nullptr;
    uint8_t *alternateRecon = nullptr, *deviceTxMap = nullptr;
    std::vector<uint8_t> txMap;
    if (adaptiveTx) {
        txMap.resize(ySize/64);
        if (cudaMalloc(&alternateLevels, ySize*sizeof(int16_t)) != cudaSuccess ||
            cudaMalloc(&alternateRecon, ySize) != cudaSuccess ||
            cudaMalloc(&deviceTxMap, txMap.size()) != cudaSuccess) {
            fprintf(stderr, "adaptive transform allocation failed\n");
            return 1;
        }
    }
    uint64_t largerTransforms = 0;
    unsigned long long *deviceMotionScores = nullptr;
    uint64_t motionChoices[MotionStepCount]{};
    if (autoMotionStep && cudaMalloc(&deviceMotionScores, sizeof(*deviceMotionScores)*MotionStepCount) != cudaSuccess) {
        fprintf(stderr, "motion prescan allocation failed\n");
        return 1;
    }
    uint64_t lastAnchor = 0, sceneCuts = 0;
    uint64_t lastKey = 0, predicted = 0, held = 0, lastWritten = 0;
    std::vector<unsigned> previousModes(blocks.size());
    inter::EntropyState interEntropy;
    std::vector<uint8_t> heldPacket;
    cfg.wantRecon = false;
    GpEncoder encoder;
    std::string err;
    if (usePframes && !kfPresetSet) apply_preset(cfg, "slow");   // key frames carry much of the quality (--kf-preset)
    if (usePframes && kfTileMp > 0) {
        // key frames of at least kfTileMp megapixels use 2^C x 2^R tiles: independent intra wavefronts and entropy
        // coders (one tile reconstructs at most one superblock diagonal at a time)
        cfg.tileMinMp = kfTileMp; cfg.tileColsLog2 = kfTileC; cfg.tileRowsLog2 = kfTileR;
    }
    if (usePframes && !pfFilters) cfg.fCdef = 0;   // enable_cdef follows the inter frames (they code no CDEF params)
    if (usePframes && pfFilters) {
        // keyframes: deblocking and CDEF searches; enable_cdef = 1 in the sequence header lets P frames use CDEF
        cfg.fLf = 2;
        cfg.fCdef = 2;
        pfCdef = true;
    }
    if (encoder.init(cfg, err)) { fprintf(stderr, "encoder init: %s\n", err.c_str()); return 1; }
    // scene-cut key frames: base-preset speed knobs (they are frequent on cut-heavy content and short-lived), with the
    // sequence-level tools of the main key-frame encoder so the sequence header stays valid
    GpEncoder cutEncoder;
    bool haveCut = false;
    if (usePframes && pfCutPreset) {
        GpConfig cc = cfg;
        if (!apply_preset(cc, pfCutPresetName)) { fprintf(stderr, "unknown --pf-cut-preset\n"); return 2; }
        cc.P.filterIntra = cfg.P.filterIntra;
        cc.fLf = cfg.fLf;
        cc.fCdef = cfg.fCdef;
        if (cutEncoder.init(cc, err)) { fprintf(stderr, "encoder init: %s\n", err.c_str()); return 1; }
        haveCut = true;
    }
    if (usePframes) {
        char perr[256];
        if (pfAnalyzer.init(w, h, perr, sizeof perr)) { fprintf(stderr, "%s\n", perr); return 1; }
        // The non-blocking compute stream shares tables initialized on the default stream.
        const auto e = cudaStreamSynchronize(nullptr);
        if (e != cudaSuccess) { fprintf(stderr, "P-frame initialization: %s\n", cudaGetErrorString(e)); return 1; }
        mfield.init(2 * ((w + 7) >> 3), 2 * ((h + 7) >> 3), kOrderHintBits);
        const int sbCols = (w + 63) / 64, sbRows = (h + 63) / 64;
        // one tile compresses best; the tile coder runs off the critical path, overlapped with later analysis
        (void)sbCols; (void)sbRows;
        if (pfTileCols < 0) pfTileCols = 0;
        if (pfTileRows < 0) pfTileRows = 0;
        pfSettings.qindex = pfQindex ? pfQindex : std::min(255, std::max(1, cfg.qindex + pfQoffset));
    }

    std::vector<uint8_t> src, padded[3];
    uint64_t frameCount = 0;
    double gpuMs = 0;
    const auto wallStart = std::chrono::steady_clock::now();
    GpEntScratch scratch;
    // ---- CUDA P/ALTREF frame path ----
    constexpr int kTfKeyMax = 16;
    auto runPframes = [&]() -> int {
        const int G = std::max(1, pfGop);
        const size_t frameBytes = ySize * 3 / 2;
        uint8_t *slotBuf[8] = {}, *spare = nullptr;
        for (int k = 0; k < 8; ++k)
            if (cudaMalloc(&slotBuf[k], frameBytes) != cudaSuccess) { fprintf(stderr, "slot allocation failed\n"); return 1; }
        if (cudaMalloc(&spare, frameBytes) != cudaSuccess) { fprintf(stderr, "slot allocation failed\n"); return 1; }
        const cudaStream_t compute = pfAnalyzer.stream();
        // slots: a mini-GOP plus temporal-filter neighbours, or a key frame and its filter neighbours
        FrameUploads inBuf(w, h, aw, ah, std::max<size_t>(size_t(G + pfTfFrames + 1), size_t(kTfKeyMax + 1)));
        if (!inBuf.init()) return 1;
        bool slotValid[8] = {};
        std::array<uint8_t *, 3> tfBuf{};
        for (int p = 0; p < 3; ++p)
            if (cudaMalloc(&tfBuf[p], sizes[p]) != cudaSuccess) { fprintf(stderr, "filter buffer allocation failed\n"); return 1; }
        std::vector<std::vector<uint8_t>> pending;   // raw frames read ahead
        bool eof = false;
        // the reader thread keeps up to a mini-GOP of frames beyond the pending ones, with their 16x16 block sums
        // (looked up by buffer address: a buffer keeps its address while it moves between pending and prevFrame)
        FrameReader reader(w, h, size_t(G) + 1);
        std::map<const uint8_t *, std::vector<int>> frameSums;
        std::vector<uint8_t> prevFrame;   // the last frame before pending (scene cuts and motion)
        auto readMore = [&](size_t want) {
            while (!eof && pending.size() < want) {
                FrameReader::Frame f;
                if (!reader.next(f)) { eof = true; break; }
                if (frameSums.size() > 4 * (size_t(G) + kTfKeyMax + 8)) {   // drop the sums of frames gone
                    std::map<const uint8_t *, std::vector<int>> live;
                    for (auto &q : pending) {
                        auto it = frameSums.find(q.data());
                        if (it != frameSums.end()) live.insert(*it);
                    }
                    auto it = frameSums.find(prevFrame.data());
                    if (it != frameSums.end()) live.insert(*it);
                    frameSums.swap(live);
                }
                frameSums[f.data.data()] = std::move(f.sums);
                pending.push_back(std::move(f.data));
            }
        };
        auto dump = [&](const uint8_t *dev) -> int {
            if (!reconDump) return 0;
            std::vector<uint8_t> host(frameBytes);
            cudaError_t e = cudaStreamSynchronize(compute);
            if (e == cudaSuccess) e = cudaMemcpy(host.data(), dev, frameBytes, cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) {
                fprintf(stderr, "P-frame reconstruction download: %s\n", cudaGetErrorString(e));
                return 1;
            }
            if (fwrite(host.data(), 1, host.size(), reconDump) != host.size()) {
                fprintf(stderr, "reconstruction write failed\n");
                return 1;
            }
            return 0;
        };
        const int pitches[3] = {aw, acw, acw};
        char perr[256];
        // ---- writer thread: CPU tile entropy coding, temporal units and IVF output, overlapped with the GPU ----
        struct Job {
            enum Kind { Coded, Tiles, Records, Raw, ShowExisting, Stop } kind = Coded;
            pf::EntropyIn ein;
            pfw::FrameParams fp;
            int buf = -1, slot0 = 0, outSlot = 8, showSlot = 0;
            std::vector<uint8_t> raw;
            std::vector<pf::Mv> mf;   // projected motion field (kept with the job for the CPU entropy path)
            std::vector<int8_t> cdefIdx;   // per-superblock CDEF presets (CPU entropy path)
            pf::EntropyOut tiles;
            bool endTU = false, resetSlots = false;
            uint64_t pts = 0;
        };
        const int kBuffers = 3;
        struct CpuEntropyBuffer {
            pf::BlockInfo *grid = nullptr;
            int16_t *levels = nullptr;
            cudaEvent_t ready = nullptr;
            ~CpuEntropyBuffer() {
                if (ready) cudaEventDestroy(ready);
                if (grid) cudaFreeHost(grid);
                if (levels) cudaFreeHost(levels);
            }
        };
        std::array<CpuEntropyBuffer, kBuffers> cpuBuf;
        std::vector<int> freeBufs;
        if (pfEntropyMode == 0) {
            for (int k = 0; k < kBuffers; ++k) {
                auto &buf = cpuBuf[k];
                cudaError_t e = cudaMallocHost(&buf.grid, pfAnalyzer.gridCount() * sizeof(pf::BlockInfo));
                if (e == cudaSuccess) e = cudaMallocHost(&buf.levels, pfAnalyzer.levelCount() * sizeof(int16_t));
                if (e == cudaSuccess) e = cudaEventCreateWithFlags(&buf.ready, cudaEventDisableTiming);
                if (e != cudaSuccess) {
                    fprintf(stderr, "CPU entropy buffer allocation: %s\n", cudaGetErrorString(e));
                    return 1;
                }
                freeBufs.push_back(k);
            }
        }
        pf::HostEntropy hostEntropy;
        if (hostEntropy.init(w, h, perr, sizeof perr)) { fprintf(stderr, "%s\n", perr); return 1; }
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<int> freeEntropy;
        for (int k = 0; k < pf::Analyzer::kEntropySlots; ++k) freeEntropy.push_back(k);
        std::deque<Job> jobs;
        bool writerBusy = false, writerFailed = false;
        std::string writerError;
        auto writer = [&]() {
            bool hostSlotValid[8] = {};
            std::vector<uint8_t> tu;
            char werr[256];
            for (;;) {
                Job job;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return !jobs.empty(); });
                    job = std::move(jobs.front());
                    jobs.pop_front();
                    writerBusy = true;
                }
                if (job.kind == Job::Stop) break;
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<uint8_t> obu;
                bool ok = true;
                if (job.kind == Job::Raw) {
                    obu = std::move(job.raw);
                    if (job.resetSlots)
                        for (auto &v : hostSlotValid) v = false;
                } else if (job.kind == Job::ShowExisting) {
                    obu = pfw::showExistingObu(job.showSlot, tu.empty());
                } else {
                    pf::EntropyOut eout;
                    if (job.kind == Job::Records) {
                        job.ein.primarySlot = persistentCdf && hostSlotValid[job.slot0] ? job.slot0 : -1;
                        job.ein.saveSlot = persistentCdf && job.outSlot < 8 ? job.outSlot : -1;
                        std::vector<uint32_t> counts;
                        std::vector<uint8_t> recs;
                        ok = !pfAnalyzer.entropyFetchRecords(job.buf, counts, recs, werr, sizeof werr);
                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            freeEntropy.push_back(job.buf);
                        }
                        cv.notify_all();
                        if (ok) ok = !hostEntropy.codeRecords(job.ein, counts, recs, eout, werr, sizeof werr);
                        if (job.outSlot < 8) hostSlotValid[job.outSlot] = persistentCdf;
                    } else if (job.kind == Job::Coded) {
                        job.ein.primarySlot = persistentCdf && hostSlotValid[job.slot0] ? job.slot0 : -1;
                        job.ein.saveSlot = persistentCdf && job.outSlot < 8 ? job.outSlot : -1;
                        auto &buf = cpuBuf[job.buf];
                        const cudaError_t e = cudaEventSynchronize(buf.ready);
                        if (e != cudaSuccess) snprintf(werr, sizeof werr, "CPU entropy download: %s", cudaGetErrorString(e));
                        ok = e == cudaSuccess && !hostEntropy.code(job.ein, buf.grid, buf.levels, eout, werr, sizeof werr);
                        if (ok && getenv("AV1GPU_PF_DUMP_GRID")) {   // debug: decision grid of each coded frame
                            static FILE *gf = fopen(getenv("AV1GPU_PF_DUMP_GRID"), "wb");
                            if (gf) fwrite(buf.grid, sizeof(pf::BlockInfo), pfAnalyzer.gridCount(), gf);
                        }
                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            freeBufs.push_back(job.buf);
                        }
                        cv.notify_all();
                        if (job.outSlot < 8) hostSlotValid[job.outSlot] = persistentCdf;
                    } else {
                        ok = !pfAnalyzer.entropyFetch(job.buf, eout, werr, sizeof werr);
                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            freeEntropy.push_back(job.buf);
                        }
                        cv.notify_all();
                    }
                    if (ok) {
                        job.fp.primary = job.ein.primarySlot >= 0;
                        job.fp.temporalDelimiter = tu.empty();
                        try {
                            obu = pfw::frameObu(job.fp, 2 * ((w + 7) >> 3), 2 * ((h + 7) >> 3), eout.tiles);
                        } catch (const std::exception &ex) {
                            snprintf(werr, sizeof werr, "P-frame syntax: %s", ex.what());
                            ok = false;
                        }
                        for (int k = 0; k < 4; ++k) { pfModes[k] += eout.modes[k]; pfBits[k] += eout.bits[k]; }
                        pfCompound += eout.compound;
                    }
                }
                if (ok) {
                    tu.insert(tu.end(), obu.begin(), obu.end());
                    if (job.endTU) {
                        ok = write_ivf_frame(outFile, tu, job.pts);
                        if (!ok) snprintf(werr, sizeof werr, "IVF write failed");
                        tu.clear();
                    }
                }
                pfWriteMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::lock_guard<std::mutex> lock(mtx);
                writerBusy = false;
                if (!ok && !writerFailed) { writerFailed = true; writerError = werr; }
                cv.notify_all();
            }
        };
        std::thread writerThread(writer);
        auto submit = [&](Job &&job) -> int {
            std::lock_guard<std::mutex> lock(mtx);
            if (writerFailed) { fprintf(stderr, "%s\n", writerError.c_str()); return 1; }
            jobs.push_back(std::move(job));
            cv.notify_all();
            return 0;
        };
        auto drain = [&]() -> int {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [&] { return (jobs.empty() && !writerBusy) || writerFailed; });
            if (writerFailed) { fprintf(stderr, "%s\n", writerError.c_str()); return 1; }
            return 0;
        };
        auto stopWriter = [&]() {
            Job stop;
            stop.kind = Job::Stop;
            {
                std::lock_guard<std::mutex> lock(mtx);
                jobs.push_back(std::move(stop));
            }
            cv.notify_all();
            writerThread.join();
        };
        struct WriterGuard {
            std::function<void()> f;
            ~WriterGuard() { f(); }
        } writerGuard{[&] {
            if (writerThread.joinable()) stopWriter();
            cudaStreamSynchronize(compute);
        }};
        std::vector<size_t> uploadOrder;
        size_t nextUpload = 0;
        const bool tmvp = persistentCdf && pfTmvp;
        constexpr int refNames[3] = {pfw::LAST_FRAME, pfw::ALTREF_FRAME, pfw::ALTREF2_FRAME};
        struct PendingMotion {
            int slot = -1, hint = 0;
            std::array<int, 8> refHints{};
        } pendingMotion;
        // The reconstruction goes to outSlot. The original input of a temporally filtered ALTREF is
        // kept in its input slot for the overlay.
        auto interFrame = [&](size_t input, int slot0, int slot1, int outSlot, int qindex, bool show,
                              bool endTU, uint64_t pts, int slot2 = -1, bool filteredSource = false) -> int {
            if (!filteredSource && !inBuf.wait(input, compute)) return 1;
            const auto &fb = filteredSource ? tfBuf : inBuf[input];
            // slot2: a third single reference (ALTREF2), used only with slot1 (--pf-refs 3)
            if (pfRefs < 3 || slot1 < 0) slot2 = -1;
            const uint8_t *refs[3] = {slotBuf[slot0], slot1 >= 0 ? slotBuf[slot1] : nullptr, slot2 >= 0 ? slotBuf[slot2] : nullptr};
            const int refCount = slot2 >= 0 ? 3 : slot1 >= 0 ? 2 : 1;
            const int orderHint = int(frameOrder & ((1u << kOrderHintBits) - 1));
            int refSlot[7];
            std::fill_n(refSlot, 7, slot0);
            if (slot1 >= 0) refSlot[pfw::ALTREF_FRAME - 1] = slot1;
            if (slot2 >= 0) refSlot[pfw::ALTREF2_FRAME - 1] = slot2;
            std::array<int, 8> refHints{};
            for (int r = 0; r < 7; ++r) refHints[r + 1] = slotHint[refSlot[r]];
            pf::Analyzer::ProjectMotion projectMotion;
            if (tmvp) {
                projectMotion = [&]() -> const pf::Mv * {
                    if (pendingMotion.slot >= 0) {
                        const auto *grid = pfAnalyzer.motionGrid(perr, sizeof perr);
                        if (!grid) return nullptr;
                        mfield.store(pendingMotion.slot, grid, pfAnalyzer.gridCols(), refNames,
                                     pendingMotion.refHints.data(), pendingMotion.hint);
                        pendingMotion.slot = -1;
                    }
                    return mfield.estimate(refSlot, refHints.data(), orderHint);
                };
            }
            pf::Settings st = pfSettings;
            st.qindex = qindex;
            // moving content codes most blocks: search rectangular partitions only where the square clearly wins
            if (pfAdaptive) st.pruneMargin = (float)adaptiveMargin(gopMotion);
            // square-children pruning: moving content only (static content refines detail in small blocks over frames)
            st.squarePrune = pfSquarePrune >= 0.f ? pfSquarePrune : pfAdaptive && gopMotion >= 0.6 ? 1.f : 0.f;
            // st.splitPrune (moving content, whole transform all zero): +0.3 points for 2-3% speed, left off
            // per-block mode cost of the evaluation: moving content pays more side information per block (larger
            // values there stop splits that win the estimate but lose in BD-rate); 2 bits for static content
            if (pfAdaptive) {
                static const double moving = getenv("AV1GPU_PF_MODEBITS_MOVING") ? atof(getenv("AV1GPU_PF_MODEBITS_MOVING")) : 12.0;
                static const double hi = getenv("AV1GPU_PF_MODEBITS_HI") ? atof(getenv("AV1GPU_PF_MODEBITS_HI")) : 1.5;
                const double t = std::min(1.0, std::max(0.0, (gopMotion - 0.3) / (hi - 0.3)));
                st.modeBits = (float)(2.0 + (moving - 2.0) * t);
            }
            const uint8_t *src[3] = {fb[0], fb[1], fb[2]};
            Job job;
            job.slot0 = slot0;
            job.outSlot = outSlot;
            job.endTU = endTU;
            job.pts = pts;
            pf::EntropyIn &ein = job.ein;
            std::copy_n(refNames, 3, ein.refName);
            ein.referenceSelect = refCount >= 2;
            ein.txModeSelect = pfSettings.txSplit;
            ein.reducedTxSet = pfSettings.txTypes < 2;
            ein.switchableFilter = pfSettings.switchableFilter;
            ein.filterIntra = cfg.P.filterIntra != 0;
            ein.qindex = qindex;
            ein.tileColsLog2 = pfTileCols;
            ein.tileRowsLog2 = pfTileRows;
            pfw::FrameParams &fp = job.fp;
            fp.qindex = qindex;
            std::copy_n(refSlot, 7, fp.refSlot);
            fp.referenceSelect = refCount >= 2;
            fp.txModeSelect = pfSettings.txSplit;
            fp.reducedTxSet = pfSettings.txTypes < 2;
            fp.switchable = pfSettings.switchableFilter;
            fp.refreshFlags = outSlot < 8 ? 1 << outSlot : 0;
            fp.showFrame = show;
            fp.persistent = persistentCdf;
            fp.orderHintBits = kOrderHintBits;
            fp.orderHint = orderHint;
            for (int k = 0; k < 8; ++k) fp.slotHint[k] = slotHint[k];
            for (int r = 1; r < 8; ++r)
                ein.signBias[r] = pfw::relDist(refHints[r], orderHint, kOrderHintBits) > 0;
            // use_ref_frame_mvs needs error_resilient_mode = 0.
            ein.useTmvp = fp.useRefFrameMvs = tmvp;
            // skip_mode: frames between a past and a future reference whose SkipModeFrame pair is our compound pair
            {
                int f0 = 0, f1 = 0;
                fp.skipModePresent = pfSkipMode && pfw::skipModeFrames(fp, f0, f1) && f0 == pfw::LAST_FRAME &&
                                     f1 == pfw::ALTREF_FRAME;
                ein.skipMode = fp.skipModePresent;
            }
            if (pfAnalyzer.analyze(src, pitches, refs, refCount, st, ein, spare, projectMotion, perr, sizeof perr)) {
                fprintf(stderr, "%s\n", perr);
                return 1;
            }
            pfMs += pfAnalyzer.lastMs();
            if (outSlot < 8 && tmvp) {
                if (pfAnalyzer.stageMotionGrid(perr, sizeof perr)) { fprintf(stderr, "%s\n", perr); return 1; }
                pendingMotion = {outSlot, orderHint, refHints};
            }
            if (nextUpload < uploadOrder.size()) {
                const size_t next = uploadOrder[nextUpload++];
                if (!inBuf.upload(next, pending[next])) return 1;
            }
            const auto tf0 = std::chrono::steady_clock::now();
            pf::FilterParams flt;
            flt.lrAllowed = pfLr;
            flt.cdefPerSb = pfCdefSb;
            if (pfFilters && pfAnalyzer.filter(src, pitches, spare, qindex, pfCdef, flt, perr, sizeof perr)) {
                fprintf(stderr, "%s\n", perr);
                return 1;
            }
            pfFilterMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tf0).count();
            ein.lrLuma = flt.lrLuma;
            fp.lrType[0] = flt.lrLuma ? 2 : 0;   // lr_type 2: RESTORE_WIENER
            pfLrFrames += flt.lrLuma;
            fp.grain = grain;
            fp.grainSeed = uint16_t(7391 + frameCount * 31847);
            for (int k = 0; k < 4; ++k) fp.lfLevel[k] = flt.lf[k];
            fp.lfSharpness = flt.sharpness;
            fp.cdef = pfCdef;
            fp.cdefDamping = flt.damping;
            fp.cdefY = flt.yPri; fp.cdefYSec = flt.ySec; fp.cdefUV = flt.uvPri; fp.cdefUVSec = flt.uvSec;
            fp.cdefBits = pfCdef ? flt.cdefBits : 0;
            std::copy(&flt.cdefPreset[0][0], &flt.cdefPreset[0][0] + 32, &fp.cdefPreset[0][0]);
            ein.cdefBits = fp.cdefBits;
            job.cdefIdx = std::move(flt.cdefIdx);
            ein.cdefIdx = job.cdefIdx.empty() ? nullptr : job.cdefIdx.data();
            fp.tileColsLog2 = pfTileCols;
            fp.tileRowsLog2 = pfTileRows;
            fp.contextTile = 0;
            // The CPU writer needs its own copy after analyze has projected the current field.
            if (tmvp && pfEntropyMode == 0) {
                job.mf = mfield.out;
                ein.motionField = job.mf.data();
            }
            if (pfEntropyMode == 1) {
                // GPU symbol modeling, host range coding (the writer decides the reference-slot CDFs)
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return !freeEntropy.empty() || writerFailed; });
                    if (writerFailed) { fprintf(stderr, "%s\n", writerError.c_str()); return 1; }
                    job.buf = freeEntropy.back();
                    freeEntropy.pop_back();
                }
                ein.recordsOnly = true;
                if (pfAnalyzer.entropyLaunch(ein, job.buf, perr, sizeof perr)) {
                    fprintf(stderr, "%s\n", perr);
                    return 1;
                }
                job.kind = Job::Records;
            } else if (pfEntropyMode == 2) {
                // GPU tile coding: reference-slot CDFs live on the device and are tracked here
                ein.primarySlot = persistentCdf && slotValid[slot0] ? slot0 : -1;
                ein.saveSlot = persistentCdf && outSlot < 8 ? outSlot : -1;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return !freeEntropy.empty() || writerFailed; });
                    if (writerFailed) { fprintf(stderr, "%s\n", writerError.c_str()); return 1; }
                    job.buf = freeEntropy.back();
                    freeEntropy.pop_back();
                }
                if (pfAnalyzer.entropyLaunch(ein, job.buf, perr, sizeof perr)) {
                    fprintf(stderr, "%s\n", perr);
                    return 1;
                }
                job.kind = Job::Tiles;
            } else {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [&] { return !freeBufs.empty() || writerFailed; });
                    if (writerFailed) { fprintf(stderr, "%s\n", writerError.c_str()); return 1; }
                    job.buf = freeBufs.back();
                    freeBufs.pop_back();
                }
                auto &buf = cpuBuf[job.buf];
                if (pfAnalyzer.download(buf.grid, buf.levels, buf.ready, perr, sizeof perr)) {
                    fprintf(stderr, "%s\n", perr);
                    return 1;
                }
                job.kind = Job::Coded;
            }
            if (submit(std::move(job))) return 1;
            if (outSlot < 8) {
                slotValid[outSlot] = persistentCdf;
                slotHint[outSlot] = orderHint;
                std::swap(slotBuf[outSlot], spare);
            }
            return 0;
        };
        int lastShown = 0;
        uint64_t lastKeyIndex = 0;
        // ---- rate control: one base quantizer per keyframe or mini-GOP period ----
        // Keyframes use baseQ, P frames baseQ + offset with the pyramid boosts. After each period the base moves by
        // ln(actual / desired) / slope, where desired corrects the accumulated error over the buffer duration.
        const double fps = double(fpsNum) / fpsDen;
        const double frameBits = targetKbps * 1000.0 / fps;
        // baseQ: the inter-frame (P) quantizer; key frames use baseQ - pfQoffset
        double baseQ = cfg.qindex, slope = 0.0146, bufferErr = 0;
        if (targetKbps > 0) {
            const double bpp = frameBits / (double(w) * h);
            baseQ = std::min(255.0, std::max(20.0, 180.0 + std::log(0.0077 / bpp) / slope));
        }
        long periodStart = ftell(outFile);
        uint64_t periodFrames = 0;
        double lastQ = baseQ, lastBpf = 0;
        auto applyQ = [&]() -> int {
            if (targetKbps <= 0) return 0;
            const int pq = std::max(1, std::min(255, (int)std::lround(baseQ)));
            const int kq = std::max(1, std::min(255, pq - pfQoffset));
            if (encoder.setQindex(kq, err)) { fprintf(stderr, "rate control: %s\n", err.c_str()); return 1; }
            pfSettings.qindex = pq;
            return 0;
        };
        auto endPeriod = [&](uint64_t frames) {
            if (targetKbps <= 0 || !frames) return;
            if (drain()) return;
            const long now = ftell(outFile);
            const double bits = double(now - periodStart) * 8.0;
            periodStart = now;
            bufferErr += bits - frameBits * frames;
            const double bpf = bits / frames;
            // refine the bits-vs-q slope from consecutive periods at different quantizers
            if (lastBpf > 0 && std::fabs(baseQ - lastQ) >= 4) {
                const double k = std::log(lastBpf / bpf) / (baseQ - lastQ);
                if (k > 0.004 && k < 0.05) slope = 0.7 * slope + 0.3 * k;
            }
            lastQ = baseQ;
            lastBpf = bpf;
            const double buffer = frameBits * fps * rcBufferSeconds;
            const double desired = frameBits * std::min(1.6, std::max(0.4, 1.0 - bufferErr / buffer));
            const double step = std::log(bpf / desired) / slope;
            baseQ = std::min(255.0, std::max(1.0, baseQ + std::max(-32.0, std::min(32.0, step))));
            rcPeriods++;
        };
        (void)periodFrames;
        bool keyPending = false;
        uint64_t keyPeriodBegin = 0;
        // scene cuts: mean absolute difference of 16x16 luma block means between consecutive frames
        auto sumsOf = [&](const std::vector<uint8_t> &f) -> const std::vector<int> & {
            auto it = frameSums.find(f.data());
            if (it == frameSums.end()) it = frameSums.emplace(f.data(), block_sums16(f, w, h)).first;
            return it->second;
        };
        // memoized by buffer address pair: valid while the frames stay pending (cleared whenever frames leave)
        std::map<std::pair<const uint8_t *, const uint8_t *>, double> diffCache;
        auto blockMeanDiff = [&](const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
            const auto key = std::make_pair(a.data(), b.data());
            const auto hit = diffCache.find(key);
            if (hit != diffCache.end()) return hit->second;
            const std::vector<int> &A = sumsOf(a), &B = sumsOf(b);
            double sum = 0;
            const int nb = (int)A.size();
            for (int i = 0; i < nb; ++i) sum += std::abs(A[i] - B[i]) / 64.0;
            const double v = nb ? sum / nb : 0.0;
            diffCache[key] = v;
            return v;
        };
        // --pf-adaptive: interpolate a setting between its static-content value (motion <= 0.3) and its moving-content
        // value (motion >= 1.5); motion is the mean blockMeanDiff of consecutive frames (independent of GOP length)
        auto adaptive = [&](double motion, double still, double moving) {
            const double t = std::min(1.0, std::max(0.0, (motion - 0.3) / 1.2));
            return still + (moving - still) * t;
        };
        auto motionOf = [&](const std::vector<uint8_t> *first, size_t from, size_t to) {   // pending[from..to]
            double sum = 0;
            int k = 0;
            const std::vector<uint8_t> *prev = first;
            for (size_t i = from; i <= to && i < pending.size(); ++i) {
                if (prev && !prev->empty()) { sum += blockMeanDiff(*prev, pending[i]); ++k; }
                prev = &pending[i];
            }
            return k ? sum / k : 0.0;
        };
        // index of the first pending frame (< limit) that starts a new scene, or limit: a large change that is not
        // followed by a comparable one (sustained motion changes every frame)
        // correlation of the 16x16 block means (subsampled) of two frames
        auto blockMeanCorr = [&](const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
            const std::vector<int> &A = sumsOf(a), &B = sumsOf(b);
            double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
            const int nb = (int)A.size();
            for (int i = 0; i < nb; ++i) {
                const int ma = A[i], mb = B[i];
                sa += ma; sb += mb; saa += double(ma) * ma; sbb += double(mb) * mb; sab += double(ma) * mb;
            }
            if (!nb) return 0.0;
            const double va = saa - sa * sa / nb, vb = sbb - sb * sb / nb, cov = sab - sa * sb / nb;
            return va > 0 && vb > 0 ? cov / std::sqrt(va * vb) : 0.0;
        };
        static const double cutCorr = getenv("AV1GPU_PF_CUTCORR") ? atof(getenv("AV1GPU_PF_CUTCORR")) : 0.7;
        // extra key-frame quality for static content (periodic key frames; AV1GPU_PF_KEYSTILL)
        static const int keyStill = getenv("AV1GPU_PF_KEYSTILL") ? atoi(getenv("AV1GPU_PF_KEYSTILL")) : 40;
        auto firstCut = [&](size_t limit) -> size_t {
            if (pfSceneCut <= 0) return limit;
            readMore(limit + 1);
            for (size_t i = 0; i < limit && i < pending.size(); ++i) {
                const std::vector<uint8_t> &prev = i ? pending[i - 1] : prevFrame;
                if (prev.empty()) continue;
                const double d = blockMeanDiff(prev, pending[i]);
                if (d <= pfSceneCut) continue;
                const double next = i + 1 < pending.size() ? blockMeanDiff(pending[i], pending[i + 1]) : 0.0;
                // a lighting change (strobe, colour wash) keeps the picture's structure: its block means stay
                // correlated (video_2's flashes 0.86-0.96, video_4's cut 0.10); only a structural change is a cut
                if (d > 2.0 * next && blockMeanCorr(prev, pending[i]) < cutCorr) return i;
            }
            return limit;
        };
        for (;;) {
            readMore(1);
            if (pending.empty()) break;
            if (!keyPending && applyQ()) return 1;
            const uint64_t periodBegin = keyPending ? keyPeriodBegin : frameCount;
            keyPending = false;
            const bool cutNow = frameCount > 0 && frameCount != lastKeyIndex && firstCut(1) == 0;
            if (cutNow) ++pfCuts;
            if (frameCount == 0 || frameCount - lastKeyIndex >= keyInterval || cutNow) {
                // scene-cut key frames sit closer to the inter-frame quality than periodic ones
                // periodic key frames: the offset adapts to the change over the next mini-GOP (static content profits
                // from a much better key frame, moving content does not)
                int keyOffset = pfCutQoffset;
                if (!cutNow) {
                    readMore(size_t(G));
                    const size_t last = std::min(pending.size(), size_t(G)) - 1;
                    keyOffset = pfAdaptive && last ? (int)std::lround(adaptive(motionOf(nullptr, 0, last), pfQoffset + keyStill, pfQoffset))
                                                   : pfQoffset;
                }
                // the key-frame offset grows with the inter quantizer (low rates invest relatively more in key frames,
                // as libaom / SVT do): offset x (q / 165)^keySlope (AV1GPU_PF_KEYSLOPE)
                static const double keySlope = getenv("AV1GPU_PF_KEYSLOPE") ? atof(getenv("AV1GPU_PF_KEYSLOPE")) : 0.0;
                if (keySlope != 0.0)
                    keyOffset = (int)std::lround(keyOffset * std::pow(pfSettings.qindex / 165.0, keySlope));
                const bool restoreQ = cutNow || keyOffset != pfQoffset;
                GpEncoder &ke = cutNow && haveCut ? cutEncoder : encoder;
                if (restoreQ && ke.setQindex(std::max(1, std::min(255, pfSettings.qindex - keyOffset)), err)) {
                    fprintf(stderr, "key frame: %s\n", err.c_str());
                    return 1;
                }
                // keyframe
                // AV1GPU_PF_TF_KEY=k: temporal filter of the key frame over the next k frames (as SVT-AV1 does); the key frame
                // codes the filtered picture
                static const int tfKey = getenv("AV1GPU_PF_TF_KEY") ? std::min(kTfKeyMax, atoi(getenv("AV1GPU_PF_TF_KEY"))) : 12;
                static const double tfKeyNoise = getenv("AV1GPU_PF_TF_KEY_NOISE") ? atof(getenv("AV1GPU_PF_TF_KEY_NOISE")) : 0.25;
                bool keyFiltered = false;
                if (tfKey > 0) {
                    readMore(size_t(tfKey) + 1);
                    const int nb = std::min<int>(tfKey, int(pending.size()) - 1);
                    if (nb > 0) {
                        for (int j = 0; j <= nb; ++j)
                            if (!inBuf.upload(j, pending[j]) || !inBuf.wait(j, compute)) return 1;
                    }
                    // only noisy content: the filter also averages real changes, which costs clean content more than
                    // it gains (video_4 / video_2, noise 0.05-0.18: +0.8% / 0; video_1 / 3 / 5, 0.37-1.4: -0.4 to -3.7%)
                    const double sigma = nb > 0 ? pfAnalyzer.noiseSigma(inBuf[0][0], pitches[0], perr, sizeof perr) : 0.0;
                    if (sigma < -1.5) { fprintf(stderr, "%s\n", perr); return 1; }
                    if (nb > 0 && sigma >= tfKeyNoise) {
                        std::vector<const uint8_t *> nbs[3];
                        for (int j = 1; j <= nb; ++j)
                            for (int p = 0; p < 3; ++p) nbs[p].push_back(inBuf[j][p]);
                        const uint8_t *const center[3] = {inBuf[0][0], inBuf[0][1], inBuf[0][2]};
                        const uint8_t *const *nbp[3] = {nbs[0].data(), nbs[1].data(), nbs[2].data()};
                        uint8_t *const outp[3] = {tfBuf[0], tfBuf[1], tfBuf[2]};
                        // strength from the key frame's own quantizer (as libaom: a near-lossless key frame is barely
                        // filtered; with the inter quantizer video_4's 60 dB logo frame lost 2.8 dB)
                        const int kfQ = std::max(1, std::min(255, pfSettings.qindex - keyOffset));
                        if (pfAnalyzer.temporalFilter(center, nbp, nb, kfQ, pfTfStrength, outp, perr, sizeof perr)) {
                            fprintf(stderr, "%s\n", perr);
                            return 1;
                        }
                        if (cudaStreamSynchronize(compute) != cudaSuccess) return 1;
                        keyFiltered = true;
                        if (getenv("AV1GPU_PF_TF_DUMP")) {   // debug: the filtered key-frame source (padded planes)
                            static FILE *tfd = fopen(getenv("AV1GPU_PF_TF_DUMP"), "wb");
                            std::vector<uint8_t> hb(sizes[0]);
                            for (int p = 0; p < 3; ++p) {
                                hb.resize(sizes[p]);
                                cudaMemcpy(hb.data(), tfBuf[p], sizes[p], cudaMemcpyDeviceToHost);
                                if (tfd) fwrite(hb.data(), 1, sizes[p], tfd);
                            }
                        }
                    }
                }
                std::vector<uint8_t> raw = std::move(pending.front());
                pending.erase(pending.begin());
                if (!keyFiltered && (!inBuf.upload(0, raw) || !inBuf.wait(0, nullptr))) return 1;
                prevFrame = std::move(raw);
                diffCache.clear();   // freed buffers may be reused at the same addresses
                GpFrameIn in;
                in.deviceRecon = reference;
                in.W = w; in.H = h; in.aw = aw; in.ah = ah;
                in.colorMatrix = 1;
                for (int p = 0; p < 3; ++p) { in.dsrc[p] = keyFiltered ? tfBuf[p] : inBuf[0][p]; in.spitch[p] = pitches[p]; }
                std::vector<GpFrameIn> inputs{in};
                std::vector<GpFrameOut> frames;
                GpTimes times;
                GpDevRecords records;
                av1g_set_video_order_hint(kOrderHintBits, int(frameCount & ((1u << kOrderHintBits) - 1)));
                for (int &v : slotHint) v = int(frameCount & ((1u << kOrderHintBits) - 1));
                const auto kt0 = std::chrono::steady_clock::now();
                if (ke.encode(inputs, frames, times, err, 0, &records)) { fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
                const auto kt1 = std::chrono::steady_clock::now();
                std::vector<std::vector<std::vector<uint8_t>>> tiles;
                double entropyMs = 0;
                int status = gp_entropy_dev(records, 0, scratch, tiles, &entropyMs, err);
                kfEncodeMs += std::chrono::duration<double, std::milli>(kt1 - kt0).count();
                kfEntropyMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - kt1).count();
                ++kfCount;
                gp_dev_records_free(records, 0);
                if (status) { fprintf(stderr, "entropy: %s\n", err.c_str()); return 1; }
                Job kj;
                kj.kind = Job::Raw;
                kj.endTU = true;
                kj.resetSlots = true;
                kj.pts = frameCount;
                if (gp_assemble_obus(&frames[0].fp, tiles[0], true, kj.raw, err, true, frameCount == 0, grain,
                                     uint16_t(7391 + frameCount * 31847))) {
                    fprintf(stderr, "write frame: %s\n", err.c_str());
                    return 1;
                }
                if (submit(std::move(kj))) return 1;
                if (restoreQ && &ke == &encoder && encoder.setQindex(targetKbps > 0 ? std::max(1, std::min(255, pfSettings.qindex - pfQoffset)) : cfg.qindex, err)) {
                    fprintf(stderr, "scene cut: %s\n", err.c_str());
                    return 1;
                }
                gpuMs += times.total + entropyMs;
                for (int k = 0; k < 8; ++k)
                    if (cudaMemcpy(slotBuf[k], reference, frameBytes, cudaMemcpyDeviceToDevice) != cudaSuccess) return 1;
                if (cudaStreamSynchronize(nullptr) != cudaSuccess) { fprintf(stderr, "key frame reference copy failed\n"); return 1; }
                for (auto &v : slotValid) v = false;
                pendingMotion.slot = -1;
                mfield.keyFrame();
                pfAnalyzer.resetTemporal();
                if (dump(slotBuf[0])) return 1;
                lastShown = 0;
                lastKeyIndex = frameCount;
                keyPending = true;
                keyPeriodBegin = frameCount;
                ++frameCount;
                continue;
            }
            const uint64_t untilKey = keyInterval - (frameCount - lastKeyIndex);
            size_t g = std::min<size_t>(G, untilKey);
            readMore(g + pfTfFrames);
            g = std::max<size_t>(1, firstCut(std::min(g, pending.size())));   // end the mini-GOP before a cut
            if (pfAdaptive && !prevFrame.empty() && g > 8 && pending.size() > 1) {
                // moving content: shorter mini-GOPs (distant references predict poorly); measured over the next
                // eight frames from the anchor
                const double m = motionOf(&prevFrame, 0, std::min<size_t>(7, pending.size() - 1));
                if (m >= gopMotion8) g = 8;
                else if (m >= gopMotion16) g = std::min<size_t>(g, 16);
            }
            const size_t n = std::min(g, pending.size());
            if (n == 1) {
                // single P frame from the last shown picture
                if (!inBuf.upload(0, pending[0])) return 1;
                frameOrder = frameCount;
                if (interFrame(0, lastShown, -1, 0, pfSettings.qindex, true, true, frameCount)) return 1;
                if (dump(slotBuf[0])) return 1;
                lastShown = 0;
                ++predicted; ++frameCount;
                if (!inBuf.finishBatch(compute)) return 1;
                prevFrame = std::move(pending[0]);
                pending.erase(pending.begin());
                diffCache.clear();
                endPeriod(frameCount - periodBegin);
                continue;
            }
            // Mini-GOP pyramid: the last frame is coded first as a hidden ALTREF; each interval (lo, hi) codes its middle
            // frame from lo (LAST) and hi (ALTREF), hidden unless it is the next frame to display, and shows coded
            // frames with show_existing_frame when their turn comes.
            const size_t avail = std::min(pending.size(), n + pfTfFrames);
            // Prime the ALTREF and any temporal-filter neighbours. Prefetch the remaining original
            // pictures in pyramid coding order, one per analysis launch.
            const size_t firstReady = n - 1 - std::min(n - 1, size_t(pfTfFrames));
            if (!inBuf.upload(n - 1, pending[n - 1])) return 1;
            for (size_t i = firstReady; i < avail; ++i)
                if (i != n - 1 && !inBuf.upload(i, pending[i])) return 1;
            uploadOrder.clear();
            nextUpload = 0;
            auto order = [&](auto &self, size_t lo, size_t hi) -> void {
                if (hi <= lo + 1) return;
                const size_t mid = (lo + hi) / 2;
                if (mid - 1 < firstReady) uploadOrder.push_back(mid - 1);
                self(self, lo, mid);
                self(self, mid, hi);
            };
            order(order, 0, n);

            bool filtered = false;
            // AV1GPU_PF_TF_NOOVERLAY: display the filtered ALTREF itself (show_existing_frame, as SVT-AV1 does by
            // default) instead of coding an overlay frame that restores the original picture
            static const bool tfNoOverlay = getenv("AV1GPU_PF_TF_NOOVERLAY") != nullptr;
            // noisy content only (as the key-frame filter): estimated luma noise of the ALTREF source
            bool tfNoisy = false;
            if (pfTfFrames > 0) {
                if (!inBuf.wait(n - 1, compute)) return 1;
                static const double tfNoise = getenv("AV1GPU_PF_TF_KEY_NOISE") ? atof(getenv("AV1GPU_PF_TF_KEY_NOISE")) : 0.25;
                const double sigma = pfAnalyzer.noiseSigma(inBuf[n - 1][0], pitches[0], perr, sizeof perr);
                if (sigma < -1.5) { fprintf(stderr, "%s\n", perr); return 1; }
                tfNoisy = sigma >= tfNoise;
            }
            if (pfTfFrames > 0 && tfNoisy) {
                std::vector<const uint8_t *> nbs[3];
                for (size_t i = firstReady; i < avail; ++i) {
                    if (!inBuf.wait(i, compute)) return 1;
                    if (i == n - 1) continue;
                    for (int p = 0; p < 3; ++p) nbs[p].push_back(inBuf[i][p]);
                }
                if (!nbs[0].empty()) {
                    const uint8_t *const center[3] = {inBuf[n - 1][0], inBuf[n - 1][1], inBuf[n - 1][2]};
                    const uint8_t *const *nbp[3] = {nbs[0].data(), nbs[1].data(), nbs[2].data()};
                    uint8_t *const outp[3] = {tfBuf[0], tfBuf[1], tfBuf[2]};
                    // ALTREF strength (AV1GPU_PF_TF_ARF_STRENGTH, else --pf-tf-strength): a filtered ALTREF is coded
                    // at high quality and referenced by the whole mini-GOP, so it wants a much weaker filter
                    static const float arfStrength = getenv("AV1GPU_PF_TF_ARF_STRENGTH") ? (float)atof(getenv("AV1GPU_PF_TF_ARF_STRENGTH")) : 0.1f;
                    if (pfAnalyzer.temporalFilter(center, nbp, int(nbs[0].size()), pfSettings.qindex,
                                                  arfStrength > 0.f ? arfStrength : pfTfStrength, outp, perr, sizeof perr)) {
                        fprintf(stderr, "%s\n", perr);
                        return 1;
                    }
                    filtered = true;
                }
            }
            std::vector<int> freeSlots;
            for (int k = 7; k >= 0; --k)
                if (k != lastShown) freeSlots.push_back(k);
            auto allocSlot = [&]() {
                if (freeSlots.empty()) throw std::runtime_error("no free reference slot");
                int k = freeSlots.back();
                freeSlots.pop_back();
                return k;
            };
            auto release = [&](int k) { if (k >= 0) freeSlots.push_back(k); };
            // ALTREF boost from the change between the anchor and the ALTREF source (--pf-adaptive)
            gopMotion = !prevFrame.empty() ? motionOf(&prevFrame, 0, n - 1) : 0.0;
            static const double movingBoost = getenv("AV1GPU_PF_BOOST_MOVING") ? atof(getenv("AV1GPU_PF_BOOST_MOVING")) : 40.0;
            // a temporally filtered ALTREF of fast-moving content (a pan of noisy footage) stays a useful reference over
            // its whole mini-GOP: it keeps a larger boost (AV1GPU_PF_BOOST_MOVING_FILTERED; video_3 -3.5%, no other clip
            // reaches the motion)
            static const double movingFiltered = getenv("AV1GPU_PF_BOOST_MOVING_FILTERED") ? atof(getenv("AV1GPU_PF_BOOST_MOVING_FILTERED")) : 100.0;
            double gopBoost = pfAdaptive && !prevFrame.empty()
                                  ? (filtered && gopMotion >= 1.5 ? movingFiltered : adaptive(gopMotion, pfAltrefBoost, movingBoost))
                                  : pfAltrefBoost;
            // short mini-GOPs (shots between cuts) amortize the boosted frames over fewer pictures (AV1GPU_PF_SHORTGOP:
            // the length below which the boost scales with n / length; 0 off)
            static const double shortLen = getenv("AV1GPU_PF_SHORTGOP") ? atof(getenv("AV1GPU_PF_SHORTGOP")) : 8.0;
            if (shortLen > 0 && double(n) < shortLen) gopBoost *= double(n) / shortLen;
            // the boost grows with the quantizer like the key-frame offset (AV1GPU_PF_BOOSTSLOPE exponent)
            static const double boostSlope = getenv("AV1GPU_PF_BOOSTSLOPE") ? atof(getenv("AV1GPU_PF_BOOSTSLOPE")) : 0.0;
            if (boostSlope != 0.0) gopBoost *= std::pow(pfSettings.qindex / 165.0, boostSlope);
            int depthMax = 0;
            while ((size_t(1) << depthMax) < n) ++depthMax;
            // boost share of a pyramid level: (1 - depth / depthMax) ^ AV1GPU_PF_LAYERGAMMA (1: linear)
            static const double layerGamma = getenv("AV1GPU_PF_LAYERGAMMA") ? atof(getenv("AV1GPU_PF_LAYERGAMMA")) : 1.0;
            auto qAt = [&](int depth) {
                const double frac = std::pow(depthMax ? 1.0 - double(depth) / depthMax : 1.0, layerGamma);
                return std::max(1, (int)std::lround(pfSettings.qindex - gopBoost * frac));
            };
            const uint64_t base = frameCount - 1;   // display index of the anchor (lastShown)
            int hiSlot = allocSlot();
            frameOrder = base + n;
            if (interFrame(n - 1, lastShown, -1, hiSlot, qAt(0), false, false, 0, -1, filtered)) return 1;
            // hi2Slot: the next anchor after hi (still held, displayed later), the third reference of the interval
            std::function<int(uint64_t, uint64_t, int, int, int, int)> rec = [&](uint64_t lo, uint64_t hi, int loSlot,
                                                                                 int hiSlotL, int depth, int hi2Slot) -> int {
                if (hi == lo + 1 && filtered && hi == base + n && !tfNoOverlay) {
                    // overlay: display the original picture coded from the filtered ALTREF (refreshes no slot)
                    frameOrder = base + n;
                    if (interFrame(n - 1, loSlot, hiSlotL, 8, std::min(255, pfSettings.qindex + pfLeafOffset),
                                   true, true, frameCount))
                        return 1;
                    if (dump(spare)) return 1;
                    ++predicted; ++frameCount;
                    release(loSlot);
                    return 0;
                }
                if (hi == lo + 1) {
                    Job se;
                    se.kind = Job::ShowExisting;
                    se.showSlot = hiSlotL;
                    se.endTU = true;
                    se.pts = frameCount;
                    if (submit(std::move(se))) return 1;
                    if (dump(slotBuf[hiSlotL])) return 1;
                    ++predicted; ++frameCount;
                    release(loSlot);
                    return 0;
                }
                const uint64_t mid = (lo + hi) / 2;
                if (mid == lo + 1) {
                    // the frame shown before an overlay is the overlay's LAST reference
                    const bool referenced = hi - mid > 1 || (filtered && hi == base + n && !tfNoOverlay);
                    const int midSlot = referenced ? allocSlot() : -1;
                    // an unreferenced shown frame refreshes no slot; its reconstruction lands in the spare buffer
                    frameOrder = mid;
                    if (interFrame(mid - base - 1, loSlot, hiSlotL, referenced ? midSlot : 8,
                                   referenced ? qAt(depth) : std::min(255, pfSettings.qindex + pfLeafOffset), true,
                                   true, frameCount, hi2Slot))
                        return 1;
                    if (dump(referenced ? slotBuf[midSlot] : spare)) return 1;
                    ++predicted; ++frameCount;
                    release(loSlot);
                    return rec(mid, hi, midSlot, hiSlotL, depth + 1, hi2Slot);
                }
                const int midSlot = allocSlot();
                frameOrder = mid;
                if (interFrame(mid - base - 1, loSlot, hiSlotL, midSlot, qAt(depth), false, false, 0, hi2Slot))
                    return 1;
                if (rec(lo, mid, loSlot, midSlot, depth + 1, hiSlotL)) return 1;
                return rec(mid, hi, midSlot, hiSlotL, depth + 1, hi2Slot);
            };
            try {
                if (rec(base, base + n, lastShown, hiSlot, 1, -1)) return 1;
            } catch (const std::exception &ex) {
                fprintf(stderr, "mini-GOP: %s\n", ex.what());
                return 1;
            }
            lastShown = hiSlot;
            if (!inBuf.finishBatch(compute)) return 1;
            prevFrame = std::move(pending[n - 1]);
            pending.erase(pending.begin(), pending.begin() + n);
            diffCache.clear();
            endPeriod(frameCount - periodBegin);
        }
        if (drain()) return 1;
        stopWriter();
        const cudaError_t e = cudaStreamSynchronize(compute);
        if (e != cudaSuccess) { fprintf(stderr, "P-frame compute: %s\n", cudaGetErrorString(e)); return 1; }
        for (auto *p : tfBuf) cudaFree(p);
        for (auto *p : slotBuf) cudaFree(p);
        cudaFree(spare);
        return 0;
    };
    if (usePframes) {
        int status = runPframes();
        if (status) return status;
    } else
    for (;;) {
        std::vector<GpFrameIn> inputs;
        while (inputs.size() < (temporal ? size_t(1) : size_t(batchSize)) && read_frame(stdin, w, h, src)) {
            const uint8_t *planes[3] = {src.data(), src.data() + (size_t)w * h,
                                        src.data() + (size_t)w * h + (size_t)cw * ch};
            const int widths[3] = {w, cw, cw}, heights[3] = {h, ch, ch}, pitches[3] = {w, cw, cw};
            const int paddedWidths[3] = {aw, acw, acw}, paddedHeights[3] = {ah, ach, ach};
            GpFrameIn in;
            in.deviceRecon = reference;
            in.W = w; in.H = h; in.aw = aw; in.ah = ah;
            in.colorMatrix = 1; // BT.709; the raw input carries no color metadata.
            for (int p = 0; p < 3; ++p) {
                pad_plane(planes[p], widths[p], heights[p], pitches[p], paddedWidths[p], paddedHeights[p], padded[p]);
                cudaError_t e = cudaMemcpy(d[p] + inputs.size() * sizes[p], padded[p].data(), sizes[p], cudaMemcpyHostToDevice);
                if (e != cudaSuccess) { fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(e)); return 1; }
                in.dsrc[p] = d[p] + inputs.size() * sizes[p]; in.spitch[p] = paddedWidths[p];
            }
            inputs.push_back(in);
        }
        if (inputs.empty()) break;
        if (canPredict && frameCount && frameCount-lastKey < keyInterval) {
            if (autoMotionStep) {
                auto status = chooseMotionStep(inputs[0].dsrc[0], reference, anchor, w, h, aw,
                                                modeCount, deviceMotionScores, motionStep);
                if (status != cudaSuccess) {
                    fprintf(stderr, "motion prescan failed: %s\n", cudaGetErrorString(status));
                    return 1;
                }
                for (int i = 0; i < MotionStepCount; ++i)
                    if (motionStepAt(i) == motionStep) ++motionChoices[i];
            }
            auto status = searchMotion(inputs[0].dsrc,reference,w,h,aw,modeCount,motionStep,deviceBlocks,blocks.data(),anchor,motionBlockSize);
            if(status!=cudaSuccess) { fprintf(stderr,"motion: %s\n",cudaGetErrorString(status)); return 1; }
            // Bound local as well as frame-average distortion to protect small text.
            uint64_t total = 0, changedSamples = 0, totalSamples = 0;
            const double rms = std::max(1.0, cfg.qindex/42.0);
            bool accept = true;
            for(const auto &block : blocks) {
                total += block.sse;
                if (sceneCutRms) {
                    totalSamples += block.samples;
                    if (block.sse > uint64_t(sceneCutRms)*sceneCutRms*block.samples)
                        changedSamples += block.samples;
                }
                if(block.sse > 9*rms*rms*block.samples) accept = false;
            }
            // Require widespread prediction failure. Isolated changed blocks
            // should normally stay inter and carry their own corrections.
            const bool sceneCut = sceneCutRms && changedSamples * 3 >= totalSamples &&
                total > uint64_t(sceneCutRms)*sceneCutRms*totalSamples;
            if (sceneCut) ++sceneCuts;
            if (!sceneCut && (useResiduals || (accept && total <= rms*rms*ySize*1.5))) {
                // Hold the displayed reconstruction only if its excess squared error
                // stays small both locally and on average (8-bit sample units).
                bool same = adaptiveFps && !grain;
                uint64_t extraError = 0;
                for(size_t i=0; same && i<blocks.size(); ++i) {
                    const auto &block = blocks[i];
                    unsigned extra = block.candidateSse[previousModes[i]] - block.sse;
                    extraError += extra;
                    same = extra <= block.samples*2;
                }
                same = same && extraError <= ySize*1.5*0.25;
                for(size_t i=0; i<blocks.size(); ++i) {
                    if(same) blocks[i].mode = previousModes[i];
                    else previousModes[i] = blocks[i].mode;
                }
                if (useResiduals) {
                    auto status = prepareInterResiduals(inputs[0].dsrc, reference, w, h, aw,
                                                       deviceBlocks, deviceLevels, residualLevels.data(), residualThreshold,
                                                       rolling ? nextReference : nullptr, motionStep, interQindex, luma8,
                                                       alternateLevels, alternateRecon, deviceTxMap,
                                                       adaptiveTx ? txMap.data() : nullptr, anchor, interRdoq, motionBlockSize);
                    if (status != cudaSuccess) {
                        fprintf(stderr, "residual transform failed: %s\n", cudaGetErrorString(status));
                        return 1;
                    }
                }
                if (adaptiveTx)
                    largerTransforms += std::count(txMap.begin(), txMap.end(), uint8_t{1});
                const bool refreshAnchor = anchorInterval && frameCount - lastAnchor >= anchorInterval;
                auto obu = inter::packet(w,h,blocks,modeCount,grain,uint16_t(7391+frameCount*31847),rolling,
                                         useResiduals ? residualLevels.data() : nullptr, (residualThreshold || interQindex), motionStep, interQindex, luma8,
                                         adaptiveTx ? txMap.data() : nullptr, useAnchor, refreshAnchor, motionBlockSize,
                                         persistentCdf ? &interEntropy : nullptr);
                if(same) {
                    heldPacket = std::move(obu);
                    ++held;
                } else {
                    if(!write_ivf_frame(outFile,obu,frameCount)) return 1;
                    if (rolling) {
                        auto status = useResiduals ? cudaSuccess :
                            reconstructMotion(reference, nextReference, w, h, deviceBlocks, motionStep, anchor, motionBlockSize);
                        if (status != cudaSuccess) {
                            fprintf(stderr, "reference reconstruction failed: %s\n", cudaGetErrorString(status));
                            return 1;
                        }
                        std::swap(reference, nextReference);
                        if (refreshAnchor) {
                            auto copy = cudaMemcpy(anchor, reference, ySize*3/2, cudaMemcpyDeviceToDevice);
                            if (copy != cudaSuccess) {
                                fprintf(stderr, "anchor refresh failed: %s\n", cudaGetErrorString(copy));
                                return 1;
                            }
                            lastAnchor = frameCount;
                        }
                    }
                    lastWritten = frameCount;
                }
                ++predicted; ++frameCount;
                continue;
            }
        }
        std::vector<GpFrameOut> frames;
        GpTimes times;
        GpDevRecords records;
        if (encoder.encode(inputs, frames, times, err, 0, &records)) {
            fprintf(stderr, "encode: %s\n", err.c_str());
            return 1;
        }
        std::vector<std::vector<std::vector<uint8_t>>> tiles;
        double entropyMs = 0;
        int status = gp_entropy_dev(records, 0, scratch, tiles, &entropyMs, err);
        gp_dev_records_free(records, 0);
        if (status) {
            fprintf(stderr, "entropy: %s\n", err.c_str());
            return 1;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::vector<uint8_t> obu;
            if (gp_assemble_obus(&frames[i].fp, tiles[i], true, obu, err, true, frameCount == 0, grain, uint16_t(7391+frameCount*31847)) ||
                !write_ivf_frame(outFile, obu, frameCount)) {
                fprintf(stderr, "write frame: %s\n", err.c_str());
                return 1;
            }
            if(canPredict) {
                if (useAnchor && cudaMemcpy(anchor, reference, ySize*3/2, cudaMemcpyDeviceToDevice) != cudaSuccess) {
                    fprintf(stderr, "anchor reference copy failed\n");
                    return 1;
                }
                lastWritten = lastKey = lastAnchor = frameCount;
                interEntropy.ready = false;
                std::fill(previousModes.begin(),previousModes.end(),0);
            }
            ++frameCount;
        }
        gpuMs += times.total + entropyMs;
    }
    // Keep the last input timestamp explicit so the final sample has one tick of duration.
    if(adaptiveFps && held && frameCount && lastWritten+1 < frameCount) {
        if(!write_ivf_frame(outFile,heldPacket,frameCount-1)) return 1;
        --held;
    }
    if(adaptiveFps) fprintf(stderr,"held %llu display ticks\n",(unsigned long long)held);
    if (sceneCutRms) fprintf(stderr, "scene cut keyframes %llu\n", (unsigned long long)sceneCuts);
    if(temporal) fprintf(stderr,"inter frames %llu, keyframes %llu\n",(unsigned long long)predicted,
                         (unsigned long long)(frameCount-predicted));
    if (targetKbps > 0 && frameCount)
        fprintf(stderr, "rate control: target %.1f kbit/s over %llu periods\n", targetKbps, (unsigned long long)rcPeriods);
    if (usePframes)
        fprintf(stderr, "P-frame bits: mode/partition %.0f, mv %.0f, luma coeff %.0f, chroma coeff %.0f\n",
                pfBits[0], pfBits[1], pfBits[2], pfBits[3]),
        fprintf(stderr, "P-frame filters %.1f ms, syntax %.1f ms, loop restoration in %d frames, scene cuts %llu\n", pfFilterMs, pfWriteMs, pfLrFrames, (unsigned long long)pfCuts),
        fprintf(stderr, "key frames %d: encode %.1f ms, entropy %.1f ms (wall)\n", kfCount, kfEncodeMs, kfEntropyMs),
        fprintf(stderr, "P-frame analysis %.1f ms; compound blocks %llu; modes nearest %llu near %llu global %llu new %llu\n", pfMs,
                (unsigned long long)pfCompound,
                (unsigned long long)pfModes[0], (unsigned long long)pfModes[1], (unsigned long long)pfModes[2],
                (unsigned long long)pfModes[3]);
    if (adaptiveTx)
        fprintf(stderr, "8x8 luma candidates selected %llu of %llu inter blocks\n",
                (unsigned long long)largerTransforms,
                (unsigned long long)(predicted * txMap.size()));
    cudaFree(deviceTxMap); cudaFree(alternateRecon); cudaFree(alternateLevels);
    cudaFree(deviceLevels);
    if (autoMotionStep) {
        for (int i = 0; i < MotionStepCount; ++i)
            fprintf(stderr, "%d-pixel motion selected on %llu searches\n",
                    motionStepAt(i), (unsigned long long)motionChoices[i]);
        cudaFree(deviceMotionScores);
    }
    cudaFree(anchor); cudaFree(nextReference); cudaFree(reference); cudaFree(deviceBlocks);
    gp_ent_scratch_free(scratch, 0);
    if (!frameCount) { fprintf(stderr, "input contains no frames\n"); return 2; }
    if (fseek(outFile, 24, SEEK_SET)) {
        fprintf(stderr, "cannot finalize IVF header\n"); return 2;
    }
    put_le(outFile, (uint32_t)frameCount, 4);
    if (fseek(outFile, 0, SEEK_END)) { fprintf(stderr, "cannot seek to IVF end\n"); return 2; }
    const long outputBytes = ftell(outFile);
    fclose(outFile);
    for (int p = 0; p < 3; ++p) cudaFree(d[p]);
    const double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wallStart).count();
    fprintf(stderr, "{\"frames\":%llu,\"fps\":%.3f,\"gpu_ms\":%.2f,\"wall_ms\":%.2f,\"output_bytes\":%lld}\n",
            (unsigned long long)frameCount, frameCount * 1000.0 / wallMs, gpuMs, wallMs,
            (long long)outputBytes);
    return 0;
}
