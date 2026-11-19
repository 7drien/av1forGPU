// Experimental CUDA AV1 video encoder. Input is raw 8-bit planar YUV420;
// all AV1 decisions, reconstruction, filters, and entropy coding use the CUDA backend.
#include "presets.h"
#include "inter_packet.h"
#include "pframe.h"
#include "pframe_writer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <deque>
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

static void pad_plane(const uint8_t *src, int w, int h, int pitch, int paddedW, int paddedH,
                      std::vector<uint8_t> &dst) {
    dst.resize((size_t)paddedW * paddedH);
    for (int y = 0; y < paddedH; ++y) {
        const uint8_t *row = src + (size_t)std::min(y, h - 1) * pitch;
        uint8_t *out = dst.data() + (size_t)y * paddedW;
        memcpy(out, row, w);
        memset(out + w, row[w - 1], paddedW - w);
    }
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ffmpeg ... -f rawvideo -pix_fmt yuv420p - | av1gpu-video WIDTH HEIGHT FPS_NUM FPS_DEN OUT.ivf [--fast] [--inter] [--inter-tx8 | --inter-tx-adaptive] [--inter-rdoq] [--persistent-cdf] [--diagonal-motion] [--motion-step 2..16 | --auto-motion-step] [--motion-block 8|16|32|64] [--rolling-reference] [--anchor-reference [--anchor-interval N]] [--lossless-inter-residual | --inter-residual-threshold N | --inter-qindex N] [--keyint 1..10000] [--scene-cut-rms 1..255] [--adaptive-fps] [--grain 0..32] [--qindex 1..255] [--pframes [--bitrate KBPS [--rc-buffer S]] [--pf-gop 1..32] [--pf-qoffset N] [--pf-altref-boost N] [--pf-leaf-offset N] [--pf-qindex N] [--pf-lambda X] [--pf-tiles C R] [--pf-cpu-entropy | --pf-gpu-entropy] [--pf-tf-frames 0..4 [--pf-tf-strength X]] [--pf-rdoq] [--pf-no-filters]]\n");
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
    int pfQindex = 0, pfQoffset = 60, pfGop = 32, pfAltrefBoost = 100, pfLeafOffset = 0;
    double targetKbps = 0, rcBufferSeconds = 2.0;
    int pfTileCols = -1, pfTileRows = -1;
    int pfEntropyMode = 1;   // 0: CPU, 1: GPU modeling + host range coder, 2: all GPU
    int pfTfFrames = 0;
    float pfTfStrength = 1.0f;
    bool pfFilters = true, pfCdef = false;
    int residualThreshold = 0, interQindex = 0;
    bool useAnchor = false;
    unsigned anchorInterval = 0;
    unsigned keyInterval = 120;
    unsigned sceneCutRms = 0;
    GpConfig cfg;
    cfg.qindex = 100;
    for (int i = 6; i < argc; ++i) {
        if (!strcmp(argv[i], "--fast")) {
            apply_preset(cfg, "turbo3");
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
            if (pfQoffset < -100 || pfQoffset > 100) {
                fprintf(stderr, "P-frame qindex offset must be in -100..100\n");
                return 2;
            }
        } else if (!strcmp(argv[i], "--pf-gop") && i + 1 < argc) {
            pfGop = atoi(argv[++i]);
            if (pfGop < 1 || pfGop > 32) { fprintf(stderr, "--pf-gop must be 1..32\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-altref-boost") && i + 1 < argc) {
            pfAltrefBoost = atoi(argv[++i]);
            if (pfAltrefBoost < 0 || pfAltrefBoost > 128) { fprintf(stderr, "--pf-altref-boost must be 0..128\n"); return 2; }
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
            if (pfTfFrames < 0 || pfTfFrames > 4) { fprintf(stderr, "--pf-tf-frames must be 0..4\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-tf-strength") && i + 1 < argc) {
            pfTfStrength = strtof(argv[++i], nullptr);
            if (!(pfTfStrength > 0.f && pfTfStrength <= 16.f)) { fprintf(stderr, "--pf-tf-strength must be in (0, 16]\n"); return 2; }
        } else if (!strcmp(argv[i], "--pf-no-rect")) {
            pfSettings.rectPartitions = false;
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
    double pfMs = 0, pfFilterMs = 0, pfWriteMs = 0;
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
    if (usePframes && pfFilters) {
        // keyframes: deblocking and CDEF searches; enable_cdef = 1 in the sequence header lets P frames use CDEF
        cfg.fLf = 2;
        cfg.fCdef = 2;
        pfCdef = true;
    }
    if (encoder.init(cfg, err)) { fprintf(stderr, "encoder init: %s\n", err.c_str()); return 1; }
    if (usePframes) {
        char perr[256];
        if (pfAnalyzer.init(w, h, perr, sizeof perr)) { fprintf(stderr, "%s\n", perr); return 1; }
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
    auto runPframes = [&]() -> int {
        const int G = std::max(1, pfGop);
        const size_t frameBytes = ySize * 3 / 2;
        uint8_t *slotBuf[8] = {}, *spare = nullptr;
        for (int k = 0; k < 8; ++k)
            if (cudaMalloc(&slotBuf[k], frameBytes) != cudaSuccess) { fprintf(stderr, "slot allocation failed\n"); return 1; }
        if (cudaMalloc(&spare, frameBytes) != cudaSuccess) { fprintf(stderr, "slot allocation failed\n"); return 1; }
        std::vector<std::array<uint8_t *, 3>> inBuf(G + pfTfFrames + 1);
        for (auto &fb : inBuf)
            for (int p = 0; p < 3; ++p)
                if (cudaMalloc(&fb[p], sizes[p]) != cudaSuccess) { fprintf(stderr, "input allocation failed\n"); return 1; }
        bool slotValid[8] = {};
        std::array<uint8_t *, 3> tfBuf{};
        for (int p = 0; p < 3; ++p)
            if (cudaMalloc(&tfBuf[p], sizes[p]) != cudaSuccess) { fprintf(stderr, "filter buffer allocation failed\n"); return 1; }
        std::vector<std::vector<uint8_t>> pending;   // raw frames read ahead
        bool eof = false;
        auto readMore = [&](size_t want) {
            while (!eof && pending.size() < want) {
                std::vector<uint8_t> raw;
                if (!read_frame(stdin, w, h, raw)) { eof = true; break; }
                pending.push_back(std::move(raw));
            }
        };
        auto upload = [&](const std::vector<uint8_t> &raw, std::array<uint8_t *, 3> &fb) -> bool {
            const uint8_t *planes[3] = {raw.data(), raw.data() + (size_t)w * h, raw.data() + (size_t)w * h + (size_t)cw * ch};
            const int widths[3] = {w, cw, cw}, heights[3] = {h, ch, ch};
            const int paddedWidths[3] = {aw, acw, acw}, paddedHeights[3] = {ah, ach, ach};
            for (int p = 0; p < 3; ++p) {
                pad_plane(planes[p], widths[p], heights[p], widths[p], paddedWidths[p], paddedHeights[p], padded[p]);
                if (cudaMemcpy(fb[p], padded[p].data(), sizes[p], cudaMemcpyHostToDevice) != cudaSuccess) return false;
            }
            return true;
        };
        auto dump = [&](const uint8_t *dev) {
            if (!reconDump) return;
            std::vector<uint8_t> host(frameBytes);
            cudaMemcpy(host.data(), dev, frameBytes, cudaMemcpyDeviceToHost);
            fwrite(host.data(), 1, host.size(), reconDump);
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
            pf::EntropyOut tiles;
            bool endTU = false, resetSlots = false;
            uint64_t pts = 0;
        };
        const int kBuffers = 3;
        std::vector<pf::BlockInfo *> gridBuf(kBuffers, nullptr);
        std::vector<int16_t *> levelBuf(kBuffers, nullptr);
        std::vector<int> freeBufs;
        for (int k = 0; k < kBuffers; ++k) {
            if (cudaMallocHost(&gridBuf[k], pfAnalyzer.gridCount() * sizeof(pf::BlockInfo)) != cudaSuccess ||
                cudaMallocHost(&levelBuf[k], pfAnalyzer.levelCount() * sizeof(int16_t)) != cudaSuccess) {
                fprintf(stderr, "pinned buffer allocation failed\n");
                return 1;
            }
            freeBufs.push_back(k);
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
                        ok = !hostEntropy.code(job.ein, gridBuf[job.buf], levelBuf[job.buf], eout, werr, sizeof werr);
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
            for (int k = 0; k < kBuffers; ++k) { cudaFreeHost(gridBuf[k]); cudaFreeHost(levelBuf[k]); }
        }};
        // Analyze, filter and write one inter frame. refSlots: slots of the AV1 references used by BlockInfo::ref
        // 0 and 1 (second < 0: single reference). The reconstruction goes to outSlot.
        auto interFrame = [&](std::array<uint8_t *, 3> &fb, int slot0, int slot1, int outSlot, int qindex, bool show,
                              bool endTU, uint64_t pts) -> int {
            const uint8_t *refs[2] = {slotBuf[slot0], slot1 >= 0 ? slotBuf[slot1] : nullptr};
            const int refCount = slot1 >= 0 ? 2 : 1;
            pf::Settings st = pfSettings;
            st.qindex = qindex;
            const uint8_t *src[3] = {fb[0], fb[1], fb[2]};
            if (pfAnalyzer.analyze(src, pitches, refs, refCount, st, spare, nullptr, nullptr, perr, sizeof perr)) {
                fprintf(stderr, "%s\n", perr);
                return 1;
            }
            pfMs += pfAnalyzer.lastMs();
            const auto tf0 = std::chrono::steady_clock::now();
            pf::FilterParams flt;
            if (pfFilters && pfAnalyzer.filter(src, pitches, spare, qindex, pfCdef, true, flt, perr, sizeof perr)) {
                fprintf(stderr, "%s\n", perr);
                return 1;
            }
            const auto tw0 = std::chrono::steady_clock::now();
            pfFilterMs += std::chrono::duration<double, std::milli>(tw0 - tf0).count();
            Job job;
            job.slot0 = slot0;
            job.outSlot = outSlot;
            job.endTU = endTU;
            job.pts = pts;
            pf::EntropyIn &ein = job.ein;
            ein.refName[0] = pfw::LAST_FRAME;
            ein.refName[1] = pfw::ALTREF_FRAME;
            ein.referenceSelect = refCount == 2;
            ein.txModeSelect = pfSettings.txSplit;
            ein.reducedTxSet = pfSettings.txTypes < 2;
            ein.qindex = qindex;
            ein.tileColsLog2 = pfTileCols;
            ein.tileRowsLog2 = pfTileRows;
            pfw::FrameParams &fp = job.fp;
            fp.qindex = qindex;
            for (int r = 0; r < 7; ++r) fp.refSlot[r] = slot0;
            if (slot1 >= 0) fp.refSlot[pfw::ALTREF_FRAME - 1] = slot1;
            fp.referenceSelect = refCount == 2;
            fp.txModeSelect = pfSettings.txSplit;
            fp.reducedTxSet = pfSettings.txTypes < 2;
            fp.refreshFlags = outSlot < 8 ? 1 << outSlot : 0;
            fp.showFrame = show;
            fp.persistent = persistentCdf;
            fp.grain = grain;
            fp.grainSeed = uint16_t(7391 + frameCount * 31847);
            for (int k = 0; k < 4; ++k) fp.lfLevel[k] = flt.lf[k];
            fp.lfSharpness = flt.sharpness;
            fp.cdef = pfCdef;
            fp.cdefDamping = flt.damping;
            fp.cdefY = flt.yPri; fp.cdefYSec = flt.ySec; fp.cdefUV = flt.uvPri; fp.cdefUVSec = flt.uvSec;
            fp.tileColsLog2 = pfTileCols;
            fp.tileRowsLog2 = pfTileRows;
            fp.contextTile = 0;
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
                if (pfAnalyzer.download(gridBuf[job.buf], levelBuf[job.buf], perr, sizeof perr)) {
                    fprintf(stderr, "%s\n", perr);
                    return 1;
                }
                job.kind = Job::Coded;
            }
            if (submit(std::move(job))) return 1;
            if (outSlot < 8) {
                slotValid[outSlot] = persistentCdf;
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
        double baseQ = cfg.qindex, slope = 0.0146, bufferErr = 0;
        if (targetKbps > 0) {
            const double bpp = frameBits / (double(w) * h);
            baseQ = std::min(230.0, std::max(20.0, 120.0 + std::log(0.0077 / bpp) / slope));
        }
        long periodStart = ftell(outFile);
        uint64_t periodFrames = 0;
        double lastQ = baseQ, lastBpf = 0;
        auto applyQ = [&]() -> int {
            if (targetKbps <= 0) return 0;
            const int kq = std::max(1, std::min(255, (int)std::lround(baseQ)));
            if (encoder.setQindex(kq, err)) { fprintf(stderr, "rate control: %s\n", err.c_str()); return 1; }
            pfSettings.qindex = std::max(1, std::min(255, kq + pfQoffset));
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
        for (;;) {
            readMore(1);
            if (pending.empty()) break;
            if (!keyPending && applyQ()) return 1;
            const uint64_t periodBegin = keyPending ? keyPeriodBegin : frameCount;
            keyPending = false;
            if (frameCount == 0 || frameCount - lastKeyIndex >= keyInterval) {
                // keyframe
                std::vector<uint8_t> raw = std::move(pending.front());
                pending.erase(pending.begin());
                if (!upload(raw, inBuf[0])) { fprintf(stderr, "upload failed\n"); return 1; }
                GpFrameIn in;
                in.deviceRecon = reference;
                in.W = w; in.H = h; in.aw = aw; in.ah = ah;
                in.colorMatrix = 1;
                for (int p = 0; p < 3; ++p) { in.dsrc[p] = inBuf[0][p]; in.spitch[p] = pitches[p]; }
                std::vector<GpFrameIn> inputs{in};
                std::vector<GpFrameOut> frames;
                GpTimes times;
                GpDevRecords records;
                if (encoder.encode(inputs, frames, times, err, 0, &records)) { fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
                std::vector<std::vector<std::vector<uint8_t>>> tiles;
                double entropyMs = 0;
                int status = gp_entropy_dev(records, 0, scratch, tiles, &entropyMs, err);
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
                gpuMs += times.total + entropyMs;
                for (int k = 0; k < 8; ++k)
                    if (cudaMemcpy(slotBuf[k], reference, frameBytes, cudaMemcpyDeviceToDevice) != cudaSuccess) return 1;
                for (auto &v : slotValid) v = false;
                pfAnalyzer.resetTemporal();
                dump(slotBuf[0]);
                lastShown = 0;
                lastKeyIndex = frameCount;
                keyPending = true;
                keyPeriodBegin = frameCount;
                ++frameCount;
                continue;
            }
            const uint64_t untilKey = keyInterval - (frameCount - lastKeyIndex);
            const size_t g = std::min<size_t>(G, untilKey);
            readMore(g + pfTfFrames);
            const size_t n = std::min(g, pending.size());
            if (n == 1) {
                // single P frame from the last shown picture
                if (!upload(pending[0], inBuf[0])) { fprintf(stderr, "upload failed\n"); return 1; }
                if (interFrame(inBuf[0], lastShown, -1, 0, pfSettings.qindex, true, true, frameCount)) return 1;
                dump(slotBuf[0]);
                lastShown = 0;
                ++predicted; ++frameCount;
                pending.erase(pending.begin());
                endPeriod(frameCount - periodBegin);
                continue;
            }
            // Mini-GOP pyramid: the last frame is coded first as a hidden ALTREF; each interval (lo, hi) codes its middle
            // frame from lo (LAST) and hi (ALTREF), hidden unless it is the next frame to display, and shows coded
            // frames with show_existing_frame when their turn comes.
            const size_t avail = std::min(pending.size(), n + pfTfFrames);
            for (size_t i = 0; i < avail; ++i)
                if (!upload(pending[i], inBuf[i])) { fprintf(stderr, "upload failed\n"); return 1; }
            // temporal filter of the ALTREF source from its neighbours (past frames of this mini-GOP, read-ahead frames)
            bool filtered = false;
            std::array<uint8_t *, 3> altSrc = inBuf[n - 1];
            if (pfTfFrames > 0) {
                std::vector<const uint8_t *> nbs[3];
                for (int d = -pfTfFrames; d <= pfTfFrames; ++d) {
                    const long idx = long(n) - 1 + d;
                    if (!d || idx < 0 || idx >= long(avail)) continue;
                    for (int p = 0; p < 3; ++p) nbs[p].push_back(inBuf[idx][p]);
                }
                if (!nbs[0].empty()) {
                    const uint8_t *const center[3] = {inBuf[n - 1][0], inBuf[n - 1][1], inBuf[n - 1][2]};
                    const uint8_t *const *nbp[3] = {nbs[0].data(), nbs[1].data(), nbs[2].data()};
                    uint8_t *const outp[3] = {tfBuf[0], tfBuf[1], tfBuf[2]};
                    if (pfAnalyzer.temporalFilter(center, nbp, int(nbs[0].size()), std::max(1, pfSettings.qindex - pfAltrefBoost),
                                                  pfTfStrength, outp, perr, sizeof perr)) {
                        fprintf(stderr, "%s\n", perr);
                        return 1;
                    }
                    altSrc = tfBuf;
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
            int depthMax = 0;
            while ((size_t(1) << depthMax) < n) ++depthMax;
            auto qAt = [&](int depth) {
                const double frac = depthMax ? 1.0 - double(depth) / depthMax : 1.0;
                return std::max(1, (int)std::lround(pfSettings.qindex - pfAltrefBoost * frac));
            };
            const uint64_t base = frameCount - 1;   // display index of the anchor (lastShown)
            int hiSlot = allocSlot();
            if (interFrame(altSrc, lastShown, -1, hiSlot, qAt(0), false, false, 0)) return 1;
            std::function<int(uint64_t, uint64_t, int, int, int)> rec = [&](uint64_t lo, uint64_t hi, int loSlot,
                                                                            int hiSlotL, int depth) -> int {
                if (hi == lo + 1 && filtered && hi == base + n) {
                    // overlay: display the original picture coded from the filtered ALTREF (refreshes no slot)
                    if (interFrame(inBuf[n - 1], loSlot, hiSlotL, 8, std::min(255, pfSettings.qindex + pfLeafOffset),
                                   true, true, frameCount))
                        return 1;
                    dump(spare);
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
                    dump(slotBuf[hiSlotL]);
                    ++predicted; ++frameCount;
                    release(loSlot);
                    return 0;
                }
                const uint64_t mid = (lo + hi) / 2;
                if (mid == lo + 1) {
                    // the frame shown before an overlay is the overlay's LAST reference
                    const bool referenced = hi - mid > 1 || (filtered && hi == base + n);
                    const int midSlot = referenced ? allocSlot() : -1;
                    // an unreferenced shown frame refreshes no slot; its reconstruction lands in the spare buffer
                    if (interFrame(inBuf[mid - base - 1], loSlot, hiSlotL, referenced ? midSlot : 8,
                                   referenced ? qAt(depth) : std::min(255, pfSettings.qindex + pfLeafOffset), true,
                                   true, frameCount))
                        return 1;
                    dump(referenced ? slotBuf[midSlot] : spare);
                    ++predicted; ++frameCount;
                    release(loSlot);
                    return rec(mid, hi, midSlot, hiSlotL, depth + 1);
                }
                const int midSlot = allocSlot();
                if (interFrame(inBuf[mid - base - 1], loSlot, hiSlotL, midSlot, qAt(depth), false, false, 0))
                    return 1;
                if (rec(lo, mid, loSlot, midSlot, depth + 1)) return 1;
                return rec(mid, hi, midSlot, hiSlotL, depth + 1);
            };
            try {
                if (rec(base, base + n, lastShown, hiSlot, 1)) return 1;
            } catch (const std::exception &ex) {
                fprintf(stderr, "mini-GOP: %s\n", ex.what());
                return 1;
            }
            lastShown = hiSlot;
            pending.erase(pending.begin(), pending.begin() + n);
            endPeriod(frameCount - periodBegin);
        }
        if (drain()) return 1;
        stopWriter();
        for (auto &fb : inBuf)
            for (auto *p : fb) cudaFree(p);
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
        fprintf(stderr, "P-frame filters %.1f ms, syntax %.1f ms\n", pfFilterMs, pfWriteMs),
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
