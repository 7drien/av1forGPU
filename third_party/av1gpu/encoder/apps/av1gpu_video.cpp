// Experimental CUDA AV1 video encoder. Input is raw 8-bit planar YUV420;
// all AV1 decisions, reconstruction, filters, and entropy coding use the CUDA backend.
#include "presets.h"
#include "inter_packet.h"
#include "pframe_writer.h"

#include <cuda_runtime.h>

#include <algorithm>
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
        fprintf(stderr, "usage: ffmpeg ... -f rawvideo -pix_fmt yuv420p - | av1gpu-video WIDTH HEIGHT FPS_NUM FPS_DEN OUT.ivf [--fast] [--inter] [--inter-tx8 | --inter-tx-adaptive] [--inter-rdoq] [--persistent-cdf] [--diagonal-motion] [--motion-step 2..16 | --auto-motion-step] [--motion-block 8|16|32|64] [--rolling-reference] [--anchor-reference [--anchor-interval N]] [--lossless-inter-residual | --inter-residual-threshold N | --inter-qindex N] [--keyint 1..10000] [--scene-cut-rms 1..255] [--adaptive-fps] [--grain 0..32] [--qindex 1..255] [--pframes [--pf-qindex N] [--pf-lambda X]]\n");
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
    int pfQindex = 0, pfQoffset = 0;
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
    std::unique_ptr<pfw::Writer> pfWriter;
    pfw::EntropyState pfEntropy;
    std::vector<pf::BlockInfo> pfInfo;
    std::vector<int16_t> pfLevels;
    double pfMs = 0, pfFilterMs = 0, pfWriteMs = 0;
    uint64_t pfModes[4] = {};
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
        pfWriter.reset(new pfw::Writer(w, h, pfAnalyzer.alignedWidth(), pfAnalyzer.alignedHeight()));
        pfSettings.qindex = pfQindex ? pfQindex : std::min(255, std::max(1, cfg.qindex + pfQoffset));
    }

    std::vector<uint8_t> src, padded[3];
    uint64_t frameCount = 0;
    double gpuMs = 0;
    const auto wallStart = std::chrono::steady_clock::now();
    GpEntScratch scratch;
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
        if (usePframes && frameCount && frameCount - lastKey < keyInterval) {
            const uint8_t *refs[1] = {reference};
            const int pitches[3] = {aw, acw, acw};
            char perr[256];
            if (pfAnalyzer.analyze(inputs[0].dsrc, pitches, refs, 1, pfSettings, nextReference, pfInfo, pfLevels,
                                   perr, sizeof perr)) {
                fprintf(stderr, "%s\n", perr);
                return 1;
            }
            pfMs += pfAnalyzer.lastMs();
            const auto tf0 = std::chrono::steady_clock::now();
            pf::FilterParams flt;
            if (pfFilters && pfAnalyzer.filter(inputs[0].dsrc, pitches, nextReference, pfSettings.qindex, pfCdef, true,
                                               flt, perr, sizeof perr)) {
                fprintf(stderr, "%s\n", perr);
                return 1;
            }
            pfw::FrameParams fp;
            for (int k = 0; k < 4; ++k) fp.lfLevel[k] = flt.lf[k];
            fp.lfSharpness = flt.sharpness;
            fp.cdef = pfCdef;
            fp.cdefDamping = flt.damping;
            fp.cdefY = flt.yPri; fp.cdefYSec = flt.ySec; fp.cdefUV = flt.uvPri; fp.cdefUVSec = flt.uvSec;
            fp.qindex = pfSettings.qindex;
            fp.refreshFlags = 1;
            fp.persistent = persistentCdf;
            fp.grain = grain;
            fp.grainSeed = uint16_t(7391 + frameCount * 31847);
            const auto tw0 = std::chrono::steady_clock::now();
            pfFilterMs += std::chrono::duration<double, std::milli>(tw0 - tf0).count();
            std::vector<uint8_t> obu;
            try {
                obu = pfWriter->frame(fp, pfInfo, pfLevels, pfEntropy);
                pfWriteMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
            } catch (const std::exception &ex) {
                fprintf(stderr, "P-frame syntax: %s\n", ex.what());
                return 1;
            }
            for (int k = 0; k < 4; ++k) pfModes[k] += pfWriter->modeCount[k];
            if (!write_ivf_frame(outFile, obu, frameCount)) return 1;
            std::swap(reference, nextReference);
            if (reconDump) {
                std::vector<uint8_t> host(ySize * 3 / 2);
                cudaMemcpy(host.data(), reference, host.size(), cudaMemcpyDeviceToHost);
                fwrite(host.data(), 1, host.size(), reconDump);
            }
            lastWritten = frameCount;
            ++predicted; ++frameCount;
            continue;
        }
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
            if (usePframes) {
                lastWritten = lastKey = frameCount;
                pfEntropy.ready = false;
                pfAnalyzer.resetTemporal();
                if (reconDump) {
                    std::vector<uint8_t> host(ySize * 3 / 2);
                    cudaMemcpy(host.data(), reference, host.size(), cudaMemcpyDeviceToHost);
                    fwrite(host.data(), 1, host.size(), reconDump);
                }
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
    if (usePframes)
        fprintf(stderr, "P-frame filters %.1f ms, syntax %.1f ms\n", pfFilterMs, pfWriteMs),
        fprintf(stderr, "P-frame analysis %.1f ms; modes nearest %llu near %llu global %llu new %llu\n", pfMs,
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
