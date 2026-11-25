#include <map>
#include <string>
#include <chrono>
// CUDA P-frame analysis (see pframe.h).
#include "pframe.h"

#include "dev_common.cuh"
#include "dev_tx.cuh"
#include "rc_inv.cuh"
#include "inter_tables.h"
#include "av1g/gpu_postfilter.h"
#include "gpu_coeff_payload.cuh"
#include "pframe_tile.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_pframe)
const GpHostTables *gp_host_tables();

namespace pf {
namespace {

constexpr int kSizes = 4;       // 8, 16, 32, 64
constexpr int kEntropySlots = Analyzer::kEntropySlots;
constexpr int kMaxRefs = 3;
constexpr float kInf = 3.0e38f;
constexpr int kFullPelPasses = 6;
constexpr int kDeblockCandidates = 5;
struct CdefStrength { int pri, sec; };
constexpr CdefStrength kCdefCandidates[] = {{0, 0}, {1, 0}, {2, 1}, {3, 1}, {4, 2}, {6, 2}, {8, 2}, {10, 4}, {12, 4}};
constexpr int kCdefCount = sizeof kCdefCandidates / sizeof kCdefCandidates[0];

struct FilterSelection {
    int lf[3], cdef[3], lrLuma;
    int cdefBits, cdefPreset[8][2];   // per-superblock CDEF: presets as (luma, chroma) candidate indices
};
// Per-superblock CDEF preset choice state (device): totals of the 9 x 9 candidate pairs, chosen pairs, sums after each.
struct CdefChoice {
    unsigned long long totals[81], totalAfter[9];
    int chosen[8], nCoded;
};

__constant__ int16_t c_filter[6][16][8];
// AV1GPU_PF_PROFILE2: per-launch GPU time by name (synchronizes after every timed launch)
static std::map<std::string, double> g_fineMs;
static double g_bitStats[4];
static struct BitReport { ~BitReport() { const double t = g_bitStats[0] + g_bitStats[1] + g_bitStats[2] + g_bitStats[3]; if (t > 0) fprintf(stderr, "inter bits: modes/partition %.1f%%, mv %.1f%%, coefficients %.1f%%, tx types %.1f%%\n", 100 * g_bitStats[0] / t, 100 * g_bitStats[1] / t, 100 * g_bitStats[2] / t, 100 * g_bitStats[3] / t); } } g_bitReport;
static bool g_fine = getenv("AV1GPU_PF_PROFILE2") != nullptr;
// thread-per-transform 4x4 split search (AV1GPU_PF_SPLIT4=0: the warp-per-transform kernel)
static bool g_split4 = !getenv("AV1GPU_PF_SPLIT4") || atoi(getenv("AV1GPU_PF_SPLIT4")) != 0;
static bool g_split8 = !getenv("AV1GPU_PF_SPLIT8") || atoi(getenv("AV1GPU_PF_SPLIT8")) != 0;
struct FineTimer {
    const char *name; std::chrono::steady_clock::time_point t;
    explicit FineTimer(const char *n) : name(n) { if (g_fine && name) { cudaDeviceSynchronize(); t = std::chrono::steady_clock::now(); } }
    ~FineTimer() { if (g_fine && name) { cudaDeviceSynchronize(); g_fineMs[name] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); } }
};
static struct FineReport { ~FineReport() { for (auto &kv : g_fineMs) fprintf(stderr, "fine %-28s %9.1f ms\n", kv.first.c_str(), kv.second); } } g_fineReport;

struct Cand {
    Mv mvR[kMaxRefs];     // best vector per reference (search)
    float costR[kMaxRefs], bitsR[kMaxRefs];
    Mv mv, mv2;           // chosen prediction
    uint8_t ref;          // 0, 1: single reference; 2: compound
    uint8_t nz;           // coded option has nonzero levels
    uint8_t txSplit;      // luma of the coded option uses four half-size transforms
    uint8_t filter;       // interp_filter: 0 EIGHTTAP, 1 SMOOTH, 2 SHARP
    uint8_t pruned;       // rectangular block of a node whose square evaluation chose skip: not searched
    // split-external evaluation of square blocks (evaluate<.., true> -> evalSplit -> evalFinal)
    float lumaD, lumaBits, dSkipAll;
    uint8_t nzWhole, wholeType;
    float mvBits;         // estimated bits of the chosen vectors
    float costSkip, costCoded;
    uint8_t skipModeTrial;   // mode-refinement trial of skip_mode: EIGHTTAP prediction (no filter search)
    uint8_t trialDepth;      // mode-refinement trial: luma transform depth to keep (0..2), 255 = search
};

struct ModeTrial {
    Cand cand;
    float oldCost, rateSaving;
    bool active;
    bool skipMode;   // the trial is skip_mode: accepted only as a skip block, compound, EIGHTTAP
    // State kept across refinement passes: cost (distortion + coefficient rate, no MV rate) of an accepted trial, and
    // the proposals rejected since (not proposed again unless their rate saving grew), so that a later pass tries the
    // next alternative of the block.
    bool refined;
    float curCost;
    static constexpr int kRej = 4;
    Mv rejMv[kRej][2];
    float rejSaving[kRej];
    int nRej, rejPass;
};

struct Frame {
    const uint8_t *src[3];
    int pitch[3];
    const uint8_t *ref[kMaxRefs];
    int w, h, aw, ah;
    const Mv *tmv[kMaxRefs];   // projected motion field of each reference (8x8 cells, w8 per row), or null
    int w8, h8;
};

__device__ __forceinline__ int round2(int x, int n) { return (x + (1 << (n - 1))) >> n; }

// Motion cost of one vector difference in bits (rough model of the AV1 MV class coding).
__device__ __forceinline__ float mvCompBits(int d) {
    if (!d) return 0.f;
    const float a = fabsf((float)d) * 0.125f;   // pixels
    return 2.5f + 2.0f * __log2f(1.f + 2.f * a);
}
__device__ __forceinline__ float mvBits(Mv mv, Mv pred) {
    const int dr = mv.row - pred.row, dc = mv.col - pred.col;
    if (!dr && !dc) return 1.0f;
    return 3.0f + mvCompBits(dr) + mvCompBits(dc);
}

// Four bytes at any address (two aligned loads and a funnel shift; may read up to 3 bytes past a + 3: the phase planes
// carry 16 bytes of padding).
__device__ __forceinline__ uint32_t ldu32(const uint8_t *a) {
    const uintptr_t ad = reinterpret_cast<uintptr_t>(a);
    const uint32_t *b = reinterpret_cast<const uint32_t *>(ad & ~uintptr_t(3));
    return __funnelshift_r(__ldg(b), __ldg(b + 1), unsigned(ad & 3) * 8);
}
// Horizontal eight-tap pass of an inter prediction (rows iy - 3 .. iy + h + 3, Round2(sum, 3) into tmp, row stride w).
// Blocks whose taps stay inside the plane (with 12 samples of margin on the right for the word loads) take four outputs
// per lane step from three word loads; others clamp every tap.
__device__ __forceinline__ void interpH(const uint8_t *ref, int pw, int ph, int ix, int iy, int w, int lw, int h,
                                        const int16_t *filt, int16_t *tmp) {
    const int lane = threadIdx.x & 31, ih = h + 7;
    int f[8];
#pragma unroll
    for (int t = 0; t < 8; t++) f[t] = filt[t];
    if (ix >= 3 && ix + w + 12 <= pw && iy >= 3 && iy + h + 4 <= ph) {
        const int lq = lw - 2, qw = 1 << lq;
        for (int u = lane; u < ih << lq; u += 32) {
            const int r = u >> lq, c = (u & (qw - 1)) << 2;
            const uint8_t *sp = ref + (iy + r - 3) * pw + ix + c - 3;
            const uint32_t w0 = ldu32(sp), w1 = ldu32(sp + 4), w2 = ldu32(sp + 8);
            int b[11];
#pragma unroll
            for (int k = 0; k < 4; k++) {
                b[k] = (w0 >> (8 * k)) & 255;
                b[4 + k] = (w1 >> (8 * k)) & 255;
                if (k < 3) b[8 + k] = (w2 >> (8 * k)) & 255;
            }
#pragma unroll
            for (int k = 0; k < 4; k++) {
                int sum = 0;
#pragma unroll
                for (int t = 0; t < 8; t++) sum += f[t] * b[k + t];
                tmp[r * w + c + k] = (int16_t)round2(sum, 3);
            }
        }
    } else {
        const int lastX = pw - 1, lastY = ph - 1;
        for (int i = lane; i < ih * w; i += 32) {
            const int r = i >> lw, c = i & (w - 1);
            const uint8_t *row = ref + min(max(iy + r - 3, 0), lastY) * pw;
            int sum = 0;
#pragma unroll
            for (int t = 0; t < 8; t++) sum += f[t] * row[min(max(ix + c + t - 3, 0), lastX)];
            tmp[i] = (int16_t)round2(sum, 3);
        }
    }
    __syncwarp();
}
// Vertical eight-tap pass: four output rows per lane step from eleven intermediate rows; emit(i, sum) per output.
template <class Emit>
__device__ __forceinline__ void interpV(const int16_t *tmp, int w, int lw, int h, const int16_t *filt, Emit emit) {
    const int lane = threadIdx.x & 31;
    int f[8];
#pragma unroll
    for (int t = 0; t < 8; t++) f[t] = filt[t];
    for (int u = lane; u < (w * h) >> 2; u += 32) {
        const int c = u & (w - 1), r0 = (u >> lw) << 2;
        int v[11];
#pragma unroll
        for (int k = 0; k < 11; k++) v[k] = tmp[(r0 + k) * w + c];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            int sum = 0;
#pragma unroll
            for (int t = 0; t < 8; t++) sum += f[t] * v[k + t];
            emit((r0 + k) * w + c, sum);
        }
    }
    __syncwarp();
}
// Out-of-line copies for the 64-wide evaluations (inlined there they push the kernels past 255 registers).
__device__ __noinline__ void interpHCall(const uint8_t *ref, int pw, int ph, int ix, int iy, int w, int lw, int h,
                                         const int16_t *filt, int16_t *tmp) {
    interpH(ref, pw, ph, ix, iy, w, lw, h, filt, tmp);
}
template <class Emit>
__device__ __noinline__ void interpVCall(const int16_t *tmp, int w, int lw, int h, const int16_t *filt, Emit emit) {
    interpV(tmp, w, lw, h, filt, emit);
}
template <bool OUT>
__device__ __forceinline__ void interpHSel(const uint8_t *ref, int pw, int ph, int ix, int iy, int w, int lw, int h,
                                           const int16_t *filt, int16_t *tmp) {
    if constexpr (OUT) interpHCall(ref, pw, ph, ix, iy, w, lw, h, filt, tmp);
    else interpH(ref, pw, ph, ix, iy, w, lw, h, filt, tmp);
}
template <bool OUT, class Emit>
__device__ __forceinline__ void interpVSel(const int16_t *tmp, int w, int lw, int h, const int16_t *filt, Emit emit) {
    if constexpr (OUT) interpVCall(tmp, w, lw, h, filt, emit);
    else interpV(tmp, w, lw, h, filt, emit);
}

// ------------------------------------------------------------------------------------------------
// Exact AV1 single-reference inter prediction (spec 7.11.3.3 / 7.11.3.4, unscaled reference, regular
// eight-tap filter; blocks of width or height 4 use the four-tap variant). One warp predicts a w x h block.
// tmp holds (h + 7) * w intermediate values.
// Subpel_Filters row of interp_filter f (0 EIGHTTAP, 1 SMOOTH, 2 SHARP) for a block side of n samples: sides of 4 use
// the four-tap rows (4: regular and sharp, 5: smooth), spec 7.11.3.4.
__device__ __forceinline__ int filterRow(int f, int n) { return n <= 4 ? (f == 1 ? 5 : 4) : f; }
template <bool OUT = false>
__device__ void warpPredict(const uint8_t *ref, int pw, int ph, int x0, int y0, int w, int h, Mv mv, int sub,
                            int16_t *tmp, uint8_t *out, int filt = 0, int hf = -1) {
    const int lane = threadIdx.x & 31;
    const int lw = 31 - __clz(w);   // block widths are powers of two
    const int X = (x0 << 4) + ((2 * mv.col) >> sub);
    const int Y = (y0 << 4) + ((2 * mv.row) >> sub);
    const int ix = X >> 4, fx = X & 15, iy = Y >> 4, fy = Y & 15;
    const int fh = filterRow(filt, w), fv = filterRow(filt, hf < 0 ? h : hf);   // hf: height of the whole block (stripes)
    const int lastX = pw - 1, lastY = ph - 1;
    if (!fx && !fy) {
        for (int i = lane; i < w * h; i += 32) {
            const int r = i >> lw, c = i & (w - 1);
            out[i] = ref[min(max(iy + r, 0), lastY) * pw + min(max(ix + c, 0), lastX)];
        }
        __syncwarp();
        return;
    }
    interpHSel<OUT>(ref, pw, ph, ix, iy, w, lw, h, c_filter[fh][fx], tmp);
    interpVSel<OUT>(tmp, w, lw, h, c_filter[fv][fy], [&](int i, int sum) { out[i] = (uint8_t)min(max(round2(sum, 11), 0), 255); });
}

// Exact AV1 compound-average prediction (spec 7.11.3.1 with InterRound1 = 7 and InterPostRound = 4):
// Clip1(Round2(p0 + p1, 5)). hp holds w * h int16 values.
template <bool OUT = false>
__device__ void warpPredictCompound(const uint8_t *ref0, const uint8_t *ref1, int pw, int ph, int x0, int y0, int w,
                                    int h, Mv mv0, Mv mv1, int sub, int16_t *tmp, int16_t *hp, uint8_t *out, int filt = 0,
                                    int hf = -1) {
    const int lane = threadIdx.x & 31;
    const int lw = 31 - __clz(w);   // block widths are powers of two
    const int fh = filterRow(filt, w), fv = filterRow(filt, hf < 0 ? h : hf);
    for (int k = 0; k < 2; k++) {
        const uint8_t *ref = k ? ref1 : ref0;
        const Mv mv = k ? mv1 : mv0;
        const int X = (x0 << 4) + ((2 * mv.col) >> sub);
        const int Y = (y0 << 4) + ((2 * mv.row) >> sub);
        const int ix = X >> 4, fx = X & 15, iy = Y >> 4, fy = Y & 15;
        interpHSel<OUT>(ref, pw, ph, ix, iy, w, lw, h, c_filter[fh][fx], tmp);
        if (!k) interpVSel<OUT>(tmp, w, lw, h, c_filter[fv][fy], [&](int i, int sum) { hp[i] = (int16_t)round2(sum, 7); });
        else interpVSel<OUT>(tmp, w, lw, h, c_filter[fv][fy], [&](int i, int sum) {
            out[i] = (uint8_t)min(max(round2(hp[i] + round2(sum, 7), 5), 0), 255);
        });
    }
}

// ------------------------------------------------------------------------------------------------
__global__ void downsample4(const uint8_t *src, int pitch, int w, int h, uint8_t *dst, int dw, int dh) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dw || y >= dh) return;
    int s = 0;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) s += src[min(4 * y + r, h - 1) * pitch + min(4 * x + c, w - 1)];
    dst[y * dw + x] = (uint8_t)((s + 8) >> 4);
}

// Exhaustive search of 8x8 blocks on the quarter-resolution luma (32x32 full-resolution areas). Each lane scores four
// horizontally adjacent candidates at once with packed byte SADs (__vsadu4) on word-aligned window rows; the costs and the
// choice (lowest cost, then lowest candidate index) are those of a plain per-candidate search.
constexpr int kCoarseWarps = 4;
constexpr int kCoarseMaxR = 48;
constexpr int kCoarseStride = ((8 + 2 * kCoarseMaxR + 3) & ~3) + 4;   // window row pitch: word aligned, 4 bytes slack
__global__ void coarseSearch(const uint8_t *cur, const uint8_t *ref, int dw, int dh, int R, int cols, int rows,
                             float lambda, Mv *out, float *outCost = nullptr) {
    __shared__ __align__(16) uint8_t win[kCoarseWarps][kCoarseStride * (8 + 2 * kCoarseMaxR)];
    __shared__ __align__(16) uint8_t blk[kCoarseWarps][64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * kCoarseWarps + warp;
    if (b >= cols * rows) return;
    const int bx = b % cols, by = b / cols, x0 = bx * 8, y0 = by * 8;
    const int ws = 8 + 2 * R, stride = ((ws + 3) & ~3) + 4;
    for (int i = lane; i < 64; i += 32)
        blk[warp][i] = cur[min(y0 + i / 8, dh - 1) * dw + min(x0 + i % 8, dw - 1)];
    for (int i = lane; i < ws * stride; i += 32) {
        const int r = i / stride, c = i - r * stride;
        win[warp][i] = ref[min(max(y0 - R + r, 0), dh - 1) * dw + min(max(x0 - R + c, 0), dw - 1)];
    }
    __syncwarp();
    uint32_t q[16];
    const uint32_t *b32 = reinterpret_cast<const uint32_t *>(blk[warp]);
#pragma unroll
    for (int i = 0; i < 16; i++) q[i] = b32[i];
    const int side = 2 * R + 1, groups = (side + 3) >> 2, words = stride >> 2;
    float best = kInf;
    int bestIdx = R * side + R;
    for (int k = lane; k < side * groups; k += 32) {
        const int dy = k / groups, dx0 = (k - dy * groups) * 4;
        const uint32_t *w = reinterpret_cast<const uint32_t *>(win[warp] + dy * stride + dx0);
        int s0 = 0, s1 = 0, s2 = 0, s3 = 0;
#pragma unroll
        for (int r = 0; r < 8; r++) {
            const uint32_t a = w[r * words], c = w[r * words + 1], e = w[r * words + 2];
            s0 += __vsadu4(q[2 * r], a) + __vsadu4(q[2 * r + 1], c);
            s1 += __vsadu4(q[2 * r], __funnelshift_r(a, c, 8)) + __vsadu4(q[2 * r + 1], __funnelshift_r(c, e, 8));
            s2 += __vsadu4(q[2 * r], __funnelshift_r(a, c, 16)) + __vsadu4(q[2 * r + 1], __funnelshift_r(c, e, 16));
            s3 += __vsadu4(q[2 * r], __funnelshift_r(a, c, 24)) + __vsadu4(q[2 * r + 1], __funnelshift_r(c, e, 24));
        }
        const int sads[4] = {s0, s1, s2, s3};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int dx = dx0 + j;
            if (dx >= side) break;
            const float cost = (float)sads[j] + lambda * (float)(abs(dx - R) + abs(dy - R));
            const int idx = dy * side + dx;
            if (cost < best || (cost == best && idx < bestIdx)) { best = cost; bestIdx = idx; }
        }
    }
    for (int o = 16; o; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bestIdx, o);
        if (ob < best || (ob == best && oi < bestIdx)) { best = ob; bestIdx = oi; }
    }
    if (!lane) {
        const int dy = bestIdx / side - R, dx = bestIdx % side - R;
        out[b] = Mv{(int16_t)(dy * 4 * 8), (int16_t)(dx * 4 * 8)};
        if (outCost) outCost[b] = best;
    }
}
// Second stage of the hierarchical coarse search: each quarter-resolution block also searches +-kCoarseAroundR around the
// vector of its 128 x 128 area found on the 1/16-resolution planes (camera pans move content further than the
// zero-centred window reaches over the 8 to 16 frames to an ALTREF), with the same cost (SAD + lambda * |offset|); the
// result replaces the zero-centred one when it is cheaper.
constexpr int kCoarseAroundR = 8;
__global__ void coarseAround(const uint8_t *cur, const uint8_t *ref, int dw, int dh, int cols, int rows, float lambda,
                             const Mv *wide, int wcols, int wrows, Mv *out, const float *cost, int R0) {
    constexpr int R = kCoarseAroundR, side = 2 * R + 1, ws = 8 + 2 * R, stride = ((ws + 3) & ~3) + 4;
    __shared__ __align__(16) uint8_t win[kCoarseWarps][stride * ws];
    __shared__ __align__(16) uint8_t blk[kCoarseWarps][64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * kCoarseWarps + warp;
    if (b >= cols * rows) return;
    const int bx = b % cols, by = b / cols, x0 = bx * 8, y0 = by * 8;
    // the vectors of the four 128 x 128 areas nearest the block (its own and the neighbours towards its quadrant)
    const int ax = bx / 4, ay = by / 4, nx = (bx & 3) < 2 ? ax - 1 : ax + 1, ny = (by & 3) < 2 ? ay - 1 : ay + 1;
    int cy[4], cx[4], nc = 0;
    for (int k = 0; k < 4; k++) {
        const int qx = min(max(k & 1 ? nx : ax, 0), wcols - 1), qy = min(max(k & 2 ? ny : ay, 0), wrows - 1);
        const Mv wm = wide[qy * wcols + qx];
        const int y = wm.row / 8, x = wm.col / 8;   // 1/16-resolution vectors are stored as quarter-resolution ones: x4
        if (abs(y) + R <= R0 && abs(x) + R <= R0) continue;   // inside the zero-centred window: already priced there
        bool dup = false;
        for (int j = 0; j < nc; j++) dup |= cy[j] == y && cx[j] == x;
        if (dup) continue;
        cy[nc] = y; cx[nc] = x; nc++;
    }
    if (!nc) return;
    for (int i = lane; i < 64; i += 32) blk[warp][i] = cur[min(y0 + i / 8, dh - 1) * dw + min(x0 + i % 8, dw - 1)];
    float best = cost[b];
    int bestY = 0, bestX = 0;
    bool found = false;
    for (int j = 0; j < nc; j++) {
        __syncwarp();
        for (int i = lane; i < ws * stride; i += 32) {
            const int r = i / stride, c = i - r * stride;
            win[warp][i] = ref[min(max(y0 + cy[j] - R + r, 0), dh - 1) * dw + min(max(x0 + cx[j] - R + c, 0), dw - 1)];
        }
        __syncwarp();
        float bj = kInf;
        int bi = 0;
        for (int k = lane; k < side * side; k += 32) {
            const int dy = k / side, dx = k % side;
            int sad = 0;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++) sad += abs((int)blk[warp][r * 8 + c] - (int)win[warp][(dy + r) * stride + dx + c]);
            const float cst = (float)sad + lambda * (float)(abs(cx[j] + dx - R) + abs(cy[j] + dy - R));
            if (cst < bj || (cst == bj && k < bi)) { bj = cst; bi = k; }
        }
        for (int o = 16; o; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, bj, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob < bj || (ob == bj && oi < bi)) { bj = ob; bi = oi; }
        }
        if (bj < best) { best = bj; bestY = cy[j] + bi / side - R; bestX = cx[j] + bi % side - R; found = true; }
    }
    if (!lane && found) out[b] = Mv{(int16_t)(bestY * 4 * 8), (int16_t)(bestX * 4 * 8)};
}

// Quarter-sample phase planes of a reference luma plane: phase (fy / 4) * 4 + fx / 4 holds the exact AV1 prediction
// sample at every integer position (two-stage filter with the spec's intermediate rounding and edge clamping).
// Quarter-sample phase planes of a reference for the motion search: the four horizontal phases of each sample (one
// thread, one tap window), then the four vertical phases of eight rows of each horizontal phase (one thread, one
// 15-row window).
__global__ void horizontalPhases(const uint8_t *ref, int w, int h, int16_t *out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const uint8_t *row = ref + size_t(y) * w;
    int v[8];
#pragma unroll
    for (int t = 0; t < 8; t++) v[t] = row[min(max(x + t - 3, 0), w - 1)];
#pragma unroll
    for (int fx = 0; fx < 4; fx++) {
        int s = 0;
#pragma unroll
        for (int t = 0; t < 8; t++) s += c_filter[0][fx * 4][t] * v[t];
        out[(size_t(fx) * h + y) * w + x] = (int16_t)round2(s, 3);
    }
}
constexpr int kPhaseRows = 8;
__global__ void verticalPhases(const int16_t *hp, int w, int h, uint8_t *out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y0 = (blockIdx.y * blockDim.y + threadIdx.y) * kPhaseRows;
    const int fx = blockIdx.z;
    if (x >= w || y0 >= h) return;
    const int16_t *col = hp + size_t(fx) * h * w + x;
    int v[kPhaseRows + 7];
#pragma unroll
    for (int k = 0; k < kPhaseRows + 7; k++) v[k] = col[size_t(min(max(y0 + k - 3, 0), h - 1)) * w];
#pragma unroll
    for (int fy = 0; fy < 4; fy++) {
        uint8_t *o = out + size_t(fy * 4 + fx) * h * w + x;
#pragma unroll
        for (int r = 0; r < kPhaseRows; r++) {
            if (y0 + r >= h) break;
            int s = 0;
#pragma unroll
            for (int t = 0; t < 8; t++) s += c_filter[0][fy * 4][t] * v[r + t];
            o[size_t(y0 + r) * w] = (uint8_t)min(max(round2(s, 11), 0), 255);
        }
    }
}

// Lanes cooperating on one block: 8 for 8x8 blocks (4 blocks per warp), else the whole warp.
// Lanes cooperating on one block in motion search: about eight four-sample SAD words per lane (8 lanes up to 256
// samples, 16 for 512, 32 for 1024), 128 (four warps, one block per CTA, sums through shared memory) for the 2048- and
// 4096-sample blocks (their SADs are memory-latency bound). Four or sixteen words per lane, 32 or 128 lanes for other
// sizes measured slower.
template <int BW, int BH> struct Grp {
    static constexpr int words = BW * BH / 4;   // four-sample SAD words per block
    static constexpr int G = BW * BH >= 2048 ? 128 : words / 8 <= 8 ? 8 : words / 8 >= 32 ? 32 : words / 8;
    __device__ static int lane() { return threadIdx.x & (G - 1); }
    __device__ static unsigned mask() {
        return G >= 32 ? 0xffffffffu : ((1u << G) - 1) << ((threadIdx.x & 31) & ~(G - 1));
    }
    __device__ static int sum(int v) {
        if constexpr (G <= 32) {
            const unsigned m = mask();
#pragma unroll
            for (int o = G / 2; o; o >>= 1) v += __shfl_xor_sync(m, v, o);
            return v;
        } else {
            __shared__ int part[G / 32];
#pragma unroll
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
            __syncthreads();
            int t = 0;
#pragma unroll
            for (int k = 0; k < G / 32; k++) t += part[k];
            __syncthreads();
            return t;
        }
    }
};
template <int V> struct Log2 { static constexpr int v = V <= 1 ? 0 : 1 + Log2<V / 2>::v; };
template <> struct Log2<1> { static constexpr int v = 0; };

template <int BW, int BH>
__device__ __forceinline__ int groupSadPlane(const uint8_t *src, int pitch, const uint8_t *plane, int w, int h, int x0,
                                             int y0, int dx, int dy) {
    constexpr int L = Log2<BW>::v, G = Grp<BW, BH>::G;
    const int lane = Grp<BW, BH>::lane();
    int s = 0;
    const bool inside = x0 + dx >= 0 && y0 + dy >= 0 && x0 + dx + BW <= w && y0 + dy + BH <= h && x0 + BW <= w && y0 + BH <= h;
    const uint8_t *q = src + size_t(y0) * pitch + x0;
    if (inside && !((reinterpret_cast<uintptr_t>(q) | unsigned(pitch)) & 3)) {   // four samples per lane step
        const uint8_t *p = plane + size_t(y0 + dy) * w + x0 + dx;
#pragma unroll 4
        for (int u = lane; u < BW * BH / 4; u += G) {
            const int i = u * 4, r = i >> L, c = i & (BW - 1);
            s = __vsadu4(*reinterpret_cast<const uint32_t *>(q + r * pitch + c), ldu32(p + r * w + c)) + s;
        }
    } else if (inside) {
        const uint8_t *p = plane + size_t(y0 + dy) * w + x0 + dx;
#pragma unroll 4
        for (int i = lane; i < BW * BH; i += G) {
            const int r = i >> L, c = i & (BW - 1);
            s += abs((int)q[r * pitch + c] - (int)p[r * w + c]);
        }
    } else {
        for (int i = lane; i < BW * BH; i += G) {
            const int r = i >> L, c = i & (BW - 1);
            const int x = x0 + c, y = y0 + r;
            if (x >= w || y >= h) continue;
            const int rx = min(max(x + dx, 0), w - 1), ry = min(max(y + dy, 0), h - 1);
            s += abs((int)src[y * pitch + x] - (int)plane[size_t(ry) * w + rx]);
        }
    }
    return Grp<BW, BH>::sum(s);
}

// SAD against source samples cached in registers, four per word (sv[k] = samples 4 * (lane + k * G) .. + 3 of the
// block, area <= 1024).
template <int BW, int BH>
__device__ __forceinline__ int groupSadCached(const uint32_t *sv, const uint8_t *src, int pitch, const uint8_t *plane,
                                              int w, int h, int x0, int y0, int dx, int dy) {
    constexpr int L = Log2<BW>::v, G = Grp<BW, BH>::G, PL = BW * BH / (4 * G);
    const int lane = Grp<BW, BH>::lane();
    const bool inside = x0 + dx >= 0 && y0 + dy >= 0 && x0 + dx + BW <= w && y0 + dy + BH <= h && x0 + BW <= w && y0 + BH <= h;
    if (!inside) return groupSadPlane<BW, BH>(src, pitch, plane, w, h, x0, y0, dx, dy);
    const uint8_t *p = plane + size_t(y0 + dy) * w + x0 + dx;
    int s = 0;
#pragma unroll
    for (int k = 0; k < PL; k++) {
        const int i = 4 * (lane + k * G), r = i >> L, c = i & (BW - 1);
        s = __vsadu4(sv[k], ldu32(p + r * w + c)) + s;
    }
    return Grp<BW, BH>::sum(s);
}

// Motion refinement per block shape from coarse, parent, temporal and zero candidates: full-pel search, then half and
// quarter sample steps on the phase planes. parentSize: side of the square block whose vector predicts this one.
template <int BW, int BH>
struct RefineCfg {
    static constexpr int warps = BW * BH >= 1024 ? 4 : 8;
    static constexpr int per() { return warps * 32 / Grp<BW, BH>::G; }   // blocks per CTA
};
template <int BW, int BH>
__global__ void __launch_bounds__(32 * RefineCfg<BW, BH>::warps) refine(Frame f, int refIdx, const uint8_t *phases,
                                                                        const Mv *coarse, int ccols, int crows,
                                                                        const Cand *parent, int pcols, int parentSize,
                                                                        const BlockInfo *temporal, int tcols, Cand *out,
                                                                        int cols, int rows, float lambda,
                                                                        bool pruneable = false,
                                                                        int fullPelPasses = kFullPelPasses) {
    using GR = Grp<BW, BH>;
    const int b = blockIdx.x * RefineCfg<BW, BH>::per() + threadIdx.x / GR::G;
    if (b >= cols * rows) return;
    const int x0 = (b % cols) * BW, y0 = (b / cols) * BH;
    Cand &o = out[b];
    if (x0 >= f.w || y0 >= f.h) {
        if (GR::lane() == 0) { o.mvR[refIdx] = Mv{0, 0}; o.costR[refIdx] = kInf; o.bitsR[refIdx] = 0.f; }
        return;
    }
    if (pruneable && o.pruned) {
        if (GR::lane() == 0) { o.mvR[refIdx] = Mv{0, 0}; o.costR[refIdx] = kInf; o.bitsR[refIdx] = 0.f; }
        return;
    }
    const uint8_t *src = f.src[0];
    const int pitch = f.pitch[0];
    const size_t planeSize = size_t(f.w) * f.h;
    int cand[12][2], n = 0;
    auto add = [&](Mv m) {
        cand[n][0] = (m.row + (m.row >= 0 ? 4 : -4)) / 8;
        cand[n][1] = (m.col + (m.col >= 0 ? 4 : -4)) / 8;
        n++;
    };
    const int cx = min((x0 + BW / 2) / 32, ccols - 1), cy = min((y0 + BH / 2) / 32, crows - 1);
    const Mv center = coarse[cy * ccols + cx];
    add(center);
    add(Mv{0, 0});
    if (cx > 0) add(coarse[cy * ccols + cx - 1]);
    if (cx + 1 < ccols) add(coarse[cy * ccols + cx + 1]);
    if (cy > 0) add(coarse[(cy - 1) * ccols + cx]);
    if (cy + 1 < crows) add(coarse[(cy + 1) * ccols + cx]);
    Mv pred = center;
    if (parent) {
        const Cand &pc = parent[(y0 / parentSize) * pcols + x0 / parentSize];
        add(pc.mvR[refIdx]);
        pred = pc.mvR[refIdx];
    }
    if (temporal) add(temporal[(y0 / 8) * tcols + x0 / 8].mv);
    // the decoder's temporal candidate at the block's first 8x8 (use_ref_frame_mvs): a search start and a second
    // predictor, so vectors equal to it cost what NEAREST / NEAR costs
    Mv tpred{0, 0};
    bool hasT = false;
    if (f.tmv[refIdx] && y0 / 8 < f.h8 && x0 / 8 < f.w8) {
        Mv t = f.tmv[refIdx][(y0 / 8) * f.w8 + x0 / 8];
        if (t.row != -32768) {
            if (t.row & 1) t.row += t.row > 0 ? -1 : 1;
            if (t.col & 1) t.col += t.col > 0 ? -1 : 1;
            tpred = t;
            hasT = true;
            add(t);
        }
    }
    constexpr int CL = BW * BH <= 1024 ? BW * BH / (4 * GR::G) : 1;   // cached source words (4 samples) per lane
    uint32_t sv[CL];
    if constexpr (BW * BH <= 1024) {
        constexpr int L = Log2<BW>::v;
#pragma unroll
        for (int k = 0; k < CL; k++) {
            const int i = 4 * (GR::lane() + k * GR::G), r = i >> L, c = i & (BW - 1);
            uint32_t v = 0;
#pragma unroll
            for (int t = 0; t < 4; t++)
                if (x0 + c + t < f.w && y0 + r < f.h) v |= uint32_t(src[(y0 + r) * pitch + x0 + c + t]) << (8 * t);
            sv[k] = v;
        }
    }
    auto costOf = [&](Mv m) {
        const int ph = (((2 * m.row) & 15) >> 2) * 4 + (((2 * m.col) & 15) >> 2);
        const uint8_t *plane = phases + ph * planeSize;
        int sad;
        if constexpr (BW * BH <= 1024)
            sad = groupSadCached<BW, BH>(sv, src, pitch, plane, f.w, f.h, x0, y0, m.col >> 3, m.row >> 3);
        else
            sad = groupSadPlane<BW, BH>(src, pitch, plane, f.w, f.h, x0, y0, m.col >> 3, m.row >> 3);
        return (float)sad + lambda * (hasT ? fminf(mvBits(m, pred), mvBits(m, tpred)) : mvBits(m, pred));
    };
    Mv bm{0, 0};
    float bc = kInf;
    auto tryMv = [&](Mv m) {
        if (abs(m.row) >= (1 << 14) - 8 || abs(m.col) >= (1 << 14) - 8) return;
        const float c = costOf(m);
        if (c < bc) { bc = c; bm = m; }
    };
    // repeated vectors cannot change the choice (ties keep the first): duplicate candidates are skipped, and so are
    // the neighbours of a new centre that were neighbours of the previous one
    for (int i = 0; i < n; i++) {
        bool seen = false;
        for (int j = 0; j < i; j++) seen |= cand[j][0] == cand[i][0] && cand[j][1] == cand[i][1];
        if (!seen) tryMv(Mv{(int16_t)(cand[i][0] * 8), (int16_t)(cand[i][1] * 8)});
    }
    Mv prev{0, 0};
    bool havePrev = false;
    for (int pass = 0; pass < fullPelPasses; pass++) {
        const Mv c0 = bm;
        for (int k = 0; k < 9; k++) {
            if (k == 4) continue;
            const Mv m{(int16_t)(c0.row + (k / 3 - 1) * 8), (int16_t)(c0.col + (k % 3 - 1) * 8)};
            if (havePrev && abs(m.row - prev.row) <= 8 && abs(m.col - prev.col) <= 8) continue;
            tryMv(m);
        }
        if (bm == c0) break;
        prev = c0;
        havePrev = true;
    }
    for (int step = 4; step >= 2; step >>= 1) {
        const Mv c0 = bm;
        for (int k = 0; k < 9; k++)
            if (k != 4) tryMv(Mv{(int16_t)(c0.row + (k / 3 - 1) * step), (int16_t)(c0.col + (k % 3 - 1) * step)});
    }
    if (hasT) tryMv(tpred);
    if (GR::lane() == 0) {
        o.mvR[refIdx] = bm;
        o.costR[refIdx] = bc;
        o.bitsR[refIdx] = hasT ? fminf(mvBits(bm, pred), mvBits(bm, tpred)) : mvBits(bm, pred);
    }
}



// Choose single reference 0, single reference 1, or their compound average by SAD + motion cost on the phase planes.
template <int BW, int BH>
__global__ void __launch_bounds__(32 * RefineCfg<BW, BH>::warps) chooseInter(Frame f, int refCount, const uint8_t *phases0,
                                                                             const uint8_t *phases1, Cand *cand,
                                                                             int cols, int rows, float lambda,
                                                                             float compoundBias) {
    using GR = Grp<BW, BH>;
    constexpr int L = Log2<BW>::v;
    const int lane = GR::lane();
    const int b = blockIdx.x * RefineCfg<BW, BH>::per() + threadIdx.x / GR::G;
    if (b >= cols * rows) return;
    Cand &c = cand[b];
    if (BW != BH && c.pruned) return;
    const int x0 = (b % cols) * BW, y0 = (b / cols) * BH;
    int choice = 0;
    float bits = c.bitsR[0];
    if (refCount > 1 && x0 < f.w && y0 < f.h) {
        // reference signalling: about one more bit for the second reference, two for compound
        const float c0 = c.costR[0], c1 = c.costR[1] + lambda * 1.0f;
        const Mv m0 = c.mvR[0], m1 = c.mvR[1];
        const size_t planeSize = size_t(f.w) * f.h;
        const uint8_t *p0 = phases0 + ((((2 * m0.row) & 15) >> 2) * 4 + (((2 * m0.col) & 15) >> 2)) * planeSize;
        const uint8_t *p1 = phases1 + ((((2 * m1.row) & 15) >> 2) * 4 + (((2 * m1.col) & 15) >> 2)) * planeSize;
        const int dx0 = m0.col >> 3, dy0 = m0.row >> 3, dx1 = m1.col >> 3, dy1 = m1.row >> 3;
        int sad = 0;
        for (int i = lane; i < BW * BH; i += GR::G) {
            const int r = i >> L, col = i & (BW - 1);
            const int x = x0 + col, y = y0 + r;
            if (x >= f.w || y >= f.h) continue;
            const int pa = p0[size_t(min(max(y + dy0, 0), f.h - 1)) * f.w + min(max(x + dx0, 0), f.w - 1)];
            const int pb = p1[size_t(min(max(y + dy1, 0), f.h - 1)) * f.w + min(max(x + dx1, 0), f.w - 1)];
            sad += abs((int)f.src[0][y * f.pitch[0] + x] - ((pa + pb + 1) >> 1));
        }
        sad = GR::sum(sad);
        const float cc = (float)sad + lambda * (c.bitsR[0] + c.bitsR[1] + compoundBias);
        float best = c0;
        if (c1 < c0 && c1 <= cc) { choice = 1; bits = c.bitsR[1] + 1.0f; best = c1; }
        else if (cc < c0) { choice = 2; bits = c.bitsR[0] + c.bitsR[1] + compoundBias; best = cc; }
        // third reference (single prediction only): about two more bits of reference signalling
        if (refCount > 2 && c.costR[2] + lambda * 2.0f < best) { choice = 3; bits = c.bitsR[2] + 2.0f; }
    }
    if (!lane) {
        c.ref = (uint8_t)choice;
        c.mv = choice == 1 ? c.mvR[1] : choice == 3 ? c.mvR[2] : c.mvR[0];
        c.mv2 = c.mvR[1];
        c.mvBits = bits;
        c.skipModeTrial = 0;
        c.trialDepth = 255;
    }
}

// ------------------------------------------------------------------------------------------------
// AV1 transform size of a w x h transform (log2 sides 2..6).
__host__ __device__ inline int txSizeOf(int lw, int lh) {
    for (int t = 0; t < 19; t++)
        if (AV1G_T(Tx_Width_Log2)[t] == lw && AV1G_T(Tx_Height_Log2)[t] == lh) return t;
    return -1;
}
// Exact inverse transform + add (DCT_DCT, square or 2:1), warp-cooperative. Halves the levels while the coefficients
// violate the conformance ranges. q: min(H,32) x min(W,32) levels; recon = Clip1(pred + residual), W x H.
template <int LW, int LH>
__device__ void warpInverseAdd(int16_t *q, int txSz, int txType, int dcQ, int acQ, int16_t *res, const uint8_t *pred,
                               uint8_t *recon) {
    constexpr int W = 1 << LW, H = 1 << LH, TW = W > 32 ? 32 : W, TH = H > 32 ? 32 : H;
    constexpr bool rect = LW - LH == 1 || LH - LW == 1;
    const int lane = threadIdx.x & 31;
    const int den = av1g_dq_denom(txSz);
    const int rk = av1g_row_kind(txType), ck = av1g_col_kind(txType);
    const int rowShift = AV1G_T(Transform_Row_Shift)[txSz];
    for (;;) {
        int any = 0;
        for (int i = lane; i < TW * TH; i += 32) any |= q[i];
        if (!__any_sync(0xffffffffu, any)) {
            for (int i = lane; i < W * H; i += 32) recon[i] = pred[i];
            __syncwarp();
            return;
        }
        uint32_t err = 0;
        for (int i = lane; i < H; i += 32) err |= rci::row<LW, 8>(i, TH, TW, q, dcQ, acQ, nullptr, den, rect, rk, rowShift, res);
        __syncwarp();
        for (int j = lane; j < W; j += 32) err |= rci::col<LH>(j, W, ck, res);
        __syncwarp();
        if (!__any_sync(0xffffffffu, err != 0)) break;
        for (int i = lane; i < TW * TH; i += 32) q[i] = (int16_t)(q[i] / 2);
        __syncwarp();
    }
    // flipped types: Residual[i][j] lands at (flipUD ? H-1-i : i, flipLR ? W-1-j : j) (spec 7.13.3 step 3)
    const int fud = av1g_flip_ud(txType), flr = av1g_flip_lr(txType);
    for (int i = lane; i < W * H; i += 32) {
        const int r = i >> LW, c = i & (W - 1);
        const int src = (fud ? H - 1 - r : r) * W + (flr ? W - 1 - c : c);
        recon[i] = (uint8_t)min(max((int)pred[i] + res[src], 0), 255);
    }
    __syncwarp();
}
__device__ __noinline__ void inverseAddAny(int lw, int lh, int16_t *q, int txSz, int txType, int dcQ, int acQ, int16_t *res,
                              const uint8_t *pred, uint8_t *recon) {
#define PF_INV(A, B) if (lw == A && lh == B) { warpInverseAdd<A, B>(q, txSz, txType, dcQ, acQ, res, pred, recon); return; }
    PF_INV(2, 2) PF_INV(3, 3) PF_INV(4, 4) PF_INV(5, 5) PF_INV(6, 6)
    PF_INV(3, 2) PF_INV(2, 3) PF_INV(4, 3) PF_INV(3, 4) PF_INV(5, 4) PF_INV(4, 5) PF_INV(6, 5) PF_INV(5, 6)
    PF_INV(4, 2) PF_INV(2, 4) PF_INV(5, 3) PF_INV(3, 5) PF_INV(6, 4) PF_INV(4, 6)
#undef PF_INV
}
// Compile-time size dispatch of the inverse (no ABI call): out-of-range sizes only occur in dead branches.
template <int LW, int LH>
__device__ __forceinline__ void inverseAddCt(int16_t *q, int txSz, int txType, int dcQ, int acQ, int16_t *res,
                                             const uint8_t *pred, uint8_t *recon) {
    constexpr int lw = LW < 2 ? 2 : LW > 6 ? 6 : LW, lh = LH < 2 ? 2 : LH > 6 ? 6 : LH;
    constexpr int dl = lw - lh;
    if constexpr (dl >= -2 && dl <= 2) warpInverseAdd<lw, lh>(q, txSz, txType, dcQ, acQ, res, pred, recon);
}

// Greedy RD-optimized quantization (one lane): in reverse scan order, each nonzero level may drop by one or to zero
// when D + lambda * R improves. Rates use the AV1 coefficient contexts of the levels already decided (later scan
// positions), matching the order in which contexts are formed. tmp: transform coefficients (row stride W + 1).
__device__ void rdoqBlock(const float *tmp, int16_t *q, const EvalIn &e, int W, float lambdaBits) {
    const int tw = W > 32 ? 32 : W, TS = W + 1, n = tw * tw;
    const uint16_t *scan = av1g_get_scan(e.txSz, AV1G_DCT_DCT);
    const int den = av1g_dq_denom(e.txSz);
    const int lt = tw == 4 ? 2 : tw == 8 ? 3 : tw == 16 ? 4 : 5;
    int eob = 0;
    for (int c = n - 1; c >= 0; c--)
        if (q[scan[c]]) { eob = c + 1; break; }
    if (!eob) return;
    const float g = g_txg[e.txSz][AV1G_DCT_DCT];
    const int txClass = av1g_tx_class(AV1G_DCT_DCT);
    for (int c = eob - 1; c >= 0; c--) {
        const int pos = scan[c];
        const int l0 = q[pos];
        if (!l0) continue;
        const int a0 = l0 < 0 ? -l0 : l0;
        const float v = fabsf(tmp[(pos >> lt) * TS + (pos & (tw - 1))]);
        const int step = pos == 0 ? e.dcQ : e.acQ;
        auto dist = [&](int a) {
            const float dq = a ? (float)av1g_dequant_coef<8>(a, step, den) : 0.f;
            const float d = v - dq;
            return g * d * d;
        };
        float best = 0.f;
        int bestA = a0;
        const int cand[2] = {a0 - 1, 0};
        for (int k = -1; k < 2; k++) {
            const int a = k < 0 ? a0 : cand[k];
            if (k >= 0 && (a < 0 || (k == 1 && a0 - 1 == 0))) continue;
            q[pos] = (int16_t)(l0 < 0 ? -a : a);
            // a zero at the last position shortens the block: price it with the new end of block
            int eobNow = eob;
            if (!a && c == eob - 1) {
                eobNow = 0;
                for (int k2 = c - 1; k2 >= 0; k2--)
                    if (q[scan[k2]]) { eobNow = k2 + 1; break; }
            }
            float bits = 0.f;
            if (c < eobNow && a) bits = coeff_bits(c, eobNow, q, scan, e.txSz, txClass, e.plane > 0, e.dcCtx, e.C);
            if (c == eob - 1 || !a) bits += base_bits(e, eobNow) - base_bits(e, eob);
            const float J = dist(a) + lambdaBits * bits;
            if (k < 0 || J < best) { best = J; bestA = a; }
        }
        q[pos] = (int16_t)(l0 < 0 ? -bestA : bestA);
        if (!bestA && c == eob - 1) {
            eob = 0;
            for (int k2 = c - 1; k2 >= 0; k2--)
                if (q[scan[k2]]) { eob = k2 + 1; break; }
            if (!eob) return;
        }
    }
}

// Rectangular blocks (HORZ / VERT / HORZ_4 / VERT_4 of an n x n node) are searched only when the node's square
// evaluation codes a residual or skip wins by less than a factor margin (default 2, AV1GPU_PF_PRUNE): a node that
// clearly prefers skip as one block rarely gains from splitting.
__global__ void pruneRect(Cand *rc, int rcols, int rrows, int bw, int bh, const Cand *parent, int pcols, int n,
                          int enabled, float margin) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= rcols * rrows) return;
    const int x0 = (b % rcols) * bw, y0 = (b / rcols) * bh;
    const Cand &p = parent[(y0 / n) * pcols + x0 / n];
    rc[b].pruned = enabled && !(p.nz && p.costCoded < p.costSkip * margin);
}
// Square children (n x n inside a 2n x 2n parent) are searched only when the parent is not a clear skip:
// same rule as pruneRect. A pruned parent prunes its children; a parent crossing the frame edge (which must split)
// never prunes them.
__global__ void pruneSquare(Cand *cc, int ccols, int crows, int n, const Cand *parent, int pcols, int w, int h,
                            float margin, int force = 0) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= ccols * crows) return;
    const int x0 = (b % ccols) * n, y0 = (b / ccols) * n;
    const int px = x0 / (2 * n) * (2 * n), py = y0 / (2 * n) * (2 * n);
    const Cand &p = parent[(py / (2 * n)) * pcols + px / (2 * n)];
    const bool mustSplit = px + n >= w || py + n >= h;
    if (force) { cc[b].pruned = !mustSplit; return; }   // level disabled: kept only where the parent must split
    cc[b].pruned = margin > 0.f && !mustSplit && (p.pruned || !(p.nz && p.costCoded < p.costSkip * margin));
}
template <int BW, int BH>
struct EvalCfg {
    static constexpr int warps = BW * BH >= 1024 ? 1 : BW * BH >= 256 ? 2 : 4;
    // split-external evaluations of one block per CTA: four warps predict a horizontal stripe each (prediction, the
    // filter search and the residual), then warp 0 alone codes the luma transform
    static constexpr int stripes = BW * BH >= 2048 ? 4 : 1;   // (32x32 and 16x64 measured slower; chroma stripes >= 4 rows)
};

struct EvalArgs {
    Frame f;
    Cand *cand;
    int cols, rows;
    uint8_t *pred, *recon;   // per-shape frame-sized buffers (tight YUV420)
    int16_t *levels;          // per-shape levels, spatial layout of the aligned planes
    uint8_t *recon2, *recon3;   // per-size luma reconstruction with depth 1 / 2 split transforms (null: no search)
    int16_t *levels2, *levels3; // per-size luma levels with depth 1 / 2 split transforms
    int qindex, qctx;
    float lambda, rnd, zbin;
    float mvWeight;
    float modeBase;      // bits charged per block for its mode info besides the vectors
    float skipBits1, skipBits0;   // skip flag cost of the skip / coded options
    int splitPrune;      // split-external: skip the split transforms of blocks whose whole transform is all zero
    int rdoq;
    // luma transform types per 4x4 unit (stride aw / 4) of the whole-block transform and of the depth 1 / 2 splits;
    // null: DCT_DCT only
    uint8_t *types1, *types2, *types3;
    float *subD, *subR;          // split-external: per block and split transform (4 depth-1, then 16 depth-2)
    uint8_t *subNz, *subType;
    float d2Margin;              // depth-2 split search only when depth 1 beats the whole transform by this factor
    int reducedTxSet;   // reduced_tx_set: luma types limited to IDTX / DCT_DCT
    int switchable;     // is_filter_switchable: choose the interpolation filter per block
    float filterBias;   // scale of the SMOOTH / SHARP bit estimate
    int txExhaustive;   // price every type of the set (else a subset)
    int splitTypes;     // split transforms: 0 same search, 1 seven-type subset, 2 DCT_DCT only
    ModeTrial *trials;   // optional sparse evaluation on the final partition
    const int *trialCount, *trialIndices;

    __device__ int blockCount() const { return trials ? *trialCount : cols * rows; }
    __device__ int blockIndex(int i) const { return trials ? trialIndices[i] : i; }

    __device__ Cand *candidate(int b, int x, int y) const {
        return trials ? &trials[(y / 8) * (f.aw / 8) + x / 8].cand : cand + b;
    }
};
// inter_tx_type default CDFs (spec Default_Inter_Tx_Type_Set1/2/3_Cdf) for the type bits of the search
__constant__ uint16_t kTxSet1Cdf[2][16] = {
    {4458, 5560, 7695, 9709, 13330, 14789, 17537, 20266, 21504, 22848, 23934, 25474, 27727, 28915, 30631, 32768},
    {1645, 2573, 4778, 5711, 7807, 8622, 10522, 15357, 17674, 20408, 22517, 25010, 27116, 28856, 30749, 32768}};
__constant__ uint16_t kTxSet2Cdf[12] = {770, 2421, 5225, 12907, 15819, 18927, 21561, 24089, 26595, 28526, 30529, 32768};
__constant__ uint16_t kTxSet3Cdf[4][2] = {{16384, 32768}, {4167, 32768}, {1998, 32768}, {748, 32768}};
__device__ float interTypeBits(int set, int sq, int idx) {
    const uint16_t *c = set == AV1G_TX_SET_INTER_1 ? kTxSet1Cdf[sq] : set == AV1G_TX_SET_INTER_2 ? kTxSet2Cdf : kTxSet3Cdf[sq];
    const int p = c[idx] - (idx ? c[idx - 1] : 0);
    return -log2f((float)max(p, 1) * (1.f / 32768.f));
}
// Choose the inter transform type of one luma transform. run(type, full) restores the residual, transforms and
// quantizes (o.D, o.R); full also reconstructs. Every type of the set is priced by the quantizer's distortion
// estimate; the winner is then reconstructed. Leaves its levels in q and reconstruction in rec, adds the type bits
// to o.R and returns the type. An all-zero transform codes no type and counts as DCT_DCT (TxTypes of an all-zero
// luma transform are DCT_DCT, which inter chroma inherits).
// Type screening score of a forward transform in tmp (row stride W + 1): sum of coefficient magnitudes in the
// pixel-domain scale of the type (a SATD-like rate proxy).
__device__ __forceinline__ float fwdScore(const float *tmp, int W, int H, int txSz, int type) {
    const int th = H > 32 ? 32 : H, tw = W > 32 ? 32 : W, lt = tw == 4 ? 2 : tw == 8 ? 3 : tw == 16 ? 4 : 5;
    float s = 0.f;
    for (int idx = threadIdx.x & 31; idx < tw * th; idx += 32) s += fabsf(tmp[(idx >> lt) * (W + 1) + (idx & (tw - 1))]);
    return warp_sum(s) * sqrtf(g_txg[txSz][type]);
}
// keep: where the caller stores this transform's levels (row stride keepStride, rows of keepW levels as in q): step 0
// leaves the DCT_DCT levels there, and a DCT_DCT winner restores them instead of transforming again (same values).
template <class Run>
__device__ int chooseLumaType(int txSz, bool enabled, int reduced, bool exhaustive, float lambda, EvalOut &o,
                              int16_t *q, int nq, Run run, int16_t *keep, int keepStride, int keepW) {
    const int set = enabled ? pft2::interTxSet(txSz, reduced) : AV1G_TX_SET_DCTONLY;
    const int sq = AV1G_T(Tx_Size_Sqr)[txSz];
    const int n = pft2::interTxCount(set);
    // One call site for run (it inlines the transform, quantizer and inverse); mode 0: forward transform only, returns
    // a coefficient-magnitude score; 1: forward + quantizer estimate; 2: full reconstruction. Steps: DCT_DCT
    // reconstruction, then (large sets) a forward-only screen of every type keeping the kTop best, the quantizer
    // estimate of the remaining candidates, and the winner's reconstruction.
    constexpr int kTop = 3;
    const bool screen = n > 4;
    int top[kTop];
    float topS[kTop];
    for (int k = 0; k < kTop; k++) { top[k] = -1; topS[k] = 3.4e38f; }
    int best = AV1G_DCT_DCT, step = 0;
    float bestJ = 0.f;
    EvalOut oDct{};
    bool kept = false;
    // steps: 0 | 1..n screen (when screen) | candidates | final
    const int candBase = 1 + (screen ? n : 0), nCand = screen ? kTop : n;
    for (;;) {
        int t = AV1G_DCT_DCT, mode = 2, cand = -1;
        if (step == 0) {
        } else if (step < candBase) {
            t = pft2::interTxInv(set, step - 1);
            mode = 0;
            if (t == AV1G_DCT_DCT || (!exhaustive && (av1g_flip_ud(t) || av1g_flip_lr(t) || t >= AV1G_V_ADST))) { step++; continue; }
        } else if (step < candBase + nCand) {
            cand = step - candBase;
            t = screen ? top[cand] : pft2::interTxInv(set, cand);
            mode = 1;
            if (t < 0 || t == AV1G_DCT_DCT || (!screen && !exhaustive && (av1g_flip_ud(t) || av1g_flip_lr(t) || t >= AV1G_V_ADST))) {
                step++;
                continue;
            }
        } else {
            t = best;
        }
        float score = 0.f;
        if (kept && step >= candBase + nCand && t == AV1G_DCT_DCT) {
            for (int i = threadIdx.x & 31; i < nq; i += 32) q[i] = keep[size_t(i / keepW) * keepStride + i % keepW];
            __syncwarp();
            o = oDct;
        } else {
            score = run(t, mode);
            // another type's reconstruction replaced the DCT_DCT one (the all-zero fallback then recomputes it)
            if (mode == 2 && t != AV1G_DCT_DCT) kept = false;
        }
        if (step == 0) {
            if (set == AV1G_TX_SET_DCTONLY) return AV1G_DCT_DCT;
            int any = 0;
            for (int i = threadIdx.x & 31; i < nq; i += 32) any |= q[i];
            // nothing left after DCT quantization: other types rarely code such a residual better
            if (!__any_sync(0xffffffffu, any)) break;
            bestJ = o.D + lambda * (o.R + interTypeBits(set, sq, pft2::interTxIndex(set, AV1G_DCT_DCT)));
            for (int i = threadIdx.x & 31; i < nq; i += 32) keep[size_t(i / keepW) * keepStride + i % keepW] = q[i];
            __syncwarp();
            oDct = o;
            kept = true;
        } else if (step < candBase) {
            // keep the kTop smallest scores (insertion)
            float sc = score;
            int tt = t;
            for (int k = 0; k < kTop; k++)
                if (sc < topS[k]) { const float fs = topS[k]; const int ft = top[k]; topS[k] = sc; top[k] = tt; sc = fs; tt = ft; }
        } else if (step < candBase + nCand) {
            const float J = o.D + lambda * (o.R + interTypeBits(set, sq, pft2::interTxIndex(set, t)));
            if (J < bestJ) { bestJ = J; best = t; }
        } else {
            int any = 0;
            for (int i = threadIdx.x & 31; i < nq; i += 32) any |= q[i];
            if (best == AV1G_DCT_DCT || __any_sync(0xffffffffu, any)) break;
            best = AV1G_DCT_DCT;   // an all-zero result codes as DCT_DCT: reconstruct that instead
            continue;
        }
        step++;
    }
    o.R += interTypeBits(set, sq, pft2::interTxIndex(set, best));
    return best;
}

// Evaluate skip and coded options of every block of one shape.
template <int BW, int BH, bool EXT>
__device__ __forceinline__ void evaluateOne(EvalArgs a, const int slot) {
    constexpr int W = EvalCfg<BW, BH>::warps;
    constexpr int AREA = BW * BH;
    constexpr int TQ = (BW > 32 ? 32 : BW) * (BH > 32 ? 32 : BH);   // stored luma coefficients
    constexpr int TS = (BH > 32 ? 32 : BH) * (BW + 1);              // forward transform scratch
    __shared__ int16_t interp[W][(BH + 7) * BW];
    __shared__ int16_t hpS[W][AREA];
    __shared__ __align__(16) uint8_t predS[W][AREA + AREA / 2];
    __shared__ __align__(16) uint8_t recS[W][AREA];
    __shared__ int16_t resS[W][AREA];
    __shared__ float tmpS[W][TS];
    __shared__ int16_t qS[W][TQ];
    constexpr int S = EXT ? EvalCfg<BW, BH>::stripes : 1;   // warps per block
    constexpr bool MW = S > 1;
    static_assert(!MW || W == 1, "striped evaluation: one block per CTA");
    const int warp = (threadIdx.x >> 5) / S, stripe = (threadIdx.x >> 5) % S, lane = threadIdx.x & 31;
    const int b = a.blockIndex(slot);
    const int x0 = (b % a.cols) * BW, y0 = (b / a.cols) * BH;
    Cand &c = *a.candidate(b, x0, y0);
    if (c.pruned) {
        if (!lane && !stripe) { c.costSkip = c.costCoded = kInf; c.nz = 0; c.txSplit = 0; c.filter = 0; }
        return;
    }
    const Frame &f = a.f;
    if (x0 >= f.w || y0 >= f.h) {
        if (!lane && !stripe) { c.costSkip = c.costCoded = kInf; c.nz = 0; c.txSplit = 0; }
        return;
    }
    // striped mode: stripe s predicts rows [s * BH / S, (s + 1) * BH / S) with its own interpolation scratch (the first
    // half in interp, the second in tmpS, which the transform uses only later)
    constexpr int SH = BH / S;
    static_assert(!MW || (2 * (SH + 7) * BW <= (BH + 7) * BW && 2 * (SH + 7) * BW * 2 <= TS * 4), "stripe scratch");
    int16_t *sInterp = !MW ? interp[warp]
                           : stripe < S / 2 ? interp[0] + stripe * (SH + 7) * BW
                                            : reinterpret_cast<int16_t *>(tmpS[0]) + (stripe - S / 2) * (SH + 7) * BW;
    __shared__ int part[S];
    const uint8_t *ref = c.ref == 1 ? f.ref[1] : c.ref == 3 ? f.ref[2] : f.ref[0];
    const uint8_t *ref2 = f.ref[1];
    const int cw = f.w >> 1, chh = f.h >> 1;
    const size_t ysz = size_t(f.w) * f.h, csz = size_t(cw) * chh;
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    float dSkip = 0.f, dCoded = 0.f, bits = 0.f;
    bool nz = false;
    float lumaD = 0.f, lumaBits = 0.f, splitD[3] = {kInf, kInf, kInf}, splitBits[3] = {0.f, 0.f, 0.f};
    bool nzSplit[3] = {false, false, false}, nzC = false;
    int wholeType = AV1G_DCT_DCT, splitTop[3] = {AV1G_DCT_DCT, AV1G_DCT_DCT, AV1G_DCT_DCT}, depth = 0, chromaType = AV1G_DCT_DCT;
    const int tstride = f.aw >> 2;
    // interpolation filter (one per block, both references): luma SSE plus approximate interp_filter bits, tried only
    // when a vector is fractional in some plane (1/16 chroma units: luma integer vectors may be chroma half-pel)
    int filt = 0;
    bool searched = false;   // the luma prediction of the chosen filter is already in predS
    if (a.switchable && !c.skipModeTrial && (((c.mv.row | c.mv.col) & 15) || (c.ref == 2 && ((c.mv2.row | c.mv2.col) & 15)))) {
        const float fbits[3] = {0.21f, 2.9f, 7.6f};   // Default_Interp_Filter_Cdf context 3 (no neighbour information)
        float bestJ = 3.4e38f;
        uint8_t *pl = predS[warp];
        const int s0 = stripe * SH;   // first row of this warp's stripe (0 unless striped)
        constexpr int RH = MW ? SH : BH;
#pragma unroll 1
        for (int t = 0; t < 3; t++) {
            // its bits alone exceed the best cost so far (SSE >= 0): this filter cannot win (same choice, no prediction)
            if (t && a.lambda * fbits[t] * a.filterBias >= bestJ) continue;
            if (c.ref == 2)
                warpPredictCompound<(BW == 64 || BH == 64)>(f.ref[0], ref2, f.w, f.h, x0, y0 + s0, BW, RH, c.mv, c.mv2, 0,
                                                            sInterp, hpS[warp] + s0 * BW, pl + s0 * BW, t, BH);
            else
                warpPredict<(BW == 64 || BH == 64)>(ref, f.w, f.h, x0, y0 + s0, BW, RH, c.mv, 0, sInterp, pl + s0 * BW, t, BH);
            int v = 0;
            for (int i = s0 * BW + lane; i < (s0 + RH) * BW; i += 32) {
                const int r = i / BW, col = i % BW;
                if (x0 + col >= f.w || y0 + r >= f.h) continue;
                const int d = (int)f.src[0][(y0 + r) * f.pitch[0] + x0 + col] - pl[i];
                v += d * d;
            }
            v = warp_sumi(v);
            if constexpr (MW) {
                if (!lane) part[stripe] = v;
                __syncthreads();
                v = 0;
#pragma unroll
                for (int k = 0; k < S; k++) v += part[k];
                __syncthreads();
            }
            const float J = (float)v + a.lambda * (t ? fbits[t] * a.filterBias : fbits[t]);
            if (J < bestJ) {
                bestJ = J;
                filt = t;
                if (t < 2) {   // keep it (recS is free until the reconstruction)
                    for (int i = s0 * BW / 4 + lane; i < (s0 + RH) * BW / 4; i += 32)
                        reinterpret_cast<uint32_t *>(recS[warp])[i] = reinterpret_cast<const uint32_t *>(pl)[i];
                    __syncwarp();
                }
            }
        }
        if (filt != 2) {
            for (int i = s0 * BW / 4 + lane; i < (s0 + RH) * BW / 4; i += 32)
                reinterpret_cast<uint32_t *>(pl)[i] = reinterpret_cast<const uint32_t *>(recS[warp])[i];
            __syncwarp();
        }
        searched = true;
    }
    if (!lane && !stripe) c.filter = (uint8_t)filt;
    // striped mode: the three planes' predictions, stored predictions and skip SSEs by all warps (luma last, so resS
    // ends with the luma residual), then warp 0 alone continues
    int sseP[3] = {0, 0, 0};
    if constexpr (MW) {
#pragma unroll
        for (int k = 0; k < 3; k++) {
            const int p = k == 2 ? 0 : k + 1;
            const int sub = p > 0, bw = p ? BW / 2 : BW, bh = p ? BH / 2 : BH, sh = bh / S, r0 = stripe * sh;
            const int lw = Log2<BW>::v - sub;
            const int pw = p ? cw : f.w, ph = p ? chh : f.h;
            const int px = x0 >> sub, py = y0 >> sub;
            uint8_t *pred = predS[0] + (p ? AREA + (p - 1) * (AREA / 4) : 0);
            const uint8_t *refP = ref + (p ? ysz + (p - 1) * csz : 0);
            if (p == 0 && searched) {
            } else if (c.ref == 2)
                warpPredictCompound<(BW == 64 || BH == 64)>(refP, ref2 + (p ? ysz + (p - 1) * csz : 0), pw, ph, px, py + r0,
                                                            bw, sh, c.mv, c.mv2, sub, sInterp, hpS[0] + r0 * bw,
                                                            pred + r0 * bw, filt, bh);
            else
                warpPredict<(BW == 64 || BH == 64)>(refP, pw, ph, px, py + r0, bw, sh, c.mv, sub, sInterp, pred + r0 * bw,
                                                    filt, bh);
            __syncthreads();
            const uint8_t *src = f.src[p];
            const int pitch = f.pitch[p];
            uint8_t *outPred = a.pred + (p ? ysz + (p - 1) * csz : 0);
            int sse = 0;
            for (int i = threadIdx.x; i < bw * bh; i += 32 * S) {
                const int r = i >> lw, col = i & (bw - 1);
                int d = 0;
                if (px + col < pw && py + r < ph) {
                    d = (int)src[(py + r) * pitch + px + col] - pred[i];
                    outPred[(py + r) * pw + px + col] = pred[i];
                }
                sse += d * d;
                resS[0][i] = (int16_t)d;
            }
            sse = warp_sumi(sse);
            if (!lane) part[stripe] = sse;
            __syncthreads();
            sse = 0;
#pragma unroll
            for (int w4 = 0; w4 < S; w4++) sse += part[w4];
            sseP[p] = sse;
            __syncthreads();
        }
        if (stripe) return;
    }
#pragma unroll
    for (int p = 0; p < 3; p++) {
        const int sub = p > 0, bw = p ? BW / 2 : BW, bh = p ? BH / 2 : BH;
        const int lw = Log2<BW>::v - sub, lh = Log2<BH>::v - sub;
        const int pw = p ? cw : f.w, ph = p ? chh : f.h;
        const int px = x0 >> sub, py = y0 >> sub;
        uint8_t *pred = predS[warp] + (p ? AREA + (p - 1) * (AREA / 4) : 0);
        const uint8_t *refP = ref + (p ? ysz + (p - 1) * csz : 0);
        if (MW || (p == 0 && searched)) {
        } else if (c.ref == 2)
            warpPredictCompound<(BW == 64 || BH == 64)>(refP, ref2 + (p ? ysz + (p - 1) * csz : 0), pw, ph, px, py, bw, bh, c.mv, c.mv2, sub,
                                interp[warp], hpS[warp], pred, filt);
        else
            warpPredict<(BW == 64 || BH == 64)>(refP, pw, ph, px, py, bw, bh, c.mv, sub, interp[warp], pred, filt);
        const uint8_t *src = f.src[p];
        const int pitch = f.pitch[p];
        int sse = 0;
        int16_t *res = resS[warp];
        float energy = 0.f;
        uint8_t *outPred = a.pred + (p ? ysz + (p - 1) * csz : 0);
        if constexpr (MW) {
            if (!p)   // the luma residual is in resS (the same per-lane order as below)
                for (int i = lane; i < bw * bh; i += 32) energy += (float)(res[i] * res[i]);
            dSkip += (float)sseP[p];
        } else {
            for (int i = lane; i < bw * bh; i += 32) {
                const int r = i >> lw, col = i & (bw - 1);
                int d = 0;
                if (px + col < pw && py + r < ph) {
                    d = (int)src[(py + r) * pitch + px + col] - pred[i];
                    outPred[(py + r) * pw + px + col] = pred[i];
                }
                sse += d * d;
                res[i] = (int16_t)d;
                energy += (float)(d * d);
            }
            dSkip += (float)warp_sumi(sse);
        }
        __syncwarp();
        if (EXT && p) continue;   // split-external mode: chroma is coded by evalFinal once the luma depth is known
        // transform of the whole block (64-point transforms keep 32 coefficients per side)
        const int tlw = lw > 6 ? 6 : lw, tlh = lh > 6 ? 6 : lh;
        const int txSz = txSizeOf(tlw, tlh);
        const int tqw = bw > 32 ? 32 : bw, tqh = bh > 32 ? 32 : bh;
        EvalIn e{};
        e.txSz = txSz; e.txType = AV1G_DCT_DCT; e.plane = p;
        e.dcQ = dcQ; e.acQ = acQ; e.qm = nullptr;
        e.rnd = a.rnd; e.zbin = a.zbin;
        e.skipCtx = p ? 7 : 0; e.dcCtx = 0; e.intraDir = -1; e.reduced = 1;
        e.C = g_cost[a.qctx];
        e.resEnergy = warp_sum(energy);
        e.qmDist = 0; e.qmRnd = 0;
        EvalOut o;
        int16_t *q = qS[warp];
        uint8_t *rec = recS[warp];
        int16_t *bak = hpS[warp];
        for (int i = lane; i < bw * bh; i += 32) bak[i] = res[i];
        __syncwarp();
        auto run = [&](int type, int mode) -> float {
            for (int i = lane; i < bw * bh; i += 32) res[i] = bak[i];
            __syncwarp();
            e.txType = type;
            if (p == 0) warp_fwd_ct<BW, BH>(res, tmpS[warp], txSz, type);
            else warp_fwd_ct<BW / 2, BH / 2>(res, tmpS[warp], txSz, type);
            if (mode == 0) return fwdScore(tmpS[warp], bw, bh, txSz, type);
            warp_quant_rate_rt<8>(tmpS[warp], q, e, o, bw, bh);
            if (mode == 1) return 0.f;
            if (a.rdoq && bw == bh && type == AV1G_DCT_DCT) {
                if (lane == 0) rdoqBlock(tmpS[warp], q, e, bw, a.lambda);
                __syncwarp();
                const uint16_t *scan = av1g_get_scan(txSz, AV1G_DCT_DCT);
                int last = 0;
                for (int k = lane; k < tqw * tqh; k += 32)
                    if (q[scan[k]]) last = k + 1;
                const int eobF = warp_max(last);
                float rb = lane == 0 ? base_bits(e, eobF) : 0.f;
                for (int k = lane; k < eobF; k += 32)
                    rb += coeff_bits(k, eobF, q, scan, txSz, av1g_tx_class(AV1G_DCT_DCT), p > 0, 0, e.C);
                o.R = warp_sum(rb);
            }
            {
                constexpr int L0 = Log2<BW>::v > 6 ? 6 : Log2<BW>::v, H0 = Log2<BH>::v > 6 ? 6 : Log2<BH>::v;
                if (p == 0) inverseAddCt<L0, H0>(q, txSz, type, dcQ, acQ, res, pred, rec);
                else inverseAddCt<Log2<BW>::v - 1, Log2<BH>::v - 1>(q, txSz, type, dcQ, acQ, res, pred, rec);
            }
            return 0.f;
        };
        auto recSse = [&]() {
            int v = 0;
            for (int i = lane; i < bw * bh; i += 32) {
                const int r = i >> lw, col = i & (bw - 1);
                if (px + col >= pw || py + r >= ph) continue;
                const int d = (int)src[(py + r) * pitch + px + col] - rec[i];
                v += d * d;
            }
            return warp_sumi(v);
        };
        if (p == 0) {
            wholeType = chooseLumaType(txSz, a.types1 != nullptr, a.reducedTxSet, a.txExhaustive != 0, a.lambda, o, q, tqw * tqh, run,
                                       a.levels + size_t(py) * f.aw + px, f.aw, tqw);
            if (a.types1)
                for (int i = lane; i < (bw >> 2) * (bh >> 2); i += 32)
                    a.types1[size_t((py >> 2) + i / (bw >> 2)) * tstride + (px >> 2) + i % (bw >> 2)] = (uint8_t)wholeType;
        } else {
            run(pft2::interChromaType(chromaType, txSz, a.reducedTxSet), 2);
        }
        bits += o.R;
        int any = 0;
        for (int i = lane; i < tqw * tqh; i += 32) any |= q[i];
        const bool anyNz = __any_sync(0xffffffffu, any);
        nz = nz || anyNz;
        if (p) nzC = nzC || anyNz;
        int sseC = 0;
        uint8_t *outRec = a.recon + (p ? ysz + (p - 1) * csz : 0);
        for (int i = lane; i < bw * bh; i += 32) {
            const int r = i >> lw, col = i & (bw - 1);
            if (px + col >= pw || py + r >= ph) continue;
            const int d = (int)src[(py + r) * pitch + px + col] - rec[i];
            sseC += d * d;
            outRec[(py + r) * pw + px + col] = rec[i];
        }
        dCoded += (float)warp_sumi(sseC);
        // spatial layout: the stored tqh x tqw coefficients sit at the top-left of the block area
        const int apw = p ? f.aw / 2 : f.aw;
        const size_t base = p ? size_t(f.aw) * f.ah + size_t(p - 1) * (f.aw / 2) * (f.ah / 2) : 0;
        int16_t *dst = a.levels + base + size_t(py) * apw + px;
        for (int i = lane; i < bw * bh; i += 32) {
            const int r = i >> lw, col = i & (bw - 1);
            dst[size_t(r) * apw + col] = (r < tqh && col < tqw) ? q[r * tqw + col] : 0;
        }
        __syncwarp();
        if (p == 0) { lumaD = dCoded; lumaBits = bits; }
        if (BW != BH && p == 0 && a.recon2 && !EXT) {
            // rectangular split: two transforms, the longer side halved (Split_Tx_Size)
            constexpr int SW = BW > BH ? BW / 2 : BW, SH = BW > BH ? BH : BH / 2;
            constexpr int lsw = Log2<SW>::v, lsh = Log2<SH>::v;
            const int stx = txSizeOf(lsw, lsh);
            float dS = 0.f, bS = 0.f;
            bool nzS = false;
            uint8_t *pq = recS[warp], *rq = recS[warp] + SW * SH;
            for (int k = 0; k < 2; k++) {
                const int qx = BW > BH ? k * SW : 0, qy = BW > BH ? 0 : k * SH;
                float en = 0.f;
                for (int i = lane; i < SW * SH; i += 32) {
                    const int r = i >> lsw, col = i & (SW - 1);
                    const int x = px + qx + col, y = py + qy + r;
                    const uint8_t pv = pred[(qy + r) * BW + qx + col];
                    pq[i] = pv;
                    int dd = 0;
                    if (x < pw && y < ph) dd = (int)src[y * pitch + x] - pv;
                    res[i] = (int16_t)dd;
                    en += (float)(dd * dd);
                }
                __syncwarp();
                EvalIn es = e;
                es.txSz = stx;
                es.skipCtx = 2;
                es.resEnergy = warp_sum(en);
                EvalOut os;
                int16_t *bk = hpS[warp];
                for (int i = lane; i < SW * SH; i += 32) bk[i] = res[i];
                __syncwarp();
                auto runS = [&](int type, int mode) -> float {
                    for (int i = lane; i < SW * SH; i += 32) res[i] = bk[i];
                    __syncwarp();
                    es.txType = type;
                    warp_fwd_ct<SW, SH>(res, tmpS[warp], stx, type);
                    if (mode == 0) return fwdScore(tmpS[warp], SW, SH, stx, type);
                    warp_quant_rate_rt<8>(tmpS[warp], q, es, os, SW, SH);
                    if (mode == 1) return 0.f;
                    inverseAddCt<lsw, lsh>(q, stx, type, dcQ, acQ, res, pq, rq);
                    return 0.f;
                };
                auto sseFn = [&]() {
                    int v = 0;
                    for (int i = lane; i < SW * SH; i += 32) {
                        const int r = i >> lsw, col = i & (SW - 1);
                        const int x = px + qx + col, y = py + qy + r;
                        if (x >= pw || y >= ph) continue;
                        const int dd = (int)src[y * pitch + x] - rq[i];
                        v += dd * dd;
                    }
                    return warp_sumi(v);
                };
                const int ty = chooseLumaType(stx, a.types2 != nullptr && a.splitTypes < 2, a.reducedTxSet, a.txExhaustive != 0 && a.splitTypes == 0, a.lambda, os, q, (SW > 32 ? 32 : SW) * (SH > 32 ? 32 : SH), runS,
                                              a.levels2 + size_t(py + qy) * apw + px + qx, apw, SW > 32 ? 32 : SW);
                if (!k) splitTop[1] = ty;
                if (a.types2)
                    for (int i = lane; i < (SW >> 2) * (SH >> 2); i += 32)
                        a.types2[size_t(((py + qy) >> 2) + i / (SW >> 2)) * tstride + ((px + qx) >> 2) + i % (SW >> 2)] = (uint8_t)ty;
                bS += os.R;
                const int sqw = SW > 32 ? 32 : SW, sqh = SH > 32 ? 32 : SH;
                int anyS = 0, sseS = 0;
                for (int i = lane; i < SW * SH; i += 32) {
                    const int r = i >> lsw, col = i & (SW - 1);
                    const int x = px + qx + col, y = py + qy + r;
                    const int16_t lv = (r < sqh && col < sqw) ? q[r * sqw + col] : 0;
                    anyS |= lv;
                    a.levels2[size_t(y) * apw + x] = lv;
                    if (x >= pw || y >= ph) continue;
                    const int dd = (int)src[y * pitch + x] - rq[i];
                    sseS += dd * dd;
                    a.recon2[y * pw + x] = rq[i];
                }
                nzS = nzS || __any_sync(0xffffffffu, anyS);
                dS += (float)warp_sumi(sseS);
                __syncwarp();
            }
            splitD[1] = dS; splitBits[1] = bS; nzSplit[1] = nzS;
        }
        if (BW == BH && p == 0 && a.recon2 && !EXT) {
            // transform split: depth 1 (four half-size transforms) and, for 16x16 and larger, depth 2 (sixteen
            // quarter-size transforms), all with the same prediction
            for (int d = 1; d <= (BW >= 16 ? 2 : 1); d++) {
                const int hn = BW >> d, lhn = Log2<BW>::v - d, per = 1 << d;
                const int stx = txSizeOf(lhn, lhn);
                uint8_t *dstRec = d == 1 ? a.recon2 : a.recon3;
                int16_t *dstLev = d == 1 ? a.levels2 : a.levels3;
                float dS = 0.f, bS = 0.f;
                bool nzS = false;
                uint8_t *pq = recS[warp], *rq = recS[warp] + hn * hn;
                for (int k = 0; k < per * per; k++) {
                    const int qx = (k % per) * hn, qy = (k / per) * hn;
                    float en = 0.f;
                    for (int i = lane; i < hn * hn; i += 32) {
                        const int r = i >> lhn, col = i & (hn - 1);
                        const int x = px + qx + col, y = py + qy + r;
                        const uint8_t pv = pred[(qy + r) * BW + qx + col];
                        pq[i] = pv;
                        int dd = 0;
                        if (x < pw && y < ph) dd = (int)src[y * pitch + x] - pv;
                        res[i] = (int16_t)dd;
                        en += (float)(dd * dd);
                    }
                    __syncwarp();
                    EvalIn es = e;
                    es.txSz = stx;
                    es.skipCtx = 2;
                    es.resEnergy = warp_sum(en);
                    EvalOut os;
                    int16_t *bk = hpS[warp];
                    for (int i = lane; i < hn * hn; i += 32) bk[i] = res[i];
                    __syncwarp();
                    auto runS = [&](int type, int mode) -> float {
                        for (int i = lane; i < hn * hn; i += 32) res[i] = bk[i];
                        __syncwarp();
                        es.txType = type;
                        if (d == 1) warp_fwd_ct<BW / 2, BW / 2>(res, tmpS[warp], stx, type);
                        else warp_fwd_ct<(BW / 4 < 4 ? 4 : BW / 4), (BW / 4 < 4 ? 4 : BW / 4)>(res, tmpS[warp], stx, type);
                        if (mode == 0) return fwdScore(tmpS[warp], hn, hn, stx, type);
                        warp_quant_rate_rt<8>(tmpS[warp], q, es, os, hn, hn);
                        if (mode == 1) return 0.f;
                        if (d == 1) inverseAddCt<Log2<BW>::v - 1, Log2<BW>::v - 1>(q, stx, type, dcQ, acQ, res, pq, rq);
                        else inverseAddCt<Log2<BW>::v - 2, Log2<BW>::v - 2>(q, stx, type, dcQ, acQ, res, pq, rq);
                        return 0.f;
                    };
                    auto sseFn = [&]() {
                        int v = 0;
                        for (int i = lane; i < hn * hn; i += 32) {
                            const int r = i >> lhn, col = i & (hn - 1);
                            const int x = px + qx + col, y = py + qy + r;
                            if (x >= pw || y >= ph) continue;
                            const int dd = (int)src[y * pitch + x] - rq[i];
                            v += dd * dd;
                        }
                        return warp_sumi(v);
                    };
                    uint8_t *tmap = d == 1 ? a.types2 : a.types3;
                    const int ty = chooseLumaType(stx, tmap != nullptr && a.splitTypes < 2, a.reducedTxSet, a.txExhaustive != 0 && a.splitTypes == 0, a.lambda, os, q, hn * hn, runS,
                                                  dstLev + size_t(py + qy) * apw + px + qx, apw, hn);
                    if (!k) splitTop[d] = ty;
                    if (tmap)
                        for (int i = lane; i < (hn >> 2) * (hn >> 2); i += 32)
                            tmap[size_t(((py + qy) >> 2) + i / (hn >> 2)) * tstride + ((px + qx) >> 2) + i % (hn >> 2)] = (uint8_t)ty;
                    bS += os.R;
                    int anyS = 0, sseS = 0;
                    for (int i = lane; i < hn * hn; i += 32) {
                        anyS |= q[i];
                        const int r = i >> lhn, col = i & (hn - 1);
                        const int x = px + qx + col, y = py + qy + r;
                        dstLev[size_t(y) * apw + x] = q[i];
                        if (x >= pw || y >= ph) continue;
                        const int dd = (int)src[y * pitch + x] - rq[i];
                        sseS += dd * dd;
                        dstRec[y * pw + x] = rq[i];
                    }
                    nzS = nzS || __any_sync(0xffffffffu, anyS);
                    dS += (float)warp_sumi(sseS);
                    __syncwarp();
                }
                splitD[d] = dS; splitBits[d] = bS; nzSplit[d] = nzS;
            }
        }
        if (p == 0 && !EXT) {
            // luma transform depth (one more bit per level for the txfm_split flags), then the chroma type: the type
            // of the top-left luma transform (inter chroma follows luma, spec 7.13.3 get_tx_type)
            float bestJ = lumaD + a.lambda * lumaBits;
            for (int d = 1; d <= 2; d++)
                if (a.recon2 && splitD[d] + a.lambda * (splitBits[d] + 1.f + 2.f * (d - 1)) < bestJ) {
                    bestJ = splitD[d] + a.lambda * (splitBits[d] + 1.f + 2.f * (d - 1));
                    depth = d;
                }
            chromaType = depth ? splitTop[depth] : wholeType;
        }
    }
    if (EXT) {
        if (!lane) { c.lumaD = lumaD; c.lumaBits = lumaBits; c.dSkipAll = dSkip; c.nzWhole = nz; c.wholeType = (uint8_t)wholeType; }
        return;
    }
    if (!lane) {
        const float modeBits = a.modeBase + c.mvBits * a.mvWeight;
        const float skipFlag1 = a.skipBits1, skipFlag0 = a.skipBits0;
        c.costSkip = dSkip + a.lambda * (modeBits + skipFlag1);
        if (depth) {
            dCoded += splitD[depth] - lumaD;
            bits += splitBits[depth] + 1.f + 2.f * (depth - 1) - lumaBits;
            nz = nzSplit[depth] || nzC;
        }
        c.txSplit = (uint8_t)depth;
        c.costCoded = dCoded + a.lambda * (modeBits + skipFlag0 + (BW == BH ? 0.f : 0.5f) + bits);
        c.nz = nz;
    }
}


// One warp per block; grid-stride, so the sparse refinement trials launch a small grid (a full-size grid of mostly
// empty CTAs delays every concurrent stream while it drains).
template <int BW, int BH, bool EXT = false, bool LOOP = false>
__global__ void __launch_bounds__(32 * EvalCfg<BW, BH>::warps * (EXT ? EvalCfg<BW, BH>::stripes : 1)) evaluate(EvalArgs a) {
    constexpr int W = EvalCfg<BW, BH>::warps, S = EXT ? EvalCfg<BW, BH>::stripes : 1;
    const int n = a.blockCount();
    const int slot = blockIdx.x * W + (threadIdx.x >> 5) / S;
    if constexpr (LOOP) {
        // striped: warps 1.. finish a block before warp 0 (its transform still uses the shared scratch)
        for (int i = slot; i < n; i += gridDim.x * W) {
            evaluateOne<BW, BH, EXT>(a, i);
            if constexpr (S > 1) __syncthreads();
        }
    } else if (slot < n) {
        evaluateOne<BW, BH, EXT>(a, slot);
    }
}

// ------------------------------------------------------------------------------------------------
// Split-external evaluation of square blocks: one warp per split transform (all blocks of a size at once), from the
// prediction evaluate<.., true> stored in a.pred. Writes the depth-1 / depth-2 levels, reconstruction and types like the
// in-kernel split path, and per transform D, R, nz and type for evalFinal.
template <int BW, int BH>
struct SplitGeo {   // split-external evaluation: split transforms per block and the largest one
    static constexpr bool sq = BW == BH;
    static constexpr int nsub = sq ? (BW >= 16 ? 20 : 4) : 2;
    static constexpr int sw = sq ? BW / 2 : (BW > BH ? BW / 2 : BW), sh = sq ? BH / 2 : (BW > BH ? BH : BH / 2);
};
template <int BW, int BH>
__device__ __forceinline__ void evalSplitOne(EvalArgs a, const int dsel, const int idx) {
    using G = SplitGeo<BW, BH>;
    constexpr int NSUB = G::nsub, SA = G::sw * G::sh;   // largest split transform area
    __shared__ int16_t resS[4][SA], bkS[4][SA], qS[4][SA];
    __shared__ uint8_t pqS[4][SA], rqS[4][SA];
    __shared__ float tmpS[4][(G::sw + 1) * G::sh];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // dsel 0: every split transform; 1: depth 1 only; 2: depth 2 only, for blocks whose depth 1 beat the whole
    // transform by a.d2Margin (the others get an infinite depth-2 cost)
    const int per0 = dsel == 1 ? (G::sq ? 4 : 2) : dsel == 2 ? 16 : NSUB, off0 = dsel == 2 ? 4 : 0;
    const int b = a.blockIndex(idx / per0), k = off0 + idx % per0;
    const Frame &f = a.f;
    const int x0 = (b % a.cols) * BW, y0 = (b / a.cols) * BH;
    const Cand &c = *a.candidate(b, x0, y0);
    if (x0 >= f.w || y0 >= f.h || c.pruned) return;
    // a refinement trial keeping its block's transform depth computes only that depth's split transforms
    if (c.trialDepth != 255 && (c.trialDepth == 0 || (c.trialDepth == 1) != (k < (G::sq ? 4 : 2)))) {
        if (!lane) { a.subD[size_t(b) * NSUB + k] = kInf; a.subR[size_t(b) * NSUB + k] = 0.f; a.subNz[size_t(b) * NSUB + k] = 0; a.subType[size_t(b) * NSUB + k] = 0; }
        return;
    }
    // split pruning (moving content): no split transform when the whole-block transform quantizes to nothing
    if (a.splitPrune && !c.nzWhole) {
        if (!lane) { a.subD[size_t(b) * NSUB + k] = kInf; a.subR[size_t(b) * NSUB + k] = 0.f; a.subNz[size_t(b) * NSUB + k] = 0; a.subType[size_t(b) * NSUB + k] = 0; }
        return;
    }
    if (dsel == 2 && c.trialDepth != 2) {
        float d1 = 0.f, r1 = 0.f;
        for (int j = 0; j < 4; j++) { d1 += a.subD[size_t(b) * NSUB + j]; r1 += a.subR[size_t(b) * NSUB + j]; }
        const float j0 = c.lumaD + a.lambda * c.lumaBits, j1 = d1 + a.lambda * (r1 + 1.f);
        if (!(j1 < a.d2Margin * j0)) {
            if (!lane) { a.subD[size_t(b) * NSUB + k] = kInf; a.subR[size_t(b) * NSUB + k] = 0.f; a.subNz[size_t(b) * NSUB + k] = 0; a.subType[size_t(b) * NSUB + k] = 0; }
            return;
        }
    }
    // square: depth 1 (k 0..3) or depth 2 (k 4..19) quadrants; rectangle: the two halves along the longer side
    const int d = k < 4 ? 1 : 2, kk = k < 4 ? k : k - 4, per = 1 << d;
    const int lsw = G::sq ? Log2<BW>::v - d : Log2<G::sw>::v, lsh = G::sq ? Log2<BH>::v - d : Log2<G::sh>::v;
    const int hw = 1 << lsw, hh = 1 << lsh;
    const int stx = txSizeOf(lsw, lsh);
    const int qx = G::sq ? (kk % per) * hw : (BW > BH ? k * hw : 0), qy = G::sq ? (kk / per) * hh : (BW > BH ? 0 : k * hh);
    const int px = x0, py = y0, pw = f.w, ph = f.h, pitch = f.pitch[0];
    const uint8_t *src = f.src[0], *predF = a.pred;
    uint8_t *dstRec = d == 1 ? a.recon2 : a.recon3;
    int16_t *dstLev = d == 1 ? a.levels2 : a.levels3;
    int16_t *res = resS[warp], *bk = bkS[warp], *q = qS[warp];
    uint8_t *pq = pqS[warp], *rq = rqS[warp];
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    const int apw = f.aw, tstride = f.aw >> 2;
    float en = 0.f;
    for (int i = lane; i < hw * hh; i += 32) {
        const int r = i >> lsw, col = i & (hw - 1);
        const int x = px + qx + col, y = py + qy + r;
        int dd = 0;
        uint8_t pv = 0;
        if (x < pw && y < ph) { pv = predF[size_t(y) * pw + x]; dd = (int)src[y * pitch + x] - pv; }
        pq[i] = pv;
        res[i] = bk[i] = (int16_t)dd;
        en += (float)(dd * dd);
    }
    __syncwarp();
    EvalIn es{};
    es.txSz = stx; es.txType = AV1G_DCT_DCT; es.plane = 0;
    es.dcQ = dcQ; es.acQ = acQ; es.qm = nullptr;
    es.rnd = a.rnd; es.zbin = a.zbin;
    es.skipCtx = 2; es.dcCtx = 0; es.intraDir = -1; es.reduced = 1;
    es.C = g_cost[a.qctx];
    es.resEnergy = warp_sum(en);
    es.qmDist = 0; es.qmRnd = 0;
    EvalOut os;
    auto runS = [&](int type, int mode) -> float {
        for (int i = lane; i < hw * hh; i += 32) res[i] = bk[i];
        __syncwarp();
        es.txType = type;
        if (!G::sq) warp_fwd_ct<G::sw, G::sh>(res, tmpS[warp], stx, type);
        else if (d == 1) warp_fwd_ct<BW / 2, BH / 2>(res, tmpS[warp], stx, type);
        else warp_fwd_ct<(BW / 4 < 4 ? 4 : BW / 4), (BH / 4 < 4 ? 4 : BH / 4)>(res, tmpS[warp], stx, type);
        if (mode == 0) return fwdScore(tmpS[warp], hw, hh, stx, type);
        warp_quant_rate_rt<8>(tmpS[warp], q, es, os, hw, hh);
        if (mode == 1) return 0.f;
        if (!G::sq) inverseAddCt<Log2<G::sw>::v, Log2<G::sh>::v>(q, stx, type, dcQ, acQ, res, pq, rq);
        else if (d == 1) inverseAddCt<Log2<BW>::v - 1, Log2<BH>::v - 1>(q, stx, type, dcQ, acQ, res, pq, rq);
        else inverseAddCt<Log2<BW>::v - 2, Log2<BH>::v - 2>(q, stx, type, dcQ, acQ, res, pq, rq);
        return 0.f;
    };
    uint8_t *tmap = d == 1 ? a.types2 : a.types3;
    const int ty = chooseLumaType(stx, tmap != nullptr && a.splitTypes < 2, a.reducedTxSet,
                                  a.txExhaustive != 0 && a.splitTypes == 0, a.lambda, os, q, hw * hh, runS,
                                  dstLev + size_t(py + qy) * apw + px + qx, apw, hw);
    if (tmap)
        for (int i = lane; i < (hw >> 2) * (hh >> 2); i += 32)
            tmap[size_t(((py + qy) >> 2) + i / (hw >> 2)) * tstride + ((px + qx) >> 2) + i % (hw >> 2)] = (uint8_t)ty;
    int anyS = 0, sseS = 0;
    for (int i = lane; i < hw * hh; i += 32) {
        anyS |= q[i];
        const int r = i >> lsw, col = i & (hw - 1);
        const int x = px + qx + col, y = py + qy + r;
        dstLev[size_t(y) * apw + x] = q[i];
        if (x >= pw || y >= ph) continue;
        const int dd = (int)src[y * pitch + x] - rq[i];
        sseS += dd * dd;
        dstRec[y * pw + x] = rq[i];
    }
    const bool nzS = __any_sync(0xffffffffu, anyS) != 0;
    const int dS = warp_sumi(sseS);
    if (!lane) {
        a.subD[size_t(b) * NSUB + k] = (float)dS;
        a.subR[size_t(b) * NSUB + k] = os.R;
        a.subNz[size_t(b) * NSUB + k] = nzS;
        a.subType[size_t(b) * NSUB + k] = (uint8_t)ty;
    }
}

template <int BW, int BH, bool LOOP = false>
__global__ void __launch_bounds__(128) evalSplit(EvalArgs a, int dsel) {
    using G = SplitGeo<BW, BH>;
    const int per0 = dsel == 1 ? (G::sq ? 4 : 2) : dsel == 2 ? 16 : G::nsub;
    const int n = a.blockCount() * per0;
    const int idx = blockIdx.x * 4 + (threadIdx.x >> 5);
    if constexpr (LOOP) {
        for (int i = idx; i < n; i += gridDim.x * 4) evalSplitOne<BW, BH>(a, dsel, i);
    } else if (idx < n) {
        evalSplitOne<BW, BH>(a, dsel, idx);
    }
}

// Thread-per-transform split search for 4x4 and 8x8 transforms: they keep at most 4 or 8 lanes of a warp busy in the
// transform passes, so here one lane runs one transform through the same steps as chooseLumaType (DCT_DCT, the
// forward-only screen of the other types, quantizer estimates of the best three, reconstruction of the winner), with
// the same arithmetic. Per-lane residual / backup / levels live in shared memory (odd word stride: conflict-free).
template <int LN>
struct TCfg {
    static constexpr int N = 1 << LN, A = N * N, stride = A + 2, threads = LN == 2 ? 128 : 64;
    static constexpr int txSz = LN == 2 ? AV1G_TX_4X4 : AV1G_TX_8X8;
};
template <int LN>
__device__ __forceinline__ void fwdT(const int16_t *bk, int type, float (&t)[TCfg<LN>::A]) {
    constexpr int N = TCfg<LN>::N;
    const int rk = av1g_row_kind(type), ck = av1g_col_kind(type);
    const int fud = av1g_flip_ud(type), flr = av1g_flip_lr(type);
    const float *Fc = g_fwd + g_fwdOff[ck][LN], *FcT = g_fwdT + g_fwdOff[ck][LN];
    const float *Fr = g_fwd + g_fwdOff[rk][LN], *FrT = g_fwdT + g_fwdOff[rk][LN];
#pragma unroll
    for (int j = 0; j < N; j++) {
        const int jj = flr ? N - 1 - j : j;
        if (ck == AV1G_K_IDT) {
#pragma unroll
            for (int i = 0; i < N; i++) t[i * N + j] = __ldg(&Fc[i * N + i]) * (float)bk[(fud ? N - 1 - i : i) * N + jj];
        } else {
            float acc[N];
#pragma unroll
            for (int i = 0; i < N; i++) acc[i] = 0.f;
#pragma unroll
            for (int k = 0; k < N; k++) {
                const float x = (float)bk[(fud ? N - 1 - k : k) * N + jj];
                const float4 *f4 = (const float4 *)(FcT + k * N);
#pragma unroll
                for (int i4 = 0; i4 < N / 4; i4++) {
                    const float4 fv = __ldg(&f4[i4]);
                    acc[4 * i4 + 0] = fmaf(fv.x, x, acc[4 * i4 + 0]);
                    acc[4 * i4 + 1] = fmaf(fv.y, x, acc[4 * i4 + 1]);
                    acc[4 * i4 + 2] = fmaf(fv.z, x, acc[4 * i4 + 2]);
                    acc[4 * i4 + 3] = fmaf(fv.w, x, acc[4 * i4 + 3]);
                }
            }
#pragma unroll
            for (int i = 0; i < N; i++) t[i * N + j] = acc[i];
        }
    }
    const float scale = g_txScale[TCfg<LN>::txSz];
#pragma unroll
    for (int i = 0; i < N; i++) {
        float acc[N];
#pragma unroll
        for (int jj = 0; jj < N; jj++) acc[jj] = 0.f;
        if (rk == AV1G_K_IDT) {
#pragma unroll
            for (int jj = 0; jj < N; jj++) acc[jj] = __ldg(&Fr[jj * N + jj]) * t[i * N + jj];
        } else {
#pragma unroll
            for (int k = 0; k < N; k++) {
                const float x = t[i * N + k];
                const float4 *f4 = (const float4 *)(FrT + k * N);
#pragma unroll
                for (int j4 = 0; j4 < N / 4; j4++) {
                    const float4 fv = __ldg(&f4[j4]);
                    acc[4 * j4 + 0] = fmaf(fv.x, x, acc[4 * j4 + 0]);
                    acc[4 * j4 + 1] = fmaf(fv.y, x, acc[4 * j4 + 1]);
                    acc[4 * j4 + 2] = fmaf(fv.z, x, acc[4 * j4 + 2]);
                    acc[4 * j4 + 3] = fmaf(fv.w, x, acc[4 * j4 + 3]);
                }
            }
        }
#pragma unroll
        for (int jj = 0; jj < N; jj++) t[i * N + jj] = acc[jj] * scale;
    }
}
// quantize (as warp_quant_rate_rt: no matrices) into q, D and R of the levels
template <int LN>
__device__ __forceinline__ void quantT(const float (&t)[TCfg<LN>::A], int16_t *q, const EvalIn &e, float &D, float &R) {
    constexpr int A = TCfg<LN>::A, txSz = TCfg<LN>::txSz;
    const int den = av1g_dq_denom(txSz);
    float dist = 0.f;
#pragma unroll
    for (int idx = 0; idx < A; idx++) {
        const float v = t[idx];
        const int step = idx == 0 ? e.dcQ : e.acQ;
        const float a = fabsf(v) * (float)den / (float)step;
        int l = a < e.zbin ? 0 : (int)(a + e.rnd);
        l = min(l, 16383);
        const int qv = v < 0 ? -l : l;
        q[idx] = (int16_t)qv;
        const float dq = l ? (float)av1g_dequant_coef<8>(qv, step, den) : 0.f;
        const float d = v - dq;
        dist = fmaf(d * d, 1.f, dist);
    }
    const uint16_t *scan = av1g_get_scan(txSz, e.txType);
    int eob = 0;
    for (int c = A - 1; c >= 0; c--)
        if (q[scan[c]]) { eob = c + 1; break; }
    float bits = base_bits(e, eob);
    const int txClass = av1g_tx_class(e.txType);
    for (int c = 0; c < eob; c++) bits += coeff_bits(c, eob, q, scan, txSz, txClass, 0, e.dcCtx, e.C);
    D = g_txg[txSz][e.txType] * dist;
    R = bits;
}
// exact inverse + add (warpInverseAdd<LN, LN> on one lane)
template <int LN>
__device__ __forceinline__ void inverseTS(int16_t *q, int type, int dcQ, int acQ, int16_t *res, const uint8_t *pred,
                                          uint8_t *rec, int ps) {
    constexpr int N = TCfg<LN>::N, A = TCfg<LN>::A, txSz = TCfg<LN>::txSz;
    const int den = av1g_dq_denom(txSz);
    const int rk = av1g_row_kind(type), ck = av1g_col_kind(type);
    const int rowShift = AV1G_T(Transform_Row_Shift)[txSz];
    for (;;) {
        int any = 0;
#pragma unroll
        for (int i = 0; i < A; i++) any |= q[i];
        if (!any) {
            for (int i = 0; i < A; i++) rec[i * ps] = pred[i * ps];
            return;
        }
        uint32_t err = 0;
        for (int i = 0; i < N; i++) err |= rci::row<LN, 8>(i, N, N, q, dcQ, acQ, nullptr, den, false, rk, rowShift, res);
        for (int j = 0; j < N; j++) err |= rci::col<LN>(j, N, ck, res);
        if (!err) break;
#pragma unroll
        for (int i = 0; i < A; i++) q[i] = (int16_t)(q[i] / 2);
    }
    const int fud = av1g_flip_ud(type), flr = av1g_flip_lr(type);
    for (int i = 0; i < A; i++) {
        const int r = i >> LN, c = i & (N - 1);
        rec[i * ps] = (uint8_t)min(max((int)pred[i * ps] + res[(fud ? N - 1 - r : r) * N + (flr ? N - 1 - c : c)], 0), 255);
    }
}
// One lane per split transform of size 2^LN (dsel as evalSplit).
template <int BW, int BH, int LN>
__global__ void __launch_bounds__(TCfg<LN>::threads) evalSplitT(EvalArgs a, int dsel) {
    using G = SplitGeo<BW, BH>;
    using T = TCfg<LN>;
    constexpr int NSUB = G::nsub, N = T::N, A = T::A, S = T::stride, NT = T::threads;
    __shared__ int16_t resS[NT * S], bkS[NT * S], qS[NT * S];
    __shared__ uint8_t pS[NT * A], rS[NT * A];
    const int per0 = dsel == 1 ? (G::sq ? 4 : 2) : dsel == 2 ? 16 : NSUB, off0 = dsel == 2 ? 4 : 0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.blockCount() * per0) return;
    const int b = a.blockIndex(idx / per0), k = off0 + idx % per0;
    const Frame &f = a.f;
    const int x0 = (b % a.cols) * BW, y0 = (b / a.cols) * BH;
    const Cand &c = *a.candidate(b, x0, y0);
    if (x0 >= f.w || y0 >= f.h || c.pruned) return;
    const size_t o = size_t(b) * NSUB + k;
    auto none = [&]() { a.subD[o] = kInf; a.subR[o] = 0.f; a.subNz[o] = 0; a.subType[o] = 0; };
    if (c.trialDepth != 255 && (c.trialDepth == 0 || (c.trialDepth == 1) != (k < (G::sq ? 4 : 2)))) { none(); return; }
    if (a.splitPrune && !c.nzWhole) { none(); return; }
    if (dsel == 2 && c.trialDepth != 2) {
        float d1 = 0.f, r1 = 0.f;
        for (int j = 0; j < 4; j++) { d1 += a.subD[size_t(b) * NSUB + j]; r1 += a.subR[size_t(b) * NSUB + j]; }
        const float j0 = c.lumaD + a.lambda * c.lumaBits, j1 = d1 + a.lambda * (r1 + 1.f);
        if (!(j1 < a.d2Margin * j0)) { none(); return; }
    }
    const int d = k < 4 ? 1 : 2, kk = k < 4 ? k : k - 4, per = 1 << d;
    const int qx = G::sq ? (kk % per) * N : (BW > BH ? k * N : 0), qy = G::sq ? (kk / per) * N : (BW > BH ? 0 : k * N);
    const int pw = f.w, ph = f.h, pitch = f.pitch[0];
    const uint8_t *src = f.src[0], *predF = a.pred;
    int16_t *res = resS + threadIdx.x * S, *bk = bkS + threadIdx.x * S, *q = qS + threadIdx.x * S;
    uint8_t *pq = pS + threadIdx.x, *rq = rS + threadIdx.x;   // element i at [i * NT]
    for (int i = 0; i < A; i++) {
        const int x = x0 + qx + (i & (N - 1)), y = y0 + qy + (i >> LN);
        int dd = 0;
        uint8_t pv = 0;
        if (x < pw && y < ph) { pv = predF[size_t(y) * pw + x]; dd = (int)src[y * pitch + x] - pv; }
        pq[i * NT] = pv;
        bk[i] = (int16_t)dd;
    }
    EvalIn e{};
    e.txSz = T::txSz; e.txType = AV1G_DCT_DCT; e.plane = 0;
    e.dcQ = av1g_dc_q(a.qindex); e.acQ = av1g_ac_q(a.qindex); e.qm = nullptr;
    e.rnd = a.rnd; e.zbin = a.zbin;
    e.skipCtx = 2; e.dcCtx = 0; e.intraDir = -1; e.reduced = 1;
    e.C = g_cost[a.qctx];
    e.qmDist = 0; e.qmRnd = 0;
    uint8_t *tmap = d == 1 ? a.types2 : a.types3;
    const bool enabled = tmap != nullptr && a.splitTypes < 2, exhaustive = a.txExhaustive != 0 && a.splitTypes == 0;
    const int set = enabled ? pft2::interTxSet(T::txSz, a.reducedTxSet) : AV1G_TX_SET_DCTONLY;
    const int sq = AV1G_T(Tx_Size_Sqr)[T::txSz];
    const int n = pft2::interTxCount(set);
    float t[A], D, R;
    auto full = [&](int type) {
        e.txType = type;
        fwdT<LN>(bk, type, t);
        quantT<LN>(t, q, e, D, R);
        inverseTS<LN>(q, type, e.dcQ, e.acQ, res, pq, rq, NT);
    };
    full(AV1G_DCT_DCT);
    int best = AV1G_DCT_DCT;
    int any = 0;
#pragma unroll
    for (int i = 0; i < A; i++) any |= q[i];
    if (set != AV1G_TX_SET_DCTONLY && any) {
        float bestJ = D + a.lambda * (R + interTypeBits(set, sq, pft2::interTxIndex(set, AV1G_DCT_DCT)));
        constexpr int kTop = 3;
        const bool screen = n > 4;
        int top[kTop] = {-1, -1, -1};
        float topS[kTop] = {3.4e38f, 3.4e38f, 3.4e38f};
        if (screen)
            for (int i = 0; i < n; i++) {
                const int ty = pft2::interTxInv(set, i);
                if (ty == AV1G_DCT_DCT || (!exhaustive && (av1g_flip_ud(ty) || av1g_flip_lr(ty) || ty >= AV1G_V_ADST))) continue;
                fwdT<LN>(bk, ty, t);
                float sc = 0.f;
#pragma unroll
                for (int j = 0; j < A; j++) sc += fabsf(t[j]);
                sc *= sqrtf(g_txg[T::txSz][ty]);
                int tt = ty;
#pragma unroll
                for (int m = 0; m < kTop; m++)
                    if (sc < topS[m]) { const float fs = topS[m]; const int ft = top[m]; topS[m] = sc; top[m] = tt; sc = fs; tt = ft; }
            }
        const int nCand = screen ? kTop : n;
        for (int ci = 0; ci < nCand; ci++) {
            const int ty = screen ? top[ci] : pft2::interTxInv(set, ci);
            if (ty < 0 || ty == AV1G_DCT_DCT || (!screen && !exhaustive && (av1g_flip_ud(ty) || av1g_flip_lr(ty) || ty >= AV1G_V_ADST)))
                continue;
            e.txType = ty;
            fwdT<LN>(bk, ty, t);
            quantT<LN>(t, q, e, D, R);
            const float J = D + a.lambda * (R + interTypeBits(set, sq, pft2::interTxIndex(set, ty)));
            if (J < bestJ) { bestJ = J; best = ty; }
        }
        full(best);
        if (best != AV1G_DCT_DCT) {
            any = 0;
#pragma unroll
            for (int i = 0; i < A; i++) any |= q[i];
            if (!any) { best = AV1G_DCT_DCT; full(best); }
        }
        R += interTypeBits(set, sq, pft2::interTxIndex(set, best));
    } else if (set != AV1G_TX_SET_DCTONLY) {
        R += interTypeBits(set, sq, pft2::interTxIndex(set, AV1G_DCT_DCT));
    }
    const int tstride = f.aw >> 2;
    if (tmap)
        for (int i = 0; i < (N >> 2) * (N >> 2); i++)
            tmap[size_t(((y0 + qy) >> 2) + i / (N >> 2)) * tstride + ((x0 + qx) >> 2) + i % (N >> 2)] = (uint8_t)best;
    int16_t *dstLev = d == 1 ? a.levels2 : a.levels3;
    uint8_t *dstRec = d == 1 ? a.recon2 : a.recon3;
    int anyS = 0, sse = 0;
    for (int i = 0; i < A; i++) {
        anyS |= q[i];
        const int x = x0 + qx + (i & (N - 1)), y = y0 + qy + (i >> LN);
        dstLev[size_t(y) * f.aw + x] = q[i];
        if (x >= pw || y >= ph) continue;
        const int dd = (int)src[y * pitch + x] - rq[i * NT];
        sse += dd * dd;
        dstRec[y * pw + x] = rq[i * NT];
    }
    a.subD[o] = (float)sse;
    a.subR[o] = R;
    a.subNz[o] = anyS != 0;
    a.subType[o] = (uint8_t)best;
}

// Split-external evaluation, last step: luma depth from the evalSplit results, chroma with the type of the top-left
// luma transform, and the block's skip / coded costs.
template <int BW, int BH>
__device__ __forceinline__ void evalFinalOne(EvalArgs a, const int slot) {
    using G = SplitGeo<BW, BH>;
    constexpr int NSUB = G::nsub, CW = BW / 2, CH = BH / 2;   // chroma block (one transform: CW, CH <= 32)
    __shared__ int16_t resS[4][CW * CH], bkS[4][CW * CH], qS[4][CW * CH];
    __shared__ uint8_t pS[4][CW * CH], recS[4][CW * CH];
    __shared__ float tmpS[4][(CW + 1) * CH];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = a.blockIndex(slot);
    const Frame &f = a.f;
    const int x0 = (b % a.cols) * BW, y0 = (b / a.cols) * BH;
    Cand &c = *a.candidate(b, x0, y0);
    if (x0 >= f.w || y0 >= f.h || c.pruned) return;
    float splitD[3] = {kInf, 0.f, 0.f}, splitBits[3] = {0.f, 0.f, 0.f};
    bool nzSplit[3] = {false, false, false};
    for (int k = 0; k < NSUB; k++) {
        const int d = k < 4 ? 1 : 2;
        splitD[d] += a.subD[size_t(b) * NSUB + k];
        splitBits[d] += a.subR[size_t(b) * NSUB + k];
        nzSplit[d] = nzSplit[d] || a.subNz[size_t(b) * NSUB + k];
    }
    if (!G::sq || BW < 16) splitD[2] = kInf;
    const float lumaD = c.lumaD, lumaBits = c.lumaBits;
    int depth = 0;
    float bestJ = lumaD + a.lambda * lumaBits;
    for (int d = 1; d <= 2; d++)
        if (splitD[d] + a.lambda * (splitBits[d] + 1.f + 2.f * (d - 1)) < bestJ) {
            bestJ = splitD[d] + a.lambda * (splitBits[d] + 1.f + 2.f * (d - 1));
            depth = d;
        }
    const int chromaType = depth == 1 ? a.subType[size_t(b) * NSUB] : depth == 2 ? a.subType[size_t(b) * NSUB + 4] : c.wholeType;
    const int cw = f.w >> 1, chh = f.h >> 1;
    const size_t ysz = size_t(f.w) * f.h, csz = size_t(cw) * chh;
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    constexpr int lw = Log2<CW>::v, lh = Log2<CH>::v;
    const int txSz = txSizeOf(lw, lh);
    float dC = 0.f, bC = 0.f;
    bool nzC = false;
    for (int p = 1; p <= 2; p++) {
        const int px = x0 >> 1, py = y0 >> 1, pitch = f.pitch[p];
        const uint8_t *src = f.src[p], *predF = a.pred + ysz + (p - 1) * csz;
        int16_t *res = resS[warp], *bk = bkS[warp], *q = qS[warp];
        uint8_t *pred = pS[warp], *rec = recS[warp];
        float en = 0.f;
        for (int i = lane; i < CW * CH; i += 32) {
            const int r = i >> lw, col = i & (CW - 1);
            int dd = 0;
            uint8_t pv = 0;
            if (px + col < cw && py + r < chh) { pv = predF[size_t(py + r) * cw + px + col]; dd = (int)src[(py + r) * pitch + px + col] - pv; }
            pred[i] = pv;
            res[i] = bk[i] = (int16_t)dd;
            en += (float)(dd * dd);
        }
        __syncwarp();
        EvalIn e{};
        e.txSz = txSz; e.plane = p;
        e.txType = pft2::interChromaType(chromaType, txSz, a.reducedTxSet);
        e.dcQ = dcQ; e.acQ = acQ; e.qm = nullptr;
        e.rnd = a.rnd; e.zbin = a.zbin;
        e.skipCtx = 7; e.dcCtx = 0; e.intraDir = -1; e.reduced = 1;
        e.C = g_cost[a.qctx];
        e.resEnergy = warp_sum(en);
        e.qmDist = 0; e.qmRnd = 0;
        EvalOut o;
        warp_fwd_ct<CW, CH>(res, tmpS[warp], txSz, e.txType);
        warp_quant_rate_rt<8>(tmpS[warp], q, e, o, CW, CH);
        inverseAddCt<lw, lh>(q, txSz, e.txType, dcQ, acQ, res, pred, rec);
        bC += o.R;
        int any = 0;
        for (int i = lane; i < CW * CH; i += 32) any |= q[i];
        nzC = nzC || __any_sync(0xffffffffu, any);
        int sseC = 0;
        uint8_t *outRec = a.recon + ysz + (p - 1) * csz;
        for (int i = lane; i < CW * CH; i += 32) {
            const int r = i >> lw, col = i & (CW - 1);
            if (px + col >= cw || py + r >= chh) continue;
            const int dd = (int)src[(py + r) * pitch + px + col] - rec[i];
            sseC += dd * dd;
            outRec[size_t(py + r) * cw + px + col] = rec[i];
        }
        dC += (float)warp_sumi(sseC);
        const int apw = f.aw / 2;
        int16_t *dst = a.levels + size_t(f.aw) * f.ah + size_t(p - 1) * (f.aw / 2) * (f.ah / 2) + size_t(py) * apw + px;
        for (int i = lane; i < CW * CH; i += 32) {
            const int r = i >> lw, col = i & (CW - 1);
            dst[size_t(r) * apw + col] = q[i];
        }
        __syncwarp();
    }
    if (!lane) {
        const float modeBits = a.modeBase + c.mvBits * a.mvWeight;
        const float skipFlag1 = a.skipBits1, skipFlag0 = a.skipBits0;
        c.costSkip = c.dSkipAll + a.lambda * (modeBits + skipFlag1);
        float dCoded = lumaD + dC, bits = lumaBits + bC;
        bool nz = c.nzWhole || nzC;
        if (depth) {
            dCoded += splitD[depth] - lumaD;
            bits += splitBits[depth] + 1.f + 2.f * (depth - 1) - lumaBits;
            nz = nzSplit[depth] || nzC;
        }
        c.txSplit = (uint8_t)depth;
        c.costCoded = dCoded + a.lambda * (modeBits + skipFlag0 + (G::sq ? 0.f : 0.5f) + bits);
        c.nz = nz;
    }
}

template <int BW, int BH, bool LOOP = false>
__global__ void __launch_bounds__(128) evalFinal(EvalArgs a) {
    const int n = a.blockCount();
    const int slot = blockIdx.x * 4 + (threadIdx.x >> 5);
    if constexpr (LOOP) {
        for (int i = slot; i < n; i += gridDim.x * 4) evalFinalOne<BW, BH>(a, i);
    } else if (slot < n) {
        evalFinalOne<BW, BH>(a, slot);
    }
}

template <int BW, int BH>
void launchEvaluation(EvalArgs a, bool splitExternal, cudaStream_t stream,
                      const char *whole, const char *split, const char *final) {
    constexpr int warps = EvalCfg<BW, BH>::warps;
    const int nb = a.cols * a.rows;
    // sparse refinement trials: a few CTAs per SM walk the trial list (grid-stride kernels)
    static const int trialGrid = [] {
        int dev = 0, sms = 1;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        return getenv("AV1GPU_PF_TRIALGRID") ? atoi(getenv("AV1GPU_PF_TRIALGRID")) * sms : 4 * sms;
    }();
    auto grid = [&](int g) { return a.trials ? std::min(g, trialGrid) : g; };
    const int blocks = (nb + warps - 1) / warps;
    if (!a.recon2 || !splitExternal) {
        if (a.trials) evaluate<BW, BH, false, true><<<grid(blocks), 32 * warps, 0, stream>>>(a);
        else evaluate<BW, BH><<<blocks, 32 * warps, 0, stream>>>(a);
        return;
    }
    {
        FineTimer t(a.trials ? nullptr : whole);
        constexpr int threads = 32 * warps * EvalCfg<BW, BH>::stripes;
        if (a.trials) evaluate<BW, BH, true, true><<<grid(blocks), threads, 0, stream>>>(a);
        else evaluate<BW, BH, true><<<blocks, threads, 0, stream>>>(a);
    }
    {
        FineTimer t(a.trials ? nullptr : split);
        // thread-per-transform kernels for dense launches (sparse refinement trials keep one warp per transform: few
        // transforms in flight are latency-bound)
        constexpr int t8 = TCfg<3>::threads;
        const bool s4 = g_split4 && !a.trials, s8 = g_split8 && !a.trials;
        if constexpr (BW == 8 && BH == 8) {
            if (s4) evalSplitT<8, 8, 2><<<(nb * 4 + 127) / 128, 128, 0, stream>>>(a, 0);
            else if (a.trials) evalSplit<BW, BH, true><<<grid((nb * 4 + 3) / 4), 128, 0, stream>>>(a, 0); else evalSplit<BW, BH><<<(nb * 4 + 3) / 4, 128, 0, stream>>>(a, 0);
        } else if constexpr (BW == 16 && BH == 16) {
            if (s8) evalSplitT<16, 16, 3><<<(nb * 4 + t8 - 1) / t8, t8, 0, stream>>>(a, 1);
            else if (a.trials) evalSplit<BW, BH, true><<<grid(nb), 128, 0, stream>>>(a, 1); else evalSplit<BW, BH><<<nb, 128, 0, stream>>>(a, 1);
            if (s4) evalSplitT<16, 16, 2><<<(nb * 16 + 127) / 128, 128, 0, stream>>>(a, 2);
            else if (a.trials) evalSplit<BW, BH, true><<<grid(nb * 4), 128, 0, stream>>>(a, 2); else evalSplit<BW, BH><<<nb * 4, 128, 0, stream>>>(a, 2);
        } else if constexpr (BW == 32 && BH == 32) {
            if (a.trials) evalSplit<BW, BH, true><<<grid(nb), 128, 0, stream>>>(a, 1); else evalSplit<BW, BH><<<nb, 128, 0, stream>>>(a, 1);
            if (s8) evalSplitT<32, 32, 3><<<(nb * 16 + t8 - 1) / t8, t8, 0, stream>>>(a, 2);
            else if (a.trials) evalSplit<BW, BH, true><<<grid(nb * 4), 128, 0, stream>>>(a, 2); else evalSplit<BW, BH><<<nb * 4, 128, 0, stream>>>(a, 2);
        } else if constexpr (BW == BH && BW >= 16) {
            if (a.trials) evalSplit<BW, BH, true><<<grid(nb), 128, 0, stream>>>(a, 1); else evalSplit<BW, BH><<<nb, 128, 0, stream>>>(a, 1);
            if (a.trials) evalSplit<BW, BH, true><<<grid(nb * 4), 128, 0, stream>>>(a, 2); else evalSplit<BW, BH><<<nb * 4, 128, 0, stream>>>(a, 2);
        } else {
            if (a.trials) evalSplit<BW, BH, true><<<grid((nb * SplitGeo<BW, BH>::nsub + 3) / 4), 128, 0, stream>>>(a, 0); else evalSplit<BW, BH><<<(nb * SplitGeo<BW, BH>::nsub + 3) / 4, 128, 0, stream>>>(a, 0);
        }
    }
    { FineTimer t(a.trials ? nullptr : final); if (a.trials) evalFinal<BW, BH, true><<<grid((nb + 3) / 4), 128, 0, stream>>>(a); else evalFinal<BW, BH><<<(nb + 3) / 4, 128, 0, stream>>>(a); }
}

// ------------------------------------------------------------------------------------------------
// Bottom-up partition choice: level s combines the four level s-1 results (parallel over blocks), then every 8x8
// unit walks down from 64x64 to the block that covers it.
struct DecideArgs {
    const Cand *cand[kSizes];
    int cols[kSizes];
    float *best[kSizes];      // best cost of the subtree rooted at each block
    uint8_t *mode[kSizes];    // 0: split, 1: none + skip, 2: none + coded, 3: HORZ, 4: VERT, 5: HORZ_4, 6: VERT_4,
                              // 7 + (0 HORZ_A, 1 HORZ_B, 2 VERT_A, 3 VERT_B)
    const Cand *rcand[10];    // rectangular blocks: (level - 1) * 2 + (0 HORZ, 1 VERT); 6 + (level - 2) * 2 + 4:1
    int rcols[10];
    int rect;
    int w, h;
    int gcols, grows;
    BlockInfo *grid;
    float lambda;
    float partBits[kSizes][2];   // [size][0: none, 1: split]
};

__global__ void decideLevel(DecideArgs a, int s, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int cols = a.cols[s];
    if (i >= cols * rows) return;
    const int n = 8 << s, x = (i % cols) * n, y = (i / cols) * n;
    if (x >= a.w || y >= a.h) { a.best[s][i] = 0.f; a.mode[s][i] = 0; return; }
    const bool hasRows = y + n / 2 < a.h, hasCols = x + n / 2 < a.w;
    float none = kInf;
    int noneMode = 1;
    if ((hasRows && hasCols) || s == 0) {
        const Cand &c = a.cand[s][i];
        const float coded = c.nz ? c.costCoded : kInf;
        noneMode = coded < c.costSkip ? 2 : 1;
        none = (noneMode == 2 ? coded : c.costSkip) + a.lambda * a.partBits[s][0];
    }
    float split = kInf;
    if (s > 0) {
        split = a.lambda * a.partBits[s][1];
        const int cc = a.cols[s - 1], cx = 2 * (i % cols), cy = 2 * (i / cols);
        for (int k = 0; k < 4; k++) split += a.best[s - 1][(cy + (k >> 1)) * cc + cx + (k & 1)];
    }
    float best = split;
    int mode = 0;
    if (none <= best) { best = none; mode = noneMode; }
    if (a.rect && s > 0 && hasRows && hasCols) {
        auto halfCost = [](const Cand &c) { return fminf(c.costSkip, c.nz ? c.costCoded : kInf); };
        const int bx = i % cols, by = i / cols;
        for (int d = 0; d < 2; d++) {
            const int rk = (s - 1) * 2 + d;
            const Cand *rc = a.rcand[rk];
            const int rcols = a.rcols[rk];
            float c2 = a.lambda * 3.0f;
            for (int k = 0; k < 2; k++)
                c2 += halfCost(d == 0 ? rc[(2 * by + k) * rcols + bx] : rc[by * rcols + 2 * bx + k]);
            if (c2 < best) { best = c2; mode = 3 + d; }
        }
        {
            // three-way partitions: two squares of the half size and one rectangular half
            const Cand *sq = a.cand[s - 1];
            const int sc = a.cols[s - 1];
            const Cand *rh = a.rcand[(s - 1) * 2], *rv = a.rcand[(s - 1) * 2 + 1];
            const int rhc = a.rcols[(s - 1) * 2], rvc = a.rcols[(s - 1) * 2 + 1];
            auto S = [&](int y, int x) { return halfCost(sq[(2 * by + y) * sc + 2 * bx + x]); };
            const float bitsAB = a.lambda * 3.5f;
            const float c[4] = {
                S(0, 0) + S(0, 1) + halfCost(rh[(2 * by + 1) * rhc + bx]),   // HORZ_A
                halfCost(rh[(2 * by) * rhc + bx]) + S(1, 0) + S(1, 1),       // HORZ_B
                S(0, 0) + S(1, 0) + halfCost(rv[by * rvc + 2 * bx + 1]),     // VERT_A
                halfCost(rv[by * rvc + 2 * bx]) + S(0, 1) + S(1, 1)};        // VERT_B
            for (int k = 0; k < 4; k++)
                if (c[k] + bitsAB < best) { best = c[k] + bitsAB; mode = 7 + k; }
        }
        if (s >= 2)
            for (int d = 0; d < 2; d++) {
                const int rk = 6 + (s - 2) * 2 + d;
                const Cand *rc = a.rcand[rk];
                const int rcols = a.rcols[rk];
                float c4 = a.lambda * 3.5f;
                for (int k = 0; k < 4; k++)
                    c4 += halfCost(d == 0 ? rc[(4 * by + k) * rcols + bx] : rc[by * rcols + 4 * bx + k]);
                if (c4 < best) { best = c4; mode = 5 + d; }
            }
    }
    a.best[s][i] = best;
    a.mode[s][i] = (uint8_t)mode;
}

__global__ void decideGrid(DecideArgs a) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x, gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= a.gcols || gy >= a.grows) return;
    BlockInfo bi{};
    if (gx * 8 < a.w && gy * 8 < a.h) {
#pragma unroll
        for (int s = kSizes - 1; s >= 0; s--) {
            const int sh = s, idx = (gy >> sh) * a.cols[s] + (gx >> sh);
            const int m = a.mode[s][idx];
            if (m >= 7) {
                // three-way node: this unit is in a half-size square or in the rectangular half
                const int t = m - 7;   // 0 HORZ_A, 1 HORZ_B, 2 VERT_A, 3 VERT_B
                const int lowHalf = t < 2 ? (gy >> (s - 1)) & 1 : (gx >> (s - 1)) & 1;
                const bool inRect = (t == 0 || t == 2) ? lowHalf : !lowHalf;
                const Cand *c;
                if (inRect) {
                    const int rk = (s - 1) * 2 + (t >= 2);
                    c = t < 2 ? &a.rcand[rk][(gy >> (s - 1)) * a.rcols[rk] + (gx >> s)]
                              : &a.rcand[rk][(gy >> s) * a.rcols[rk] + (gx >> (s - 1))];
                    bi.log2 = (uint8_t)(3 + s);
                    bi.part = (uint8_t)(t < 2 ? 1 : 2);
                } else {
                    c = &a.cand[s - 1][(gy >> (s - 1)) * a.cols[s - 1] + (gx >> (s - 1))];
                    bi.log2 = (uint8_t)(2 + s);
                    bi.part = 0;
                }
                const bool coded = c->nz && c->costCoded < c->costSkip;
                bi.txSplit = coded ? c->txSplit : 0;
                bi.mv = c->mv;
                bi.mv2 = c->mv2;
                bi.skip = (uint8_t)!coded;
                bi.ref = c->ref;
                bi.filter = c->filter;
                bi.ab = (uint8_t)(AV1G_PARTITION_HORZ_A + t);
                bi.abLog2 = (uint8_t)(3 + s);
                break;
            }
            if (m >= 3) {
                const int rk = m <= 4 ? (s - 1) * 2 + (m - 3) : 6 + (s - 2) * 2 + (m - 5);
                const int sh = m <= 4 ? s - 1 : s - 2;   // log2 of 8x8 units per block along the split direction
                const int ridx = (m == 3 || m == 5) ? (gy >> sh) * a.rcols[rk] + (gx >> s) : (gy >> s) * a.rcols[rk] + (gx >> sh);
                const Cand &c = a.rcand[rk][ridx];
                bi.mv = c.mv;
                bi.mv2 = c.mv2;
                bi.txSplit = 0;
                bi.log2 = (uint8_t)(3 + s);
                bi.part = (uint8_t)(m - 2);
                bi.txSplit = (c.nz && c.costCoded < c.costSkip) ? c.txSplit : 0;
                bi.skip = (uint8_t)!(c.nz && c.costCoded < c.costSkip);
                bi.ref = c.ref;
                bi.filter = c.filter;
                break;
            }
            if (m || s == 0) {
                const Cand &c = a.cand[s][idx];
                bi.mv = c.mv;
                bi.mv2 = c.mv2;
                bi.txSplit = m == 2 ? c.txSplit : 0;
                bi.log2 = (uint8_t)(3 + s);
                bi.skip = (uint8_t)(m != 2);
                bi.ref = c.ref;
                bi.filter = c.filter;
                break;
            }
        }
    }
    a.grid[gy * a.gcols + gx] = bi;
}

// Gather the chosen reconstruction and levels. Shapes: 0..3 squares 8..64, 4 + (level - 1) * 2 + (0 HORZ, 1 VERT).
constexpr int kShapes = 14;   // + 10 + (level - 2) * 2 + (0 HORZ_4, 1 VERT_4)
__device__ int blockShape(const BlockInfo &bi) {
    const int s = bi.log2 - 3;
    return !bi.part ? s : bi.part <= 2 ? 4 + (s - 1) * 2 + (bi.part - 1) : 10 + (s - 2) * 2 + (bi.part - 3);
}
struct CommitArgs {
    const BlockInfo *grid;
    int gcols, w, h, aw, ah;
    const uint8_t *pred[kShapes], *recon[kShapes];
    const int16_t *levels[kShapes];
    const uint8_t *recon2[kShapes], *recon3[kSizes];
    const int16_t *levels2[kShapes], *levels3[kSizes];
    uint8_t *out;
    int16_t *outLevels;
    const uint8_t *changed;  // optional: commit only blocks replaced by mode refinement
};
template <class T, int K>
__device__ __forceinline__ T pickN(int s, const T (&v)[K]) {
    T r = v[0];
#pragma unroll
    for (int k = 1; k < K; k++)
        if (s == k) r = v[k];
    return r;
}
// Luma transform types of the chosen blocks into the grid (one nibble per 4x4 of each 8x8 unit).
struct TypeArgs {
    BlockInfo *grid;
    int gcols, grows, tstride, enabled;
    const uint8_t *types1[kShapes], *types2[kShapes], *types3[kSizes];
    const uint8_t *changed;
};
__global__ void fillTxTypes(TypeArgs a) {
    const int gx = blockIdx.x * blockDim.x + threadIdx.x, gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= a.gcols || gy >= a.grows) return;
    if (a.changed && !a.changed[gy * a.gcols + gx]) return;
    BlockInfo &bi = a.grid[gy * a.gcols + gx];
    uint16_t v = 0;
    if (a.enabled && bi.log2 && !bi.skip) {
        const int s = bi.log2 - 3;
        const int shape = blockShape(bi);
        const uint8_t *m = bi.txSplit == 2 ? pickN(s, a.types3) : bi.txSplit ? pickN(shape, a.types2) : pickN(shape, a.types1);
        for (int q = 0; q < 4; q++)
            v |= uint16_t(m[size_t(gy * 2 + (q >> 1)) * a.tstride + gx * 2 + (q & 1)] & 15) << (4 * q);
    }
    bi.txTypes = v;
}
// One sample of plane p at (x, y) of the aligned area: its level and (inside the frame) its reconstruction.
__device__ __forceinline__ void commitSample(const CommitArgs &a, int p, int x, int y, int gyc, int gxc) {
    const int sub = p > 0;
    const int apw = a.aw >> sub;
    const BlockInfo bi = a.grid[gyc * a.gcols + gxc];
    const size_t lbase = p ? size_t(a.aw) * a.ah + size_t(p - 1) * (a.aw / 2) * (a.ah / 2) : 0;
    const int s = bi.log2 - 3;
    const int shape = blockShape(bi);
    int16_t level = 0;
    if (bi.log2 && !bi.skip) {
        if (p == 0 && bi.txSplit) level = (bi.txSplit == 2 ? pickN(s, a.levels3) : pickN(shape, a.levels2))[size_t(y) * apw + x];
        else level = pickN(shape, a.levels)[lbase + size_t(y) * apw + x];
    }
    a.outLevels[lbase + size_t(y) * apw + x] = level;
    const int pw = a.w >> sub, ph = a.h >> sub;
    if (x >= pw || y >= ph) return;
    const size_t off = (p ? size_t(a.w) * a.h + size_t(p - 1) * pw * ph : 0) + size_t(y) * pw + x;
    if (p == 0 && !bi.skip && bi.txSplit) {
        a.out[off] = (bi.txSplit == 2 ? pickN(s, a.recon3) : pickN(shape, a.recon2))[off];
        return;
    }
    a.out[off] = (bi.skip ? pickN(shape, a.pred) : pickN(shape, a.recon))[off];
}
__global__ void commit(CommitArgs a) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    const int p = blockIdx.z, sub = p > 0;
    const int apw = a.aw >> sub, aph = a.ah >> sub;
    if (x >= apw || y >= aph) return;
    // samples beyond the frame (aligned area) belong to the block covering the last row / column of cells: their levels
    // are coefficients of transforms that start inside the frame (the grid is only filled inside the frame)
    const int gyc = min((y << sub) >> 3, (a.h + 7) / 8 - 1), gxc = min((x << sub) >> 3, (a.w + 7) / 8 - 1);
    if (a.changed && !a.changed[gyc * a.gcols + gxc]) return;
    commitSample(a, p, x, y, gyc, gxc);
}
// Refinement passes: one warp per 8 x 8 unit of the aligned area (64 luma, 2 x 16 chroma samples), only where its cell
// changed (a per-sample grid spent most of its time on unchanged cells).
__global__ void __launch_bounds__(256) commitUnits(CommitArgs a) {
    const int u = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int ucols = a.aw / 8, urows = a.ah / 8;
    if (u >= ucols * urows) return;
    const int ux = u % ucols, uy = u / ucols;
    const int gyc = min(uy, (a.h + 7) / 8 - 1), gxc = min(ux, (a.w + 7) / 8 - 1);
    if (!a.changed[gyc * a.gcols + gxc]) return;
    for (int i = lane; i < 64; i += 32) commitSample(a, 0, ux * 8 + (i & 7), uy * 8 + (i >> 3), gyc, gxc);
    commitSample(a, 1 + (lane >> 4), ux * 4 + (lane & 3), uy * 4 + ((lane >> 2) & 3), gyc, gxc);
}

// ------------------------------------------------------------------------------------------------
// In-loop filter maps (spec Skips, LoopfilterTxSizes, cdef_idx) from the decision grid. Every transform equals
// its block, so all transform edges are block edges and the intra-frame filter rules apply unchanged.
__device__ __forceinline__ int imin6(int v) { return v > 6 ? 6 : v; }
__global__ void buildFilterMaps(const BlockInfo *grid, int gcols, av1g_pf_frame f) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= f.MiCols || r >= f.MiRows) return;
    const BlockInfo bi = grid[(r >> 1) * gcols + (c >> 1)];
    const size_t k = size_t(r) * f.MiCols + c;
    f.skip[k] = bi.skip;
    const int lw = bi.log2 - (bi.part == 2) - 2 * (bi.part == 4), lh = bi.log2 - (bi.part == 1) - 2 * (bi.part == 3);
    const int sp = bi.skip ? 0 : bi.txSplit;
    const int tlw = imin6(lw), tlh = imin6(lh);
    f.lfTx[0][k] = (uint8_t)(lw == lh ? txSizeOf(tlw - sp, tlh - sp)
                                      : txSizeOf(tlw - (sp && lw > lh), tlh - (sp && lh > lw)));
    if (!(r & 1) && !(c & 1)) {
        const int cc = f.MiCols >> 1;
        const size_t kc = size_t(r >> 1) * cc + (c >> 1);
        f.lfTx[1][kc] = f.lfTx[2][kc] = (uint8_t)txSizeOf(lw - 1, lh - 1);
    }
}
__global__ void buildCdefIdx(const BlockInfo *grid, int gcols, int grows, av1g_pf_frame f) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= f.sbCols * f.sbRows) return;
    const int sx = i % f.sbCols, sy = i / f.sbCols;
    int coded = 0;
    for (int y = sy * 8; y < min(sy * 8 + 8, grows); y++)
        for (int x = sx * 8; x < min(sx * 8 + 8, gcols); x++) {
            const BlockInfo bi = grid[y * gcols + x];
            if (bi.log2 && !bi.skip && x * 8 < f.W && y * 8 < f.H) coded = 1;
        }
    f.cdefIdx[i] = coded ? 0 : -1;
}
// Sum of squared differences of a plane: four samples per thread (32 x 8 threads cover 128 x 8 samples), one atomic
// per CTA.
__global__ void __launch_bounds__(256) planeSse(const uint8_t *src, int pitch, const uint8_t *buf, int w, int h,
                                                unsigned long long *out) {
    __shared__ int red[8];
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 4, y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = 0;
    if (y < h) {
#pragma unroll
        for (int k = 0; k < 4; k++)
            if (x + k < w) {
                const int e = (int)src[y * pitch + x + k] - (int)buf[y * w + x + k];
                d += e * e;
            }
    }
    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if ((tid & 31) == 0) red[tid >> 5] = d;
    __syncthreads();
    if (!tid) {
        unsigned long long t = 0;
        for (int k = 0; k < 8; k++) t += (unsigned)red[k];
        if (t) atomicAdd(out, t);
    }
}

// ------------------------------------------------------------------------------------------------
// GPU tile entropy coding: one thread per AV1 tile.
// Inter tile entropy coding in four stages (pframe_tile.cuh): three parallel phases over superblocks (mode info and
// context maps, mode choice, symbol records), then one serial range coder per tile over the records.
struct TileJob {
    pft2::FrameSyntax fs;
    const BlockInfo *grid;
    const pf::Mv *mf;
    const int16_t *levels;
    pft2::MiInfo *mi;
    pft2::EntropyState *frameInit;       // default CDFs: symbol cdf ids and mode-choice costs
    pft2::EntropyState *coderInit;       // CDFs the range coder starts from (primary slot or defaults)
    uint32_t *packOff, *tileCount;       // records-only mode: per-block destination, per-tile record counts
    pft2::SymRec *packed;
    pft2::EntropyState *save;            // receives the frame-end CDFs of tile contextTile, or null
    int contextTile;
    int tileCols, tileRows;
    int colStart[65], rowStart[65];
    int sbCols, sbRows;
    uint8_t *ctxLevel[3], *ctxDc[3];
    int ctxStride[3];
    uint32_t *sbCount, *sbOffset;   // per block-list slot (superblock * 64 + index): record count and offset
    pft2::BlockEntry *entries;      // 64 per superblock
    uint32_t *entryCount;           // per superblock
    pft2::SymRec *recs;
    uint32_t recCap;
    int *error;
    uint16_t *ecBuf;
    size_t ecCapPerTile;
    uint8_t *out;
    int *sizes;
};
void setTileLayout(TileJob &j, int miCols, int miRows, int cl, int rl) {
    j.fs.miCols = miCols;
    j.fs.miRows = miRows;
    j.sbCols = (miCols + 15) >> 4;
    j.sbRows = (miRows + 15) >> 4;
    const int wSb = (j.sbCols + (1 << cl) - 1) >> cl, hSb = (j.sbRows + (1 << rl) - 1) >> rl;
    for (int sb = 0; sb < j.sbCols; sb += wSb) j.colStart[j.tileCols++] = sb * 16;
    j.colStart[j.tileCols] = miCols;
    for (int sb = 0; sb < j.sbRows; sb += hSb) j.rowStart[j.tileRows++] = sb * 16;
    j.rowStart[j.tileRows] = miRows;
}
__device__ __host__ inline void tileOfSb(const TileJob &j, int miRow, int miCol, int &tr, int &tc) {
    tc = 0;
    while (tc + 1 < j.tileCols && j.colStart[tc + 1] <= miCol) ++tc;
    tr = 0;
    while (tr + 1 < j.tileRows && j.rowStart[tr + 1] <= miRow) ++tr;
}
// Default inter-frame CDFs (spec Default_*_Cdf) of the adaptive symbols in pft2::InterCdfs.
static void defaultInterCdfs(pft2::InterCdfs &d) {
    memcpy(d.isInter, pft::Default_Is_Inter_Cdf, sizeof d.isInter);
    memcpy(d.singleRef, pft::Default_Single_Ref_Cdf, sizeof d.singleRef);
    memcpy(d.newMv, pft::Default_New_Mv_Cdf, sizeof d.newMv);
    memcpy(d.zeroMv, pft::Default_Zero_Mv_Cdf, sizeof d.zeroMv);
    memcpy(d.refMv, pft::Default_Ref_Mv_Cdf, sizeof d.refMv);
    memcpy(d.drl, pft::Default_Drl_Mode_Cdf, sizeof d.drl);
    memcpy(d.mvJoint, pft::Default_Mv_Joint_Cdf, sizeof d.mvJoint);
    for (int c = 0; c < 2; c++) {
        memcpy(d.mvSign[c], pft::Default_Mv_Sign_Cdf, sizeof d.mvSign[c]);
        memcpy(d.mvClass[c], pft::Default_Mv_Class_Cdf[c], sizeof d.mvClass[c]);
        memcpy(d.class0Bit[c], pft::Default_Mv_Class0_Bit_Cdf, sizeof d.class0Bit[c]);
        memcpy(d.class0Fr[c], pft::Default_Mv_Class0_Fr_Cdf[c], sizeof d.class0Fr[c]);
        memcpy(d.mvFr[c], pft::Default_Mv_Fr_Cdf[c], sizeof d.mvFr[c]);
        memcpy(d.mvBit[c], pft::Default_Mv_Bit_Cdf, sizeof d.mvBit[c]);
    }
    memcpy(d.txSet3, pft::Default_Inter_Tx_Type_Set3_Cdf, sizeof d.txSet3);
    memcpy(d.txSet1, pft::Default_Inter_Tx_Type_Set1_Cdf, sizeof d.txSet1);
    memcpy(d.interpFilter, pft::Default_Interp_Filter_Cdf, sizeof d.interpFilter);
    { const uint16_t uw[3] = {11570, 32768, 0}; memcpy(d.useWiener, uw, sizeof uw); }
    memcpy(d.txSet2, pft::Default_Inter_Tx_Type_Set2_Cdf, sizeof d.txSet2);
    memcpy(d.compMode, pft::Default_Comp_Mode_Cdf, sizeof d.compMode);
    memcpy(d.compRefType, pft::Default_Comp_Ref_Type_Cdf, sizeof d.compRefType);
    memcpy(d.compRef, pft::Default_Comp_Ref_Cdf, sizeof d.compRef);
    memcpy(d.compBwdRef, pft::Default_Comp_Bwd_Ref_Cdf, sizeof d.compBwdRef);
    memcpy(d.compoundMode, pft::Default_Compound_Mode_Cdf, sizeof d.compoundMode);
    memcpy(d.txfmSplit, pft::Default_Txfm_Split_Cdf, sizeof d.txfmSplit);
    memcpy(d.skipMode, pft2::kDefaultSkipModeCdf, sizeof d.skipMode);
    memcpy(d.yMode, pft::Default_Y_Mode_Cdf, sizeof d.yMode);
}
__global__ void initFrameCdfs(pft2::EntropyState *out, const pft2::EntropyState *primary, const pft2::InterCdfs *defaults,
                              int qindex) {
    if (threadIdx.x || blockIdx.x) return;
    if (primary) {
        *out = *primary;
    } else {
        av1g_cdfs_init(&out->cdf, qindex);
        out->inter = *defaults;
    }
}
__device__ __host__ inline void setupMotionCoder(pft2::TileCoder &t, const TileJob &j, int sb) {
    const int miRow = (sb / j.sbCols) * 16, miCol = (sb % j.sbCols) * 16;
    int tr, tc;
    tileOfSb(j, miRow, miCol, tr, tc);
    t.fs = &j.fs;
    t.grid = j.grid;
    t.mf = j.mf;
    t.mi = j.mi;
    t.rowStart = j.rowStart[tr]; t.rowEnd = j.rowStart[tr + 1];
    t.colStart = j.colStart[tc]; t.colEnd = j.colStart[tc + 1];
}
__device__ __host__ inline void setupCoder(pft2::TileCoder &t, const TileJob &j, int sb, int16_t *qv) {
    setupMotionCoder(t, j, sb);
    t.levels = j.levels;
    t.cdf = &j.frameInit->cdf;
    t.ic = &j.frameInit->inter;
    t.base = reinterpret_cast<const uint16_t *>(j.frameInit);
    for (int p = 0; p < 3; p++) { t.ctxLevel[p] = j.ctxLevel[p]; t.ctxDc[p] = j.ctxDc[p]; t.ctxStride[p] = j.ctxStride[p]; }
    t.qv = qv;
    t.recs = nullptr;
    t.nrec = 0;
    t.recCap = 0xffffffffu;
    t.entries = nullptr;
    t.nEntries = 0;
    t.st = pft2::TileStats{};
}
// Phase 1: one thread per superblock (mode info, context maps, decode-order block list).
__global__ void tilePhase1(TileJob j) {
    const int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= j.sbCols * j.sbRows) return;
    pft2::TileCoder t;
    setupMotionCoder(t, j, sb);
    t.st = {};
    t.phase = 1;
    t.entries = j.entries + size_t(sb) * 64;
    t.superblock((sb / j.sbCols) * 16, (sb % j.sbCols) * 16, (uint32_t)sb);
    if (t.st.error && j.error) atomicMax(j.error, t.st.error);
    j.entryCount[sb] = t.nEntries;
}

__global__ void refinementModeInfo(TileJob j, const uint8_t *changed) {
    const int sb = blockIdx.x, k = threadIdx.x;
    if (k >= (int)j.entryCount[sb]) return;
    const auto &e = j.entries[sb * 64 + k];
    // later passes: a block's mode info changes only where the previous pass accepted a trial (changed cells)
    if (changed && !changed[(e.r / 2) * j.fs.gcols + e.c / 2]) return;
    pft2::TileCoder t;
    t.fs = &j.fs;
    t.mi = j.mi;
    t.curOrder = e.order;
    t.fillModeInfo(e.r, e.c, e.lw, e.lh, j.grid[(e.r / 2) * j.fs.gcols + e.c / 2]);
}

struct RefinementArgs {
    Frame f;
    const uint8_t *phases[kMaxRefs];
    const Cand *candidates[kShapes];
    int cols[kShapes];
    ModeTrial *trials;
    int *trialCount;
    int *trialIndices[kShapes];
    float lambda, origMvWeight, screenWeight;
    int pass;
    int keepDepth;   // trials keep the block's transform depth (no split search)
};

struct MotionOption {
    Mv mv[2];
    float bits;
    int skipMode;   // skip_mode: compound NEAREST pair of LAST / ALTREF, no residual, EIGHTTAP
};
struct MotionOptions {
    // 13 implied compound pairs including mixed NEW modes, plus the searched vector pair, plus skip_mode.
    MotionOption option[15];
    int count;
    int same;   // equal to the menu of the previous pass: the screen and trial would repeat their outcome
};

// Price each distinct vector pair once. A mixed compound mode keeps the searched vector on its NEWMV list.
__device__ int motionOptions(pft2::TileCoder &t, const pft2::TileCoder::RefPair &rp, const pft2::TileCoder::Stack &s,
                             const BlockInfo &bi, MotionOption *out) {
    using namespace pft2;
    out[0] = {{bi.mv, rp.compound ? bi.mv2 : Mv{0, 0}}, kInf, 0};
    int n = 1;
    const int first = rp.compound ? NEAREST_NEARESTMV : NEARESTMV, last = rp.compound ? NEW_NEWMV : NEWMV;
    for (int mode = first; mode <= last; ++mode) {
        for (int idx = t.firstMvIdx(mode); idx <= t.lastMvIdx(s, mode); ++idx) {
            Mv target[2] = {out[0].mv[0], out[0].mv[1]};
            for (int list = 0; list < 1 + rp.compound; ++list) {
                const int m = t.modeOfList(mode, list);
                if (m != NEWMV) target[list] = t.modePredictor(s, m, idx, list);
            }
            Mv pred[2];
            const float bits = t.modeCost(s, rp.compound, mode, idx, target, pred);
            int k = 0;
            while (k < n && !(target[0] == out[k].mv[0] && target[1] == out[k].mv[1])) ++k;
            if (k == n) out[n++] = {{target[0], target[1]}, bits, 0};
            else out[k].bits = fminf(out[k].bits, bits);
        }
    }
    return n;
}

__global__ void priceMotion(TileJob j, pft2::InterCdfs *defaults, MotionOptions *menus, int pass, float skipModeCredit,
                            const uint8_t *changed) {
    const int sb = blockIdx.x, k = threadIdx.x;
    if (k >= (int)j.entryCount[sb]) return;
    const auto &b = j.entries[sb * 64 + k];
    if (pass > 0) {
        // the menu depends on the block and its MV-stack neighbourhood (rows up to 5 MI above, columns up to 5 MI to
        // the left, the top-right unit): unchanged by the previous pass, it is the same menu
        const int gy = b.r / 2, gx = b.c / 2, w8 = max(1, (1 << b.lw) >> 3), h8 = max(1, (1 << b.lh) >> 3);
        const int gcols = j.fs.gcols, grows = j.sbRows * 8;
        bool any = false;
        for (int y = max(gy - 3, 0); y < min(gy + h8, grows) && !any; y++)
            for (int x = max(gx - 3, 0); x <= min(gx + w8, gcols - 1); x++) any |= changed[y * gcols + x] != 0;
        if (!any) { menus[sb * 64 + k].same = true; return; }
    }
    const BlockInfo &bi = j.grid[(b.r / 2) * j.fs.gcols + b.c / 2];
    pft2::TileCoder t;
    setupMotionCoder(t, j, sb);
    t.ic = defaults;
    t.curOrder = b.order;
    const auto rp = t.blockRefs(bi);
    pft2::TileCoder::Stack stack;
    t.findMvStack(stack, b.r, b.c, 1 << (b.lw - 2), 1 << (b.lh - 2), rp);
    t.setParallelMvCtx(stack);
    MotionOption options[15];
    int n = motionOptions(t, rp, stack, bi, options);
    if (j.fs.skipMode && b.lw >= 3 && b.lh >= 3) {
        // skip_mode: the compound stack of LAST / ALTREF, NEAREST pair at index 0. Its rate is the skip_mode symbol
        // (context 1 of the default CDF) less what the block codes besides its mode and vectors and skip_mode
        // replaces: references, skip flag and interpolation filter, about 4.5 bits
        BlockInfo cb = bi;
        cb.ref = 2;
        const auto crp = t.blockRefs(cb);
        pft2::TileCoder::Stack cs;
        t.findMvStack(cs, b.r, b.c, 1 << (b.lw - 2), 1 << (b.lh - 2), crp);
        const Mv p0 = t.modePredictor(cs, pft2::NEARESTMV, 0, 0), p1 = t.modePredictor(cs, pft2::NEARESTMV, 0, 1);
        options[n++] = {{p0, p1}, pft2::symCost(defaults->skipMode[1], 1) - skipModeCredit, 1};
    }
    int count = 1;
    for (int i = 1; i < n; ++i)
        if (options[i].bits < options[0].bits) options[count++] = options[i];
    MotionOptions &menu = menus[sb * 64 + k];
    bool same = pass > 0 && menu.count == count;
    for (int i = 0; same && i < count; ++i)
        same = menu.option[i].mv[0] == options[i].mv[0] && menu.option[i].mv[1] == options[i].mv[1] &&
               menu.option[i].bits == options[i].bits && menu.option[i].skipMode == options[i].skipMode;
    for (int i = 0; i < count; ++i) menu.option[i] = options[i];
    menu.count = count;
    menu.same = same;
}

// Screen with 4x4 luma SATD on the existing search planes. Final acceptance uses exact interpolation,
// chroma and the quantized residual, including on boundaries and for compound prediction.
constexpr int kProposeWarps = 4;
__device__ int motionSatd(const RefinementArgs &a, const MotionOption &option, int ref, int x0, int y0, int lw,
                          int lh) {
    __shared__ int partial[kProposeWarps];
    const Frame &f = a.f;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tw = 1 << (lw - 2), nt = 1 << (lw + lh - 4);
    const int nref = ref == 2 ? 2 : 1;
    const uint8_t *planes[2];
    for (int i = 0; i < nref; ++i) {
        const Mv mv = option.mv[i];
        const int phase = ((mv.row & 6) << 1) + ((mv.col & 6) >> 1);
        planes[i] = a.phases[i ? 1 : ref == 3 ? 2 : ref == 1 ? 1 : 0] + size_t(phase) * f.w * f.h;
    }
    int sum = 0;
    for (int tile = warp * 2 + (lane >> 4); tile < nt; tile += 2 * kProposeWarps) {
        const int x = x0 + (tile & (tw - 1)) * 4 + (lane & 3), y = y0 + (tile / tw) * 4 + ((lane & 15) >> 2);
        int d = 0;
        if (x < f.w && y < f.h) {
            int pred = 0;
            for (int i = 0; i < nref; ++i) {
                const Mv mv = option.mv[i];
                const int px = min(max(x + (mv.col >> 3), 0), f.w - 1), py = min(max(y + (mv.row >> 3), 0), f.h - 1);
                pred += planes[i][size_t(py) * f.w + px];
            }
            d = f.src[0][size_t(y) * f.pitch[0] + x] - (nref == 2 ? (pred + 1) >> 1 : pred);
        }
        for (int stride = 1; stride < 16; stride *= 2) {
            const int other = __shfl_xor_sync(0xffffffffu, d, stride);
            d = (lane & stride) ? other - d : d + other;
        }
        sum += abs(d);
    }
    sum = warp_sumi(sum);
    if (!lane) partial[warp] = sum;
    __syncthreads();
    int total = 0;
    for (int w = 0; w < kProposeWarps; ++w) total += partial[w];
    __syncthreads();
    return total;
}

// One CTA per block: its warps share the SATD tiles (large blocks bound the kernel's duration otherwise).
__device__ __forceinline__ void proposeOne(TileJob j, RefinementArgs a, const MotionOptions *menus, const int e) {
    const int sb = e / 64, lane = threadIdx.x;
    if (e % 64 >= (int)j.entryCount[sb]) return;
    const auto &b = j.entries[e];
    const int gy = b.r / 2, gx = b.c / 2, pos = gy * j.fs.gcols + gx;
    ModeTrial &trial = a.trials[pos];
    // pass 0 initializes the block's cross-pass state (thread 0); the other threads must not read it then
    const bool retry = a.pass > 0 && trial.rejPass == a.pass - 1;
    const int nRej = a.pass > 0 ? trial.nRej : 0;
    __syncthreads();
    if (!lane) {
        trial.active = false;
        if (!a.pass) { trial.refined = false; trial.nRej = 0; trial.rejPass = -2; }
    }
    const MotionOptions &menu = menus[e];
    if (menu.count == 1 || (menu.same && !retry)) return;
    const BlockInfo &bi = j.grid[pos];
    const float origBits = menu.option[0].bits;
    int best = 0;
    float bestScore = kInf;
    for (int i = 1; i < menu.count; ++i) {
        const MotionOption option = menu.option[i];
        bool rejected = false;
        for (int r = 0; r < nRej; ++r)
            rejected |= option.mv[0] == trial.rejMv[r][0] && option.mv[1] == trial.rejMv[r][1] &&
                        origBits - option.bits <= trial.rejSaving[r];
        if (rejected) continue;
        const int satd = motionSatd(a, option, option.skipMode ? 2 : bi.ref, gx * 8, gy * 8, b.lw, b.lh);
        const float score = satd * 0.125f + a.screenWeight * option.bits;
        if (score < bestScore) { bestScore = score; best = i; }
    }
    const MotionOption orig = menu.option[0];
    const int satd = motionSatd(a, orig, bi.ref, gx * 8, gy * 8, b.lw, b.lh);
    if (!lane && bestScore < satd * 0.125f + a.screenWeight * orig.bits) {
        const MotionOption option = menu.option[best];
        const float saving = orig.bits - option.bits;
        const int shape = blockShape(bi);
        const int index = (gy >> (b.lh - 3)) * a.cols[shape] + (gx >> (b.lw - 3));
        const Cand &old = a.candidates[shape][index];
        if (trial.refined) trial.oldCost = trial.curCost;
        else trial.oldCost = (bi.skip ? old.costSkip : old.costCoded) - a.lambda * old.mvBits * a.origMvWeight;
        trial.cand.mv = option.mv[0];
        trial.cand.mv2 = option.mv[1];
        trial.cand.ref = option.skipMode ? 2 : bi.ref;
        trial.cand.skipModeTrial = (uint8_t)option.skipMode;
        trial.cand.trialDepth = a.keepDepth ? (bi.skip ? 0 : bi.txSplit) : 255;
        trial.skipMode = option.skipMode != 0;
        trial.cand.pruned = 0;
        trial.cand.mvBits = 0;
        trial.rateSaving = saving;
        trial.active = true;
        a.trialIndices[shape][atomicAdd(a.trialCount + shape, 1)] = index;
    }
}

// kProposeSlices CTAs per superblock, each taking every kProposeSlices-th of its entries (superblocks hold few blocks
// where 64x64 blocks dominate: a CTA per entry slot left most CTAs empty); the CTA's threads share each block's tiles.
constexpr int kProposeSlices = 8;
__global__ void __launch_bounds__(32 * kProposeWarps) proposeMotion(TileJob j, RefinementArgs a,
                                                                    const MotionOptions *menus) {
    const int sb = blockIdx.x / kProposeSlices, slice = blockIdx.x % kProposeSlices, n = (int)j.entryCount[sb];
    for (int k = slice; k < n; k += kProposeSlices) {
        proposeOne(j, a, menus, sb * 64 + k);
        __syncthreads();
    }
}

__global__ void acceptMotion(TileJob j, ModeTrial *trials, BlockInfo *grid, uint8_t *changed, float lambda, int pass) {
    const int sb = blockIdx.x, k = threadIdx.x;
    if (k >= (int)j.entryCount[sb]) return;
    const auto &b = j.entries[sb * 64 + k];
    const int gx = b.c / 2, gy = b.r / 2, cols = j.fs.gcols;
    ModeTrial &trial = trials[gy * cols + gx];
    if (!trial.active) return;
    const Cand &c = trial.cand;
    // a skip_mode trial is only valid without residual
    const bool skip = trial.skipMode || !c.nz || c.costSkip <= c.costCoded;
    const float cost = skip ? c.costSkip : c.costCoded;
    if (cost >= trial.oldCost + lambda * trial.rateSaving) {
        const int r = trial.nRej < ModeTrial::kRej ? trial.nRej++ : pass % ModeTrial::kRej;
        trial.rejMv[r][0] = c.mv;
        trial.rejMv[r][1] = c.mv2;
        trial.rejSaving[r] = trial.rateSaving;
        trial.rejPass = pass;
        return;
    }
    trial.refined = true;
    trial.curCost = cost;
    trial.nRej = 0;
    const int endY = min(gy + (1 << (b.lh - 3)), j.fs.miRows / 2);
    const int endX = min(gx + (1 << (b.lw - 3)), j.fs.miCols / 2);
    for (int y = gy; y < endY; ++y)
        for (int x = gx; x < endX; ++x) {
            BlockInfo &bi = grid[y * cols + x];
            bi.mv = c.mv;
            bi.mv2 = c.mv2;
            bi.skip = skip;
            bi.txSplit = skip ? 0 : c.txSplit;
            bi.filter = c.filter;
            if (trial.skipMode) bi.ref = 2;
            changed[y * cols + x] = 1;
        }
}

// ------------------------------------------------------------------------------------------------
// Intra blocks in the inter frame. Square blocks (8x8 .. 64x64) inside the frame may switch to intra prediction: luma
// DC / V / H / SMOOTH / SMOOTH_V / SMOOTH_H / PAETH (they read no above-right or below-left samples, and the intra edge
// filter leaves them unchanged), UV_DC_PRED chroma, one DCT_DCT transform per plane. intraDecide chooses with the inter
// reconstruction around each block (all blocks in parallel); intraRecon then predicts and reconstructs the chosen
// blocks from their final neighbours in decode order: a superblock with intra blocks waits for its left, above and
// above-left neighbours (one launch, superblocks taken by ticket in wavefront order).
// The last three (D135, D113, D157: 90 < angle < 180, angle_delta 0) read only AboveRow[-1 .. w-1] and
// LeftCol[-1 .. h-1] too; they need enable_intra_edge_filter = 0 (IntraArgs::directional).
constexpr int kIntraModes = 10, kIntraModesNoDir = 7;
// Luma candidates: the modes above (angle_delta 0), then directional modes with an angle delta that keeps the angle
// strictly between 90 and 180 degrees (V +1..+3, H -1..-3, D135 / D113 / D157 +-1..3), so no above-right / below-left
// sample is read. Code: mode | d << 4, d = angle_delta + 4 (0: delta 0).
#define PF_LD(m, d) ((m) | ((d) + 4) << 4)
constexpr int kLumaCands = 34;
__constant__ uint8_t c_lumaCands[kLumaCands] = {
    AV1G_DC_PRED, AV1G_V_PRED, AV1G_H_PRED, AV1G_SMOOTH_PRED, AV1G_SMOOTH_V_PRED, AV1G_SMOOTH_H_PRED, AV1G_PAETH_PRED,
    AV1G_D135_PRED, AV1G_D113_PRED, AV1G_D157_PRED,
    PF_LD(AV1G_V_PRED, 1), PF_LD(AV1G_V_PRED, 2), PF_LD(AV1G_V_PRED, 3),
    PF_LD(AV1G_H_PRED, -1), PF_LD(AV1G_H_PRED, -2), PF_LD(AV1G_H_PRED, -3),
    PF_LD(AV1G_D135_PRED, -3), PF_LD(AV1G_D135_PRED, -2), PF_LD(AV1G_D135_PRED, -1),
    PF_LD(AV1G_D135_PRED, 1), PF_LD(AV1G_D135_PRED, 2), PF_LD(AV1G_D135_PRED, 3),
    PF_LD(AV1G_D113_PRED, -3), PF_LD(AV1G_D113_PRED, -2), PF_LD(AV1G_D113_PRED, -1),
    PF_LD(AV1G_D113_PRED, 1), PF_LD(AV1G_D113_PRED, 2), PF_LD(AV1G_D113_PRED, 3),
    PF_LD(AV1G_D157_PRED, -3), PF_LD(AV1G_D157_PRED, -2), PF_LD(AV1G_D157_PRED, -1),
    PF_LD(AV1G_D157_PRED, 1), PF_LD(AV1G_D157_PRED, 2), PF_LD(AV1G_D157_PRED, 3)};
#undef PF_LD
__device__ __forceinline__ int lumaDelta(int code) { return code >> 4 ? (code >> 4) - 4 : 0; }
__constant__ uint8_t c_intraModes[kIntraModes] = {AV1G_DC_PRED, AV1G_V_PRED, AV1G_H_PRED, AV1G_SMOOTH_PRED,
                                                   AV1G_SMOOTH_V_PRED, AV1G_SMOOTH_H_PRED, AV1G_PAETH_PRED,
                                                   AV1G_D135_PRED, AV1G_D113_PRED, AV1G_D157_PRED};
struct IntraArgs {
    TileJob j;   // grid, decode-order block lists, tile layout
    BlockInfo *grid;   // writable view of j.grid
    const uint8_t *src[3];
    int pitch[3];
    uint8_t *recon;       // tight YUV420 (reconstruction before the in-loop filters)
    int16_t *levels;      // aligned planes
    int w, h, aw, ah;
    int qindex, qctx, reduced;
    int txSplit;   // TX_MODE_SELECT: intra blocks may split their luma transform
    float splitGate;
    int topK64;    // topK of blocks with a 64-sample side (-1: topK)
    int uvSad64;   // blocks with a 64-sample side: chroma by SAD screen (not every mode by RD)
    int topK;      // RD-check this many best luma modes by SAD (3: the third, second and first)
    int alt3;      // also RD-check the third best luma mode by SAD
    int alt2;      // RD-check the second best luma mode by SAD (else the best non-directional one)
    int uvModes;   // choose the chroma mode (else UV_DC_PRED)
    float uvBits;
    int cfl;        // CfL chroma candidate (blocks up to 32x32)
    float cflBits;  // extra bits charged to CfL (signs and alphas)
    int uvAll;
    int uvTip;    // the chosen chroma mode's cost (not UV_DC's) enters the intra / inter decision    // every chroma mode by RD (else UV_DC against the best by SAD)
    float margin;
    float sadGate;
    float earlyGate;
    int splitDct;   // split luma transforms of the reconstruction use DCT_DCT (no type search)
    int est;   // rank luma candidates by the quantizer estimate (DCT_DCT, no reconstruction), code the winner in full
    int satd;   // rank luma modes by 4x4 Hadamard SATD (else subsampled SAD)
    int splitIntra;        // 64x64 blocks may become four 32x32 intra blocks
    float splitPartBits;   // extra partition bits of that split   // > 0: skip blocks whose best intra SAD^2 / N^2 exceeds sadGate * the inter cost   // intra wins when margin * its cost is below the inter cost
    int deltaK;   // angle-delta candidates RD-checked (best by SAD) after the base modes
    int nLuma;    // luma candidates (with the angle deltas: kLumaCands)
    int nModes;   // candidate intra modes (kIntraModesNoDir without the directional ones)   // extra bits charged to a chroma mode other than UV_DC_PRED
    float rnd, zbin, lambda, intraBits, modeBase, skipBits0, skipBits1, mvWeight;
    const Cand *cand[kSizes];
    int cols[kSizes];
    const Cand *rcand[10];     // rectangular blocks (DecideArgs::rcand layout)
    int rcols[10];
    const ModeTrial *trials;   // refinement state (accepted trials' costs), or null
    uint32_t *sel;   // per grid unit (block top-left): mode + 1 | type << 4 | tx_depth << 8 | uv << 10 | d << 16, 0 inter
    int *done;
    unsigned *ticket;
    int tag;
};
template <int W, int H = W>
struct IntraScratch {
    static constexpr int TW = W > 32 ? 32 : W, TH = H > 32 ? 32 : H, CW = W / 2, CH = H / 2, TCH = CH > 32 ? 32 : CH;
    uint8_t ttype[16];   // luma transform types of a split block (raster order); first: same offset for every size
    uint8_t pred[W * H], rec[W * H];
    int16_t res[W * H];
    float tmp[TH * (W + 1)];
    int16_t q[TW * TH];
    uint8_t above[W + 4], left[H + 4];   // [0] = AboveRow[-1] / LeftCol[-1]
    // chroma planes (own buffers: a warp per plane in intraRecon)
    uint8_t predC[2][CW * CH], recC[2][CW * CH];
    int16_t qC[2][CW * CH], resC[2][CW * CH];
    float tmpC[2][TCH * (CW + 1)];
    uint8_t aboveC[2][CW + 4], leftC[2][CH + 4];
};
// AV1GPU_PF_INTRA_WORK: samples processed by the intra decision, [0] luma coding, [1] chroma coding, [2] CfL alpha
// search, [3] SAD screens (a deterministic work measure).
__device__ unsigned long long g_intraWork[4];
__device__ __forceinline__ void intraWork(int k, int n) { if ((threadIdx.x & 31) == 0) atomicAdd(&g_intraWork[k], (unsigned long long)n); }
// Rate-distortion terms of one plane of an intra block.
struct IntraPart { float dSkip, dCoded, rate; int nz, type, noSkip; };
// AboveRow[-1 .. n-1] and LeftCol[-1 .. n-1] of an n x n block at (x, y) of a plane (spec 7.11.2; the samples are
// inside the frame, so the above-right and frame limits do not matter for these indices). L2 loads: other
// superblocks' samples were written by other CTAs of the same launch.
__device__ void intraEdges(const uint8_t *plane, int pitch, int x, int y, int nA, int nL, bool haveA, bool haveL,
                           uint8_t *ab, uint8_t *lb) {
    const int lane = threadIdx.x & 31;
    for (int i = lane; i < nA; i += 32)
        ab[1 + i] = haveA ? __ldcg(plane + size_t(y - 1) * pitch + x + i) : haveL ? __ldcg(plane + size_t(y) * pitch + x - 1) : 127;
    for (int i = lane; i < nL; i += 32)
        lb[1 + i] = haveL ? __ldcg(plane + size_t(y + i) * pitch + x - 1) : haveA ? __ldcg(plane + size_t(y - 1) * pitch + x) : 129;
    if (!lane) {
        const uint8_t c = haveA && haveL ? __ldcg(plane + size_t(y - 1) * pitch + x - 1)
                        : haveA ? __ldcg(plane + size_t(y - 1) * pitch + x)
                        : haveL ? __ldcg(plane + size_t(y) * pitch + x - 1) : 128;
        ab[0] = lb[0] = c;
    }
    __syncwarp();
}
__device__ __forceinline__ int intraPixel(int code, int dc, int i, int j, int lw, int lh, const uint8_t *ab, const uint8_t *lb) {
    const int mode = code & 15;   // code: mode | (angle_delta + 4) << 4 for luma candidates with a delta
    return mode == AV1G_DC_PRED ? dc
         : av1g_pred_pixel(mode, av1g_is_directional(mode) ? AV1G_T(Mode_To_Angle)[mode] + 3 * lumaDelta(code) : 0, i, j, lw, lh, ab, lb);
}
// Luma of an intra block (one warp). mode < 0: choose the luma mode by SAD first; forcedType < 0: search the transform
// type of the intra set (8x8 / 16x16) by the quantizer's estimate.
template <int LW, int LH = LW>
__device__ IntraPart intraLuma(const IntraArgs &a, IntraScratch<1 << LW, 1 << LH> &S, int x0, int y0, bool haveA, bool haveL, int &mode,
                               int forcedType, int *altMode = nullptr, float sseGate = kInf, int *alt3Mode = nullptr,
                               int *sadOut = nullptr) {
    constexpr int W = 1 << LW, H = 1 << LH, TW = W > 32 ? 32 : W, TH = H > 32 ? 32 : H;
    const int lane = threadIdx.x & 31;
    const uint8_t *src = a.src[0];
    const int sp = a.pitch[0];
    intraEdges(a.recon, a.w, x0, y0, W, H, haveA, haveL, S.above, S.left);
    const uint8_t *ab = S.above + 1, *lb = S.left + 1;
    const int dcY = av1g_dc_value<8>(ab, lb, haveL, haveA, LW, LH);
    if (mode < 0) {
        int best = 0x7fffffff, bestNd = 0x7fffffff, third = 0x7fffffff;
        for (int k = 0; k < a.nLuma; k++) {
            const int m = c_lumaCands[k];
            // SAD screen on every other row and column of blocks >= 16x16 (the choice is checked by RD after), or
            // with AV1GPU_PF_INTRA_SATD the 4x4 Hadamard SATD of the residual over the whole block
            constexpr int SS = W >= 16 && H >= 16 ? 1 : 0, LS = LW - SS, WS = W >> SS, HS = H >> SS;
            int sad = 0;
            if (a.satd) {
                for (int b4 = lane; b4 < (W / 4) * (H / 4); b4 += 32) {
                    const int r0 = (b4 / (W / 4)) * 4, c0 = (b4 % (W / 4)) * 4;
                    int d[16];
#pragma unroll
                    for (int t = 0; t < 16; t++)
                        d[t] = (int)src[(y0 + r0 + t / 4) * sp + x0 + c0 + t % 4] - intraPixel(m, dcY, r0 + t / 4, c0 + t % 4, LW, LH, ab, lb);
#pragma unroll
                    for (int rr = 0; rr < 4; rr++) {   // rows
                        int *v = d + 4 * rr;
                        const int a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
                        v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
                    }
#pragma unroll
                    for (int cc = 0; cc < 4; cc++) {   // columns
                        const int a0 = d[cc] + d[4 + cc], a1 = d[cc] - d[4 + cc], a2 = d[8 + cc] + d[12 + cc], a3 = d[8 + cc] - d[12 + cc];
                        sad += abs(a0 + a2) + abs(a1 + a3) + abs(a0 - a2) + abs(a1 - a3);
                    }
                }
            } else {
                for (int i = lane; i < WS * HS; i += 32) {
                    const int r = (i >> LS) << SS, c = (i & (WS - 1)) << SS;
                    sad += abs((int)src[(y0 + r) * sp + x0 + c] - intraPixel(m, dcY, r, c, LW, LH, ab, lb));
                }
            }
            intraWork(3, WS * HS);
            sad = warp_sumi(sad);
            if (sadOut) sadOut[k] = sad;
            if (a.alt2) {   // runner-up by SAD (checked by RD in intraBlock)
                if (sad < best) {
                    third = bestNd; if (alt3Mode && altMode) *alt3Mode = *altMode;
                    bestNd = best; if (altMode) *altMode = mode; best = sad; mode = m;
                } else if (sad < bestNd) {
                    third = bestNd; if (alt3Mode && altMode) *alt3Mode = *altMode;
                    bestNd = sad; if (altMode) *altMode = m;
                } else if (sad < third) {
                    third = sad; if (alt3Mode) *alt3Mode = m;
                }
            } else {
                if (sad < best) { best = sad; mode = m; }
                if (k < kIntraModesNoDir && sad < bestNd) { bestNd = sad; if (altMode) *altMode = m; }
            }
        }
        // the prediction error is at least SAD^2 / N^2: far above the inter cost, intra cannot win
        if ((float)best * (float)best * (1.f / (W * H)) > sseGate) {
            IntraPart r{};
            r.dCoded = r.dSkip = kInf;
            return r;
        }
    }
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    float en = 0.f;
    for (int i = lane; i < W * H; i += 32) {
        const int r = i >> LW, c = i & (W - 1);
        const int p = intraPixel(mode, dcY, r, c, LW, LH, ab, lb);
        const int d = (int)src[(y0 + r) * sp + x0 + c] - p;
        S.pred[i] = (uint8_t)p;
        S.res[i] = (int16_t)d;
        en += (float)(d * d);
    }
    __syncwarp();
    EvalIn e{};
    e.txSz = txSizeOf(LW, LH);
    e.txType = AV1G_DCT_DCT;
    e.plane = 0;
    e.dcQ = dcQ; e.acQ = acQ; e.qm = nullptr;
    e.rnd = a.rnd; e.zbin = a.zbin;
    e.skipCtx = 0; e.dcCtx = 0; e.intraDir = mode & 15; e.reduced = a.reduced;
    e.C = g_cost[a.qctx];
    intraWork(0, W * H);
    e.resEnergy = warp_sum(en);
    e.qmDist = 0; e.qmRnd = 0;
    EvalOut o;
    int type = AV1G_DCT_DCT;
    const int set = av1g_get_tx_set_intra(e.txSz, a.reduced);
    if (forcedType >= 0) {
        type = forcedType;
    } else if (forcedType != -3 && set != AV1G_TX_SET_DCTONLY) {   // -3: DCT_DCT estimate (candidate ranking)
        float bestJ = kInf;
        const int n = set == AV1G_TX_SET_INTRA_1 ? 7 : 5;
        for (int k = 0; k < n; k++) {
            const int t = set == AV1G_TX_SET_INTRA_1 ? AV1G_T(Tx_Type_Intra_Inv_Set1)[k] : AV1G_T(Tx_Type_Intra_Inv_Set2)[k];
            e.txType = t;
            warp_fwd_ct<W, H>(S.res, S.tmp, e.txSz, t);
            warp_quant_rate_rt<8>(S.tmp, S.q, e, o, W, H);
            const float J = o.D + a.lambda * o.R;
            if (J < bestJ) { bestJ = J; type = t; }
        }
    }
    e.txType = type;
    warp_fwd_ct<W, H>(S.res, S.tmp, e.txSz, type);
    warp_quant_rate_rt<8>(S.tmp, S.q, e, o, W, H);
    if (forcedType == -3) {   // the quantizer's distortion estimate, no reconstruction
        int any = 0;
        for (int i = lane; i < TW * TH; i += 32) any |= S.q[i];
        IntraPart r;
        r.nz = __any_sync(0xffffffffu, any != 0);
        r.dCoded = o.D;
        r.dSkip = e.resEnergy;
        r.rate = o.R;
        r.type = AV1G_DCT_DCT;
        r.noSkip = 0;
        return r;
    }
    inverseAddCt<LW, LH>(S.q, e.txSz, type, dcQ, acQ, S.res, S.pred, S.rec);
    int sse = 0, any = 0;
    for (int i = lane; i < W * H; i += 32) {
        const int r = i >> LW, c = i & (W - 1);
        const int d = (int)src[(y0 + r) * sp + x0 + c] - S.rec[i];
        sse += d * d;
    }
    for (int i = lane; i < TW * TH; i += 32) any |= S.q[i];
    IntraPart r;
    r.nz = __any_sync(0xffffffffu, any != 0);
    r.dCoded = (float)warp_sumi(sse);
    r.dSkip = e.resEnergy;
    r.rate = o.R;
    r.type = r.nz ? type : AV1G_DCT_DCT;   // an all-zero luma transform codes no type
    r.noSkip = 0;
    return r;
}
// Edges of a transform at (x, y) inside the intra block at (x0, y0) (n x n): samples of the block come from its local
// reconstruction S.rec (earlier transforms in raster order), the others from the frame.
template <int N>
__device__ void intraEdgesLocal(const IntraArgs &a, const IntraScratch<N> &S, int x0, int y0, int x, int y, int n, bool haveA,
                                bool haveL, uint8_t *ab, uint8_t *lb) {
    const int lane = threadIdx.x & 31;
    auto at = [&](int xx, int yy) -> uint8_t {
        if (xx >= x0 && yy >= y0) return S.rec[(yy - y0) * N + xx - x0];
        return __ldcg(a.recon + size_t(yy) * a.w + xx);
    };
    for (int i = lane; i < n; i += 32) {
        ab[1 + i] = haveA ? at(x + i, y - 1) : haveL ? at(x - 1, y) : 127;
        lb[1 + i] = haveL ? at(x - 1, y + i) : haveA ? at(x, y - 1) : 129;
    }
    if (!lane) {
        const uint8_t c = haveA && haveL ? at(x - 1, y - 1) : haveA ? at(x, y - 1) : haveL ? at(x - 1, y) : 128;
        ab[0] = lb[0] = c;
    }
    __syncwarp();
}
// Luma of an intra block with tx_depth D (2^D x 2^D transforms of side N >> D, predicted one after the other in raster
// order from the reconstruction of the earlier ones), one warp, for a given mode. Transform types are searched per
// transform (S.ttype). With write, each transform's levels go to the frame's level planes. Such blocks never code skip.
template <int L, int D>
__device__ IntraPart intraLumaSplit(const IntraArgs &a, IntraScratch<1 << L> &S, int x0, int y0, bool haveA, bool haveL,
                                    int mode, bool write) {
    constexpr int N = 1 << L, LT = L - D, T = 1 << LT, PER = 1 << D;
    static_assert(T <= 32 && T >= 8, "split transforms of 8..32");
    const int lane = threadIdx.x & 31;
    const uint8_t *src = a.src[0];
    const int sp = a.pitch[0];
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    EvalIn e{};
    e.txSz = txSizeOf(LT, LT);
    e.plane = 0;
    e.dcQ = dcQ; e.acQ = acQ; e.qm = nullptr;
    e.rnd = a.rnd; e.zbin = a.zbin;
    e.skipCtx = 4; e.dcCtx = 0; e.intraDir = mode & 15; e.reduced = a.reduced;
    e.C = g_cost[a.qctx];
    e.qmDist = 0; e.qmRnd = 0;
    const int set = av1g_get_tx_set_intra(e.txSz, a.reduced);
    float dTot = 0.f, rTot = 0.f;
    int anyTot = 0;
    uint8_t *recT = reinterpret_cast<uint8_t *>(S.res + T * T);   // compact reconstruction of one transform
    for (int k = 0; k < PER * PER; k++) {
        const int tx = (k % PER) * T, ty = (k / PER) * T;
        intraEdgesLocal<N>(a, S, x0, y0, x0 + tx, y0 + ty, T, ty > 0 || haveA, tx > 0 || haveL, S.above, S.left);
        const uint8_t *ab = S.above + 1, *lb = S.left + 1;
        const int dc = av1g_dc_value<8>(ab, lb, tx > 0 || haveL, ty > 0 || haveA, LT, LT);
        float en = 0.f;
        for (int i = lane; i < T * T; i += 32) {
            const int r = i >> LT, c = i & (T - 1);
            const int pv = intraPixel(mode, dc, r, c, LT, LT, ab, lb);
            const int d = (int)src[(y0 + ty + r) * sp + x0 + tx + c] - pv;
            S.pred[i] = (uint8_t)pv;
            S.res[i] = (int16_t)d;
            en += (float)(d * d);
        }
        __syncwarp();
        e.resEnergy = warp_sum(en);
        EvalOut o;
        int type = AV1G_DCT_DCT;
        if (set != AV1G_TX_SET_DCTONLY && write && !a.splitDct) {   // the decision pass estimates with DCT_DCT
            float bestJ = kInf;
            const int n = set == AV1G_TX_SET_INTRA_1 ? 7 : 5;
            for (int t = 0; t < n; t++) {
                const int tt = set == AV1G_TX_SET_INTRA_1 ? AV1G_T(Tx_Type_Intra_Inv_Set1)[t] : AV1G_T(Tx_Type_Intra_Inv_Set2)[t];
                e.txType = tt;
                warp_fwd_ct<T, T>(S.res, S.tmp, e.txSz, tt);
                warp_quant_rate_rt<8>(S.tmp, S.q, e, o, T, T);
                const float J = o.D + a.lambda * o.R;
                if (J < bestJ) { bestJ = J; type = tt; }
            }
        }
        e.txType = type;
        warp_fwd_ct<T, T>(S.res, S.tmp, e.txSz, type);
        warp_quant_rate_rt<8>(S.tmp, S.q, e, o, T, T);
        inverseAddCt<LT, LT>(S.q, e.txSz, type, dcQ, acQ, S.res, S.pred, recT);
        int sse = 0, any = 0;
        for (int i = lane; i < T * T; i += 32) {
            const int r = i >> LT, c = i & (T - 1);
            const int d = (int)src[(y0 + ty + r) * sp + x0 + tx + c] - recT[i];
            sse += d * d;
            S.rec[(ty + r) * N + tx + c] = recT[i];
            any |= S.q[i];
            if (write) a.levels[size_t(y0 + ty + r) * a.aw + x0 + tx + c] = S.q[i];
        }
        const bool nz = __any_sync(0xffffffffu, any != 0);
        if (!lane) S.ttype[k] = (uint8_t)(nz ? type : AV1G_DCT_DCT);
        anyTot |= nz;
        dTot += (float)warp_sumi(sse);
        rTot += o.R;
        __syncwarp();
    }
    IntraPart r;
    r.nz = anyTot;
    r.dCoded = dTot;
    r.dSkip = kInf;
    r.rate = rTot + 1.f * D;   // tx_depth beyond 0 (approximate)
    r.type = AV1G_DCT_DCT;
    r.noSkip = 1;
    return r;
}
// Chroma plane p (1, 2) of an intra block, UV_DC_PRED (one warp).
// CfL (spec 7.11.5, 4:2:0): the luma value of chroma sample (r, c) from the block's W-wide luma reconstruction, and
// the block average of those values (all lanes).
__device__ __forceinline__ int cflLumaAt(const uint8_t *luma, int W, int r, int c) {
    const uint8_t *l = luma + (2 * r) * W + 2 * c;
    return (l[0] + l[1] + l[W] + l[W + 1]) << 1;
}
template <int LCW, int LCH>
__device__ __forceinline__ int cflLumaAvg(const uint8_t *luma) {
    constexpr int CW = 1 << LCW, CH = 1 << LCH;
    int t = 0;
    for (int i = threadIdx.x & 31; i < CW * CH; i += 32) t += cflLumaAt(luma, 2 * CW, i >> LCW, i & (CW - 1));
    return av1g_round2(warp_sumi(t), LCW + LCH);
}
__device__ __forceinline__ int cflPixel(int dc, int alpha, int v, int avg) {
    return min(max(dc + av1g_round2signed(alpha * (v - avg), 6), 0), 255);
}
template <int LW, int LH = LW>
__device__ IntraPart intraChroma(const IntraArgs &a, IntraScratch<1 << LW, 1 << LH> &S, int p, int x0, int y0, bool haveA, bool haveL,
                                 int uvMode, int cflAlpha = 0, const uint8_t *cflLuma = nullptr) {
    constexpr int CW = (1 << LW) / 2, CH = (1 << LH) / 2, LCW = LW - 1, LCH = LH - 1;
    const int lane = threadIdx.x & 31;
    const int cw = a.w >> 1, ch = a.h >> 1, k = p - 1;
    const uint8_t *plane = a.recon + size_t(a.w) * a.h + size_t(k) * cw * ch;
    const uint8_t *cs = a.src[p];
    const int cp = a.pitch[p], cx = x0 >> 1, cy = y0 >> 1;
    intraEdges(plane, cw, cx, cy, CW, CH, haveA, haveL, S.aboveC[k], S.leftC[k]);
    const uint8_t *ab = S.aboveC[k] + 1, *lb = S.leftC[k] + 1;
    const int dc = av1g_dc_value<8>(ab, lb, haveL, haveA, LCW, LCH);
    uint8_t *pr = S.predC[k], *rc = S.recC[k];
    int16_t *q = S.qC[k], *res = S.resC[k];
    const bool cfl = uvMode == AV1G_UV_CFL_PRED;
    const int avg = cfl ? cflLumaAvg<LCW, LCH>(cflLuma) : 0;
    float en = 0.f;
    for (int i = lane; i < CW * CH; i += 32) {
        const int r = i >> LCW, c = i & (CW - 1);
        const int pv = cfl ? cflPixel(dc, cflAlpha, cflLumaAt(cflLuma, 2 * CW, r, c), avg)
                           : intraPixel(uvMode, dc, r, c, LCW, LCH, ab, lb);
        const int d = (int)cs[(cy + r) * cp + cx + c] - pv;
        pr[i] = (uint8_t)pv;
        res[i] = (int16_t)d;
        en += (float)(d * d);
    }
    __syncwarp();
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    EvalIn e{};
    e.txSz = txSizeOf(LCW, LCH);
    e.txType = av1g_chroma_tx_type(e.txSz, uvMode, a.reduced);   // Mode_To_Txfm[UVMode] when in the set
    e.plane = p;
    e.dcQ = dcQ; e.acQ = acQ; e.qm = nullptr;
    e.rnd = a.rnd; e.zbin = a.zbin;
    e.skipCtx = 7; e.dcCtx = 0; e.intraDir = 0; e.reduced = a.reduced;
    e.C = g_cost[a.qctx];
    intraWork(1, CW * CH);
    e.resEnergy = warp_sum(en);
    e.qmDist = 0; e.qmRnd = 0;
    EvalOut o;
    warp_fwd_ct<CW, CH>(res, S.tmpC[k], e.txSz, e.txType);
    warp_quant_rate_rt<8>(S.tmpC[k], q, e, o, CW, CH);
    inverseAddCt<LCW, LCH>(q, e.txSz, e.txType, dcQ, acQ, res, pr, rc);
    int sse = 0, any = 0;
    for (int i = lane; i < CW * CH; i += 32) {
        const int r = i >> LCW, c = i & (CW - 1);
        const int d = (int)cs[(cy + r) * cp + cx + c] - rc[i];
        sse += d * d;
        any |= q[i];
    }
    IntraPart r;
    r.nz = __any_sync(0xffffffffu, any != 0);
    r.dCoded = (float)warp_sumi(sse);
    r.dSkip = e.resEnergy;
    r.rate = o.R;
    r.type = AV1G_DCT_DCT;
    r.noSkip = 0;
    return r;
}
// CfL alpha of plane p (-16..16, 1/8 units) with the least prediction SSE, from a W-wide luma reconstruction.
template <int LW, int LH>
__device__ int cflBestAlpha(const IntraArgs &a, IntraScratch<1 << LW, 1 << LH> &S, int p, int x0, int y0, bool haveA, bool haveL,
                            const uint8_t *luma) {
    constexpr int CW = (1 << LW) / 2, CH = (1 << LH) / 2, LCW = LW - 1, LCH = LH - 1;
    const int lane = threadIdx.x & 31;
    const int cw = a.w >> 1, ch = a.h >> 1, k = p - 1;
    const uint8_t *plane = a.recon + size_t(a.w) * a.h + size_t(k) * cw * ch;
    const uint8_t *cs = a.src[p];
    const int cp = a.pitch[p], cx = x0 >> 1, cy = y0 >> 1;
    intraEdges(plane, cw, cx, cy, CW, CH, haveA, haveL, S.aboveC[k], S.leftC[k]);
    const int dc = av1g_dc_value<8>(S.aboveC[k] + 1, S.leftC[k] + 1, haveL, haveA, LCW, LCH);
    const int avg = cflLumaAvg<LCW, LCH>(luma);
    int sse[33] = {};
    for (int i = lane; i < CW * CH; i += 32) {
        const int r = i >> LCW, c = i & (CW - 1);
        const int v = cflLumaAt(luma, 2 * CW, r, c), sv = cs[(cy + r) * cp + cx + c];
#pragma unroll
        for (int al = -16; al <= 16; al++) {
            const int d = sv - cflPixel(dc, al, v, avg);
            sse[al + 16] += d * d;
        }
    }
    intraWork(2, CW * CH * 33);
    int best = 0x7fffffff, bestA = 0;
#pragma unroll
    for (int al = 0; al < 33; al++) {
        const int t = warp_sumi(sse[al]);
        if (t < best) { best = t; bestA = al - 16; }
    }
    return bestA;
}
// Skip or coded, from the three planes' terms; returns the rate-distortion cost (mode bits included).
__device__ __forceinline__ float intraCombine(const IntraArgs &a, const IntraPart *pt, bool &skip) {
    const bool nz = pt[0].nz || pt[1].nz || pt[2].nz;
    const float jCoded = pt[0].dCoded + pt[1].dCoded + pt[2].dCoded + a.lambda * (pt[0].rate + pt[1].rate + pt[2].rate + a.skipBits0);
    const float jSkip = pt[0].dSkip + pt[1].dSkip + pt[2].dSkip + a.lambda * a.skipBits1;
    skip = pt[0].noSkip ? false : !nz || jSkip <= jCoded;
    return (skip ? jSkip : jCoded) + a.lambda * (a.intraBits + a.modeBase);
}
// Store plane p of an intra block (reconstruction and levels of the chosen option).
template <int LW, int LH = LW>
__device__ void intraWrite(const IntraArgs &a, const IntraScratch<1 << LW, 1 << LH> &S, int p, int x0, int y0, bool skip, int depth = 0) {
    constexpr int W = 1 << LW, H = 1 << LH, TW = W > 32 ? 32 : W, TH = H > 32 ? 32 : H, CW = W / 2, CH = H / 2, LCW = LW - 1;
    const int lane = threadIdx.x & 31;
    if (p == 0 && depth) {   // split luma: levels already stored, the block's reconstruction in S.rec
        for (int i = lane; i < W * H; i += 32) a.recon[size_t(y0 + (i >> LW)) * a.w + x0 + (i & (W - 1))] = S.rec[i];
        return;
    }
    if (p == 0) {
        for (int i = lane; i < W * H; i += 32) {
            const int r = i >> LW, c = i & (W - 1);
            a.recon[size_t(y0 + r) * a.w + x0 + c] = skip ? S.pred[i] : S.rec[i];
            a.levels[size_t(y0 + r) * a.aw + x0 + c] = (!skip && r < TH && c < TW) ? S.q[r * TW + c] : (int16_t)0;
        }
        return;
    }
    const int cw = a.w >> 1, ch = a.h >> 1, k = p - 1;
    uint8_t *plane = a.recon + size_t(a.w) * a.h + size_t(k) * cw * ch;
    int16_t *lv = a.levels + size_t(a.aw) * a.ah + size_t(k) * (a.aw / 2) * (a.ah / 2);
    const int cx = x0 >> 1, cy = y0 >> 1;
    for (int i = lane; i < CW * CH; i += 32) {
        const int r = i >> LCW, c = i & (CW - 1);
        plane[size_t(cy + r) * cw + cx + c] = skip ? S.predC[k][i] : S.recC[k][i];
        lv[size_t(cy + r) * (a.aw / 2) + cx + c] = skip ? (int16_t)0 : S.qC[k][i];
    }
}
// Chroma mode of an intra block by SAD over both planes (one warp).
template <int LW, int LH = LW>
__device__ int intraChromaMode(const IntraArgs &a, IntraScratch<1 << LW, 1 << LH> &S, int x0, int y0, bool haveA, bool haveL) {
    constexpr int CW = (1 << LW) / 2, CH = (1 << LH) / 2, LCW = LW - 1, LCH = LH - 1;
    const int lane = threadIdx.x & 31;
    const int cw = a.w >> 1, ch = a.h >> 1, cx = x0 >> 1, cy = y0 >> 1;
    int sad[kIntraModes] = {};
    for (int k = 0; k < 2; k++) {
        const uint8_t *plane = a.recon + size_t(a.w) * a.h + size_t(k) * cw * ch;
        intraEdges(plane, cw, cx, cy, CW, CH, haveA, haveL, S.aboveC[k], S.leftC[k]);
        const uint8_t *ab = S.aboveC[k] + 1, *lb = S.leftC[k] + 1, *cs = a.src[k + 1];
        const int cp = a.pitch[k + 1];
        const int dc = av1g_dc_value<8>(ab, lb, haveL, haveA, LCW, LCH);
        constexpr int SS = CW >= 16 && CH >= 16 ? 1 : 0, LS = LCW - SS, WS = CW >> SS, HS = CH >> SS;
        for (int m = 0; m < a.nModes; m++)
            for (int i = lane; i < WS * HS; i += 32) {
                const int r = (i >> LS) << SS, c = (i & (WS - 1)) << SS;
                sad[m] += abs((int)cs[(cy + r) * cp + cx + c] - intraPixel(c_intraModes[m], dc, r, c, LCW, LCH, ab, lb));
            }
    }
    int best = 0x7fffffff, mode = AV1G_DC_PRED;
    for (int m = 0; m < a.nModes; m++) {
        const int v = warp_sumi(sad[m]);
        if (v < best) { best = v; mode = c_intraModes[m]; }
    }
    return mode;
}
// Decision pass: one warp codes all three planes; returns the cost and the mode / luma type.
template <int LW, int LH = LW>
__device__ float intraBlock(const IntraArgs &a, IntraScratch<1 << LW, 1 << LH> &S, int x0, int y0, bool haveA, bool haveL, int &mode,
                            int &type, int &depth, float jInter, int &uvMode, int &cflU, int &cflV) {
    IntraPart pt[3];
    cflU = cflV = 0;
    int alt = -1, alt3 = -1, sads[kLumaCands];
    const int cft = a.est ? -3 : -1;   // candidate ranking: estimate only (the winner is coded in full below)
    pt[0] = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, mode, cft, &alt, a.sadGate > 0.f ? a.sadGate * jInter : kInf,
                              a.alt3 ? &alt3 : nullptr, sads);
    if (pt[0].dSkip >= kInf) return kInf;
    // the first luma candidate already puts intra far above the inter cost: skip the other candidates, chroma and CfL
    // (heuristic, AV1GPU_PF_INTRA_EARLYGATE; 0 off)
    if (a.earlyGate > 0.f) {
        const float lb = fminf(pt[0].dCoded + a.lambda * (pt[0].rate + a.skipBits0), pt[0].dSkip + a.lambda * a.skipBits1) +
                         a.lambda * (a.intraBits + a.modeBase);
        if (lb > a.earlyGate * jInter) return kInf;
    }
    // candidate comparison; with the estimate ranking the runner-up is kept too (coded in full below)
    int runner = -1;
    float runnerJ = kInf;
    auto consider = [&](const IntraPart &q, int m) {
        const float jq = fminf(q.dCoded + a.lambda * q.rate, q.dSkip);
        const float jb = fminf(pt[0].dCoded + a.lambda * pt[0].rate, pt[0].dSkip);
        if (jq <= jb) { runner = mode; runnerJ = jb; pt[0] = q; mode = m; }
        else if (jq < runnerJ) { runner = m; runnerJ = jq; }
    };
    if (alt >= 0 && alt != mode) {
        // the SAD choice is one of the directional modes: keep it only if it also wins by rate-distortion
        int m2 = alt;
        const IntraPart q = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, m2, cft);
        consider(q, alt);
    }
    if (alt3 >= 0 && alt3 != mode && alt3 != alt) {   // third best by SAD (AV1GPU_PF_INTRA_ALT3)
        int m3 = alt3;
        const IntraPart q = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, m3, cft);
        consider(q, alt3);
    }
    // AV1GPU_PF_INTRA_TOPK > 3: the next best modes by SAD too
    const int topK = (LW == 6 || LH == 6) && a.topK64 >= 0 ? a.topK64 : a.topK;   // 64-sample sides: lighter search
    if (topK > 0 && pt[0].dSkip < kInf) {
        const int first = mode, second = alt, third = alt3;
        auto jl = [&](const IntraPart &v) { return fminf(v.dCoded + a.lambda * v.rate, v.dSkip); };
        // base modes (angle_delta 0) first, then the deltaK best angle-delta candidates by SAD
        for (int rank = 0; rank < topK + a.deltaK; rank++) {
            const bool deltas = rank >= topK;
            int pick = -1, ps = 0x7fffffff;   // the smallest SAD among the candidates not yet tried
            for (int k = deltas ? a.nModes : 0; k < (deltas ? a.nLuma : a.nModes); k++) {
                const int m = c_lumaCands[k];
                if (m == first || m == second || m == third || sads[k] < 0) continue;
                if (sads[k] < ps) { ps = sads[k]; pick = k; }
            }
            if (pick < 0) break;
            sads[pick] = -1;
            int mk = c_lumaCands[pick];
            const IntraPart q = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, mk, cft);
            consider(q, c_lumaCands[pick]);
        }
    }
    if (a.est) {   // the chosen mode in full: transform type search and exact reconstruction (and the runner-up)
        if (a.est >= 2 && runner >= 0 && runner != mode) {
            int mr = runner;
            const IntraPart q = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, mr, -1);
            int mf = mode;
            const IntraPart b = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, mf, -1);
            const float jq = fminf(q.dCoded + a.lambda * q.rate, q.dSkip), jb = fminf(b.dCoded + a.lambda * b.rate, b.dSkip);
            if (jq < jb) { pt[0] = q; mode = runner; }
            else pt[0] = b;
        } else {
            int mf = mode;
            pt[0] = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, mf, -1);
        }
    }
    {
        // lower bound of the final cost (chroma adds distortion and bits; splits are only tried below splitGate x the
        // inter cost): above the threshold the decision cannot change, so skip chroma and splits
        const float lb = fminf(pt[0].dCoded + a.lambda * (pt[0].rate + a.skipBits0), pt[0].dSkip + a.lambda * a.skipBits1) +
                         a.lambda * (a.intraBits + a.modeBase);
        if (lb * a.margin >= jInter && (!a.txSplit || lb > a.splitGate * jInter || LW != LH || LW < 4)) return kInf;
    }
    // chroma: UV_DC_PRED against the best mode by SAD, by rate-distortion (one extra bit for the other mode)
    const bool uvAll = a.uvAll && !((LW == 6 || LH == 6) && a.uvSad64);
    const int uvCand = a.uvModes && !uvAll ? intraChromaMode<LW, LH>(a, S, x0, y0, haveA, haveL) : AV1G_DC_PRED;
    uvMode = AV1G_DC_PRED;
    pt[1] = intraChroma<LW, LH>(a, S, 1, x0, y0, haveA, haveL, AV1G_DC_PRED);
    pt[2] = intraChroma<LW, LH>(a, S, 2, x0, y0, haveA, haveL, AV1G_DC_PRED);
    if (a.uvModes && uvAll) {   // every chroma mode by rate-distortion (AV1GPU_PF_INTRA_UVALL)
        auto jc = [&](const IntraPart &q) { return fminf(q.dCoded + a.lambda * q.rate, q.dSkip); };
        float best = jc(pt[1]) + jc(pt[2]);
        IntraPart bu = pt[1], bv = pt[2];
        for (int k = 1; k < a.nModes; k++) {
            const int m = c_intraModes[k];
            const IntraPart u = intraChroma<LW, LH>(a, S, 1, x0, y0, haveA, haveL, m);
            const IntraPart v = intraChroma<LW, LH>(a, S, 2, x0, y0, haveA, haveL, m);
            const float j = jc(u) + jc(v) + a.lambda * a.uvBits;
            if (j < best) { best = j; uvMode = m; bu = u; bv = v; }
        }
        if (a.uvTip) { pt[1] = bu; pt[2] = bv; }   // the chosen chroma mode's cost enters the decision
        // CfL (blocks up to 32x32): the luma reconstruction of the chosen luma mode, the best alphas by SSE
        if constexpr ((1 << LW) <= 32 && (1 << LH) <= 32) {
            if (a.cfl) {
                int mt = mode;
                intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, mt, pt[0].type);   // S.rec: the luma reconstruction
                const int aU = cflBestAlpha<LW, LH>(a, S, 1, x0, y0, haveA, haveL, S.rec);
                const int aV = cflBestAlpha<LW, LH>(a, S, 2, x0, y0, haveA, haveL, S.rec);
                if (aU || aV) {
                    const IntraPart u = intraChroma<LW, LH>(a, S, 1, x0, y0, haveA, haveL, AV1G_UV_CFL_PRED, aU, S.rec);
                    const IntraPart v = intraChroma<LW, LH>(a, S, 2, x0, y0, haveA, haveL, AV1G_UV_CFL_PRED, aV, S.rec);
                    const float j = jc(u) + jc(v) + a.lambda * a.cflBits;
                    if (j < best) { best = j; uvMode = AV1G_UV_CFL_PRED; cflU = aU; cflV = aV; }
                }
            }
        }
    } else if (uvCand != AV1G_DC_PRED) {
        const IntraPart u = intraChroma<LW, LH>(a, S, 1, x0, y0, haveA, haveL, uvCand);
        const IntraPart v = intraChroma<LW, LH>(a, S, 2, x0, y0, haveA, haveL, uvCand);
        auto jc = [&](const IntraPart &q) { return fminf(q.dCoded + a.lambda * q.rate, q.dSkip); };
        if (jc(u) + jc(v) + a.lambda * a.uvBits < jc(pt[1]) + jc(pt[2])) uvMode = uvCand;
    }
    // the intra / inter decision and the luma choices use the UV_DC_PRED cost (a better chroma mode only
    // refines blocks that are intra anyway: letting it tip the decision lost on video_2)
    bool skip;
    float J = intraCombine(a, pt, skip);
    type = skip ? AV1G_DCT_DCT : pt[0].type;
    depth = 0;
    // luma transform splits (8x8 .. 32x32 transforms) with the same mode
    // splits only where intra is close to winning (AV1GPU_PF_INTRA_SPLITGATE, default 1.3)
    if (!a.txSplit || J > a.splitGate * jInter) return J;
    if constexpr (LW == LH && LW >= 4) {
        pt[0] = intraLumaSplit<LW, 1>(a, S, x0, y0, haveA, haveL, mode, false);
        const float j1 = intraCombine(a, pt, skip);
        if (j1 < J) { J = j1; depth = 1; type = 0; }
    }
    if constexpr (LW == LH && LW >= 5) {
        pt[0] = intraLumaSplit<LW, 2>(a, S, x0, y0, haveA, haveL, mode, false);
        const float j2 = intraCombine(a, pt, skip);
        if (j2 < J) { J = j2; depth = 2; type = 0; }
    }
    return J;
}

__device__ __forceinline__ void intraAvail(const TileJob &j, int miRow, int miCol, bool &haveA, bool &haveL) {
    int tr, tc;
    tileOfSb(j, miRow, miCol, tr, tc);
    haveA = miRow > j.rowStart[tr];
    haveL = miCol > j.colStart[tc];
}
// Warps per CTA of the intra decision of a W x H block (shared memory per warp grows with the area).
template <int LW, int LH> struct IntraCfg {
    static constexpr int area = (1 << LW) * (1 << LH), warps = area >= 2048 ? 1 : area >= 512 ? 2 : 4;
};
// Log2 width and height of a grid unit's block (square, HORZ / VERT halves, HORZ_4 / VERT_4 quarters).
__device__ __forceinline__ void blockDims(const BlockInfo &bi, int &lw, int &lh) {
    lw = bi.log2 - (bi.part == 2) - 2 * (bi.part == 4);
    lh = bi.log2 - (bi.part == 1) - 2 * (bi.part == 3);
}
// Evaluation candidate of the block at grid unit (gx, gy): a square of cand[], or a rectangle of rcand[].
__device__ __forceinline__ const Cand &blockCand(const IntraArgs &a, const BlockInfo &bi, int gx, int gy) {
    int lw, lh;
    blockDims(bi, lw, lh);
    if (!bi.part) {
        const int K = bi.log2 - 3;
        return a.cand[K][(gy >> K) * a.cols[K] + (gx >> K)];
    }
    const int s = bi.log2 - 3;
    const int rk = bi.part <= 2 ? (s - 1) * 2 + (bi.part - 1) : 6 + (s - 2) * 2 + (bi.part - 3);
    return a.rcand[rk][(gy >> (lh - 3)) * a.rcols[rk] + (gx >> (lw - 3))];
}
template <int LW, int LH>
__global__ void __launch_bounds__(32 * IntraCfg<LW, LH>::warps) intraDecide(IntraArgs a) {
    constexpr int W = 1 << LW, H = 1 << LH, WARPS = IntraCfg<LW, LH>::warps, UW = W / 8, UH = H / 8;
    constexpr int L = LW;   // square blocks: their size
    __shared__ IntraScratch<W, H> S[WARPS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gcols = a.j.fs.gcols, bcols = (gcols + UW - 1) / UW;
    const int idx = blockIdx.x * WARPS + warp;
    const int gx = (idx % bcols) * UW, gy = (idx / bcols) * UH;
    if (gy * 8 >= a.h) return;
    const BlockInfo bi = a.j.grid[gy * gcols + gx];
    const int x0 = gx * 8, y0 = gy * 8;
    int blw, blh;
    blockDims(bi, blw, blh);
    if (!bi.log2 || blw != LW || blh != LH || x0 + W > a.w || y0 + H > a.h) return;
    const Cand &c = blockCand(a, bi, gx, gy);
    float jInter = bi.skip ? c.costSkip : c.costCoded;
    if (a.trials) {
        const ModeTrial &t = a.trials[gy * gcols + gx];
        if (t.refined) jInter = t.curCost + a.lambda * c.mvBits * a.mvWeight;
    }
    // an intra block costs at least its mode bits: blocks already cheaper keep inter prediction
    if (jInter <= a.lambda * (a.intraBits + a.modeBase + fminf(a.skipBits0, a.skipBits1))) {
        if (!lane) a.sel[gy * gcols + gx] = 0;
        return;
    }
    bool haveA, haveL;
    intraAvail(a.j, gy * 2, gx * 2, haveA, haveL);
    int mode = -1;
    int type = 0, depth = 0, uvMode = 0, cflU = 0, cflV = 0;
    const float jIntra = intraBlock<LW, LH>(a, S[warp], x0, y0, haveA, haveL, mode, type, depth, jInter, uvMode, cflU, cflV);
    const bool useIntra = jIntra * a.margin < jInter;
    // bits 19-24 / 25-30: CfL alphas + 16
    uint32_t v = useIntra ? (uint32_t)(((mode & 15) + 1) | type << 4 | depth << 8 | uvMode << 10 | (mode >> 4) << 16 |
                                       (uint32_t)(cflU + 16) << 19 | (uint32_t)(cflV + 16) << 25) : 0;
    if constexpr (LW == 6 && LH == 6) {
        // four 32x32 intra blocks instead (partition SPLIT), each with its own modes; the quadrants' neighbours inside
        // the block are approximated by the inter reconstruction, as the outer ones are
        if (a.splitIntra && jIntra < kInf) {
            auto &S32 = *reinterpret_cast<IntraScratch<32> *>(&S[warp]);
            float j4 = a.lambda * a.splitPartBits;
            uint32_t cs[4];
            for (int q = 0; q < 4 && j4 < kInf; q++) {
                int m = -1, t = 0, d = 0, uv = 0, cu = 0, cv = 0;
                const float jq = intraBlock<5, 5>(a, S32, x0 + (q & 1) * 32, y0 + (q >> 1) * 32, (q >> 1) ? true : haveA,
                                               (q & 1) ? true : haveL, m, t, d, kInf, uv, cu, cv);
                j4 += jq;
                cs[q] = (uint32_t)(((m & 15) + 1) | t << 4 | d << 8 | uv << 10 | (m >> 4) << 16 | (uint32_t)(cu + 16) << 19 |
                                   (uint32_t)(cv + 16) << 25);
            }
            if (j4 * a.margin < jInter && j4 < (useIntra ? jIntra : kInf)) {
                if (!lane) {
                    a.sel[gy * gcols + gx] = cs[0] | 1u << 14;   // bit 14: the 64x64 block splits
                    a.sel[gy * gcols + gx + 4] = cs[1];
                    a.sel[(gy + 4) * gcols + gx] = cs[2];
                    a.sel[(gy + 4) * gcols + gx + 4] = cs[3];
                }
                return;
            }
        }
    }
    if (!lane) a.sel[gy * gcols + gx] = v;
}
// 64x64 blocks whose intra decision chose four 32x32 intra blocks: rewrite their grid units (the block lists are
// rebuilt after this).
__global__ void applyIntraSplit(BlockInfo *grid, uint32_t *sel, int gcols, int grows) {
    const int gx = (blockIdx.x * blockDim.x + threadIdx.x) * 8, gy = (blockIdx.y * blockDim.y + threadIdx.y) * 8;
    if (gx >= gcols || gy >= grows) return;
    if (!(sel[gy * gcols + gx] & (1u << 14))) return;
    sel[gy * gcols + gx] &= ~(1u << 14);
    for (int y = gy; y < min(gy + 8, grows); y++)
        for (int x = gx; x < min(gx + 8, gcols); x++) {
            BlockInfo &b = grid[y * gcols + x];
            b.log2 = 5;
            b.part = 0;
            b.ab = 0;
            b.abLog2 = 0;
        }
}
// Reconstruction pass helper: warp w codes plane w of the block, then all decide skip from the shared terms.
template <int LW, int LH = LW>
__device__ void intraReconBlock(const IntraArgs &a, IntraScratch<1 << LW, 1 << LH> &S, IntraPart *pt, int x0, int y0, bool haveA,
                                bool haveL, int mode, int type, int depth, int uvMode, int cflU, int cflV, int &skipOut,
                                int &typeOut) {
    const int warp = threadIdx.x >> 5;
    const bool cfl = uvMode == AV1G_UV_CFL_PRED;   // chroma waits for the luma reconstruction
    IntraPart r;
    if (warp == 0) {
        if constexpr (LW == LH && LW >= 5) { if (depth == 2) r = intraLumaSplit<LW, 2>(a, S, x0, y0, haveA, haveL, mode, true); }
        if constexpr (LW == LH && LW >= 4) { if (depth == 1) r = intraLumaSplit<LW, 1>(a, S, x0, y0, haveA, haveL, mode, true); }
        if (depth == 0) r = intraLuma<LW, LH>(a, S, x0, y0, haveA, haveL, mode, type);
    }
    if (cfl) __syncthreads();
    if (warp > 0) r = intraChroma<LW, LH>(a, S, warp, x0, y0, haveA, haveL, uvMode, warp == 1 ? cflU : cflV, S.rec);
    if ((threadIdx.x & 31) == 0) pt[warp] = r;
    __syncthreads();
    bool skip;
    intraCombine(a, pt, skip);
    // a skipped CfL block predicts chroma from the luma prediction (its reconstruction)
    if (cfl && skip && warp > 0) intraChroma<LW, LH>(a, S, warp, x0, y0, haveA, haveL, uvMode, warp == 1 ? cflU : cflV, S.pred);
    intraWrite<LW, LH>(a, S, warp, x0, y0, skip, depth);
    skipOut = skip;
    typeOut = skip ? AV1G_DCT_DCT : pt[0].type;
}
__global__ void __launch_bounds__(96) intraRecon(IntraArgs a) {
    __shared__ IntraScratch<64> S;
    __shared__ IntraPart pt[3];
    __shared__ int sbShared;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int sbCols = a.j.sbCols, sbRows = a.j.sbRows, gcols = a.j.fs.gcols;
    if (!tid) {
        int t = (int)atomicAdd(a.ticket, 1u);
        for (int d = 0;; d++) {   // the t-th superblock in (anti-diagonal, row) order
            const int r0 = max(0, d - (sbCols - 1)), r1 = min(d, sbRows - 1), n = r1 - r0 + 1;
            if (t < n) { sbShared = (r0 + t) * sbCols + d - (r0 + t); break; }
            t -= n;
        }
    }
    __syncthreads();
    const int sb = sbShared, sr = sb / sbCols, sc = sb % sbCols;
    const int n = (int)a.j.entryCount[sb];
    const pft2::BlockEntry *list = a.j.entries + size_t(sb) * 64;
    bool any = false;
    for (int k = tid; k < n; k += 96) any |= a.sel[(list[k].r >> 1) * gcols + (list[k].c >> 1)] != 0;
    if (__syncthreads_or(any)) {
        if (!tid) {
            const volatile int *done = a.done;
            if (sr > 0) while (done[(sr - 1) * sbCols + sc] != a.tag) {}
            if (sc > 0) while (done[sr * sbCols + sc - 1] != a.tag) {}
            if (sr > 0 && sc > 0) while (done[(sr - 1) * sbCols + sc - 1] != a.tag) {}
            __threadfence();
        }
        __syncthreads();
        for (int k = 0; k < n; k++) {
            const pft2::BlockEntry be = list[k];
            const int gx = be.c >> 1, gy = be.r >> 1;
            const int s = a.sel[gy * gcols + gx];
            if (!s) continue;
            bool haveA, haveL;
            intraAvail(a.j, be.r, be.c, haveA, haveL);
            const int mode = ((s & 15) - 1) | ((s >> 16) & 7) << 4, ft = (s >> 4) & 15, depth = (s >> 8) & 3,
                      uvMode = (s >> 10) & 15, cflU = int((s >> 19) & 63) - 16, cflV = int((s >> 25) & 63) - 16;
            int skip = 0, type = 0;
            const int x0 = gx * 8, y0 = gy * 8;
#define PF_IRB(LW, LH)                                                                                         \
    if (be.lw == LW && be.lh == LH)                                                                            \
        intraReconBlock<LW, LH>(a, *reinterpret_cast<IntraScratch<1 << LW, 1 << LH> *>(&S), pt, x0, y0, haveA, haveL, \
                                mode, ft, depth, uvMode, cflU, cflV, skip, type);
            PF_IRB(3, 3) PF_IRB(4, 4) PF_IRB(5, 5) PF_IRB(6, 6)
            PF_IRB(4, 3) PF_IRB(3, 4) PF_IRB(5, 4) PF_IRB(4, 5) PF_IRB(6, 5) PF_IRB(5, 6)
            PF_IRB(5, 3) PF_IRB(3, 5) PF_IRB(6, 4) PF_IRB(4, 6)
#undef PF_IRB
            if (warp == 0) {
                const int u = 1 << (be.lw - 3), uh = 1 << (be.lh - 3);
                for (int i = lane; i < u * uh; i += 32) {
                    BlockInfo &bi = a.grid[(gy + i / u) * gcols + gx + i % u];
                    bi.intra = 1;
                    bi.yMode = (uint8_t)(mode & 15);
                    bi.yAngle = (int8_t)lumaDelta(mode);
                    bi.uvMode = (uint8_t)uvMode;
                    bi.cflU = (int8_t)cflU;
                    bi.cflV = (int8_t)cflV;
                    bi.skip = (uint8_t)skip;
                    bi.txSplit = (uint8_t)depth;
                    int t = type;
                    if (depth) {   // the type of the split transform covering this 8x8 unit
                        const int lt = be.lw - depth, per = 1 << depth;
                        t = S.ttype[(((i / u) * 8) >> lt) * per + (((i % u) * 8) >> lt)];
                    }
                    bi.txTypes = (uint16_t)(t * 0x1111);
                    bi.filter = 0;
                }
            }
            __syncthreads();   // the next block reads this one's reconstruction and reuses the scratch
        }
    }
    __threadfence();
    __syncthreads();
    if (!tid) atomicExch(a.done + sb, a.tag);
}

// Phases 2 and 3: one thread per block of the lists.
__device__ __host__ inline uint32_t blockRecCap(const pft2::BlockEntry &e) {
    return 96u + 24u * (1u << e.lw) * (1u << e.lh);   // 16 records per sample over luma and chroma
}
// Fixed record regions: superblock sb owns [sb * sbCap, (sb + 1) * sbCap); its blocks follow in decode order.
__global__ void blockOffsets(TileJob j, uint32_t sbCap) {
    const int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= j.sbCols * j.sbRows) return;
    uint32_t o = uint32_t(sb) * sbCap;
    for (uint32_t k = 0; k < j.entryCount[sb]; k++) { j.sbOffset[sb * 64 + k] = o; o += blockRecCap(j.entries[sb * 64 + k]); }
}
__global__ void tilePhaseBlocks(TileJob j, int phase, int write) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int sb = e >> 6;
    if (sb >= j.sbCols * j.sbRows) return;
    if ((e & 63) >= (int)j.entryCount[sb]) { if (phase == 3) j.sbCount[e] = 0; return; }
    int16_t qv[1024];
    pft2::TileCoder t;
    setupCoder(t, j, sb, qv);
    t.phase = phase;
    t.recs = write ? j.recs + j.sbOffset[e] : nullptr;
    t.recCap = write ? blockRecCap(j.entries[e]) : 0xffffffffu;
    t.sbList = j.entries + size_t(sb) * 64;
    t.runEntry(j.entries[e]);
    if (t.st.error) atomicMax(j.error, t.st.error);
    if (phase == 3) j.sbCount[e] = t.nrec;
}
// Record offsets: per superblock sums (one thread each), a serial scan over superblocks, then per-block offsets.
__global__ void prefixSb(TileJob j, uint32_t *sbTotal) {
    const int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= j.sbCols * j.sbRows) return;
    uint32_t t = 0;
    for (int k = 0; k < 64; k++) t += j.sbCount[sb * 64 + k];
    sbTotal[sb] = t;
}
__global__ void prefixScan(TileJob j, uint32_t *sbTotal) {
    if (threadIdx.x || blockIdx.x) return;
    const int n = j.sbCols * j.sbRows;
    uint32_t total = 0;
    for (int i = 0; i < n; i++) { const uint32_t v = sbTotal[i]; sbTotal[i] = total; total += v; }
    j.sbOffset[n * 64] = total;
    if (total > j.recCap) atomicMax(j.error, 100);
}
__global__ void prefixBlocks(TileJob j, const uint32_t *sbStart) {
    const int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= j.sbCols * j.sbRows) return;
    uint32_t o = sbStart[sb];
    for (int k = 0; k < 64; k++) { j.sbOffset[sb * 64 + k] = o; o += j.sbCount[sb * 64 + k]; }
}
// Records-only mode: pack every tile's records contiguously in decode order (tile after tile).
__global__ void packOffsets(TileJob j) {
    if (threadIdx.x || blockIdx.x) return;
    uint32_t total = 0;
    for (int t = 0; t < j.tileCols * j.tileRows; t++) {
        const uint32_t start = total;
        const int tc = t % j.tileCols, tr = t / j.tileCols;
        for (int r = j.rowStart[tr]; r < j.rowStart[tr + 1]; r += 16)
            for (int c = j.colStart[tc]; c < j.colStart[tc + 1]; c += 16) {
                const int sb = (r / 16) * j.sbCols + c / 16;
                for (uint32_t b = 0; b < j.entryCount[sb]; b++) {
                    j.packOff[sb * 64 + b] = total;
                    total += j.sbCount[sb * 64 + b];
                }
            }
        j.tileCount[t] = total - start;
    }
}
__global__ void packRecords(TileJob j) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const int sb = e >> 6;
    if (sb >= j.sbCols * j.sbRows || (e & 63) >= (int)j.entryCount[sb]) return;
    const uint32_t src = j.sbOffset[e], dst = j.packOff[e], n = j.sbCount[e];
    for (uint32_t i = 0; i < n; i++) j.packed[dst + i] = j.recs[src + i];
}
// Two warps per tile. Warp 0 gathers 32 records at a time and turns them into range-coder intervals: lanes whose
// records share a CDF form a group (__match_any_sync) whose leader adapts that CDF in record order, so distinct
// contexts adapt in parallel. Warp 1 lane 0 range-codes the previous chunk's intervals meanwhile (double buffer).
struct CodeIv { uint16_t fl, fh; uint8_t s, n, type, pad; };
constexpr int kCodeChunk = 256;
__global__ void __launch_bounds__(64) tileCode(TileJob j) {
    __shared__ pft2::EntropyState es;
    __shared__ pft2::SymRec buf[kCodeChunk];
    __shared__ CodeIv iv[2][kCodeChunk];
    __shared__ int ivCount[2];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (t >= j.tileCols * j.tileRows) return;
    if (*j.error) { if (!tid) j.sizes[t] = -1; return; }
    {
        const uint32_t *src = reinterpret_cast<const uint32_t *>(j.coderInit);
        uint32_t *dst = reinterpret_cast<uint32_t *>(&es);
        for (int i = tid; i < int(sizeof(es) / 4); i += 64) dst[i] = src[i];
    }
    if (!tid) ivCount[0] = ivCount[1] = 0;
    __syncthreads();
    av1g_ec ec;
    if (tid == 32) av1g_ec_init(&ec, j.ecBuf + j.ecCapPerTile * t, (uint32_t)j.ecCapPerTile);
    const int tc = t % j.tileCols, tr = t / j.tileCols;
    uint16_t *base = reinterpret_cast<uint16_t *>(&es);
    // producer cursor (warp 0, warp-uniform): superblock position, entry, record
    int r = j.rowStart[tr], c = j.colStart[tc];
    uint32_t b = 0, k = 0, last = 0;
    bool started = false, finished = false;
    auto nextSegment = [&]() -> bool {   // advance to a non-empty segment; false at the end of the tile
        for (;;) {
            if (started) {
                if (++b >= j.entryCount[(r / 16) * j.sbCols + c / 16]) {
                    b = 0;
                    c += 16;
                    if (c >= j.colStart[tc + 1]) { c = j.colStart[tc]; r += 16; }
                }
            }
            started = true;
            if (r >= j.rowStart[tr + 1]) return false;
            const int sb = (r / 16) * j.sbCols + c / 16;
            if (b >= j.entryCount[sb]) continue;
            k = j.sbOffset[sb * 64 + b];
            last = k + j.sbCount[sb * 64 + b];
            if (k < last) return true;
        }
    };
    for (int step = 0;; step++) {
        const int cur = step & 1;
        if (warp == 0) {
            int n = 0;
            while (!finished && n < kCodeChunk) {
                if (k >= last && !nextSegment()) { finished = true; break; }
                const int take = min(kCodeChunk - n, int(last - k));
                for (int q = lane; q < take; q += 32) buf[n + q] = j.recs[k + q];
                k += take;
                n += take;
            }
            __syncwarp();
            for (int s0 = 0; s0 < n; s0 += 32) {
              if (s0 + lane < n) {
                const pft2::SymRec rec = buf[s0 + lane];
                const int type = rec.kind >> 5;
                const uint32_t key = (type == 0 || type == 2) ? rec.cdf : 0x10000u + lane;
                const unsigned grp = __match_any_sync(__activemask(), key);
                if (lane == __ffs(grp) - 1) {
                    for (unsigned m = grp; m; m &= m - 1) {
                        const int i = s0 + __ffs(m) - 1;
                        const pft2::SymRec q = buf[i];
                        const int ty = q.kind >> 5, nn = q.kind & 31;
                        CodeIv o;
                        o.type = uint8_t(ty);
                        o.pad = 0;
                        if (ty == 0) {
                            uint16_t *cdf = base + q.cdf;
                            o.fl = uint16_t(q.sym > 0 ? 32768u - cdf[q.sym - 1] : 32768u);   // 32768 wraps to 0
                            o.fh = uint16_t(32768u - cdf[q.sym]);
                            o.s = q.sym; o.n = uint8_t(nn);
                            if (q.sym == 0) o.pad = 1;   // fl == 32768
                            av1g_update_cdf(cdf, q.sym, nn);
                        } else if (ty == 2) {
                            o.fl = uint16_t(av1g_split_or_psum(base + q.cdf, q.sym));
                            o.fh = 0; o.s = 1; o.n = 2;
                        } else if (ty == 1) {
                            o.fl = q.sym ? 16384 : 0; o.pad = q.sym ? 0 : 1;
                            o.fh = q.sym ? 0 : 16384; o.s = q.sym; o.n = 2;
                        } else {
                            o.fl = 0; o.fh = 0; o.s = q.sym; o.n = uint8_t(nn);
                        }
                        iv[cur][i] = o;
                    }
                }
              }
              __syncwarp();
            }
            if (!lane) ivCount[cur] = n;
        } else if (tid == 32 && step > 0) {
            const int prev = cur ^ 1, n = ivCount[prev];
            for (int i = 0; i < n; i++) {
                const CodeIv o = iv[prev][i];
                if (o.type == 3) {
                    for (int q = o.n - 1; q >= 0; q--) av1g_ec_bool(&ec, (o.s >> q) & 1);
                } else {
                    av1g_ec_encode_q15(&ec, o.pad ? 32768u : o.fl, o.fh, o.s, o.n);
                }
            }
        }
        __syncthreads();
        const int produced = ivCount[cur];
        __syncthreads();
        if (produced == 0) break;   // the previous chunk is coded and nothing is left
    }
    if (tid == 32) {
        const int n = av1g_ec_done(&ec, j.out + j.ecCapPerTile * t);
        j.sizes[t] = n;
        if (n < 0) atomicMax(j.error, 6);
        if (t == j.contextTile && j.save) {
            pft2::clearCounters(base, (int)(sizeof(es) / 2));
            *j.save = es;
        }
    }
}
// Pack the tiles contiguously (block t copies tile t after the sizes of the tiles before it).
__global__ void packTiles(const uint8_t *out, size_t capPerTile, const int *sizes, int count, uint8_t *packed) {
    const int t = blockIdx.x;
    size_t off = 0;
    for (int k = 0; k < t; k++) off += sizes[k] > 0 ? sizes[k] : 0;
    const int n = sizes[t];
    for (int i = threadIdx.x; i < n; i += blockDim.x) packed[off + i] = out[capPerTile * t + i];
}

// ------------------------------------------------------------------------------------------------
// Temporal filter (ALTREF source denoising). One CTA per 16x16 luma block: for every neighbour frame, a full-pel search
// of +-kTfRange around the coarse vector refined to quarter-pel (8-tap regular filter; integer alignment blurs fine
// texture), then the motion-compensated samples are blended with non-local-means weights
// as in libaom: exp(-(5 * window error + block error) / 120 / decay), the window error being the mean squared
// difference over the 5x5 window of the pixel (inside the block), decay = (0.5 + ln(2 sigma + 5)) * q decay * strength
// with sigma the estimated noise of the centre frame and the q decay from the inter-frame quantizer.
constexpr int kTfMax = 16, kTfRange = 6, kTfPad = 4, kTfWin = 16 + 2 * (kTfRange + kTfPad);
struct TfArgs {
    const uint8_t *center[3];
    const uint8_t *nb[kTfMax][3];
    const Mv *coarse[kTfMax];
    int count, w, h, aw, ccols, crows;
    const unsigned long long *noise;   // luma noise estimate: sum of |Laplacian| over flat pixels, pixel count
    float qDecay, strength;
    uint8_t *out[3];
};
// libaom av1_estimate_noise_from_single_plane: mean |3x3 Laplacian| over pixels whose Sobel gradient is below 50.
__global__ void estimateNoise(const uint8_t *p, int pitch, int w, int h, unsigned long long *out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1, y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    unsigned v = 0, n = 0;
    if (x < w - 1 && y < h - 1) {
        const uint8_t *c = p + size_t(y) * pitch + x;
        const int gx = (c[-pitch - 1] - c[-pitch + 1]) + (c[pitch - 1] - c[pitch + 1]) + 2 * (c[-1] - c[1]);
        const int gy = (c[-pitch - 1] - c[pitch - 1]) + (c[-pitch + 1] - c[pitch + 1]) + 2 * (c[-pitch] - c[pitch]);
        if (abs(gx) + abs(gy) < 50) {
            v = abs(4 * c[0] - 2 * (c[-1] + c[1] + c[-pitch] + c[pitch]) +
                    (c[-pitch - 1] + c[-pitch + 1] + c[pitch - 1] + c[pitch + 1]));
            n = 1;
        }
    }
    for (int o = 16; o; o >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, o); n += __shfl_xor_sync(0xffffffffu, n, o); }
    if ((threadIdx.x & 31) == 0 && n) { atomicAdd(out, (unsigned long long)v); atomicAdd(out + 1, (unsigned long long)n); }
}
__global__ void __launch_bounds__(256) temporalFilterKernel(TfArgs a) {
    __shared__ __align__(16) uint8_t cur[256];
    __shared__ __align__(16) uint8_t win[kTfWin * kTfWin];
    __shared__ int hsumS[3][24][16];   // horizontal pass of the three column phases of a quarter-pel step
    __shared__ int sads[(2 * kTfRange + 1) * (2 * kTfRange + 1)];
    __shared__ float wsum[256], acc[256], wl[256];
    __shared__ int d2[256];
    __shared__ float cw[2][64], cacc[2][64];
    __shared__ int ccen[2][64], cpred[2][64], cd2[2][64];
    __shared__ int bestOff, bestSub;
    __shared__ float blockMse;
    __shared__ int subSad[8];
    __shared__ float warpE[8];
    const int t = threadIdx.x;
    const int bx = blockIdx.x * 16, by = blockIdx.y * 16;
    const int px = bx + (t & 15), py = by + (t >> 4);
    const int cx = min(px, a.w - 1), cy = min(py, a.h - 1);
    cur[t] = a.center[0][cy * a.aw + cx];
    wsum[t] = 1.f;
    acc[t] = cur[t];
    const int caw = a.aw / 2, cw2 = a.w / 2, ch2 = a.h / 2;
    if (t < 64) {
        const int ux = min(bx / 2 + (t & 7), cw2 - 1), uy = min(by / 2 + (t >> 3), ch2 - 1);
        for (int p = 0; p < 2; p++) {
            cw[p][t] = 1.f;
            ccen[p][t] = a.center[1 + p][uy * caw + ux];
            cacc[p][t] = (float)ccen[p][t];
        }
    }
    float decay;
    {
        const double sum = (double)a.noise[0], cnt = (double)a.noise[1];
        const float sigma = cnt >= 16 ? (float)(sum / (6.0 * cnt) * 1.25331413731550025) : 0.f;   // sqrt(pi / 2)
        decay = (0.5f + logf(2.f * sigma + 5.f)) * a.qDecay * a.strength;
    }
    const float inv = 1.f / (120.f * decay);
    __syncthreads();
    const int side = 2 * kTfRange + 1;
    for (int j = 0; j < a.count; j++) {
        const Mv cm = a.coarse[j][min((by + 8) / 32, a.crows - 1) * a.ccols + min((bx + 8) / 32, a.ccols - 1)];
        const int mx = cm.col >> 3, my = cm.row >> 3;
        const uint8_t *ny = a.nb[j][0];
        for (int i = t; i < kTfWin * kTfWin; i += 256) {
            const int r = i / kTfWin, c = i % kTfWin;
            const int x = min(max(bx + mx - kTfRange - kTfPad + c, 0), a.w - 1);
            const int y = min(max(by + my - kTfRange - kTfPad + r, 0), a.h - 1);
            win[i] = ny[y * a.aw + x];
        }
        __syncthreads();
        static_assert(kTfWin % 4 == 0, "word rows");
        if (t < side * side) {   // four samples per step: aligned words of the window, funnel-shifted to the offset
            const int oy = t / side, ox = t % side;
            const int c0 = kTfPad + ox, sh = (c0 & 3) * 8;
            const uint32_t *cw32 = reinterpret_cast<const uint32_t *>(cur);
            int s = 0;
            for (int r = 0; r < 16; r++) {
                const uint32_t *wr = reinterpret_cast<const uint32_t *>(win + (kTfPad + oy + r) * kTfWin) + (c0 >> 2);
                uint32_t w0 = wr[0];
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    const uint32_t w1 = wr[k + 1];
                    s = __vsadu4(cw32[r * 4 + k], __funnelshift_r(w0, w1, sh)) + s;
                    w0 = w1;
                }
            }
            sads[t] = s;
        }
        __syncthreads();
        if (t == 0) {
            int best = 0;
            for (int k = 1; k < side * side; k++)
                if (sads[k] < sads[best] || (sads[k] == sads[best] && abs(k - side * side / 2) < abs(best - side * side / 2)))
                    best = k;
            bestOff = best;
        }
        __syncthreads();
        const int oy = bestOff / side, ox = bestOff % side;
        // quarter-pel refinement: half-pel ring, then quarter-pel ring around the best (1/16 units)
        const int r0p = t >> 4, c0p = t & 15;
        auto predAt = [&](int y16, int x16) {   // window position of the block origin, 1/16 sample units
            const int iy = (y16 >> 4) + r0p, ix = (x16 >> 4) + c0p, fy = y16 & 15, fx = x16 & 15;
            int v = 0;
            for (int r = 0; r < 8; r++) {
                const uint8_t *row = win + (iy + r - 3) * kTfWin + ix - 3;
                int hsum = 0;
                for (int c = 0; c < 8; c++) hsum += c_filter[0][fx][c] * row[c];
                v += c_filter[0][fy][r] * hsum;
            }
            return min(max((v + 8192) >> 14, 0), 255);
        };
        int by16 = (kTfPad + oy) * 16, bx16 = (kTfPad + ox) * 16;
        for (int step = 8; step >= 4; step >>= 1) {
            int best = 0, bestS = 0;
            // the horizontal pass (full precision) of the three column phases over the rows of all three row phases,
            // then each position's vertical pass: the same integer sums as predAt
            const int rowBase = ((by16 - step) >> 4) - 3;
            for (int i = t; i < 3 * 24 * 16; i += 256) {
                const int dxi = i / (24 * 16), r = (i / 16) % 24, c = i % 16;
                const int x16 = bx16 + (dxi - 1) * step, fx = x16 & 15, ix = (x16 >> 4) + c;
                const uint8_t *row = win + min(rowBase + r, kTfWin - 1) * kTfWin + ix - 3;   // (an unused 24th row may lie past the window)
                int hsum = 0;
                for (int k = 0; k < 8; k++) hsum += c_filter[0][fx][k] * row[k];
                hsumS[dxi][r][c] = hsum;
            }
            __syncthreads();
            for (int k = 0; k <= 8; k++) {
                const int dy = (k / 3 - 1) * step;
                const int y16 = by16 + dy, fy = y16 & 15, iy = (y16 >> 4) + r0p - 3 - rowBase;
                int v = 0;
                for (int r = 0; r < 8; r++) v += c_filter[0][fy][r] * hsumS[k % 3][iy + r][c0p];
                const int pv = min(max((v + 8192) >> 14, 0), 255);
                int s;
                int e = abs((int)cur[t] - pv);
                for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
                if ((t & 31) == 0) subSad[t >> 5] = e;
                __syncthreads();
                s = 0;
                for (int w8 = 0; w8 < 8; w8++) s += subSad[w8];
                __syncthreads();
                if (k == 0 || s < bestS) { bestS = s; best = k; }
            }
            by16 += (best / 3 - 1) * step;
            bx16 += (best % 3 - 1) * step;
        }
        const int nv = predAt(by16, bx16);
        const int d = (int)cur[t] - nv;
        d2[t] = d * d;
        float e = (float)(d * d);
        for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
        if ((t & 31) == 0) warpE[t >> 5] = e;
        __syncthreads();
        if (t == 0) {
            float tot = 0.f;
            for (int k = 0; k < 8; k++) tot += warpE[k];
            blockMse = tot / 256.f;
        }
        __syncthreads();
        // 5x5 window error inside the block
        const int r0 = t >> 4, c0 = t & 15;
        int we = 0, wn = 0;
        for (int r = max(r0 - 2, 0); r <= min(r0 + 2, 15); r++)
            for (int c = max(c0 - 2, 0); c <= min(c0 + 2, 15); c++) { we += d2[r * 16 + c]; wn++; }
        const float wpix = __expf(-fminf(7.f, (5.f * (float)we / (float)wn + blockMse) * inv));
        wl[t] = wpix;
        acc[t] += wpix * (float)nv;
        wsum[t] += wpix;
        __syncthreads();
        if (t < 64) {
            // luma displacement in 1/16 samples = chroma displacement in 1/32: use the 1/16 phase of half of it
            const int dy16 = (my - kTfRange - kTfPad) * 16 + by16, dx16 = (mx - kTfRange - kTfPad) * 16 + bx16;
            const int cy16 = dy16 >> 1, cx16 = dx16 >> 1;
            const int ux = bx / 2 + (t & 7), uy = by / 2 + (t >> 3);
            const int iy = uy + (cy16 >> 4), ix = ux + (cx16 >> 4), fy = cy16 & 15, fx = cx16 & 15;
            for (int p = 0; p < 2; p++) {
                const uint8_t *pl = a.nb[j][1 + p];
                int v = 0;
                for (int r = 0; r < 8; r++) {
                    const int yy = min(max(iy + r - 3, 0), ch2 - 1);
                    int hsum = 0;
                    for (int c = 0; c < 8; c++) hsum += c_filter[0][fx][c] * pl[yy * caw + min(max(ix + c - 3, 0), cw2 - 1)];
                    v += c_filter[0][fy][r] * hsum;
                }
                v = min(max((v + 8192) >> 14, 0), 255);
                cpred[p][t] = v;
                cd2[p][t] = (ccen[p][t] - v) * (ccen[p][t] - v);
            }
        }
        __syncthreads();
        if (t < 64) {
            // chroma weights from the plane's own 3x3 window error plus the colocated 2x2 luma errors (as libaom
            // adds luma to the chroma window): a colour change over unchanged luma must not be averaged
            const int cr = t >> 3, cc = t & 7, l = cr * 32 + cc * 2;
            const int lumaSse = d2[l] + d2[l + 1] + d2[l + 16] + d2[l + 17];
            for (int p = 0; p < 2; p++) {
                int sse = lumaSse, n = 4;
                for (int r = max(cr - 1, 0); r <= min(cr + 1, 7); r++)
                    for (int c = max(cc - 1, 0); c <= min(cc + 1, 7); c++) { sse += cd2[p][r * 8 + c]; n++; }
                const float wc = __expf(-fminf(7.f, (5.f * (float)sse / (float)n + blockMse) * inv));
                cacc[p][t] += wc * (float)cpred[p][t];
                cw[p][t] += wc;
            }
        }
        __syncthreads();
    }
    if (px < a.aw && py < a.h) {
        const int v = (int)(acc[t] / wsum[t] + 0.5f);
        a.out[0][py * a.aw + px] = (uint8_t)min(max(v, 0), 255);
    }
    if (t < 64) {
        const int ux = bx / 2 + (t & 7), uy = by / 2 + (t >> 3);
        if (ux < caw && uy < ch2)
            for (int p = 0; p < 2; p++)
                a.out[1 + p][uy * caw + ux] = (uint8_t)min(max((int)(cacc[p][t] / cw[p][t] + 0.5f), 0), 255);
    }
}
// Replicate the last row / column into the padding of a filtered padded frame.
__global__ void padFrame(uint8_t *plane, int pitch, int w, int h, int ph) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= pitch || y >= ph || (x < w && y < h)) return;
    plane[y * pitch + x] = plane[min(y, h - 1) * pitch + min(x, w - 1)];
}

struct EntropySlot {
    BlockInfo *grid = nullptr;
    int16_t *levels = nullptr;
    uint8_t *tileOut = nullptr, *packed = nullptr;
    int *sizes = nullptr;
    int *error = nullptr;
    uint32_t *tileCount = nullptr;
    pft2::SymRec *recs = nullptr;    // records-only mode output
    pf::Mv *mf = nullptr;            // motion field of the frame (use_ref_frame_mvs)
    int8_t *cdef = nullptr;          // per-superblock CDEF presets of the frame
    bool recordsOnly = false;
    cudaEvent_t staged = nullptr, done = nullptr;
    cudaEvent_t profile[5] = {};
    cudaStream_t fetch = nullptr;
    int nTiles = 0;
};

struct State {
    uint8_t *ds[1 + kMaxRefs] = {};
    Mv *coarse[kMaxRefs] = {};
    // hierarchical coarse search: 1/16-resolution planes of the current picture and of one reference at a time, their
    // vectors, and the quarter-resolution costs
    uint8_t *ds16[2] = {};
    Mv *coarse16 = nullptr;
    float *coarseCost = nullptr;
    int dw16 = 0, dh16 = 0, c16cols = 0, c16rows = 0;
    Cand *cand[kSizes] = {};
    uint8_t *pred[kSizes] = {}, *recon[kSizes] = {};
    int16_t *levels[kSizes] = {}, *outLevels = nullptr;
    uint8_t *recon2[kSizes] = {}, *recon3[kSizes] = {};
    int16_t *levels3[kSizes] = {};
    uint8_t *types1[kShapes] = {}, *types2[kShapes] = {}, *types3[kSizes] = {};   // luma tx types per 4x4
    float *subD[kSizes] = {}, *subR[kSizes] = {};       // split-external evaluation of squares
    uint8_t *subNz[kSizes] = {}, *subType[kSizes] = {};
    float *rsubD[10] = {}, *rsubR[10] = {};
    uint8_t *rsubNz[10] = {}, *rsubType[10] = {};
    Cand *rcand[10] = {};
    uint8_t *rpred[10] = {}, *rrecon[10] = {};
    int16_t *rlevels[10] = {};
    int rcols[10] = {}, rrows[10] = {};
    uint8_t *rrecon2[10] = {};
    int16_t *rlevels2[10] = {};
    int16_t *levels2[kSizes] = {};
    BlockInfo *grid = nullptr, *prevGrid = nullptr;
    ModeTrial *modeTrials = nullptr;
    uint8_t *modeChanged = nullptr;
    bool refineStats = getenv("AV1GPU_PF_REFINE_STATS") != nullptr;
    pft2::MiInfo *modeMi = nullptr;
    pft2::BlockEntry *modeEntries = nullptr;
    uint32_t *modeEntryCount = nullptr;
 uint32_t *intraSel = nullptr;       // per grid unit: intra mode + 1 chosen for the block starting there, 0 inter
    int *intraDone = nullptr;          // per superblock: frame tag once its intra blocks are reconstructed
    unsigned *intraTicket = nullptr;
    int intraTag = 0;
    MotionOptions *modeMenus = nullptr;
    int *modeCounts = nullptr, *modeIndices[kShapes] = {};
    bool havePrev = false;
    int dw = 0, dh = 0, ccols = 0, crows = 0;
    int cols[kSizes] = {}, rows[kSizes] = {};
    int gcols = 0, grows = 0;
    bool tablesReady = false;
    void *pfScratch = nullptr;
    uint8_t *filterCandidates = nullptr, *cdefOut = nullptr;
    FilterSelection *filterDev = nullptr, *filterHost = nullptr;
    Mv *mfDev = nullptr, *mfHost = nullptr;
    int8_t *cdefIdxDev = nullptr;               // per-superblock CDEF presets of the last filtered frame
    unsigned long long *cdefSbErr = nullptr;    // [candidate][superblock][luma, chroma] SSE
    unsigned long long *cdefCur = nullptr;      // preset choice scratch, one per superblock
    CdefChoice *cdefChoice = nullptr;
    bool useTmvp = false;
    BlockInfo *motionGridHost = nullptr;
    cudaStream_t compute = nullptr;
    cudaEvent_t mfUploaded = nullptr, motionGridReady = nullptr;
    // mode refinement trials: the per-shape evaluations are independent and mostly latency bound, so they run
    // concurrently, one stream per shape (forked from and joined into the compute stream)
    cudaStream_t shapeStream[kShapes] = {};
    cudaEvent_t trialFork = nullptr, trialJoin[kShapes] = {}, sqDone[kSizes] = {};
    float *lrCoef = nullptr; int8_t *lrQ = nullptr; unsigned long long *lrSse = nullptr; uint8_t *lrOut = nullptr;
    int lrUnitRows = 1, lrUnitCols = 1;
    unsigned long long *sse = nullptr;
    float *best[kSizes] = {};
    uint8_t *mode[kSizes] = {};
    // tile entropy coding
    pft2::EntropyState *slots = nullptr, *tileWork = nullptr, *staging = nullptr;
    pft2::InterCdfs *defaultInter = nullptr;
    pft2::MiInfo *mi = nullptr;
    pft2::TileStats *tileStats = nullptr;
    uint8_t *ctxArrays = nullptr, *tileOut = nullptr, *packed = nullptr;
    uint16_t *ecBuf = nullptr;
    int *tileSizes = nullptr;
    size_t ecCap = 0, ctxStride = 0;
    int maxTiles = 64;
    cudaStream_t entStream = nullptr;
    pft2::EntropyState *frameInit = nullptr, *coderInit = nullptr;
    uint32_t *packOff = nullptr;
    uint8_t *ctxLevel[3] = {}, *ctxDc[3] = {};
    int ctxStride4[3] = {}, ctxRows4[3] = {};
    uint32_t *sbCount = nullptr, *sbOffset = nullptr, *entryCount = nullptr;
    pft2::BlockEntry *entries = nullptr;
    uint32_t *sbTotal = nullptr;
    pft2::SymRec *recs = nullptr;
    uint32_t recCap = 0;
    EntropySlot eslot[kEntropySlots];
    uint8_t *tfDs[kTfMax + 1] = {};
    Mv *tfCoarse[kTfMax] = {};
    unsigned long long *tfNoise = nullptr;
    int16_t *hphase = nullptr;
    uint8_t *phases[kMaxRefs] = {};
    bool profile = false;
    cudaEvent_t ev[10] = {};
    double stageMs[8] = {};
    bool profilePending = false, profileFiltered = false;
    int filterFrames = 0;
    double entMs[4] = {};
    int entFrames = 0;
    int frames = 0;
};

static void collectAnalysisProfile(State &s) {
    if (!s.profilePending) return;
    cudaEventSynchronize(s.ev[s.profileFiltered ? 9 : 7]);
    for (int i = 0; i < 7; i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, s.ev[i], s.ev[i + 1]);
        s.stageMs[i] += ms;
    }
    s.frames++;
    if (s.profileFiltered) {
        float ms = 0;
        cudaEventElapsedTime(&ms, s.ev[8], s.ev[9]);
        s.stageMs[7] += ms;
        s.filterFrames++;
    }
    s.profilePending = s.profileFiltered = false;
}

static void collectEntropyProfile(State &s, const EntropySlot &es) {
    if (!s.profile) return;
    for (int i = 0; i < 4; i++) {
        float ms = 0;
        cudaEventElapsedTime(&ms, es.profile[i], es.profile[i + 1]);
        s.entMs[i] += ms;
    }
    s.entFrames++;
}

}  // namespace

Analyzer::~Analyzer() {
    auto *s = static_cast<State *>(state);
    if (!s) return;
    if (s->compute) cudaStreamSynchronize(s->compute);
    if (s->entStream) cudaStreamSynchronize(s->entStream);
    collectAnalysisProfile(*s);
    for (auto *p : s->ds) cudaFree(p);
    for (auto *p : s->coarse) cudaFree(p);
    for (auto *p : s->cand) cudaFree(p);
    for (auto *p : s->pred) cudaFree(p);
    for (auto *p : s->recon) cudaFree(p);
    for (auto *p : s->levels) cudaFree(p);
    for (auto *p : s->recon2) cudaFree(p);
    for (int k = 0; k < kSizes; k++) { cudaFree(s->recon3[k]); cudaFree(s->levels3[k]); }
    for (int k = 0; k < 10; k++) { cudaFree(s->rrecon2[k]); cudaFree(s->rlevels2[k]); }
    for (int k = 0; k < 10; k++) { cudaFree(s->rcand[k]); cudaFree(s->rpred[k]); cudaFree(s->rrecon[k]); cudaFree(s->rlevels[k]); }
    for (auto *p : s->levels2) cudaFree(p);
    cudaFree(s->outLevels);
    for (int k = 0; k < kShapes; k++) { cudaFree(s->types1[k]); cudaFree(s->types2[k]); }
    for (int k = 0; k < kSizes; k++) { cudaFree(s->types3[k]); cudaFree(s->subD[k]); cudaFree(s->subR[k]); cudaFree(s->subNz[k]); cudaFree(s->subType[k]); }
    for (int k = 0; k < 10; k++) { cudaFree(s->rsubD[k]); cudaFree(s->rsubR[k]); cudaFree(s->rsubNz[k]); cudaFree(s->rsubType[k]); }
    cudaFree(s->grid);
    cudaFree(s->prevGrid);
    cudaFree(s->modeTrials); cudaFree(s->modeChanged); cudaFree(s->modeMi);
    cudaFree(s->modeEntries); cudaFree(s->modeEntryCount);
    cudaFree(s->intraSel); cudaFree(s->intraDone); cudaFree(s->intraTicket);
    cudaFree(s->modeMenus); cudaFree(s->modeCounts); cudaFree(s->modeIndices[0]);
    cudaFree(s->pfScratch);
    cudaFree(s->filterCandidates);
    cudaFree(s->filterDev);
    if (s->filterHost) cudaFreeHost(s->filterHost);
    if (s->motionGridHost) cudaFreeHost(s->motionGridHost);
    if (s->mfHost) cudaFreeHost(s->mfHost);
    if (s->motionGridReady) cudaEventDestroy(s->motionGridReady);
    if (s->mfUploaded) cudaEventDestroy(s->mfUploaded);
    if (s->compute) cudaStreamDestroy(s->compute);
    for (auto stream : s->shapeStream) if (stream) cudaStreamDestroy(stream);
    for (auto event : s->trialJoin) if (event) cudaEventDestroy(event);
    if (s->trialFork) cudaEventDestroy(s->trialFork);
    for (auto event : s->sqDone) if (event) cudaEventDestroy(event);
    cudaFree(s->cdefOut);
    cudaFree(s->mfDev);
    cudaFree(s->lrCoef); cudaFree(s->lrQ); cudaFree(s->lrSse); cudaFree(s->lrOut);
    cudaFree(s->sse);
    cudaFree(s->cdefSbErr); cudaFree(s->cdefCur); cudaFree(s->cdefChoice);
    cudaFree(s->hphase);
    for (void *p : {(void *)s->slots, (void *)s->tileWork, (void *)s->staging, (void *)s->defaultInter, (void *)s->mi,
                    (void *)s->tileStats, (void *)s->ecBuf})
        cudaFree(p);
    for (auto &es : s->eslot) {
        cudaFree(es.grid); cudaFree(es.levels); cudaFree(es.tileOut); cudaFree(es.packed); cudaFree(es.sizes); cudaFree(es.error); cudaFree(es.tileCount); cudaFree(es.recs); cudaFree(es.mf); cudaFree(es.cdef);
        if (es.staged) cudaEventDestroy(es.staged);
        if (es.done) cudaEventDestroy(es.done);
        if (es.fetch) cudaStreamDestroy(es.fetch);
        for (auto event : es.profile) if (event) cudaEventDestroy(event);
    }
    if (s->entStream) cudaStreamDestroy(s->entStream);
    cudaFree(s->entries); cudaFree(s->entryCount); cudaFree(s->sbTotal);
    cudaFree(s->frameInit); cudaFree(s->coderInit); cudaFree(s->packOff); cudaFree(s->sbCount); cudaFree(s->sbOffset); cudaFree(s->recs);
    for (int p = 0; p < 3; p++) { cudaFree(s->ctxLevel[p]); cudaFree(s->ctxDc[p]); }
    for (auto *p : s->tfDs) cudaFree(p);
    for (auto *p : s->ds16) cudaFree(p);
    cudaFree(s->coarse16);
    cudaFree(s->coarseCost);
    for (auto *p : s->tfCoarse) cudaFree(p);
    cudaFree(s->tfNoise);
    for (auto *p : s->best) cudaFree(p);
    for (auto *p : s->mode) cudaFree(p);
    for (auto *p : s->phases) cudaFree(p);
    if (s->profile && s->frames) {
        const char *names[] = {"downsample+coarse", "refine", "evaluate", "decide", "commit", "mode refinement", "download"};
        for (int i = 0; i < 7; i++) fprintf(stderr, "pframe %-18s %7.2f ms/frame\n", names[i], s->stageMs[i] / s->frames);
        if (s->filterFrames) fprintf(stderr, "pframe %-18s %7.2f ms/frame\n", "filters", s->stageMs[7] / s->filterFrames);
    }
    for (auto &e : s->ev) if (e) cudaEventDestroy(e);
    if (s->profile && s->entFrames) {
        const char *names[] = {"entropy phase 1", "entropy phase 2", "entropy phase 3", "entropy coder"};
        for (int i = 0; i < 4; i++) fprintf(stderr, "pframe %-18s %7.2f ms/frame\n", names[i], s->entMs[i] / s->entFrames);
    }
    delete s;
}

int Analyzer::init(int width, int height, char *err, int errLen) {
    if (width < 16 || height < 16 || width % 8 || height % 8) {
        snprintf(err, errLen, "P-frame analysis requires dimensions that are multiples of 8 and at least 16");
        return -1;
    }
    w = width; h = height;
    aw = (w + 63) & ~63; ah = (h + 63) & ~63;
    auto *s = new State;
    state = s;
    cudaError_t e = cudaSuccess;
    auto alloc = [&](auto **p, size_t bytes) {
        if (e == cudaSuccess) e = cudaMalloc((void **)p, bytes);
        if (e == cudaSuccess) e = cudaMemset(*p, 0, bytes);
    };
    s->dw = (w + 3) / 4; s->dh = (h + 3) / 4;
    for (auto *&p : s->ds) alloc(&p, size_t(s->dw) * s->dh);
    s->ccols = (s->dw + 7) / 8; s->crows = (s->dh + 7) / 8;
    for (auto *&p : s->coarse) alloc(&p, sizeof(Mv) * s->ccols * s->crows);
    for (auto *&p : s->tfDs) alloc(&p, size_t(s->dw) * s->dh);
    for (auto *&p : s->tfCoarse) alloc(&p, sizeof(Mv) * s->ccols * s->crows);
    s->dw16 = (s->dw + 3) / 4; s->dh16 = (s->dh + 3) / 4;
    s->c16cols = (s->dw16 + 7) / 8; s->c16rows = (s->dh16 + 7) / 8;
    for (auto *&p : s->ds16) alloc(&p, size_t(s->dw16) * s->dh16);
    alloc(&s->coarse16, sizeof(Mv) * s->c16cols * s->c16rows);
    alloc(&s->coarseCost, sizeof(float) * s->ccols * s->crows);
    alloc(&s->tfNoise, 2 * sizeof(unsigned long long));
    const size_t frame = size_t(w) * h * 3 / 2, aframe = size_t(aw) * ah * 3 / 2;
    for (int k = 0; k < kSizes; k++) {
        const int n = 8 << k;
        s->cols[k] = aw / n; s->rows[k] = ah / n;
        alloc(&s->cand[k], sizeof(Cand) * s->cols[k] * s->rows[k]);
        alloc(&s->best[k], sizeof(float) * s->cols[k] * s->rows[k]);
        alloc(&s->mode[k], s->cols[k] * s->rows[k]);
        alloc(&s->pred[k], frame);
        alloc(&s->recon[k], frame);
        alloc(&s->levels[k], aframe * sizeof(int16_t));
        alloc(&s->recon2[k], size_t(w) * h);
        alloc(&s->recon3[k], size_t(w) * h);
        alloc(&s->levels3[k], size_t(aw) * ah * sizeof(int16_t));
        alloc(&s->levels2[k], size_t(aw) * ah * sizeof(int16_t));
    }
    for (int k = 0; k < 10; k++) {
        // 0..5: HORZ / VERT halves of 16, 32, 64 nodes; 6..9: HORZ_4 / VERT_4 quarters of 32, 64 nodes
        const int n = k < 6 ? 16 << (k / 2) : 32 << ((k - 6) / 2), div = k < 6 ? 2 : 4;
        const int bw = k & 1 ? n / div : n, bh = k & 1 ? n : n / div;
        s->rcols[k] = aw / bw; s->rrows[k] = ah / bh;
        alloc(&s->rcand[k], sizeof(Cand) * s->rcols[k] * s->rrows[k]);
        alloc(&s->rpred[k], frame);
        alloc(&s->rrecon[k], frame);
        alloc(&s->rlevels[k], aframe * sizeof(int16_t));
        alloc(&s->rrecon2[k], size_t(w) * h);
        alloc(&s->rlevels2[k], size_t(aw) * ah * sizeof(int16_t));
        const size_t rn = size_t(s->rcols[k]) * s->rrows[k] * 2;
        alloc(&s->rsubD[k], sizeof(float) * rn); alloc(&s->rsubR[k], sizeof(float) * rn);
        alloc(&s->rsubNz[k], rn); alloc(&s->rsubType[k], rn);
    }
    alloc(&s->outLevels, aframe * sizeof(int16_t));
    for (int k = 0; k < kShapes; k++) { alloc(&s->types1[k], size_t(aw / 4) * (ah / 4)); alloc(&s->types2[k], size_t(aw / 4) * (ah / 4)); }
    for (int k = 0; k < kSizes; k++) {
        alloc(&s->types3[k], size_t(aw / 4) * (ah / 4));
        const size_t nsub = size_t(s->cols[k]) * s->rows[k] * 20;
        alloc(&s->subD[k], sizeof(float) * nsub); alloc(&s->subR[k], sizeof(float) * nsub);
        alloc(&s->subNz[k], nsub); alloc(&s->subType[k], nsub);
    }
    s->gcols = aw / 8; s->grows = ah / 8;
    alloc(&s->grid, sizeof(BlockInfo) * s->gcols * s->grows);
    alloc(&s->prevGrid, sizeof(BlockInfo) * s->gcols * s->grows);
    {
        av1g_frame_params fp{};
        fp.width = w; fp.height = h; fp.chroma_format = AV1G_CHROMA_420; fp.bit_depth = 8; fp.enable_cdef = 1;
        alloc((uint8_t **)&s->pfScratch, av1g_gpu_pf_scratch_size(&fp));
        // Deblock and CDEF reuse this workspace. All candidates are fully overwritten before use.
        if (e == cudaSuccess) e = cudaMalloc(&s->filterCandidates, frame * kCdefCount);
        alloc(&s->filterDev, sizeof(FilterSelection));
        if (e == cudaSuccess) e = cudaMallocHost(&s->filterHost, sizeof(FilterSelection));
        if (e == cudaSuccess) e = cudaMallocHost(&s->motionGridHost, gridCount() * sizeof(BlockInfo));
        if (e == cudaSuccess) e = cudaMallocHost(&s->mfHost, sizeof(Mv) * size_t(w / 8) * (h / 8) * 8);
        alloc(&s->cdefOut, frame);
        alloc(&s->mfDev, sizeof(Mv) * 8 * size_t((w + 7) >> 3) * ((h + 7) >> 3));
        s->lrUnitRows = std::max((h + 32) / 64, 1); s->lrUnitCols = std::max((w + 32) / 64, 1);
        const int nu = s->lrUnitRows * s->lrUnitCols;
        alloc(&s->lrCoef, sizeof(float) * 6 * nu); alloc(&s->lrQ, 6 * nu);
        alloc(&s->lrSse, sizeof(unsigned long long) * 2 * nu); alloc(&s->lrOut, size_t(w) * h);
        alloc(&s->sse, sizeof(unsigned long long) * 3 * kCdefCount);
        alloc(&s->cdefSbErr, sizeof(unsigned long long) * 2 * kCdefCount * size_t((w + 63) / 64) * ((h + 63) / 64));
        alloc(&s->cdefCur, sizeof(unsigned long long) * size_t((w + 63) / 64) * ((h + 63) / 64));
        alloc(&s->cdefChoice, sizeof(CdefChoice));
        alloc(&s->hphase, sizeof(int16_t) * 4 * size_t(w) * h);
        for (auto *&p : s->phases) alloc(&p, 16 * size_t(w) * h + 16);   // + 16: ldu32 over-read
    }
    {
        const int miCols = 2 * ((w + 7) >> 3), miRows = 2 * ((h + 7) >> 3);
        alloc(&s->slots, sizeof(pft2::EntropyState) * 8);
        alloc(&s->tileWork, sizeof(pft2::EntropyState) * s->maxTiles);
        alloc(&s->staging, sizeof(pft2::EntropyState));
        alloc(&s->defaultInter, sizeof(pft2::InterCdfs));
        alloc(&s->mi, sizeof(pft2::MiInfo) * miCols * miRows);
        alloc(&s->tileStats, sizeof(pft2::TileStats) * s->maxTiles);
        s->ctxStride = size_t(miCols + miRows + 64);
        // the precarry buffer holds one entry per output byte: a generous bound per tile (any tile may hold the frame)
        s->ecCap = size_t(aw) * ah * 3 / 2 + 65536;
        alloc(&s->ecBuf, sizeof(uint16_t) * s->ecCap * 2);
        {
            // lowest priority: the entropy kernels fill SMs the analysis of later frames leaves idle
            int lo = 0, hi = 0;
            cudaDeviceGetStreamPriorityRange(&lo, &hi);
            if (e == cudaSuccess) e = cudaStreamCreateWithPriority(&s->entStream, cudaStreamNonBlocking, lo);
        }
        alloc(&s->frameInit, sizeof(pft2::EntropyState));
        alloc(&s->coderInit, sizeof(pft2::EntropyState));
        for (int p = 0; p < 3; p++) {
            s->ctxStride4[p] = (p ? aw / 2 : aw) / 4;
            s->ctxRows4[p] = (p ? ah / 2 : ah) / 4;
            alloc(&s->ctxLevel[p], size_t(s->ctxStride4[p]) * s->ctxRows4[p]);
            alloc(&s->ctxDc[p], size_t(s->ctxStride4[p]) * s->ctxRows4[p]);
        }
        const int nSb = (aw / 64) * (ah / 64);
        alloc(&s->modeTrials, sizeof(ModeTrial) * nSb * 64);
        alloc(&s->modeChanged, nSb * 64);
        alloc(&s->modeMi, sizeof(pft2::MiInfo) * miCols * miRows);
        alloc(&s->modeEntries, sizeof(pft2::BlockEntry) * nSb * 64);
        alloc(&s->modeEntryCount, sizeof(uint32_t) * nSb);
        alloc(&s->intraSel, gridCount() * sizeof(uint32_t));
        alloc(&s->intraDone, sizeof(int) * nSb);
        alloc(&s->intraTicket, sizeof(unsigned));
        if (e == cudaSuccess) e = cudaMemset(s->intraDone, 0, sizeof(int) * nSb);
        alloc(&s->modeMenus, sizeof(MotionOptions) * nSb * 64);
        alloc(&s->modeCounts, sizeof(int) * kShapes);
        size_t offsets[kShapes + 1] = {};
        for (int k = 0; k < kSizes; ++k) offsets[k + 1] = offsets[k] + size_t(s->cols[k]) * s->rows[k];
        for (int k = 0; k < 10; ++k)
            offsets[kSizes + k + 1] = offsets[kSizes + k] + size_t(s->rcols[k]) * s->rrows[k];
        alloc(&s->modeIndices[0], sizeof(int) * offsets[kShapes]);
        if (e == cudaSuccess)
            for (int k = 1; k < kShapes; ++k) s->modeIndices[k] = s->modeIndices[0] + offsets[k];
        alloc(&s->sbCount, sizeof(uint32_t) * (nSb * 64 + 1));
        alloc(&s->sbOffset, sizeof(uint32_t) * (nSb * 64 + 1));
        alloc(&s->entries, sizeof(pft2::BlockEntry) * nSb * 64);
        alloc(&s->entryCount, sizeof(uint32_t) * nSb);
        alloc(&s->sbTotal, sizeof(uint32_t) * nSb);
        alloc(&s->packOff, sizeof(uint32_t) * nSb * 64);
        s->recCap = uint32_t(nSb) * (64u * 96u + 24u * 4096u);
        alloc(&s->recs, sizeof(pft2::SymRec) * s->recCap);
        for (auto &es : s->eslot) alloc(&es.recs, sizeof(pft2::SymRec) * s->recCap);
        for (auto &es : s->eslot) {
            alloc(&es.grid, gridCount() * sizeof(BlockInfo));
            alloc(&es.mf, sizeof(pf::Mv) * 8 * size_t(miRows / 2) * (miCols / 2));
            alloc(&es.cdef, size_t((w + 63) / 64) * ((h + 63) / 64));
            alloc(&es.levels, levelCount() * sizeof(int16_t));
            alloc(&es.tileOut, s->ecCap * 2);
            alloc(&es.packed, s->ecCap * 2);
            alloc(&es.sizes, sizeof(int) * (s->maxTiles + 1));
            alloc(&es.error, sizeof(int));
            alloc(&es.tileCount, sizeof(uint32_t) * s->maxTiles);
            if (e == cudaSuccess) e = cudaEventCreateWithFlags(&es.staged, cudaEventDisableTiming);
            if (e == cudaSuccess) e = cudaEventCreateWithFlags(&es.done, cudaEventDisableTiming);
            if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&es.fetch, cudaStreamNonBlocking);
        }
        pft2::InterCdfs d{};
        defaultInterCdfs(d);
        if (e == cudaSuccess) e = cudaMemcpy(s->defaultInter, &d, sizeof d, cudaMemcpyHostToDevice);
    }
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(c_filter, pft::Subpel_Filters, sizeof(pft::Subpel_Filters));
    if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&s->compute, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaEventCreateWithFlags(&s->motionGridReady, cudaEventDisableTiming);
    if (e == cudaSuccess) e = cudaEventCreateWithFlags(&s->mfUploaded, cudaEventDisableTiming);
    if (e == cudaSuccess) e = cudaEventCreateWithFlags(&s->trialFork, cudaEventDisableTiming);
    for (int k = 0; k < kSizes; k++)
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&s->sqDone[k], cudaEventDisableTiming);
    for (int k = 0; k < kShapes; k++) {
        if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&s->shapeStream[k], cudaStreamNonBlocking);
        if (e == cudaSuccess) e = cudaEventCreateWithFlags(&s->trialJoin[k], cudaEventDisableTiming);
    }
    s->profile = getenv("AV1GPU_PF_PROFILE") != nullptr;
    if (s->profile) {
        for (auto &ev : s->ev) if (e == cudaSuccess) e = cudaEventCreate(&ev);
        for (auto &slot : s->eslot)
            for (auto &ev : slot.profile) if (e == cudaSuccess) e = cudaEventCreate(&ev);
    }
    if (e == cudaSuccess) e = cudaStreamSynchronize(nullptr);   // finish default-stream initialization
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame allocation: %s", cudaGetErrorString(e));
        return -1;
    }
    if (gp_upload_tables_pframe(gp_host_tables())) {
        snprintf(err, errLen, "P-frame table upload failed");
        return -1;
    }
    return 0;
}

// Hierarchical coarse search of one reference: the 1/16-resolution planes (newCur: the current one too) searched over
// +-r16 steps of 16 full samples (48: +-768; half the quarter-level lambda per step measured best), the
// quarter-resolution search over +-R around zero (16: +-64 samples, the smaller window measured better than +-192), then
// +-kCoarseAroundR around the vectors of the four nearest 1/16-resolution areas where those windows reach beyond the
// zero-centred one. Against the +-192 zero-centred search alone: video_3 -30%, video_4 -4%, video_2 -3%, video_1 +0.6%.
static void hierCoarse(State *s, cudaStream_t st, const uint8_t *curDs, const uint8_t *refDs, bool newCur, int R,
                       float lambda, Mv *out) {
    static const int r16 = getenv("AV1GPU_PF_RANGE16") ? atoi(getenv("AV1GPU_PF_RANGE16")) : 48;
    static const float lambda16 = getenv("AV1GPU_PF_LAMBDA16") ? (float)atof(getenv("AV1GPU_PF_LAMBDA16")) : 0.5f;
    const dim3 t16(16, 16), g16((s->dw16 + 15) / 16, (s->dh16 + 15) / 16);
    const int nq = s->ccols * s->crows;
    if (r16 > 0) {
        if (newCur) downsample4<<<g16, t16, 0, st>>>(curDs, s->dw, s->dw, s->dh, s->ds16[0], s->dw16, s->dh16);
        downsample4<<<g16, t16, 0, st>>>(refDs, s->dw, s->dw, s->dh, s->ds16[1], s->dw16, s->dh16);
        const int n16 = s->c16cols * s->c16rows;
        coarseSearch<<<(n16 + kCoarseWarps - 1) / kCoarseWarps, 32 * kCoarseWarps, 0, st>>>(
            s->ds16[0], s->ds16[1], s->dw16, s->dh16, std::min(kCoarseMaxR, r16), s->c16cols, s->c16rows, lambda * lambda16,
            s->coarse16);
    }
    coarseSearch<<<(nq + kCoarseWarps - 1) / kCoarseWarps, 32 * kCoarseWarps, 0, st>>>(
        curDs, refDs, s->dw, s->dh, R, s->ccols, s->crows, lambda, out, s->coarseCost);
    if (r16 > 0)
        coarseAround<<<(nq + kCoarseWarps - 1) / kCoarseWarps, 32 * kCoarseWarps, 0, st>>>(
            curDs, refDs, s->dw, s->dh, s->ccols, s->crows, lambda, s->coarse16, s->c16cols, s->c16rows, out,
            s->coarseCost, R);
}

int Analyzer::analyze(const uint8_t *const src[3], const int pitch[3], const uint8_t *const *refs, int refCount,
                      const Settings &st, const EntropyIn &syntax, uint8_t *recon, const ProjectMotion &projectMotion,
                      char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    const cudaStream_t stream = s->compute;
    collectAnalysisProfile(*s);
    const auto t0 = std::chrono::steady_clock::now();
    refCount = refCount < 1 ? 1 : refCount > kMaxRefs ? kMaxRefs : refCount;
    Frame f{};
    for (int p = 0; p < 3; p++) { f.src[p] = src[p]; f.pitch[p] = pitch[p]; }
    for (int r = 0; r < refCount; r++) f.ref[r] = refs[r];
    f.w8 = (w + 7) >> 3; f.h8 = (h + 7) >> 3;
    f.w = w; f.h = h; f.aw = aw; f.ah = ah;
    const float lambda = gp_lambda<8>(st.qindex, st.lambdaMul);
    const float lambdaSad = sqrtf(lambda);
    const dim3 t16(16, 16);
    const dim3 gds((s->dw + 15) / 16, (s->dh + 15) / 16);
    auto mark = [&](int i) { if (s->profile) cudaEventRecord(s->ev[i], stream); };
    mark(0);
    downsample4<<<gds, t16, 0, stream>>>(src[0], pitch[0], w, h, s->ds[0], s->dw, s->dh);
    const int searchRange = getenv("AV1GPU_PF_RANGE") ? atoi(getenv("AV1GPU_PF_RANGE")) : st.searchRange;
    const int R = std::min(kCoarseMaxR, std::max(2, searchRange / 4));
    const int nCoarse = s->ccols * s->crows;
    for (int r = 0; r < refCount; r++) {
        downsample4<<<gds, t16, 0, stream>>>(refs[r], w, w, h, s->ds[1 + r], s->dw, s->dh);
        hierCoarse(s, stream, s->ds[0], s->ds[1 + r], r == 0, R, lambdaSad * 0.5f, s->coarse[r]);
    }
    mark(1);
    s->useTmvp = syntax.useTmvp;
    if (s->useTmvp) {
        // Coarse search does not need the projected vectors. Its kernels run while the host stores the previous
        // referenced grid and projects this frame. The pinned upload is ordered before refinement.
        if (!projectMotion) {
            snprintf(err, errLen, "P-frame TMVP requires a motion field projection");
            return -1;
        }
        const Mv *field = projectMotion();
        if (!field) return -1;
        const size_t cells = size_t(f.h8) * f.w8, bytes = 8 * cells * sizeof(Mv);
        cudaError_t e = cudaEventSynchronize(s->mfUploaded);
        if (e == cudaSuccess) {
            memcpy(s->mfHost, field, bytes);
            e = cudaMemcpyAsync(s->mfDev, s->mfHost, bytes, cudaMemcpyHostToDevice, stream);
        }
        if (e == cudaSuccess) e = cudaEventRecord(s->mfUploaded, stream);
        if (e != cudaSuccess) {
            snprintf(err, errLen, "P-frame motion field upload: %s", cudaGetErrorString(e));
            return -1;
        }
        for (int r = 0; r < refCount; r++) f.tmv[r] = s->mfDev + size_t(syntax.refName[r]) * cells;
    }
    const BlockInfo *temporal = s->havePrev ? s->prevGrid : nullptr;
    // without the 8x8 level (Settings::blocks8 false): 8x8 blocks only where their 16x16 node crosses the frame edge
    // and must split; the others are pruned before motion search
    if (!st.blocks8)
        pruneSquare<<<(s->cols[0] * s->rows[0] + 127) / 128, 128, 0, stream>>>(s->cand[0], s->cols[0], s->rows[0], 8,
                                                                            s->cand[1], s->cols[1], w, h, 0.f, 1);
    for (int r = 0; r < refCount; r++) {
        { FineTimer ft_("phase planes");
        horizontalPhases<<<dim3((w + 31) / 32, (h + 7) / 8), dim3(32, 8), 0, stream>>>(refs[r], w, h, s->hphase);
        verticalPhases<<<dim3((w + 31) / 32, (h + 8 * kPhaseRows - 1) / (8 * kPhaseRows), 4), dim3(32, 8), 0, stream>>>(
            s->hphase, w, h, s->phases[r]); }
#define PF_REFINE(K, N)                                                                                         \
    FineTimer ft##K##_("refine " #N "x" #N);                                                                    \
    refine<N, N><<<(s->cols[K] * s->rows[K] + RefineCfg<N, N>::per() - 1) / RefineCfg<N, N>::per(),               \
                   32 * RefineCfg<N, N>::warps, 0, stream>>>(f, r, s->phases[r], s->coarse[r], s->ccols, s->crows,             \
                                                  K == 3 ? nullptr : s->cand[K + 1], K == 3 ? 0 : s->cols[K + 1],    \
                                                  2 * N, temporal, s->gcols, s->cand[K], s->cols[K], s->rows[K], lambdaSad, \
                                                  K == 0 && !st.blocks8, sqPasses)
        static const int sqPasses = getenv("AV1GPU_PF_SQPASSES") ? atoi(getenv("AV1GPU_PF_SQPASSES")) : kFullPelPasses;
        PF_REFINE(3, 64);
        PF_REFINE(2, 32);
        PF_REFINE(1, 16);
        PF_REFINE(0, 8);
#undef PF_REFINE
    }
#define PF_CHOOSE(C, CO, R, BW, BH)                                                                             \
    { FineTimer ft_("choose " #BW "x" #BH);                                                                     \
    chooseInter<BW, BH><<<(CO * R + RefineCfg<BW, BH>::per() - 1) / RefineCfg<BW, BH>::per(),                     \
                          32 * RefineCfg<BW, BH>::warps, 0, stream>>>(f, refCount, s->phases[0], s->phases[1], C, CO, R,        \
                                                           lambdaSad, st.compoundBias); }
    PF_CHOOSE(s->cand[3], s->cols[3], s->rows[3], 64, 64);
    PF_CHOOSE(s->cand[2], s->cols[2], s->rows[2], 32, 32);
    PF_CHOOSE(s->cand[1], s->cols[1], s->rows[1], 16, 16);
    PF_CHOOSE(s->cand[0], s->cols[0], s->rows[0], 8, 8);
    mark(2);
    EvalArgs ea{};
    ea.f = f;
    ea.qindex = st.qindex;
    ea.qctx = av1g_coeff_cdf_q_ctx(st.qindex);
    ea.lambda = lambda;
    ea.rnd = getenv("AV1GPU_PF_RND") ? (float)atof(getenv("AV1GPU_PF_RND")) : st.rnd;
    ea.zbin = getenv("AV1GPU_PF_ZBIN") ? (float)atof(getenv("AV1GPU_PF_ZBIN")) : st.zbin;
    ea.mvWeight = getenv("AV1GPU_PF_MVW") ? (float)atof(getenv("AV1GPU_PF_MVW")) : 0.5f;
    ea.modeBase = getenv("AV1GPU_PF_MODEBITS") ? (float)atof(getenv("AV1GPU_PF_MODEBITS")) : st.modeBits;
    ea.skipBits1 = getenv("AV1GPU_PF_SKIP1") ? (float)atof(getenv("AV1GPU_PF_SKIP1")) : 0.5f;
    ea.skipBits0 = getenv("AV1GPU_PF_SKIP0") ? (float)atof(getenv("AV1GPU_PF_SKIP0")) : 1.5f;
    ea.splitPrune = getenv("AV1GPU_PF_SPLITPRUNE") ? atoi(getenv("AV1GPU_PF_SPLITPRUNE")) : st.splitPrune;
    ea.rdoq = st.rdoq;
    ea.d2Margin = getenv("AV1GPU_PF_D2MARGIN") ? (float)atof(getenv("AV1GPU_PF_D2MARGIN")) : 1.0f;
    static const bool splitExternal = !getenv("AV1GPU_PF_INKERNEL") ;
    bool splitSearch = st.txSplit;   // evaluations search split transforms (the refinement trials may not)
    ea.reducedTxSet = st.txTypes == 1;
    ea.switchable = st.switchableFilter;
    ea.filterBias = getenv("AV1GPU_PF_FBIAS") ? (float)atof(getenv("AV1GPU_PF_FBIAS")) : 16.f;
    ea.txExhaustive = st.txTypes == 3;
    ea.splitTypes = getenv("AV1GPU_PF_SPLITTX") ? atoi(getenv("AV1GPU_PF_SPLITTX")) : st.splitTypes;
#define PF_EVAL(K, N)                                                                                         \
    ea.cand = s->cand[K]; ea.cols = s->cols[K]; ea.rows = s->rows[K];                                        \
    ea.pred = s->pred[K]; ea.recon = s->recon[K]; ea.levels = s->levels[K];                                   \
    ea.recon2 = splitSearch ? s->recon2[K] : nullptr; ea.levels2 = s->levels2[K];                             \
    ea.recon3 = s->recon3[K]; ea.levels3 = s->levels3[K];                                                     \
    ea.types1 = st.txTypes ? s->types1[K] : nullptr; ea.types2 = st.txTypes ? s->types2[K] : nullptr;         \
    ea.types3 = st.txTypes ? s->types3[K] : nullptr;                                                          \
    ea.subD = s->subD[K]; ea.subR = s->subR[K]; ea.subNz = s->subNz[K]; ea.subType = s->subType[K];           \
    ea.trialCount = s->modeCounts + (K); ea.trialIndices = s->modeIndices[K];                                 \
    { FineTimer t(ea.trials ? "mode trial " #N "x" #N : "evaluate " #N "x" #N);                                      \
      launchEvaluation<N, N>(ea, splitExternal, stream, "eval whole " #N, "eval split " #N, "eval final " #N); }
#define PF_REVAL(RK, BW, BH)                                                                                  \
    ea.cand = s->rcand[RK]; ea.cols = s->rcols[RK]; ea.rows = s->rrows[RK];                                 \
    ea.pred = s->rpred[RK]; ea.recon = s->rrecon[RK]; ea.levels = s->rlevels[RK];                              \
    ea.recon2 = splitSearch ? s->rrecon2[RK] : nullptr; ea.levels2 = s->rlevels2[RK];                         \
    ea.types1 = st.txTypes ? s->types1[4 + RK] : nullptr; ea.types2 = st.txTypes ? s->types2[4 + RK] : nullptr; \
    ea.types3 = nullptr;                                                                                     \
    ea.subD = s->rsubD[RK]; ea.subR = s->rsubR[RK]; ea.subNz = s->rsubNz[RK]; ea.subType = s->rsubType[RK];   \
    ea.trialCount = s->modeCounts + 4 + (RK); ea.trialIndices = s->modeIndices[4 + RK];                        \
    { FineTimer t(ea.trials ? "mode trial " #BW "x" #BH : "evaluate " #BW "x" #BH);                             \
      launchEvaluation<BW, BH>(ea, splitExternal, stream, "eval whole " #BW "x" #BH, "eval split " #BW "x" #BH, \
                               "eval final " #BW "x" #BH); }
#define PF_EVAL_RECTS() \
    PF_REVAL(0, 16, 8); PF_REVAL(1, 8, 16); PF_REVAL(2, 32, 16); PF_REVAL(3, 16, 32);                           \
    PF_REVAL(4, 64, 32); PF_REVAL(5, 32, 64); PF_REVAL(6, 32, 8); PF_REVAL(7, 8, 32);                            \
    PF_REVAL(8, 64, 16); PF_REVAL(9, 16, 64)
    // The four square evaluations run concurrently, one stream each (they are latency bound and write per-shape
    // buffers only); the rectangular searches of a square size (pruning, refinement, choice, evaluation) follow that
    // size's square evaluation on one stream per rectangle. Everything joins the compute stream before the decision.
    {
        cudaError_t fe = cudaEventRecord(s->trialFork, stream);
        for (int k = 0; fe == cudaSuccess && k < kShapes; k++) fe = cudaStreamWaitEvent(s->shapeStream[k], s->trialFork, 0);
        if (fe != cudaSuccess) {
            snprintf(err, errLen, "P-frame evaluation fork: %s", cudaGetErrorString(fe));
            return -1;
        }
    }
#define PF_ON(K, BODY) { const cudaStream_t stream = s->shapeStream[K]; BODY; }
    // square pruning (Settings::squarePrune margin, AV1GPU_PF_SQPRUNE overrides, 0 off): sizes in order
    // 64 -> 32 -> 16 -> 8, each size's children pruned from its result; otherwise the four sizes run concurrently
    // (every square block searched)
    const float sqMargin = getenv("AV1GPU_PF_SQPRUNE") ? (float)atof(getenv("AV1GPU_PF_SQPRUNE")) : st.squarePrune;
    for (int K = kSizes - 1; K >= 0; K--) {
        const cudaStream_t ks = s->shapeStream[K];
        if (K < kSizes - 1) {
            if (sqMargin > 0.f) cudaStreamWaitEvent(ks, s->sqDone[K + 1], 0);
            pruneSquare<<<(s->cols[K] * s->rows[K] + 127) / 128, 128, 0, ks>>>(s->cand[K], s->cols[K], s->rows[K], 8 << K,
                                                                           s->cand[K + 1], s->cols[K + 1], w, h, sqMargin,
                                                                           K == 0 && !st.blocks8);
        }
        if (K == 3) PF_ON(3, PF_EVAL(3, 64));
        if (K == 2) PF_ON(2, PF_EVAL(2, 32));
        if (K == 1) PF_ON(1, PF_EVAL(1, 16));
        if (K == 0) PF_ON(0, PF_EVAL(0, 8));
        cudaEventRecord(s->sqDone[K], ks);
    }
    if (st.rectPartitions) {
        for (int k = 0; k < 10; k++) {
            const int K = k < 6 ? k / 2 + 1 : (k - 6) / 2 + 2;
            const int n = 8 << K, div = k < 6 ? 2 : 4;
            const int bw = k & 1 ? n / div : n, bh = k & 1 ? n : n / div;
            cudaStreamWaitEvent(s->shapeStream[4 + k], s->sqDone[K], 0);
            // 4:1 shapes (k >= 6) off: margin 0 prunes every block (their searches return at once)
            const bool off = (k >= 6 && !st.rect4) || (k < 2 && !st.blocks8);
            pruneRect<<<(s->rcols[k] * s->rrows[k] + 127) / 128, 128, 0, s->shapeStream[4 + k]>>>(s->rcand[k], s->rcols[k], s->rrows[k], bw, bh,
                                                                     s->cand[K], s->cols[K], n, st.pruneRect || off,
                                                                     off ? 0.f : getenv("AV1GPU_PF_PRUNE") ? (float)atof(getenv("AV1GPU_PF_PRUNE")) : st.pruneMargin);
        }
        // full-pel search passes of the rectangles (they start from their square's vector; AV1GPU_PF_RECTPASSES)
        static const int rectPasses = getenv("AV1GPU_PF_RECTPASSES") ? atoi(getenv("AV1GPU_PF_RECTPASSES")) : kFullPelPasses;
        for (int r = 0; r < refCount; r++) {
        #define PF_RREFINE(RK, K, BW, BH)                                                                               \
            { FineTimer ft_("refine " #BW "x" #BH);                                                                     \
            refine<BW, BH><<<(s->rcols[RK] * s->rrows[RK] + RefineCfg<BW, BH>::per() - 1) / RefineCfg<BW, BH>::per(),   \
                             32 * RefineCfg<BW, BH>::warps, 0, s->shapeStream[4 + (RK)]>>>(f, r, s->phases[r], s->coarse[r], s->ccols, s->crows, \
                                                              s->cand[K], s->cols[K], 8 << K, temporal, s->gcols,            \
                                                              s->rcand[RK], s->rcols[RK], s->rrows[RK], lambdaSad, true, rectPasses); }
                {
                    PF_RREFINE(0, 1, 16, 8) PF_RREFINE(1, 1, 8, 16)
                    PF_RREFINE(2, 2, 32, 16) PF_RREFINE(3, 2, 16, 32)
                    PF_RREFINE(4, 3, 64, 32) PF_RREFINE(5, 3, 32, 64)
                    PF_RREFINE(6, 2, 32, 8) PF_RREFINE(7, 2, 8, 32)
                    PF_RREFINE(8, 3, 64, 16) PF_RREFINE(9, 3, 16, 64)
                }
        #undef PF_RREFINE

        }
    {
        PF_ON(4, PF_CHOOSE(s->rcand[0], s->rcols[0], s->rrows[0], 16, 8)); PF_ON(5, PF_CHOOSE(s->rcand[1], s->rcols[1], s->rrows[1], 8, 16));
        PF_ON(6, PF_CHOOSE(s->rcand[2], s->rcols[2], s->rrows[2], 32, 16)); PF_ON(7, PF_CHOOSE(s->rcand[3], s->rcols[3], s->rrows[3], 16, 32));
        PF_ON(8, PF_CHOOSE(s->rcand[4], s->rcols[4], s->rrows[4], 64, 32)); PF_ON(9, PF_CHOOSE(s->rcand[5], s->rcols[5], s->rrows[5], 32, 64));
        PF_ON(10, PF_CHOOSE(s->rcand[6], s->rcols[6], s->rrows[6], 32, 8)); PF_ON(11, PF_CHOOSE(s->rcand[7], s->rcols[7], s->rrows[7], 8, 32));
        PF_ON(12, PF_CHOOSE(s->rcand[8], s->rcols[8], s->rrows[8], 64, 16)); PF_ON(13, PF_CHOOSE(s->rcand[9], s->rcols[9], s->rrows[9], 16, 64));
    }
    PF_ON(4, PF_REVAL(0, 16, 8)); PF_ON(5, PF_REVAL(1, 8, 16)); PF_ON(6, PF_REVAL(2, 32, 16)); PF_ON(7, PF_REVAL(3, 16, 32));
    PF_ON(8, PF_REVAL(4, 64, 32)); PF_ON(9, PF_REVAL(5, 32, 64)); PF_ON(10, PF_REVAL(6, 32, 8)); PF_ON(11, PF_REVAL(7, 8, 32));
    PF_ON(12, PF_REVAL(8, 64, 16)); PF_ON(13, PF_REVAL(9, 16, 64));
    }
    {
        const int nShapes = st.rectPartitions ? kShapes : kSizes;
        cudaError_t je = cudaSuccess;
        for (int k = 0; je == cudaSuccess && k < nShapes; k++) {
            je = cudaEventRecord(s->trialJoin[k], s->shapeStream[k]);
            if (je == cudaSuccess) je = cudaStreamWaitEvent(stream, s->trialJoin[k], 0);
        }
        if (je != cudaSuccess) {
            snprintf(err, errLen, "P-frame evaluation join: %s", cudaGetErrorString(je));
            return -1;
        }
    }
#undef PF_ON
#undef PF_CHOOSE
    mark(3);
    DecideArgs da{};
    for (int k = 0; k < kSizes; k++) {
        da.cand[k] = s->cand[k]; da.cols[k] = s->cols[k]; da.best[k] = s->best[k]; da.mode[k] = s->mode[k];
    }
    for (int k = 0; k < 10; k++) { da.rcand[k] = s->rcand[k]; da.rcols[k] = s->rcols[k]; }
    da.rect = st.rectPartitions;
    da.w = w; da.h = h; da.gcols = s->gcols; da.grows = s->grows; da.grid = s->grid; da.lambda = lambda;
    // approximate partition symbol costs (bits)
    static const float splitScale = getenv("AV1GPU_PF_SPLITBITS") ? (float)atof(getenv("AV1GPU_PF_SPLITBITS")) : 1.f;
    const float part[kSizes][2] = {{1.0f, 0.f}, {1.0f, 2.5f * splitScale}, {1.0f, 2.5f * splitScale}, {1.0f, 2.0f * splitScale}};
    memcpy(da.partBits, part, sizeof part);
    for (int k = 0; k < kSizes; k++)
        decideLevel<<<(s->cols[k] * s->rows[k] + 127) / 128, 128, 0, stream>>>(da, k, s->rows[k]);
    decideGrid<<<dim3((s->gcols + 15) / 16, (s->grows + 15) / 16), t16, 0, stream>>>(da);
    mark(4);
    CommitArgs ca{};
    ca.grid = s->grid; ca.gcols = s->gcols; ca.w = w; ca.h = h; ca.aw = aw; ca.ah = ah;
    for (int k = 0; k < kSizes; k++) {
        ca.pred[k] = s->pred[k]; ca.recon[k] = s->recon[k]; ca.levels[k] = s->levels[k];
        ca.recon2[k] = s->recon2[k]; ca.levels2[k] = s->levels2[k];
        ca.recon3[k] = s->recon3[k]; ca.levels3[k] = s->levels3[k];
    }
    for (int k = 0; k < 10; k++) {
        ca.pred[4 + k] = s->rpred[k]; ca.recon[4 + k] = s->rrecon[k]; ca.levels[4 + k] = s->rlevels[k];
        ca.recon2[4 + k] = s->rrecon2[k]; ca.levels2[4 + k] = s->rlevels2[k];
    }
    ca.out = recon;
    ca.outLevels = s->outLevels;
    const dim3 commitGrid((aw + 15) / 16, (ah + 15) / 16, 3);
    const dim3 typeGrid((s->gcols + 15) / 16, (s->grows + 15) / 16);
    TypeArgs ta{};
    ta.grid = s->grid; ta.gcols = s->gcols; ta.grows = s->grows; ta.tstride = aw / 4; ta.enabled = st.txTypes;
    for (int k = 0; k < kShapes; k++) { ta.types1[k] = s->types1[k]; ta.types2[k] = s->types2[k]; }
    for (int k = 0; k < kSizes; k++) ta.types3[k] = s->types3[k];
    commit<<<commitGrid, t16, 0, stream>>>(ca);
    fillTxTypes<<<typeGrid, t16, 0, stream>>>(ta);
    mark(5);
    if (st.modeRefine && st.modeRefinePasses > 0) {
        TileJob j{};
        setTileLayout(j, f.w8 * 2, f.h8 * 2, syntax.tileColsLog2, syntax.tileRowsLog2);
        j.fs.gcols = s->gcols;
        std::copy_n(syntax.refName, 3, j.fs.refName);
        std::copy_n(syntax.signBias, 8, j.fs.signBias);
        j.fs.useTmvp = s->useTmvp;
        j.fs.skipMode = syntax.skipMode;
        j.grid = s->grid;
        j.mf = s->mfDev;
        j.mi = s->modeMi;
        j.entries = s->modeEntries;
        j.entryCount = s->modeEntryCount;
        const int nSb = j.sbCols * j.sbRows;
        const float origMvWeight = ea.mvWeight;
        static const int trialFast = getenv("AV1GPU_PF_TRIALFAST") ? atoi(getenv("AV1GPU_PF_TRIALFAST")) : 0;
        // extra bits credited to a skip_mode option (it also saves the references, skip flag and filter, but 0
        // measured best: 4.5 / 2 / -2 were worse)
        static const float skipModeCredit = getenv("AV1GPU_PF_SMCREDIT") ? (float)atof(getenv("AV1GPU_PF_SMCREDIT")) : 0.f;
        for (int pass = 0; pass < st.modeRefinePasses; ++pass) {
        cudaError_t e = cudaMemsetAsync(s->modeCounts, 0, sizeof(int) * kShapes, stream);

        if (e != cudaSuccess) {
            snprintf(err, errLen, "P-frame mode refinement: %s", cudaGetErrorString(e));
            return -1;
        }
        {
            FineTimer t("mode refinement proposals");
            if (!pass) tilePhase1<<<(nSb + 63) / 64, 64, 0, stream>>>(j);
            refinementModeInfo<<<nSb, 64, 0, stream>>>(j, pass ? s->modeChanged : nullptr);
            RefinementArgs ra{};
            ra.f = f;
            std::copy_n(s->phases, refCount, ra.phases);
            std::copy_n(s->cand, kSizes, ra.candidates);
            std::copy_n(s->rcand, 10, ra.candidates + kSizes);
            std::copy_n(s->cols, kSizes, ra.cols);
            std::copy_n(s->rcols, 10, ra.cols + kSizes);
            ra.trials = s->modeTrials;
            ra.trialCount = s->modeCounts;
            std::copy_n(s->modeIndices, kShapes, ra.trialIndices);
            ra.lambda = lambda;
            ra.origMvWeight = origMvWeight;
            ra.screenWeight = 3.f * lambdaSad;
            ra.pass = pass;
            ra.keepDepth = (trialFast & 16) != 0;
            priceMotion<<<nSb, 64, 0, stream>>>(j, s->defaultInter, s->modeMenus, pass, skipModeCredit, s->modeChanged);
            proposeMotion<<<nSb * kProposeSlices, 32 * kProposeWarps, 0, stream>>>(j, ra, s->modeMenus);
        }
        // The original output is already committed. Trials reuse the per-shape evaluation buffers,
        // and only accepted blocks overwrite their reconstruction, levels and transform types.
        ea.trials = s->modeTrials;
        ea.mvWeight = 0.f;
        // experiment (AV1GPU_PF_TRIALFAST bits): 1 DCT-only split transforms, 2 seven-type whole transforms,
        // 4 no interpolation-filter search in the trials
        if (trialFast & 1) ea.splitTypes = 2;
        if (trialFast & 2) ea.txExhaustive = 0;
        if (trialFast & 4) ea.switchable = 0;
        if (trialFast & 8) splitSearch = false;   // no split-transform search in the trials
        {
            e = cudaEventRecord(s->trialFork, stream);
            const int nTrialShapes = st.rectPartitions ? kShapes : kSizes;
            for (int k = 0; e == cudaSuccess && k < nTrialShapes; k++)
                e = cudaStreamWaitEvent(s->shapeStream[k], s->trialFork, 0);
            if (e != cudaSuccess) {
                snprintf(err, errLen, "P-frame mode refinement: %s", cudaGetErrorString(e));
                return -1;
            }
            {
#define PF_TRIAL(K, EVAL) { const cudaStream_t stream = s->shapeStream[K]; EVAL; }
                PF_TRIAL(3, PF_EVAL(3, 64));
                PF_TRIAL(2, PF_EVAL(2, 32));
                PF_TRIAL(1, PF_EVAL(1, 16));
                PF_TRIAL(0, PF_EVAL(0, 8));
                if (st.rectPartitions) {
                    PF_TRIAL(8, PF_REVAL(4, 64, 32)); PF_TRIAL(9, PF_REVAL(5, 32, 64));
                    PF_TRIAL(12, PF_REVAL(8, 64, 16)); PF_TRIAL(13, PF_REVAL(9, 16, 64));
                    PF_TRIAL(6, PF_REVAL(2, 32, 16)); PF_TRIAL(7, PF_REVAL(3, 16, 32));
                    PF_TRIAL(10, PF_REVAL(6, 32, 8)); PF_TRIAL(11, PF_REVAL(7, 8, 32));
                    PF_TRIAL(4, PF_REVAL(0, 16, 8)); PF_TRIAL(5, PF_REVAL(1, 8, 16));
                }
#undef PF_TRIAL
            }
            for (int k = 0; e == cudaSuccess && k < nTrialShapes; k++) {
                e = cudaEventRecord(s->trialJoin[k], s->shapeStream[k]);
                if (e == cudaSuccess) e = cudaStreamWaitEvent(stream, s->trialJoin[k], 0);
            }
            if (e != cudaSuccess) {
                snprintf(err, errLen, "P-frame mode refinement: %s", cudaGetErrorString(e));
                return -1;
            }
        }
        {
            FineTimer t("mode refinement commit");
            // the previous pass's changes were read by refinementModeInfo and priceMotion
            cudaMemsetAsync(s->modeChanged, 0, nSb * 64, stream);
            acceptMotion<<<nSb, 64, 0, stream>>>(j, s->modeTrials, s->grid, s->modeChanged, lambda, pass);
            ca.changed = ta.changed = s->modeChanged;
            commitUnits<<<((aw / 8) * (ah / 8) + 7) / 8, 256, 0, stream>>>(ca);
            fillTxTypes<<<typeGrid, t16, 0, stream>>>(ta);
        }
        if (s->refineStats) {
            int counts[kShapes];
            cudaMemcpyAsync(counts, s->modeCounts, sizeof(counts), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            fprintf(stderr, "refine pass %d trials:", pass);
            int total = 0;
            for (int k = 0; k < kShapes; k++) { fprintf(stderr, " %d", counts[k]); total += counts[k]; }
            fprintf(stderr, " total %d\n", total);
        }
        }
    }
    if (st.intra) {
        // intra blocks (see intraDecide / intraRecon)
        FineTimer t("intra blocks");
        IntraArgs ia{};
        setTileLayout(ia.j, f.w8 * 2, f.h8 * 2, syntax.tileColsLog2, syntax.tileRowsLog2);
        ia.j.fs.gcols = s->gcols;
        ia.j.grid = s->grid;
        ia.grid = s->grid;
        ia.j.entries = s->modeEntries;
        ia.j.entryCount = s->modeEntryCount;
        const int nSb = ia.j.sbCols * ia.j.sbRows;
        if (!(st.modeRefine && st.modeRefinePasses > 0)) tilePhase1<<<(nSb + 63) / 64, 64, 0, stream>>>(ia.j);
        for (int p = 0; p < 3; p++) { ia.src[p] = src[p]; ia.pitch[p] = pitch[p]; }
        ia.recon = recon;
        ia.levels = s->outLevels;
        ia.w = w; ia.h = h; ia.aw = aw; ia.ah = ah;
        ia.qindex = st.qindex; ia.qctx = ea.qctx; ia.reduced = ea.reducedTxSet;
        ia.rnd = ea.rnd; ia.zbin = ea.zbin; ia.lambda = lambda;
        ia.txSplit = syntax.txModeSelect && !getenv("AV1GPU_PF_INTRA_NOSPLIT");
        ia.uvModes = !getenv("AV1GPU_PF_INTRA_UVDC");
        ia.uvAll = !getenv("AV1GPU_PF_INTRA_UVSAD");
        ia.cfl = !getenv("AV1GPU_PF_INTRA_NOCFL");   // video_1 -0.32%, video_2 -0.12%
        ia.cflBits = getenv("AV1GPU_PF_INTRA_CFLBITS") ? (float)atof(getenv("AV1GPU_PF_INTRA_CFLBITS")) : 5.f;
        ia.uvTip = getenv("AV1GPU_PF_INTRA_UVTIP") != nullptr;   // every chroma mode by RD: video_2 -0.28%
        ia.topK64 = getenv("AV1GPU_PF_INTRA_TOPK64") ? atoi(getenv("AV1GPU_PF_INTRA_TOPK64")) : -1;
        ia.uvSad64 = getenv("AV1GPU_PF_INTRA_UVSAD64") != nullptr;
        ia.topK = getenv("AV1GPU_PF_INTRA_TOPK") ? atoi(getenv("AV1GPU_PF_INTRA_TOPK")) : 10;   // every mode: video_2 -2.3% over 3
        ia.alt3 = !getenv("AV1GPU_PF_INTRA_NOALT3");   // video_1 -0.09%, video_2 -0.52%
        ia.alt2 = !getenv("AV1GPU_PF_INTRA_NOALT2");   // video_1 -0.24%, video_2 -0.48%
        ia.splitIntra = getenv("AV1GPU_PF_INTRA_QUAD") != nullptr;   // off: 6% of the encode for -0.07% (video_2)
        ia.splitPartBits = getenv("AV1GPU_PF_INTRA_QUADBITS") ? (float)atof(getenv("AV1GPU_PF_INTRA_QUADBITS")) : 1.5f;
        ia.satd = getenv("AV1GPU_PF_INTRA_SATD") != nullptr;
        ia.est = getenv("AV1GPU_PF_INTRA_EST") ? atoi(getenv("AV1GPU_PF_INTRA_EST")) : 0;
        ia.splitDct = getenv("AV1GPU_PF_INTRA_SPLITDCT") != nullptr;
        ia.earlyGate = getenv("AV1GPU_PF_INTRA_EARLYGATE") ? (float)atof(getenv("AV1GPU_PF_INTRA_EARLYGATE")) : 1.5f;   // -3.8% time, +0.11% BD (video_1 / video_2)
        ia.sadGate = getenv("AV1GPU_PF_INTRA_SADGATE") ? (float)atof(getenv("AV1GPU_PF_INTRA_SADGATE")) : 0.f;
        ia.margin = getenv("AV1GPU_PF_INTRA_MARGIN") ? (float)atof(getenv("AV1GPU_PF_INTRA_MARGIN")) : 1.f;
        ia.nModes = st.intraDirectional && !getenv("AV1GPU_PF_INTRA_NODIR") ? kIntraModes : kIntraModesNoDir;
        ia.nLuma = ia.nModes == kIntraModes && getenv("AV1GPU_PF_INTRA_DELTA") ? kLumaCands : ia.nModes;   // off: neutral
        ia.deltaK = getenv("AV1GPU_PF_INTRA_DELTAK") ? atoi(getenv("AV1GPU_PF_INTRA_DELTAK")) : 3;
        ia.uvBits = getenv("AV1GPU_PF_INTRA_UVBITS") ? (float)atof(getenv("AV1GPU_PF_INTRA_UVBITS")) : 1.f;
        ia.splitGate = getenv("AV1GPU_PF_INTRA_SPLITGATE") ? (float)atof(getenv("AV1GPU_PF_INTRA_SPLITGATE")) : 1.3f;
        ia.intraBits = getenv("AV1GPU_PF_INTRABITS") ? (float)atof(getenv("AV1GPU_PF_INTRABITS")) : st.intraBits;
        ia.modeBase = ea.modeBase; ia.skipBits0 = ea.skipBits0; ia.skipBits1 = ea.skipBits1;
        ia.mvWeight = getenv("AV1GPU_PF_MVW") ? (float)atof(getenv("AV1GPU_PF_MVW")) : 0.5f;
        for (int k = 0; k < kSizes; k++) { ia.cand[k] = s->cand[k]; ia.cols[k] = s->cols[k]; }
        for (int k = 0; k < 10; k++) { ia.rcand[k] = s->rcand[k]; ia.rcols[k] = s->rcols[k]; }
        ia.trials = st.modeRefine && st.modeRefinePasses > 0 ? s->modeTrials : nullptr;
        ia.sel = s->intraSel;
        ia.done = s->intraDone;
        ia.ticket = s->intraTicket;
        ia.tag = ++s->intraTag;
        cudaMemsetAsync(s->intraSel, 0, gridCount() * sizeof(uint32_t), stream);
        cudaMemsetAsync(s->intraTicket, 0, sizeof(unsigned), stream);
        const int gc = s->gcols, gr = s->grows;
        { FineTimer t2("intra decide");
        // the four sizes are independent (they read the reconstruction, write their own block positions): one shape
        // stream each, joined before the reconstruction pass
        // rectangular shapes too unless AV1GPU_PF_INTRA_NORECT
        static const bool intraRect = !getenv("AV1GPU_PF_INTRA_NORECT");
        const int nStreams = intraRect ? kShapes : 4;
        cudaEventRecord(s->trialFork, stream);
        for (int k = 0; k < nStreams; k++) cudaStreamWaitEvent(s->shapeStream[k], s->trialFork, 0);
        int si = 0;
#define PF_IDEC(LW, LH)                                                                                            \
        {                                                                                                          \
            constexpr int uw = 1 << (LW - 3), uh = 1 << (LH - 3), wp = IntraCfg<LW, LH>::warps;                    \
            const int n = ((gc + uw - 1) / uw) * ((gr + uh - 1) / uh);                                             \
            intraDecide<LW, LH><<<(n + wp - 1) / wp, 32 * wp, 0, s->shapeStream[si++]>>>(ia);                      \
        }
        PF_IDEC(6, 6) PF_IDEC(5, 5) PF_IDEC(4, 4) PF_IDEC(3, 3)
        if (intraRect) {
            PF_IDEC(4, 3) PF_IDEC(3, 4) PF_IDEC(5, 4) PF_IDEC(4, 5) PF_IDEC(6, 5) PF_IDEC(5, 6)
            PF_IDEC(5, 3) PF_IDEC(3, 5) PF_IDEC(6, 4) PF_IDEC(4, 6)
        }
#undef PF_IDEC
        for (int k = 0; k < nStreams; k++) {
            cudaEventRecord(s->trialJoin[k], s->shapeStream[k]);
            cudaStreamWaitEvent(stream, s->trialJoin[k], 0);
        }
        }
        if (ia.splitIntra) {
            applyIntraSplit<<<dim3((gc + 127) / 128, (gr + 127) / 128), dim3(16, 16), 0, stream>>>(s->grid, s->intraSel, gc, gr);
            tilePhase1<<<(nSb + 63) / 64, 64, 0, stream>>>(ia.j);   // block lists of the new partition
        }
        { FineTimer t3("intra recon");
        intraRecon<<<nSb, 96, 0, stream>>>(ia); }
        static const bool intraWorkStats = getenv("AV1GPU_PF_INTRA_WORK") != nullptr;
        if (intraWorkStats) {
            static unsigned long long tot[4];
            static int nf;
            unsigned long long wk[4], zero[4] = {};
            cudaStreamSynchronize(stream);
            cudaMemcpyFromSymbol(wk, g_intraWork, sizeof wk);
            cudaMemcpyToSymbol(g_intraWork, zero, sizeof zero);
            for (int k = 0; k < 4; k++) tot[k] += wk[k];
            if (++nf % 20 == 0)
                fprintf(stderr, "intra work per frame (Msamples): luma %.2f chroma %.2f cfl %.2f sad %.2f\n", tot[0] / 1e6 / nf,
                        tot[1] / 1e6 / nf, tot[2] / 1e6 / nf, tot[3] / 1e6 / nf);
        }
        static const bool intraStats = getenv("AV1GPU_PF_INTRA_STATS") != nullptr;
        if (intraStats) {   // share of the frame area in intra blocks, by size (AV1GPU_PF_INTRA_STATS)
            static long long area[5], frames;
            std::vector<BlockInfo> g(gridCount());
            cudaMemcpyAsync(g.data(), s->grid, g.size() * sizeof(BlockInfo), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            long long fa = 0, fi = 0, fr = 0;
            for (const BlockInfo &b : g) if (b.log2) { area[4]++; fa++; fr += b.part || b.ab; if (b.intra) { area[b.log2 - 3]++; fi++; } }
            if (getenv("AV1GPU_PF_INTRA_STATS")[0] == '2')
                fprintf(stderr, "intra frame %lld: %.1f%% intra, %.1f%% rectangular / three-way\n", frames, 100.0 * fi / fa, 100.0 * fr / fa);
            if (++frames % 30 == 0)
                fprintf(stderr, "intra area: %.2f%% (8x8 %.2f, 16 %.2f, 32 %.2f, 64 %.2f)\n", 100.0 * (area[0] + area[1] + area[2] + area[3]) / area[4],
                        100.0 * area[0] / area[4], 100.0 * area[1] / area[4], 100.0 * area[2] / area[4], 100.0 * area[3] / area[4]);
        }
    }
#undef PF_EVAL_RECTS
#undef PF_REVAL
#undef PF_EVAL
    mark(6);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(s->prevGrid, s->grid, gridCount() * sizeof(BlockInfo), cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame analysis: %s", cudaGetErrorString(e));
        return -1;
    }
    s->havePrev = true;
    mark(7);
    s->profilePending = s->profile;
    elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

// Entropy coding of the last analyzed frame, asynchronous: the decisions are staged into entropy slot k (on the
// analysis stream), then the tile kernel runs on its own stream, overlapping the analysis of later frames. Kernels of
// that stream run in coding order, so the reference-slot CDFs they read and write stay consistent.
int Analyzer::entropyLaunch(const EntropyIn &in, int k, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    const int miCols = 2 * ((w + 7) >> 3), miRows = 2 * ((h + 7) >> 3);
    TileJob j{};
    setTileLayout(j, miCols, miRows, in.tileColsLog2, in.tileRowsLog2);
    const int nTiles = j.tileCols * j.tileRows;
    if (nTiles > s->maxTiles || k < 0 || k >= kEntropySlots) {
        snprintf(err, errLen, "too many tiles (%d) or bad entropy slot", nTiles);
        return -1;
    }
    EntropySlot &es = s->eslot[k];
    es.nTiles = nTiles;
    cudaError_t e = cudaMemcpyAsync(es.grid, s->grid, gridCount() * sizeof(BlockInfo), cudaMemcpyDeviceToDevice, s->compute);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(es.levels, s->outLevels, levelCount() * sizeof(int16_t), cudaMemcpyDeviceToDevice, s->compute);
    if (e == cudaSuccess && in.useTmvp && s->useTmvp) {
        const size_t n = 8 * size_t(miRows / 2) * (miCols / 2);
        e = cudaMemcpyAsync(es.mf, s->mfDev, n * sizeof(Mv), cudaMemcpyDeviceToDevice, s->compute);
    }
    if (e == cudaSuccess && in.cdefBits && s->cdefIdxDev)
        e = cudaMemcpyAsync(es.cdef, s->cdefIdxDev, size_t((w + 63) / 64) * ((h + 63) / 64), cudaMemcpyDeviceToDevice,
                            s->compute);
    if (e == cudaSuccess) e = cudaEventRecord(es.staged, s->compute);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(s->entStream, es.staged, 0);
    j.fs.refName[0] = in.refName[0];
    j.fs.refName[1] = in.refName[1];
    j.fs.refName[2] = in.refName[2];
    j.fs.referenceSelect = in.referenceSelect;
    j.fs.txModeSelect = in.txModeSelect;
    j.fs.reducedTxSet = in.reducedTxSet;
    j.fs.switchable = in.switchableFilter;
    j.fs.skipMode = in.skipMode;
    j.fs.filterIntra = in.filterIntra;
    j.fs.cdefBits = in.cdefBits;
    j.fs.cdefIdx = es.cdef;
    j.fs.lrLuma = in.lrLuma; j.fs.width = w; j.fs.height = h;
    for (int k = 0; k < 8; k++) j.fs.signBias[k] = in.signBias[k];
    j.fs.useTmvp = in.useTmvp && s->useTmvp;
    j.mf = es.mf;
    j.fs.aw = aw; j.fs.ah = ah; j.fs.gcols = s->gcols;
    j.grid = es.grid;
    j.levels = es.levels;
    j.mi = s->mi;
    j.frameInit = s->frameInit;
    j.coderInit = s->coderInit;
    j.packOff = s->packOff; j.tileCount = es.tileCount; j.packed = es.recs;
    j.save = in.saveSlot >= 0 ? s->slots + in.saveSlot : nullptr;
    j.contextTile = std::min(in.contextTile, nTiles - 1);
    for (int p = 0; p < 3; p++) { j.ctxLevel[p] = s->ctxLevel[p]; j.ctxDc[p] = s->ctxDc[p]; j.ctxStride[p] = s->ctxStride4[p]; }
    j.sbCount = s->sbCount; j.sbOffset = s->sbOffset;
    j.entries = s->entries; j.entryCount = s->entryCount;
    j.recs = s->recs; j.recCap = s->recCap;
    j.error = es.error;
    j.ecCapPerTile = s->ecCap * 2 / nTiles;
    j.ecBuf = s->ecBuf;
    j.out = es.tileOut;
    j.sizes = es.sizes;
    cudaStream_t st = s->entStream;
    if (e == cudaSuccess) e = cudaMemsetAsync(es.error, 0, sizeof(int), st);
    for (int p = 0; p < 3 && e == cudaSuccess; p++) {
        const size_t bytes = size_t(s->ctxStride4[p]) * s->ctxRows4[p];
        e = cudaMemsetAsync(s->ctxLevel[p], 0, bytes, st);
        if (e == cudaSuccess) e = cudaMemsetAsync(s->ctxDc[p], 0, bytes, st);
    }
    // costs and cdf ids from the default CDFs; the coder starts from the primary slot, copied before this frame may
    // overwrite it as its save slot
    initFrameCdfs<<<1, 1, 0, st>>>(s->frameInit, nullptr, s->defaultInter, in.qindex);
    if (!in.recordsOnly)
        initFrameCdfs<<<1, 1, 0, st>>>(s->coderInit, in.primarySlot >= 0 ? s->slots + in.primarySlot : nullptr,
                                       s->defaultInter, in.qindex);
    const int nSb = j.sbCols * j.sbRows;
    auto emark = [&](int i) { if (s->profile) cudaEventRecord(es.profile[i], st); };
    emark(0);
    tilePhase1<<<(nSb + 63) / 64, 64, 0, st>>>(j);
    emark(1);
    tilePhaseBlocks<<<nSb, 64, 0, st>>>(j, 4, 0);
    tilePhaseBlocks<<<nSb, 64, 0, st>>>(j, 2, 0);
    emark(2);
    blockOffsets<<<(nSb + 127) / 128, 128, 0, st>>>(j, s->recCap / nSb);
    tilePhaseBlocks<<<nSb, 64, 0, st>>>(j, 3, 1);
    emark(3);
    es.recordsOnly = in.recordsOnly;
    if (in.recordsOnly) {
        packOffsets<<<1, 1, 0, st>>>(j);
        packRecords<<<nSb, 64, 0, st>>>(j);
        emark(4);
        if (e == cudaSuccess) e = cudaEventRecord(es.done, st);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) {
            snprintf(err, errLen, "tile entropy launch: %s", cudaGetErrorString(e));
            return -1;
        }
        return 0;
    }
    tileCode<<<nTiles, 64, 0, st>>>(j);
    emark(4);
    packTiles<<<nTiles, 256, 0, st>>>(es.tileOut, j.ecCapPerTile, es.sizes, nTiles, es.packed);
    if (e == cudaSuccess) e = cudaEventRecord(es.done, s->entStream);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {
        snprintf(err, errLen, "tile entropy launch: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

int Analyzer::entropyFetch(int k, EntropyOut &out, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    EntropySlot &es = s->eslot[k];
    const int nTiles = es.nTiles;
    std::vector<int> sizes(nTiles);
    int error = 0;
    // the fetch stream does not wait for later analysis or entropy work
    cudaError_t e = cudaEventSynchronize(es.done);
    if (e == cudaSuccess) collectEntropyProfile(*s, es);
    if (e == cudaSuccess) e = cudaMemcpyAsync(sizes.data(), es.sizes, sizeof(int) * nTiles, cudaMemcpyDeviceToHost, es.fetch);
    if (e == cudaSuccess) e = cudaMemcpyAsync(&error, es.error, sizeof(int), cudaMemcpyDeviceToHost, es.fetch);
    if (e == cudaSuccess) e = cudaStreamSynchronize(es.fetch);
    if (e == cudaSuccess && error) {
        snprintf(err, errLen, "tile entropy coding failed (error %d%s)", error,
                 error == 9 ? ": symbol buffer overflow, use --pf-cpu-entropy" : "");
        return -1;
    }
    size_t total = 0;
    for (int t = 0; t < nTiles; t++) total += sizes[t];
    std::vector<uint8_t> packed(total);
    if (e == cudaSuccess && total) e = cudaMemcpyAsync(packed.data(), es.packed, total, cudaMemcpyDeviceToHost, es.fetch);
    if (e == cudaSuccess) e = cudaStreamSynchronize(es.fetch);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "tile entropy: %s", cudaGetErrorString(e));
        return -1;
    }
    out.tiles.assign(nTiles, {});
    size_t off = 0;
    out.compound = 0;
    for (int q = 0; q < 4; q++) { out.bits[q] = 0; out.modes[q] = 0; }
    for (int t = 0; t < nTiles; t++) {
        out.tiles[t].assign(packed.begin() + off, packed.begin() + off + sizes[t]);
        off += sizes[t];
    }
    return 0;
}

int Analyzer::entropyFetchRecords(int k, std::vector<uint32_t> &tileCounts, std::vector<uint8_t> &records, char *err,
                                  int errLen) {
    auto *s = static_cast<State *>(state);
    EntropySlot &es = s->eslot[k];
    tileCounts.assign(es.nTiles, 0);
    int error = 0;
    cudaError_t e = cudaEventSynchronize(es.done);
    if (e == cudaSuccess) collectEntropyProfile(*s, es);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(tileCounts.data(), es.tileCount, sizeof(uint32_t) * es.nTiles, cudaMemcpyDeviceToHost, es.fetch);
    if (e == cudaSuccess) e = cudaMemcpyAsync(&error, es.error, sizeof(int), cudaMemcpyDeviceToHost, es.fetch);
    if (e == cudaSuccess) e = cudaStreamSynchronize(es.fetch);
    if (e == cudaSuccess && error) {
        snprintf(err, errLen, "inter symbol modeling failed (error %d%s)", error,
                 error == 9 ? ": symbol buffer overflow, use --pf-cpu-entropy" : "");
        return -1;
    }
    size_t total = 0;
    for (uint32_t c : tileCounts) total += c;
    records.resize(total * sizeof(pft2::SymRec));
    if (e == cudaSuccess && total)
        e = cudaMemcpyAsync(records.data(), es.recs, records.size(), cudaMemcpyDeviceToHost, es.fetch);
    if (e == cudaSuccess) e = cudaStreamSynchronize(es.fetch);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "symbol records: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

int Analyzer::entropy(const EntropyIn &in, EntropyOut &out, char *err, int errLen) {
    return entropyLaunch(in, 0, err, errLen) ? -1 : entropyFetch(0, out, err, errLen);
}

cudaStream_t Analyzer::stream() const {
    return static_cast<State *>(state)->compute;
}

int Analyzer::stageMotionGrid(char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    cudaError_t e = cudaMemcpyAsync(s->motionGridHost, s->prevGrid, gridCount() * sizeof(BlockInfo),
                                    cudaMemcpyDeviceToHost, s->compute);
    if (e == cudaSuccess) e = cudaEventRecord(s->motionGridReady, s->compute);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame motion grid download: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

const BlockInfo *Analyzer::motionGrid(char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    const cudaError_t e = cudaEventSynchronize(s->motionGridReady);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame motion grid download: %s", cudaGetErrorString(e));
        return nullptr;
    }
    return s->motionGridHost;
}

int Analyzer::download(BlockInfo *info, int16_t *levels, cudaEvent_t done, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    cudaError_t e = cudaMemcpyAsync(info, s->grid, gridCount() * sizeof(BlockInfo), cudaMemcpyDeviceToHost, s->compute);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(levels, s->outLevels, levelCount() * sizeof(int16_t), cudaMemcpyDeviceToHost, s->compute);
    if (e == cudaSuccess) e = cudaEventRecord(done, s->compute);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame download: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

// ---- CPU tile entropy ----
namespace {
struct HostState {
    pft2::EntropyState slots[8];
    pft2::InterCdfs defaults;
    std::vector<pft2::MiInfo> mi;
    std::vector<uint8_t> ctx;
    std::vector<uint16_t> ecBuf;
    std::vector<uint8_t> out;
    pft2::EntropyState work, init;
    int16_t qv[1024];
    std::vector<uint8_t> ctxLevel[3], ctxDc[3];
    std::vector<pft2::SymRec> recs;
    std::vector<pft2::BlockEntry> entries;
};
}  // namespace

HostEntropy::~HostEntropy() { delete static_cast<HostState *>(state); }

int HostEntropy::init(int width, int height, char *, int) {
    w = width; h = height;
    aw = (w + 63) & ~63; ah = (h + 63) & ~63;
    auto *s = new HostState;
    state = s;
    const int miCols = 2 * ((w + 7) >> 3), miRows = 2 * ((h + 7) >> 3);
    s->mi.resize(size_t(miCols) * miRows);
    for (int p = 0; p < 3; p++) {
        s->ctxLevel[p].resize(size_t((p ? aw / 2 : aw) / 4) * ((p ? ah / 2 : ah) / 4));
        s->ctxDc[p].resize(s->ctxLevel[p].size());
    }
    s->ecBuf.resize(size_t(aw) * ah * 3 / 2 + 65536);
    s->out.resize(s->ecBuf.size());
    pft2::InterCdfs &d = s->defaults;
    defaultInterCdfs(d);
    return 0;
}

int HostEntropy::code(const EntropyIn &in, const BlockInfo *grid, const int16_t *levels, EntropyOut &out, char *err,
                      int errLen) {
    auto *s = static_cast<HostState *>(state);
    const int miCols = 2 * ((w + 7) >> 3), miRows = 2 * ((h + 7) >> 3);
    const int sbCols = (miCols + 15) >> 4, sbRows = (miRows + 15) >> 4;
    TileJob j{};
    setTileLayout(j, miCols, miRows, in.tileColsLog2, in.tileRowsLog2);
    const int nTiles = j.tileCols * j.tileRows;
    pft2::FrameSyntax fs{};
    fs.refName[0] = in.refName[0];
    fs.refName[1] = in.refName[1];
    fs.refName[2] = in.refName[2];
    fs.referenceSelect = in.referenceSelect;
    fs.txModeSelect = in.txModeSelect;
    fs.reducedTxSet = in.reducedTxSet;
    fs.switchable = in.switchableFilter;
    fs.skipMode = in.skipMode;
    fs.filterIntra = in.filterIntra;
    fs.cdefBits = in.cdefBits;
    fs.cdefIdx = in.cdefIdx;
    fs.lrLuma = in.lrLuma; fs.width = w; fs.height = h;
    for (int k = 0; k < 8; k++) fs.signBias[k] = in.signBias[k];
    fs.useTmvp = in.useTmvp && in.motionField;
    fs.miRows = miRows; fs.miCols = miCols; fs.aw = aw; fs.ah = ah; fs.gcols = aw / 8;
    // costs and cdf ids from the default CDFs (as on the GPU, so both paths decide alike)
    pft2::EntropyState &init = s->init;
    av1g_cdfs_init(&init.cdf, in.qindex);
    init.inter = s->defaults;
    for (int p = 0; p < 3; p++) {
        std::fill(s->ctxLevel[p].begin(), s->ctxLevel[p].end(), 0);
        std::fill(s->ctxDc[p].begin(), s->ctxDc[p].end(), 0);
    }
    const int nSb = sbCols * sbRows;
    j.fs = fs;
    j.grid = grid;
    j.mf = in.motionField;
    j.levels = levels;
    j.mi = s->mi.data();
    j.frameInit = &init;
    for (int p = 0; p < 3; p++) {
        j.ctxLevel[p] = s->ctxLevel[p].data();
        j.ctxDc[p] = s->ctxDc[p].data();
        j.ctxStride[p] = (p ? aw / 2 : aw) / 4;
    }
    s->entries.resize(size_t(nSb) * 64);
    std::vector<uint32_t> entryCount(nSb), count(size_t(nSb) * 64 + 1, 0), offset(size_t(nSb) * 64 + 1, 0);
    out.compound = 0;
    for (int k = 0; k < 4; k++) { out.bits[k] = 0; out.modes[k] = 0; }
    int error = 0;
    for (int sb = 0; sb < nSb && !error; sb++) {
        pft2::TileCoder t;
        setupCoder(t, j, sb, s->qv);
        t.fs = &fs;
        t.phase = 1;
        t.entries = s->entries.data() + size_t(sb) * 64;
        t.superblock((sb / sbCols) * 16, (sb % sbCols) * 16, (uint32_t)sb);
        error = t.st.error;
        entryCount[sb] = t.nEntries;
    }
    auto blocks = [&](int phase, bool write) {
        for (int sb = 0; sb < nSb && !error; sb++)
            for (uint32_t k = 0; k < entryCount[sb] && !error; k++) {
                const size_t e = size_t(sb) * 64 + k;
                pft2::TileCoder t;
                setupCoder(t, j, sb, s->qv);
                t.fs = &fs;
                t.phase = phase;
                t.recs = write ? s->recs.data() + offset[e] : nullptr;
                t.sbList = s->entries.data() + size_t(sb) * 64;
                t.runEntry(s->entries[e]);
                error = t.st.error;
                if (phase == 2) {
                    for (int q = 0; q < 4; q++) out.modes[q] += t.st.modeCount[q];
                    out.compound += t.st.compound;
                }
                if (phase == 3 && !write) count[e] = t.nrec;
            }
    };
    blocks(4, false);
    blocks(2, false);
    blocks(3, false);
    if (!error) {
        for (size_t e = 0; e < size_t(nSb) * 64; e++) offset[e + 1] = offset[e] + count[e];
        if (s->recs.size() < offset[size_t(nSb) * 64]) s->recs.resize(offset[size_t(nSb) * 64]);
        blocks(3, true);
    }
    if (error) {
        snprintf(err, errLen, "inter tile syntax failed (error %d)", error);
        return -1;
    }
    // pack the records tile after tile in decode order and range-code them
    std::vector<uint32_t> tileCounts(nTiles, 0);
    std::vector<uint8_t> packed;
    for (int t = 0; t < nTiles; t++) {
        const int tc = t % j.tileCols, tr = t / j.tileCols;
        for (int r = j.rowStart[tr]; r < j.rowStart[tr + 1]; r += 16)
            for (int c = j.colStart[tc]; c < j.colStart[tc + 1]; c += 16) {
                const int sb = (r / 16) * sbCols + c / 16;
                const uint32_t first = offset[size_t(sb) * 64], last = offset[size_t(sb) * 64 + 64];
                const uint8_t *p0 = reinterpret_cast<const uint8_t *>(s->recs.data() + first);
                packed.insert(packed.end(), p0, p0 + (last - first) * sizeof(pft2::SymRec));
                tileCounts[t] += last - first;
            }
    }
    return codeRecords(in, tileCounts, packed, out, err, errLen);
}

int HostEntropy::codeRecords(const EntropyIn &in, const std::vector<uint32_t> &tileCounts,
                             const std::vector<uint8_t> &records, EntropyOut &out, char *err, int errLen) {
    auto *s = static_cast<HostState *>(state);
    const int nTiles = int(tileCounts.size());
    const pft2::SymRec *recs = reinterpret_cast<const pft2::SymRec *>(records.data());
    pft2::EntropyState start;
    if (in.primarySlot >= 0) {
        start = s->slots[in.primarySlot];
    } else {
        av1g_cdfs_init(&start.cdf, in.qindex);
        start.inter = s->defaults;
    }
    out.tiles.assign(nTiles, {});
    const int contextTile = std::min(in.contextTile, nTiles - 1);
    size_t at = 0;
    for (int t = 0; t < nTiles; t++) {
        s->work = start;
        uint16_t *base = reinterpret_cast<uint16_t *>(&s->work);
        av1g_ec ec;
        av1g_ec_init(&ec, s->ecBuf.data(), (uint32_t)s->ecBuf.size());
        static const bool stats = getenv("AV1GPU_PF_BITSTATS") != nullptr;
        if (stats) {
            // bits by syntax group (exact adaptive costs): 0 partition / modes / refs, 1 motion vectors, 2 coefficients
            // (txb_skip, eob, levels, signs, Golomb literals), 3 other literals / transform types
            const size_t interOff = offsetof(pft2::EntropyState, inter) / 2;
            const size_t mvLo = interOff + offsetof(pft2::InterCdfs, mvJoint) / 2, mvHi = interOff + offsetof(pft2::InterCdfs, txSet3) / 2;
            const size_t txLo = mvHi, txHi = interOff + offsetof(pft2::InterCdfs, compMode) / 2;
            const size_t partHi = offsetof(av1g_cdfs, txb_skip) / 2;
            for (uint32_t k = 0; k < tileCounts[t]; k++) {
                const pft2::SymRec r = recs[at + k];
                const int type = r.kind >> 5, n = r.kind & 31;
                float b;
                int cat;
                if (type == 0) {
                    const uint16_t *c = base + r.cdf;
                    b = -std::log2(std::max(1, c[r.sym] - (r.sym ? c[r.sym - 1] : 0)) / 32768.0f);
                    cat = r.cdf >= interOff ? (r.cdf >= mvLo && r.cdf < mvHi ? 1 : r.cdf >= txLo && r.cdf < txHi ? 3 : 0)
                                            : (r.cdf < partHi ? 0 : 2);
                } else if (type == 3) { b = (float)n; cat = 2; }
                else { b = 1.f; cat = type == 2 ? 0 : 2; }
                g_bitStats[cat] += b;
                pft2::codeRecord(&ec, base, r);
            }
        } else
        for (uint32_t k = 0; k < tileCounts[t]; k++) pft2::codeRecord(&ec, base, recs[at + k]);
        at += tileCounts[t];
        const int n = av1g_ec_done(&ec, s->out.data());
        if (n < 0) {
            snprintf(err, errLen, "tile %d overflow", t);
            return -1;
        }
        out.tiles[t].assign(s->out.begin(), s->out.begin() + n);
        if (t == contextTile && in.saveSlot >= 0) {
            pft2::clearCounters(base, (int)(sizeof(s->work) / 2));
            s->slots[in.saveSlot] = s->work;
        }
    }
    return 0;
}

int Analyzer::temporalFilter(const uint8_t *const center[3], const uint8_t *const *const neighbors[3], int count,
                             int qindex, float strength, uint8_t *const out[3], char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    count = std::min(count, kTfMax);
    const dim3 t16(16, 16);
    const dim3 gds((s->dw + 15) / 16, (s->dh + 15) / 16);
    downsample4<<<gds, t16, 0, s->compute>>>(center[0], aw, w, h, s->tfDs[0], s->dw, s->dh);
    const int nCoarse = s->ccols * s->crows;
    TfArgs a{};
    for (int j = 0; j < count; j++) {
        downsample4<<<gds, t16, 0, s->compute>>>(neighbors[0][j], aw, w, h, s->tfDs[1 + j], s->dw, s->dh);
        hierCoarse(s, s->compute, s->tfDs[0], s->tfDs[1 + j], j == 0, 12, 2.f, s->tfCoarse[j]);
        for (int p = 0; p < 3; p++) a.nb[j][p] = neighbors[p][j];
        a.coarse[j] = s->tfCoarse[j];
    }
    for (int p = 0; p < 3; p++) { a.center[p] = center[p]; a.out[p] = out[p]; }
    a.count = count; a.w = w; a.h = h; a.aw = aw; a.ccols = s->ccols; a.crows = s->crows;
    cudaMemsetAsync(s->tfNoise, 0, 2 * sizeof(unsigned long long), s->compute);
    estimateNoise<<<dim3((w + 29) / 32, (h + 5) / 8), dim3(32, 8), 0, s->compute>>>(center[0], aw, w, h, s->tfNoise);
    a.noise = s->tfNoise;
    // libaom: q factor = ac quantizer / 4 of the inter frames; full strength from 20, weaker below
    const float qf = av1g_ac_q(qindex) / 4.f;
    a.qDecay = qf >= 128.f ? 0.5f * (qf / 64.f) * (qf / 64.f) : std::min(1.f, std::max(1e-5f, (qf / 20.f) * (qf / 20.f)));
    a.strength = strength;
    temporalFilterKernel<<<dim3((w + 15) / 16, (h + 15) / 16), 256, 0, s->compute>>>(a);
    for (int p = 0; p < 3; p++) {
        const int pw = p ? w / 2 : w, phh = p ? h / 2 : h, pitch = p ? aw / 2 : aw, pah = p ? ah / 2 : ah;
        padFrame<<<dim3((pitch + 31) / 32, (pah + 7) / 8), dim3(32, 8), 0, s->compute>>>(out[p], pitch, pw, phh, pah);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        snprintf(err, errLen, "temporal filter: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

double Analyzer::noiseSigma(const uint8_t *luma, int pitch, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    unsigned long long host[2] = {0, 0};
    cudaError_t e = cudaMemsetAsync(s->tfNoise, 0, 2 * sizeof(unsigned long long), s->compute);
    estimateNoise<<<dim3((w + 29) / 32, (h + 5) / 8), dim3(32, 8), 0, s->compute>>>(luma, pitch, w, h, s->tfNoise);
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaMemcpyAsync(host, s->tfNoise, sizeof host, cudaMemcpyDeviceToHost, s->compute);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s->compute);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "noise estimate: %s", cudaGetErrorString(e));
        return -2.0;
    }
    return host[1] >= 16 ? double(host[0]) / (6.0 * double(host[1])) * 1.25331413731550025 : -1.0;
}

void Analyzer::resetTemporal() {
    if (state) static_cast<State *>(state)->havePrev = false;
}


// ------------------------------------------------------------------------------------------------
// Loop restoration: luma Wiener filter in 64-sample units (spec 7.17). The unit of a sample uses the stripe offset:
// unitRow = Min(unitRows - 1, (y + 8) / 64). Samples outside the 64-row stripe of the sample being filtered come from
// the deblocked frame (two rows at most), inside it from the CDEF output.
struct LrArgs {
    const uint8_t *dbk, *cdef;   // luma planes (w x h, pitch w)
    const uint8_t *src;
    int srcPitch;
    int w, h, unitRows, unitCols;
    float *coef;                 // per unit: [pass][3] filter taps / 128 (pass 0 vertical, 1 horizontal)
    int8_t *q;                   // per unit: quantized taps [2][3]
    unsigned long long *sse;     // per unit: [0] unfiltered, [1] filtered
    uint8_t *out;                // filtered luma
};
__device__ __forceinline__ int lrSample(const LrArgs &a, int x, int y, int stripeStart, int stripeEnd) {
    x = min(max(x, 0), a.w - 1);
    y = min(max(y, 0), a.h - 1);
    if (y < stripeStart) return a.dbk[max(stripeStart - 2, y) * a.w + x];
    if (y > stripeEnd) return a.dbk[min(stripeEnd + 2, y) * a.w + x];
    return a.cdef[y * a.w + x];
}
__device__ __forceinline__ int lrUnitOf(const LrArgs &a, int x, int y, int *ur, int *uc) {
    *ur = min(a.unitRows - 1, (y + 8) / 64);
    *uc = min(a.unitCols - 1, x / 64);
    return *ur * a.unitCols + *uc;
}
// Normal equations of one pass for one unit (one CTA per unit, grid-stride over the frame rows of the unit): solve
// the free taps (a, b, c) of the pass `dir` (0 vertical, 1 horizontal) with the other pass fixed, then store them.
__global__ void __launch_bounds__(256) lrSolve(LrArgs a, int dir) {
    const int u = blockIdx.x, ur = u / a.unitCols, uc = u % a.unitCols;
    const int y0 = ur == 0 ? 0 : ur * 64 - 8, y1 = ur == a.unitRows - 1 ? a.h : ur * 64 + 56;
    const int x0 = uc * 64, x1 = uc == a.unitCols - 1 ? a.w : uc * 64 + 64;
    float fixed[7];
    {
        const float *c = a.coef + u * 6 + (dir ? 0 : 3);   // the other pass
        fixed[0] = fixed[6] = c[0]; fixed[1] = fixed[5] = c[1]; fixed[2] = fixed[4] = c[2];
        fixed[3] = 1.f - 2.f * (c[0] + c[1] + c[2]);
    }
    float m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, v[3] = {0, 0, 0};
    const int W = x1 - x0, n = W * (y1 - y0);
    // a unit lies in one 64-row stripe: its samples (3 of margin) once in shared memory
    constexpr int kTw = 96 + 6, kTh = 104 + 6;   // largest unit: 95 x 103
    __shared__ uint8_t tile[kTh * kTw];
    const int ss0 = -8 + ((y0 + 8) / 64) * 64, se0 = ss0 + 63;
    const bool tiled = y1 - 1 <= se0;   // the bottom unit row may reach into the next stripe: per-sample stripes then
    if (tiled) {
        const int tw = W + 6, th = y1 - y0 + 6;
        for (int i = threadIdx.x; i < tw * th; i += blockDim.x)
            tile[(i / tw) * kTw + i % tw] = (uint8_t)lrSample(a, x0 + i % tw - 3, y0 + i / tw - 3, ss0, se0);
        __syncthreads();
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int xi = i % W, yi = i / W, x = x0 + xi, y = y0 + yi;
        float t[7];
        if (tiled) {
            const uint8_t *c0 = tile + (yi + 3) * kTw + xi + 3;
            for (int k = 0; k < 7; k++) {   // filtered along the fixed direction, at offset k - 3 along the solved one
                float s = 0.f;
                for (int j = 0; j < 7; j++)
                    s += fixed[j] * (float)(dir ? c0[(j - 3) * kTw + k - 3] : c0[(k - 3) * kTw + j - 3]);
                t[k] = s;
            }
        } else {
            const int sn = (y + 8) / 64, ss = -8 + sn * 64, se = ss + 63;
            for (int k = 0; k < 7; k++) {
                float s = 0.f;
                for (int j = 0; j < 7; j++)
                    s += fixed[j] * (float)(dir ? lrSample(a, x + k - 3, y + j - 3, ss, se) : lrSample(a, x + j - 3, y + k - 3, ss, se));
                t[k] = s;
            }
        }
        const float f[3] = {t[0] + t[6] - 2.f * t[3], t[1] + t[5] - 2.f * t[3], t[2] + t[4] - 2.f * t[3]};
        const float r = (float)a.src[y * a.srcPitch + x] - t[3];
        for (int p = 0; p < 3; p++) {
            v[p] += f[p] * r;
            for (int q = 0; q < 3; q++) m[p * 3 + q] += f[p] * f[q];
        }
    }
    __shared__ float red[12][8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int k = 0; k < 12; k++) {
        float x = k < 9 ? m[k] : v[k - 9];
        for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
        if (!lane) red[k][warp] = x;
    }
    __syncthreads();
    if (threadIdx.x) return;
    double M[9], V[3];
    for (int k = 0; k < 12; k++) {
        double x = 0;
        for (int w = 0; w < 8; w++) x += red[k][w];
        if (k < 9) M[k] = x; else V[k - 9] = x;
    }
    const double reg = 1e-3 * (M[0] + M[4] + M[8]) + 1e-6;
    M[0] += reg; M[4] += reg; M[8] += reg;
    const double det = M[0] * (M[4] * M[8] - M[5] * M[7]) - M[1] * (M[3] * M[8] - M[5] * M[6]) + M[2] * (M[3] * M[7] - M[4] * M[6]);
    float *out = a.coef + u * 6 + (dir ? 3 : 0);
    if (fabs(det) < 1e-12) { out[0] = out[1] = out[2] = 0.f; return; }
    double x[3];
    for (int c = 0; c < 3; c++) {   // Cramer
        double Mc[9];
        for (int k = 0; k < 9; k++) Mc[k] = (k % 3 == c) ? V[k / 3] : M[k];
        x[c] = (Mc[0] * (Mc[4] * Mc[8] - Mc[5] * Mc[7]) - Mc[1] * (Mc[3] * Mc[8] - Mc[5] * Mc[6]) + Mc[2] * (Mc[3] * Mc[7] - Mc[4] * Mc[6])) / det;
    }
    for (int c = 0; c < 3; c++) out[c] = (float)x[c];
}
// Quantize the taps to the coded ranges (Wiener_Taps_Min / Max) and clear the SSE accumulators.
__global__ void lrQuantize(LrArgs a) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= a.unitRows * a.unitCols) return;
    const int lo[3] = {-5, -23, -17}, hi[3] = {10, 8, 46};
    for (int p = 0; p < 2; p++)
        for (int j = 0; j < 3; j++)
            a.q[u * 6 + p * 3 + j] = (int8_t)min(max((int)lrintf(a.coef[u * 6 + p * 3 + j] * 128.f), lo[j]), hi[j]);
    a.sse[2 * u] = a.sse[2 * u + 1] = 0;
}
// Exact Wiener filter (spec 7.17.4, 8-bit: InterRound0 3, InterRound1 11) of every luma sample with its unit's
// quantized taps, and the per-unit SSE with and without it. One CTA per 32 x 8 tile: tiles never straddle a 64-row
// stripe (stripes start at -8 + 64 n) or a unit, so the tile shares one set of taps and one stripe; the samples go to
// shared memory once, then a horizontal and a vertical pass, and one SSE atomic per CTA.
constexpr int kLrTw = 32, kLrTh = 8;
__global__ void __launch_bounds__(kLrTw * kLrTh) lrApply(LrArgs a) {
    __shared__ uint8_t smp[kLrTh + 6][kLrTw + 6];
    __shared__ int iv[kLrTh + 6][kLrTw];
    __shared__ unsigned long long red[2][kLrTw * kLrTh / 32];
    const int x0 = blockIdx.x * kLrTw, y0 = blockIdx.y * kLrTh, tid = threadIdx.y * kLrTw + threadIdx.x;
    int ur, uc;
    const int u = lrUnitOf(a, x0, y0, &ur, &uc);
    const int sn = (y0 + 8) / 64, ss = -8 + sn * 64, se = ss + 63;
    for (int i = tid; i < (kLrTh + 6) * (kLrTw + 6); i += kLrTw * kLrTh) {
        const int r = i / (kLrTw + 6), c = i % (kLrTw + 6);
        smp[r][c] = (uint8_t)lrSample(a, x0 + c - 3, y0 + r - 3, ss, se);
    }
    const int8_t *q = a.q + u * 6;
    const int vf[7] = {q[0], q[1], q[2], 128 - 2 * (q[0] + q[1] + q[2]), q[2], q[1], q[0]};
    const int hf[7] = {q[3], q[4], q[5], 128 - 2 * (q[3] + q[4] + q[5]), q[5], q[4], q[3]};
    __syncthreads();
    for (int i = tid; i < (kLrTh + 6) * kLrTw; i += kLrTw * kLrTh) {
        const int r = i / kLrTw, c = i % kLrTw;
        int hs = 0;
#pragma unroll
        for (int t = 0; t < 7; t++) hs += hf[t] * smp[r][c + t];
        iv[r][c] = min(max((hs + 4) >> 3, -2048), 6143);   // Round2(s, 3), Clip3(-offset, limit - offset)
    }
    __syncthreads();
    const int x = x0 + threadIdx.x, y = y0 + threadIdx.y;
    unsigned long long e0 = 0, e1 = 0;
    if (x < a.w && y < a.h) {
        int sum = 0;
#pragma unroll
        for (int r = 0; r < 7; r++) sum += vf[r] * iv[threadIdx.y + r][threadIdx.x];
        const int v = min(max((sum + 1024) >> 11, 0), 255);
        a.out[y * a.w + x] = (uint8_t)v;
        const int src = a.src[y * a.srcPitch + x], cd = a.cdef[y * a.w + x];
        e0 = (unsigned long long)((src - cd) * (src - cd));
        e1 = (unsigned long long)((src - v) * (src - v));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) { e0 += __shfl_xor_sync(0xffffffffu, e0, o); e1 += __shfl_xor_sync(0xffffffffu, e1, o); }
    if ((tid & 31) == 0) { red[0][tid >> 5] = e0; red[1][tid >> 5] = e1; }
    __syncthreads();
    if (!tid) {
        unsigned long long s0 = 0, s1 = 0;
        for (int k = 0; k < kLrTw * kLrTh / 32; k++) { s0 += red[0][k]; s1 += red[1][k]; }
        atomicAdd(&a.sse[2 * u], s0);
        atomicAdd(&a.sse[2 * u + 1], s1);
    }
}
// Per unit: keep the filter when its SSE saving pays for its bits; write the decision into the grid (the unit's
// top-left 8x8 entry) and the chosen samples into dst.
__global__ void lrDecide(LrArgs a, float bitCost, BlockInfo *grid, int gcols, uint8_t *dst, FilterSelection *selection) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= a.w || y >= a.h) return;
    int ur, uc;
    const int u = lrUnitOf(a, x, y, &ur, &uc);
    const float gain = (float)a.sse[2 * u] - (float)a.sse[2 * u + 1];
    const bool use = gain > bitCost;
    dst[y * a.w + x] = use ? a.out[y * a.w + x] : a.cdef[y * a.w + x];
    if (x == uc * 64 && y == ur * 64) {
        if (use) atomicExch(&selection->lrLuma, 1);
        BlockInfo &b = grid[(ur * 8) * gcols + uc * 8];
        b.lrUse = use;
        for (int k = 0; k < 6; k++) b.lrCoef[k / 3][k % 3] = use ? a.q[u * 6 + k] : 0;
    }
}

// Both filters evaluate every candidate from the same input. Deblocking is plane-local and CDEF chroma
// uses the same luma directions for all strengths, so choosing each plane from the saved pixels is exact.
template<bool Cdef>
__global__ void selectFilter(const unsigned long long *sse, int count, bool zeroLevel, FilterSelection *selection) {
    int *indices = Cdef ? selection->cdef : selection->lf;
    for (int p = 0; p < (Cdef ? 2 : 3); ++p) {
        unsigned long long best = ~0ull;
        for (int i = 0; i < count; ++i) {
            auto error = sse[3 * i + p];
            if (Cdef && p == 1) error += sse[3 * i + 2];
            if (error < best) {
                best = error;
                indices[p] = i;
            }
        }
    }
    if (Cdef) indices[2] = indices[1];
    // AV1 signals chroma deblocking levels only when at least one luma level is nonzero.
    if (!Cdef && zeroLevel && indices[0] == 0) indices[1] = indices[2] = 0;
}

// Per-superblock CDEF: SSE of every candidate strength on each coded 64x64 superblock, luma and chroma (both planes).
__global__ void cdefSbSse(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, int p0, int p1, int p2,
                          const uint8_t *cands, size_t frame, int w, int h, int sbCols, int nSb, const int8_t *cdefIdx,
                          unsigned long long *out) {
    __shared__ unsigned long long red[2][8];
    const int sb = blockIdx.x, k = blockIdx.y, t = threadIdx.x;
    unsigned long long *o = out + (size_t(k) * nSb + sb) * 2;
    if (cdefIdx[sb] < 0) { if (!t) o[0] = o[1] = 0; return; }
    const int sx = (sb % sbCols) * 64, sy = (sb / sbCols) * 64, cw = w / 2, ch = h / 2;
    const uint8_t *y = cands + k * frame, *u = y + size_t(w) * h, *v = u + size_t(cw) * ch;
    unsigned long long ly = 0, lc = 0;
    for (int i = t; i < 64 * 64; i += 256) {
        const int x = sx + (i & 63), yy = sy + (i >> 6);
        if (x < w && yy < h) { const int d = (int)s0[yy * p0 + x] - (int)y[yy * w + x]; ly += d * d; }
    }
    for (int i = t; i < 32 * 32; i += 256) {
        const int x = sx / 2 + (i & 31), yy = sy / 2 + (i >> 5);
        if (x < cw && yy < ch) {
            const int du = (int)s1[yy * p1 + x] - (int)u[yy * cw + x], dv = (int)s2[yy * p2 + x] - (int)v[yy * cw + x];
            lc += du * du + dv * dv;
        }
    }
    for (int off = 16; off; off >>= 1) { ly += __shfl_xor_sync(0xffffffffu, ly, off); lc += __shfl_xor_sync(0xffffffffu, lc, off); }
    if ((t & 31) == 0) { red[0][t >> 5] = ly; red[1][t >> 5] = lc; }
    __syncthreads();
    if (!t) {
        unsigned long long a = 0, b = 0;
        for (int i = 0; i < 8; i++) { a += red[0][i]; b += red[1][i]; }
        o[0] = a; o[1] = b;
    }
}
// Greedy choice of up to 8 presets (luma candidate, chroma candidate) over the coded superblocks: each step adds the
// pair that lowers the summed best-preset SSE most; 1, 2, 4 or 8 presets are kept by SSE + lambda * index bits; then
// each coded superblock takes its best preset. Exact integer sums (deterministic). Per step: one CTA per pair sums its
// SSE over the superblocks (cdefPairTotals), then one CTA picks the pair and updates the per-superblock minima.
__device__ __forceinline__ unsigned long long cdefPairErr(const unsigned long long *err, int nSb, int K, int pair, int sb) {
    return err[(size_t(pair / K) * nSb + sb) * 2] + err[(size_t(pair % K) * nSb + sb) * 2 + 1];
}
__global__ void cdefChooseInit(int nSb, const int8_t *cdefIdx, unsigned long long *cur, CdefChoice *ch) {
    __shared__ int coded;
    if (!threadIdx.x) coded = 0;
    __syncthreads();
    int my = 0;
    for (int sb = threadIdx.x; sb < nSb; sb += blockDim.x) { cur[sb] = ~0ull; my += cdefIdx[sb] >= 0; }
    atomicAdd(&coded, my);
    __syncthreads();
    if (!threadIdx.x) ch->nCoded = coded;
}
__global__ void cdefPairTotals(const unsigned long long *err, int nSb, int K, const int8_t *cdefIdx,
                               const unsigned long long *cur, CdefChoice *ch) {
    __shared__ unsigned long long red[8];
    const int p = blockIdx.x, t = threadIdx.x;
    unsigned long long sum = 0;
    for (int sb = t; sb < nSb; sb += 256)
        if (cdefIdx[sb] >= 0) { const unsigned long long v = cdefPairErr(err, nSb, K, p, sb); sum += v < cur[sb] ? v : cur[sb]; }
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if ((t & 31) == 0) red[t >> 5] = sum;
    __syncthreads();
    if (!t) { unsigned long long a = 0; for (int i = 0; i < 8; i++) a += red[i]; ch->totals[p] = a; }
}
__global__ void cdefPick(const unsigned long long *err, int nSb, int K, const int8_t *cdefIdx, unsigned long long *cur,
                         CdefChoice *ch, int step) {
    __shared__ int pick;
    if (!threadIdx.x) {
        int best = 0;
        for (int p = 1; p < K * K; p++) if (ch->totals[p] < ch->totals[best]) best = p;
        pick = best;
        ch->chosen[step] = best;
        ch->totalAfter[step + 1] = ch->totals[best];
    }
    __syncthreads();
    for (int sb = threadIdx.x; sb < nSb; sb += blockDim.x)
        if (cdefIdx[sb] >= 0) { const unsigned long long v = cdefPairErr(err, nSb, K, pick, sb); if (v < cur[sb]) cur[sb] = v; }
}
__global__ void cdefFinish(const unsigned long long *err, int nSb, int K, int8_t *cdefIdx, float lambda,
                           const CdefChoice *ch, FilterSelection *sel) {
    __shared__ int nKeep;
    if (!threadIdx.x) {
        int best = 1;
        double bestCost = 1e300;
        for (int b = 0; b <= 3; b++) {
            const int n = 1 << b;
            const double c = (double)ch->totalAfter[n] + (double)lambda * b * ch->nCoded;
            if (c < bestCost) { bestCost = c; best = n; }
        }
        nKeep = best;
        sel->cdefBits = best == 1 ? 0 : best == 2 ? 1 : best == 4 ? 2 : 3;
        for (int i = 0; i < 8; i++) { sel->cdefPreset[i][0] = ch->chosen[i] / K; sel->cdefPreset[i][1] = ch->chosen[i] % K; }
    }
    __syncthreads();
    for (int sb = threadIdx.x; sb < nSb; sb += blockDim.x) {
        if (cdefIdx[sb] < 0) continue;
        int bi = 0;
        unsigned long long bv = cdefPairErr(err, nSb, K, ch->chosen[0], sb);
        for (int i = 1; i < nKeep; i++) { const unsigned long long v = cdefPairErr(err, nSb, K, ch->chosen[i], sb); if (v < bv) { bv = v; bi = i; } }
        cdefIdx[sb] = (int8_t)bi;
    }
}
// Per-superblock gather of the chosen candidate (CDEF reads only the deblocked frame, so this equals filtering each
// superblock with its preset); superblocks without coded blocks keep candidate 0 (no filtering).
__global__ void gatherCdef(const uint8_t *cands, size_t frame, int w, int h, int sbCols, const int8_t *cdefIdx,
                           const FilterSelection *sel, uint8_t *recon, uint8_t *lumaOut) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= frame) return;
    const size_t ysz = size_t(w) * h, csz = size_t(w / 2) * (h / 2);
    int sb, p;
    if (i < ysz) { p = 0; const int y = int(i / w), x = int(i % w); sb = (y >> 6) * sbCols + (x >> 6); }
    else { p = i < ysz + csz ? 1 : 2; const size_t j = i - ysz - (p - 1) * csz; const int y = int(j / (w / 2)), x = int(j % (w / 2)); sb = (y >> 5) * sbCols + (x >> 5); }
    const int idx = cdefIdx[sb];
    const int cand = idx < 0 ? 0 : sel->cdefPreset[idx][p > 0];
    (p == 0 ? lumaOut : recon)[i] = cands[frame * cand + i];
}
template<bool Cdef>
__global__ void gatherFilter(const uint32_t *candidates, size_t luma, size_t chroma, uint32_t *recon,
                             uint32_t *lumaOut, const FilterSelection *selection) {
    const size_t frame = luma + 2 * chroma;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= frame) return;
    const int p = i < luma ? 0 : i < luma + chroma ? 1 : 2;
    const int candidate = Cdef ? selection->cdef[p] : selection->lf[p];
    uint32_t *dst = p == 0 ? lumaOut : recon;
    dst[i] = candidates[frame * candidate + i];
}

int Analyzer::filter(const uint8_t *const src[3], const int pitch[3], uint8_t *recon, int qindex, bool cdefAllowed,
                     FilterParams &out, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    const cudaStream_t stream = s->compute;
    av1g_frame_params fp{};
    fp.width = w; fp.height = h; fp.chroma_format = AV1G_CHROMA_420; fp.bit_depth = 8;
    fp.enable_cdef = cdefAllowed;
    av1g_pf_frame f;
    av1g_gpu_pf_attach(&fp, s->pfScratch, &f);
    const dim3 t16(16, 16);
    cudaError_t e = cudaSuccess;
    if (s->profile) e = cudaEventRecord(s->ev[8], stream);
    if (e == cudaSuccess) e = cudaMemsetAsync(s->filterDev, 0, sizeof(FilterSelection), stream);
    buildFilterMaps<<<dim3((f.MiCols + 15) / 16, (f.MiRows + 15) / 16), t16, 0, stream>>>(s->grid, s->gcols, f);
    buildCdefIdx<<<(f.sbCols * f.sbRows + 63) / 64, 64, 0, stream>>>(s->grid, s->gcols, s->grows, f);
    const int cw = w / 2, ch = h / 2;
    const size_t ysz = size_t(w) * h, csz = size_t(cw) * ch, frame = ysz + 2 * csz;
    const int pw[3] = {w, cw, cw}, ph[3] = {h, ch, ch};
    auto sseOf = [&](uint8_t *const planes[3], int slot) {
        for (int p = 0; p < 3; p++)
            planeSse<<<dim3((pw[p] + 127) / 128, (ph[p] + 7) / 8), dim3(32, 8), 0, stream>>>(
                src[p], pitch[p], planes[p], pw[p], ph[p], s->sse + 3 * slot + p);
        return cudaGetLastError();
    };
    auto *candidates = reinterpret_cast<const uint32_t *>(s->filterCandidates);
    auto *dstWords = reinterpret_cast<uint32_t *>(recon);
    const unsigned blocks = (frame / sizeof(uint32_t) + 255) / 256;

    // Keep the candidate order and strict SSE comparisons so tied candidates keep the first level.
    const int acq = av1g_ac_q(qindex);
    const int guess = std::max(0, std::min(63, (int)(((long)acq * 20723 + 1015158 + (1 << 17)) >> 18)));
    constexpr int deltas[kDeblockCandidates] = {-12, -6, 0, 6, 12};
    int levels[kDeblockCandidates], n = 0;
    if (e == cudaSuccess) e = cudaMemsetAsync(s->sse, 0, sizeof(unsigned long long) * 3 * kDeblockCandidates, stream);
    for (int d : deltas) {
        if (e != cudaSuccess) break;
        const int L = std::max(0, std::min(63, guess + d));
        if (d && L == std::max(0, std::min(63, guess + d - (d > 0 ? 6 : -6)))) continue;
        uint8_t *work = s->filterCandidates + n * frame;
        e = cudaMemcpyAsync(work, recon, frame, cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) break;
        f.plane[0] = work; f.plane[1] = work + ysz; f.plane[2] = work + ysz + csz;
        for (int p = 0; p < 3; p++) f.stride[p] = pw[p];
        for (int k = 0; k < 4; k++) f.lfLevel[k] = L;
        e = (cudaError_t)av1g_gpu_deblock(&f, stream);
        if (e == cudaSuccess) e = sseOf(f.plane, n);
        levels[n++] = L;
    }
    if (e == cudaSuccess) {
        selectFilter<false><<<1, 1, 0, stream>>>(s->sse, n, levels[0] == 0, s->filterDev);
        gatherFilter<false><<<blocks, 256, 0, stream>>>(candidates, ysz / 4, csz / 4, dstWords, dstWords, s->filterDev);
        e = cudaGetLastError();
    }

    out.cdef = cdefAllowed;
    if (cdefAllowed && e == cudaSuccess) {
        out.damping = 3 + std::min(3, qindex >> 6);
        f.plane[0] = recon; f.plane[1] = recon + ysz; f.plane[2] = recon + ysz + csz;
        f.cdefDamping = out.damping;
        e = (cudaError_t)av1g_gpu_cdef_dir(&f, stream);
        if (e == cudaSuccess) e = cudaMemsetAsync(s->sse, 0, sizeof(unsigned long long) * 3 * kCdefCount, stream);
        if (out.cdefPerSb && e == cudaSuccess) {
            // per-superblock choice: every candidate in one launch (candidate 0 also copies the superblocks without CDEF)
            int pri[kCdefCount], sec[kCdefCount];
            for (int k = 0; k < kCdefCount; k++) { pri[k] = kCdefCandidates[k].pri; sec[k] = kCdefCandidates[k].sec; }
            uint8_t *dst[3] = {s->filterCandidates, s->filterCandidates + ysz, s->filterCandidates + ysz + csz};
            e = (cudaError_t)av1g_gpu_cdef_search(&f, dst, pw, frame, kCdefCount, pri, sec, stream);
        }
        for (int k = 0; k < kCdefCount && e == cudaSuccess && !out.cdefPerSb; k++) {
            f.cdefYPri[0] = f.cdefUVPri[0] = kCdefCandidates[k].pri;
            f.cdefYSec[0] = f.cdefUVSec[0] = kCdefCandidates[k].sec;
            auto *candidate = s->filterCandidates + k * frame;
            uint8_t *dst[3] = {candidate, candidate + ysz, candidate + ysz + csz};
            // per-superblock choice: candidates after the first skip the superblocks without CDEF (the gather takes
            // them from candidate 0) and their whole-frame SSEs (the choice uses the per-superblock ones)
            if (out.cdefPerSb && k > 0) {
                e = (cudaError_t)av1g_gpu_cdef_coded(&f, dst, pw, stream);
            } else {
                e = (cudaError_t)av1g_gpu_cdef(&f, dst, pw, stream);
                if (e == cudaSuccess && !out.cdefPerSb) e = sseOf(dst, k);
            }
        }
        if (e == cudaSuccess && out.cdefPerSb) {
            const int nSb = f.sbCols * f.sbRows;
            cdefSbSse<<<dim3(nSb, kCdefCount), 256, 0, stream>>>(src[0], src[1], src[2], pitch[0], pitch[1], pitch[2],
                                                               s->filterCandidates, frame, w, h, f.sbCols, nSb, f.cdefIdx,
                                                               s->cdefSbErr);
            cdefChooseInit<<<1, 256, 0, stream>>>(nSb, f.cdefIdx, s->cdefCur, s->cdefChoice);
            for (int step = 0; step < 8; step++) {
                cdefPairTotals<<<kCdefCount * kCdefCount, 256, 0, stream>>>(s->cdefSbErr, nSb, kCdefCount, f.cdefIdx,
                                                                            s->cdefCur, s->cdefChoice);
                cdefPick<<<1, 256, 0, stream>>>(s->cdefSbErr, nSb, kCdefCount, f.cdefIdx, s->cdefCur, s->cdefChoice, step);
            }
            cdefFinish<<<1, 256, 0, stream>>>(s->cdefSbErr, nSb, kCdefCount, f.cdefIdx, gp_lambda<8>(qindex, 0.7f),
                                             s->cdefChoice, s->filterDev);
            uint8_t *lumaOut = out.lrAllowed ? s->cdefOut : recon;
            gatherCdef<<<unsigned((frame + 255) / 256), 256, 0, stream>>>(s->filterCandidates, frame, w, h, f.sbCols,
                                                                          f.cdefIdx, s->filterDev, recon, lumaOut);
            s->cdefIdxDev = f.cdefIdx;
            e = cudaGetLastError();
        } else if (e == cudaSuccess) {
            selectFilter<true><<<1, 1, 0, stream>>>(s->sse, kCdefCount, false, s->filterDev);
            auto *lumaOut = out.lrAllowed ? reinterpret_cast<uint32_t *>(s->cdefOut) : dstWords;
            gatherFilter<true><<<blocks, 256, 0, stream>>>(candidates, ysz / 4, csz / 4, dstWords, lumaOut, s->filterDev);
            e = cudaGetLastError();
        }
    }

    if (out.lrAllowed && e == cudaSuccess) {
        // Luma Wiener uses both the deblocked luma in recon and the CDEF luma in cdefOut.
        LrArgs la{};
        la.dbk = recon; la.cdef = cdefAllowed ? s->cdefOut : recon;
        la.src = src[0]; la.srcPitch = pitch[0];
        la.w = w; la.h = h; la.unitRows = s->lrUnitRows; la.unitCols = s->lrUnitCols;
        la.coef = s->lrCoef; la.q = s->lrQ; la.sse = s->lrSse; la.out = s->lrOut;
        const int nu = la.unitRows * la.unitCols;
        const float lrBits = getenv("AV1GPU_PF_LRBITS") ? (float)atof(getenv("AV1GPU_PF_LRBITS")) : 26.f;
        e = cudaMemsetAsync(s->lrCoef, 0, sizeof(float) * 6 * nu, stream);
        if (e == cudaSuccess) {
            lrSolve<<<nu, 256, 0, stream>>>(la, 1);   // horizontal given an identity vertical pass
            lrSolve<<<nu, 256, 0, stream>>>(la, 0);   // vertical given the horizontal pass
            lrSolve<<<nu, 256, 0, stream>>>(la, 1);
            lrQuantize<<<(nu + 127) / 128, 128, 0, stream>>>(la);
            const dim3 g16((w + 15) / 16, (h + 15) / 16);
            lrApply<<<dim3((w + kLrTw - 1) / kLrTw, (h + kLrTh - 1) / kLrTh), dim3(kLrTw, kLrTh), 0, stream>>>(la);
            lrDecide<<<g16, t16, 0, stream>>>(la, gp_lambda<8>(qindex, 0.7f) * lrBits, s->grid, s->gcols, recon,
                                             s->filterDev);
            e = cudaGetLastError();
        }
    }
    if (e == cudaSuccess && s->profile) e = cudaEventRecord(s->ev[9], stream);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(s->filterHost, s->filterDev, sizeof(FilterSelection), cudaMemcpyDeviceToHost, stream);
    const bool perSb = cdefAllowed && out.cdefPerSb;
    if (perSb) out.cdefIdx.resize(size_t(f.sbCols) * f.sbRows);
    if (e == cudaSuccess && perSb)
        e = cudaMemcpyAsync(out.cdefIdx.data(), f.cdefIdx, out.cdefIdx.size(), cudaMemcpyDeviceToHost, stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame filters: %s", cudaGetErrorString(e));
        return -1;
    }
    s->profileFiltered = s->profile;
    const auto &selection = *s->filterHost;
    out.lf[0] = out.lf[1] = levels[selection.lf[0]];
    out.lf[2] = levels[selection.lf[1]];
    out.lf[3] = levels[selection.lf[2]];
    out.sharpness = 0;
    if (cdefAllowed && perSb) {
        out.cdefBits = selection.cdefBits;
        for (int i = 0; i < 8; i++) {
            out.cdefPreset[i][0] = kCdefCandidates[selection.cdefPreset[i][0]].pri;
            out.cdefPreset[i][1] = kCdefCandidates[selection.cdefPreset[i][0]].sec;
            out.cdefPreset[i][2] = kCdefCandidates[selection.cdefPreset[i][1]].pri;
            out.cdefPreset[i][3] = kCdefCandidates[selection.cdefPreset[i][1]].sec;
        }
        out.yPri = out.cdefPreset[0][0]; out.ySec = out.cdefPreset[0][1];
        out.uvPri = out.cdefPreset[0][2]; out.uvSec = out.cdefPreset[0][3];
    } else if (cdefAllowed) {
        out.cdefBits = 0;
        out.yPri = kCdefCandidates[selection.cdef[0]].pri;
        out.ySec = kCdefCandidates[selection.cdef[0]].sec;
        out.uvPri = kCdefCandidates[selection.cdef[1]].pri;
        out.uvSec = kCdefCandidates[selection.cdef[1]].sec;
    }
    out.lrLuma = selection.lrLuma != 0;
    return 0;
}

}  // namespace pf
