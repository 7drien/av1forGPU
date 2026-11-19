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
constexpr int kMaxRefs = 2;
constexpr float kInf = 3.0e38f;
constexpr int kFullPelPasses = 6;

__constant__ int16_t c_filter[6][16][8];

struct Cand {
    Mv mvR[kMaxRefs];     // best vector per reference (search)
    float costR[kMaxRefs], bitsR[kMaxRefs];
    Mv mv, mv2;           // chosen prediction
    uint8_t ref;          // 0, 1: single reference; 2: compound
    uint8_t nz;           // coded option has nonzero levels
    uint8_t txSplit;      // luma of the coded option uses four half-size transforms
    float mvBits;         // estimated bits of the chosen vectors
    float costSkip, costCoded;
};

struct Frame {
    const uint8_t *src[3];
    int pitch[3];
    const uint8_t *ref[kMaxRefs];
    int w, h, aw, ah;
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

// ------------------------------------------------------------------------------------------------
// Exact AV1 single-reference inter prediction (spec 7.11.3.3 / 7.11.3.4, unscaled reference, regular
// eight-tap filter; blocks of width or height 4 use the four-tap variant). One warp predicts a w x h block.
// tmp holds (h + 7) * w intermediate values.
__device__ void warpPredict(const uint8_t *ref, int pw, int ph, int x0, int y0, int w, int h, Mv mv, int sub,
                            int16_t *tmp, uint8_t *out) {
    const int lane = threadIdx.x & 31;
    const int X = (x0 << 4) + ((2 * mv.col) >> sub);
    const int Y = (y0 << 4) + ((2 * mv.row) >> sub);
    const int ix = X >> 4, fx = X & 15, iy = Y >> 4, fy = Y & 15;
    const int fh = w <= 4 ? 4 : 0, fv = h <= 4 ? 4 : 0;
    const int lastX = pw - 1, lastY = ph - 1;
    if (!fx && !fy) {
        for (int i = lane; i < w * h; i += 32) {
            const int r = i / w, c = i - r * w;
            out[i] = ref[min(max(iy + r, 0), lastY) * pw + min(max(ix + c, 0), lastX)];
        }
        __syncwarp();
        return;
    }
    const int ih = h + 7;
    for (int i = lane; i < ih * w; i += 32) {
        const int r = i / w, c = i - r * w;
        const uint8_t *row = ref + min(max(iy + r - 3, 0), lastY) * pw;
        int s = 0;
#pragma unroll
        for (int t = 0; t < 8; t++) s += c_filter[fh][fx][t] * row[min(max(ix + c + t - 3, 0), lastX)];
        tmp[i] = (int16_t)round2(s, 3);
    }
    __syncwarp();
    for (int i = lane; i < w * h; i += 32) {
        const int r = i / w, c = i - r * w;
        int s = 0;
#pragma unroll
        for (int t = 0; t < 8; t++) s += c_filter[fv][fy][t] * tmp[(r + t) * w + c];
        out[i] = (uint8_t)min(max(round2(s, 11), 0), 255);
    }
    __syncwarp();
}

// Exact AV1 compound-average prediction (spec 7.11.3.1 with InterRound1 = 7 and InterPostRound = 4):
// Clip1(Round2(p0 + p1, 5)). hp holds w * h int16 values.
__device__ void warpPredictCompound(const uint8_t *ref0, const uint8_t *ref1, int pw, int ph, int x0, int y0, int w,
                                    int h, Mv mv0, Mv mv1, int sub, int16_t *tmp, int16_t *hp, uint8_t *out) {
    const int lane = threadIdx.x & 31;
    const int fh = w <= 4 ? 4 : 0, fv = h <= 4 ? 4 : 0;
    const int lastX = pw - 1, lastY = ph - 1;
    const int ih = h + 7;
    for (int k = 0; k < 2; k++) {
        const uint8_t *ref = k ? ref1 : ref0;
        const Mv mv = k ? mv1 : mv0;
        const int X = (x0 << 4) + ((2 * mv.col) >> sub);
        const int Y = (y0 << 4) + ((2 * mv.row) >> sub);
        const int ix = X >> 4, fx = X & 15, iy = Y >> 4, fy = Y & 15;
        for (int i = lane; i < ih * w; i += 32) {
            const int r = i / w, c = i - r * w;
            const uint8_t *row = ref + min(max(iy + r - 3, 0), lastY) * pw;
            int s = 0;
#pragma unroll
            for (int t = 0; t < 8; t++) s += c_filter[fh][fx][t] * row[min(max(ix + c + t - 3, 0), lastX)];
            tmp[i] = (int16_t)round2(s, 3);
        }
        __syncwarp();
        for (int i = lane; i < w * h; i += 32) {
            const int r = i / w, c = i - r * w;
            int s = 0;
#pragma unroll
            for (int t = 0; t < 8; t++) s += c_filter[fv][fy][t] * tmp[(r + t) * w + c];
            const int v = round2(s, 7);
            if (!k) hp[i] = (int16_t)v;
            else out[i] = (uint8_t)min(max(round2(hp[i] + v, 5), 0), 255);
        }
        __syncwarp();
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

// Exhaustive search of 8x8 blocks on the quarter-resolution luma (32x32 full-resolution areas).
constexpr int kCoarseWarps = 4;
constexpr int kCoarseMaxR = 24;
__global__ void coarseSearch(const uint8_t *cur, const uint8_t *ref, int dw, int dh, int R, int cols, int rows,
                             float lambda, Mv *out) {
    __shared__ uint8_t win[kCoarseWarps][(8 + 2 * kCoarseMaxR) * (8 + 2 * kCoarseMaxR)];
    __shared__ uint8_t blk[kCoarseWarps][64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * kCoarseWarps + warp;
    if (b >= cols * rows) return;
    const int bx = b % cols, by = b / cols, x0 = bx * 8, y0 = by * 8;
    const int ws = 8 + 2 * R;
    for (int i = lane; i < 64; i += 32)
        blk[warp][i] = cur[min(y0 + i / 8, dh - 1) * dw + min(x0 + i % 8, dw - 1)];
    for (int i = lane; i < ws * ws; i += 32) {
        const int r = i / ws, c = i - r * ws;
        win[warp][i] = ref[min(max(y0 - R + r, 0), dh - 1) * dw + min(max(x0 - R + c, 0), dw - 1)];
    }
    __syncwarp();
    const int side = 2 * R + 1;
    float best = kInf;
    int bestIdx = R * side + R;
    for (int k = lane; k < side * side; k += 32) {
        const int dy = k / side, dx = k - dy * side;
        int s = 0;
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++) s += abs((int)blk[warp][r * 8 + c] - (int)win[warp][(dy + r) * ws + dx + c]);
        const float cost = (float)s + lambda * (float)(abs(dx - R) + abs(dy - R));
        if (cost < best || (cost == best && k < bestIdx)) { best = cost; bestIdx = k; }
    }
    for (int o = 16; o; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bestIdx, o);
        if (ob < best || (ob == best && oi < bestIdx)) { best = ob; bestIdx = oi; }
    }
    if (!lane) {
        const int dy = bestIdx / side - R, dx = bestIdx % side - R;
        out[b] = Mv{(int16_t)(dy * 4 * 8), (int16_t)(dx * 4 * 8)};
    }
}

// Quarter-sample phase planes of a reference luma plane: phase (fy / 4) * 4 + fx / 4 holds the exact AV1 prediction
// sample at every integer position (two-stage filter with the spec's intermediate rounding and edge clamping).
__global__ void horizontalPhases(const uint8_t *ref, int w, int h, int16_t *out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    const int fx = blockIdx.z;
    if (x >= w || y >= h) return;
    const uint8_t *row = ref + size_t(y) * w;
    int s = 0;
#pragma unroll
    for (int t = 0; t < 8; t++) s += c_filter[0][fx * 4][t] * row[min(max(x + t - 3, 0), w - 1)];
    out[(size_t(fx) * h + y) * w + x] = (int16_t)round2(s, 3);
}
__global__ void verticalPhases(const int16_t *hp, int w, int h, uint8_t *out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    const int phase = blockIdx.z, fy = phase >> 2, fx = phase & 3;
    if (x >= w || y >= h) return;
    const int16_t *col = hp + size_t(fx) * h * w + x;
    int s = 0;
#pragma unroll
    for (int t = 0; t < 8; t++) s += c_filter[0][fy * 4][t] * col[size_t(min(max(y + t - 3, 0), h - 1)) * w];
    out[(size_t(phase) * h + y) * w + x] = (uint8_t)min(max(round2(s, 11), 0), 255);
}

// Lanes cooperating on one block: 8 for 8x8 blocks (4 blocks per warp), else the whole warp.
template <int BW, int BH> struct Grp {
    static constexpr int G = BW * BH == 64 ? 8 : 32;
    __device__ static int lane() { return threadIdx.x & (G - 1); }
    __device__ static unsigned mask() {
        return G == 32 ? 0xffffffffu : ((1u << G) - 1) << ((threadIdx.x & 31) & ~(G - 1));
    }
    __device__ static int sum(int v) {
        const unsigned m = mask();
#pragma unroll
        for (int o = G / 2; o; o >>= 1) v += __shfl_xor_sync(m, v, o);
        return v;
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
    if (inside) {
        const uint8_t *p = plane + size_t(y0 + dy) * w + x0 + dx;
        const uint8_t *q = src + size_t(y0) * pitch + x0;
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

// SAD against source samples cached in registers (sv[k] = sample lane + k * G of the block, area <= 1024).
template <int BW, int BH>
__device__ __forceinline__ int groupSadCached(const uint8_t *sv, const uint8_t *src, int pitch, const uint8_t *plane,
                                              int w, int h, int x0, int y0, int dx, int dy) {
    constexpr int L = Log2<BW>::v, G = Grp<BW, BH>::G, PL = BW * BH / G;
    const int lane = Grp<BW, BH>::lane();
    const bool inside = x0 + dx >= 0 && y0 + dy >= 0 && x0 + dx + BW <= w && y0 + dy + BH <= h && x0 + BW <= w && y0 + BH <= h;
    if (!inside) return groupSadPlane<BW, BH>(src, pitch, plane, w, h, x0, y0, dx, dy);
    const uint8_t *p = plane + size_t(y0 + dy) * w + x0 + dx;
    int s = 0;
#pragma unroll
    for (int k = 0; k < PL; k++) {
        const int i = lane + k * G, r = i >> L, c = i & (BW - 1);
        s += abs((int)sv[k] - (int)p[r * w + c]);
    }
    return Grp<BW, BH>::sum(s);
}

// Motion refinement per block shape from coarse, parent, temporal and zero candidates: full-pel search, then half and
// quarter sample steps on the phase planes. parentSize: side of the square block whose vector predicts this one.
template <int BW, int BH>
struct RefineCfg {
    static constexpr int warps = BW * BH >= 4096 ? 2 : BW * BH >= 1024 ? 4 : 8;
    static constexpr int per() { return warps * 32 / Grp<BW, BH>::G; }   // blocks per CTA
};
template <int BW, int BH>
__global__ void __launch_bounds__(32 * RefineCfg<BW, BH>::warps) refine(Frame f, int refIdx, const uint8_t *phases,
                                                                        const Mv *coarse, int ccols, int crows,
                                                                        const Cand *parent, int pcols, int parentSize,
                                                                        const BlockInfo *temporal, int tcols, Cand *out,
                                                                        int cols, int rows, float lambda) {
    using GR = Grp<BW, BH>;
    const int b = blockIdx.x * RefineCfg<BW, BH>::per() + threadIdx.x / GR::G;
    if (b >= cols * rows) return;
    const int x0 = (b % cols) * BW, y0 = (b / cols) * BH;
    Cand &o = out[b];
    if (x0 >= f.w || y0 >= f.h) {
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
    constexpr int CL = BW * BH <= 1024 ? BW * BH / GR::G : 1;   // cached source samples per lane
    uint8_t sv[CL];
    if constexpr (BW * BH <= 1024) {
        constexpr int L = Log2<BW>::v;
#pragma unroll
        for (int k = 0; k < CL; k++) {
            const int i = GR::lane() + k * GR::G, r = i >> L, c = i & (BW - 1);
            sv[k] = (x0 + c < f.w && y0 + r < f.h) ? src[(y0 + r) * pitch + x0 + c] : 0;
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
        return (float)sad + lambda * mvBits(m, pred);
    };
    Mv bm{0, 0};
    float bc = kInf;
    auto tryMv = [&](Mv m) {
        if (abs(m.row) >= (1 << 14) - 8 || abs(m.col) >= (1 << 14) - 8) return;
        const float c = costOf(m);
        if (c < bc) { bc = c; bm = m; }
    };
    for (int i = 0; i < n; i++) tryMv(Mv{(int16_t)(cand[i][0] * 8), (int16_t)(cand[i][1] * 8)});
    for (int pass = 0; pass < kFullPelPasses; pass++) {
        const Mv c0 = bm;
        for (int k = 0; k < 9; k++)
            if (k != 4) tryMv(Mv{(int16_t)(c0.row + (k / 3 - 1) * 8), (int16_t)(c0.col + (k % 3 - 1) * 8)});
        if (bm == c0) break;
    }
    for (int step = 4; step >= 2; step >>= 1) {
        const Mv c0 = bm;
        for (int k = 0; k < 9; k++)
            if (k != 4) tryMv(Mv{(int16_t)(c0.row + (k / 3 - 1) * step), (int16_t)(c0.col + (k % 3 - 1) * step)});
    }
    if (GR::lane() == 0) {
        o.mvR[refIdx] = bm;
        o.costR[refIdx] = bc;
        o.bitsR[refIdx] = mvBits(bm, pred);
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
        if (c1 < c0 && c1 <= cc) { choice = 1; bits = c.bitsR[1] + 1.0f; }
        else if (cc < c0) { choice = 2; bits = c.bitsR[0] + c.bitsR[1] + compoundBias; }
    }
    if (!lane) {
        c.ref = (uint8_t)choice;
        c.mv = choice == 1 ? c.mvR[1] : c.mvR[0];
        c.mv2 = c.mvR[1];
        c.mvBits = bits;
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
__device__ void warpInverseAdd(int16_t *q, int txSz, int dcQ, int acQ, int16_t *res, const uint8_t *pred, uint8_t *recon) {
    constexpr int W = 1 << LW, H = 1 << LH, TW = W > 32 ? 32 : W, TH = H > 32 ? 32 : H;
    constexpr bool rect = LW - LH == 1 || LH - LW == 1;
    const int lane = threadIdx.x & 31;
    const int den = av1g_dq_denom(txSz);
    const int rk = av1g_row_kind(AV1G_DCT_DCT), ck = av1g_col_kind(AV1G_DCT_DCT);
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
    for (int i = lane; i < W * H; i += 32) recon[i] = (uint8_t)min(max((int)pred[i] + res[i], 0), 255);
    __syncwarp();
}
__device__ void inverseAddAny(int lw, int lh, int16_t *q, int txSz, int dcQ, int acQ, int16_t *res, const uint8_t *pred,
                              uint8_t *recon) {
#define PF_INV(A, B) if (lw == A && lh == B) { warpInverseAdd<A, B>(q, txSz, dcQ, acQ, res, pred, recon); return; }
    PF_INV(2, 2) PF_INV(3, 3) PF_INV(4, 4) PF_INV(5, 5) PF_INV(6, 6)
    PF_INV(3, 2) PF_INV(2, 3) PF_INV(4, 3) PF_INV(3, 4) PF_INV(5, 4) PF_INV(4, 5) PF_INV(6, 5) PF_INV(5, 6)
    PF_INV(4, 2) PF_INV(2, 4) PF_INV(5, 3) PF_INV(3, 5) PF_INV(6, 4) PF_INV(4, 6)
#undef PF_INV
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

template <int BW, int BH>
struct EvalCfg {
    static constexpr int warps = BW * BH >= 1024 ? 1 : BW * BH >= 256 ? 2 : 4;
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
    int rdoq;
};

// Evaluate skip and coded options of every block of one shape.
template <int BW, int BH>
__global__ void __launch_bounds__(32 * EvalCfg<BW, BH>::warps) evaluate(EvalArgs a) {
    constexpr int W = EvalCfg<BW, BH>::warps;
    constexpr int AREA = BW * BH;
    constexpr int TQ = (BW > 32 ? 32 : BW) * (BH > 32 ? 32 : BH);   // stored luma coefficients
    constexpr int TS = (BH > 32 ? 32 : BH) * (BW + 1);              // forward transform scratch
    __shared__ int16_t interp[W][(BH + 7) * BW];
    __shared__ int16_t hpS[W][AREA];
    __shared__ uint8_t predS[W][AREA + AREA / 2];
    __shared__ uint8_t recS[W][AREA];
    __shared__ int16_t resS[W][AREA];
    __shared__ float tmpS[W][TS];
    __shared__ int16_t qS[W][TQ];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * W + warp;
    if (b >= a.cols * a.rows) return;
    Cand &c = a.cand[b];
    const int x0 = (b % a.cols) * BW, y0 = (b / a.cols) * BH;
    const Frame &f = a.f;
    if (x0 >= f.w || y0 >= f.h) {
        if (!lane) { c.costSkip = c.costCoded = kInf; c.nz = 0; c.txSplit = 0; }
        return;
    }
    const uint8_t *ref = c.ref == 1 ? f.ref[1] : f.ref[0];
    const uint8_t *ref2 = f.ref[1];
    const int cw = f.w >> 1, chh = f.h >> 1;
    const size_t ysz = size_t(f.w) * f.h, csz = size_t(cw) * chh;
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    float dSkip = 0.f, dCoded = 0.f, bits = 0.f;
    bool nz = false;
    float lumaD = 0.f, lumaBits = 0.f, splitD[3] = {kInf, kInf, kInf}, splitBits[3] = {0.f, 0.f, 0.f};
    bool nzSplit[3] = {false, false, false}, nzC = false;
#pragma unroll
    for (int p = 0; p < 3; p++) {
        const int sub = p > 0, bw = p ? BW / 2 : BW, bh = p ? BH / 2 : BH;
        const int lw = Log2<BW>::v - sub, lh = Log2<BH>::v - sub;
        const int pw = p ? cw : f.w, ph = p ? chh : f.h;
        const int px = x0 >> sub, py = y0 >> sub;
        uint8_t *pred = predS[warp] + (p ? AREA + (p - 1) * (AREA / 4) : 0);
        const uint8_t *refP = ref + (p ? ysz + (p - 1) * csz : 0);
        if (c.ref == 2)
            warpPredictCompound(refP, ref2 + (p ? ysz + (p - 1) * csz : 0), pw, ph, px, py, bw, bh, c.mv, c.mv2, sub,
                                interp[warp], hpS[warp], pred);
        else
            warpPredict(refP, pw, ph, px, py, bw, bh, c.mv, sub, interp[warp], pred);
        const uint8_t *src = f.src[p];
        const int pitch = f.pitch[p];
        int sse = 0;
        int16_t *res = resS[warp];
        float energy = 0.f;
        uint8_t *outPred = a.pred + (p ? ysz + (p - 1) * csz : 0);
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
        __syncwarp();
        // transform of the whole block (64-point transforms keep 32 coefficients per side)
        const int tlw = lw > 6 ? 6 : lw, tlh = lh > 6 ? 6 : lh;
        const int txSz = txSizeOf(tlw, tlh);
        const int tqw = bw > 32 ? 32 : bw, tqh = bh > 32 ? 32 : bh;
        EvalIn e{};
        e.txSz = txSz; e.txType = AV1G_DCT_DCT; e.plane = p;
        e.dcQ = dcQ; e.acQ = acQ; e.qm = nullptr;
        e.rnd = a.rnd; e.zbin = a.zbin;
        e.skipCtx = p ? 7 : 0; e.dcCtx = 0; e.intraDir = 0; e.reduced = 1;
        e.C = g_cost[a.qctx];
        e.resEnergy = warp_sum(energy);
        e.qmDist = 0; e.qmRnd = 0;
        EvalOut o;
        int16_t *q = qS[warp];
        warp_fwd_rt(res, tmpS[warp], bw, bh, txSz, AV1G_DCT_DCT);
        warp_quant_rate_rt<8>(tmpS[warp], q, e, o, bw, bh);
        if (a.rdoq && bw == bh) {
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
        bits += o.R;
        uint8_t *rec = recS[warp];
        inverseAddAny(tlw, tlh, q, txSz, dcQ, acQ, res, pred, rec);
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
        if (BW != BH && p == 0 && a.recon2) {
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
                warp_fwd_rt(res, tmpS[warp], SW, SH, stx, AV1G_DCT_DCT);
                warp_quant_rate_rt<8>(tmpS[warp], q, es, os, SW, SH);
                bS += os.R;
                inverseAddAny(lsw, lsh, q, stx, dcQ, acQ, res, pq, rq);
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
        if (BW == BH && p == 0 && a.recon2) {
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
                    warp_fwd_rt(res, tmpS[warp], hn, hn, stx, AV1G_DCT_DCT);
                    warp_quant_rate_rt<8>(tmpS[warp], q, es, os, hn, hn);
                    bS += os.R;
                    inverseAddAny(lhn, lhn, q, stx, dcQ, acQ, res, pq, rq);
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
    }
    if (!lane) {
        const float modeBits = 2.f + c.mvBits * a.mvWeight;
        const float skipFlag1 = 0.5f, skipFlag0 = 1.5f;
        c.costSkip = dSkip + a.lambda * (modeBits + skipFlag1);
        // split depth: about one more bit per level for the txfm_split flags
        int depth = 0;
        float bestJ = lumaD + a.lambda * lumaBits;
        for (int d = 1; d <= 2; d++)
            if (a.recon2 && splitD[d] + a.lambda * (splitBits[d] + 1.f + 2.f * (d - 1)) < bestJ) {
                bestJ = splitD[d] + a.lambda * (splitBits[d] + 1.f + 2.f * (d - 1));
                depth = d;
            }
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
                break;
            }
        }
    }
    a.grid[gy * a.gcols + gx] = bi;
}

// Gather the chosen reconstruction and levels. Shapes: 0..3 squares 8..64, 4 + (level - 1) * 2 + (0 HORZ, 1 VERT).
constexpr int kShapes = 14;   // + 10 + (level - 2) * 2 + (0 HORZ_4, 1 VERT_4)
struct CommitArgs {
    const BlockInfo *grid;
    int gcols, w, h, aw, ah;
    const uint8_t *pred[kShapes], *recon[kShapes];
    const int16_t *levels[kShapes];
    const uint8_t *recon2[kShapes], *recon3[kSizes];
    const int16_t *levels2[kShapes], *levels3[kSizes];
    uint8_t *out;
    int16_t *outLevels;
};
template <class T, int K>
__device__ __forceinline__ T pickN(int s, const T (&v)[K]) {
    T r = v[0];
#pragma unroll
    for (int k = 1; k < K; k++)
        if (s == k) r = v[k];
    return r;
}
__global__ void commit(CommitArgs a) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    const int p = blockIdx.z, sub = p > 0;
    const int apw = a.aw >> sub, aph = a.ah >> sub;
    if (x >= apw || y >= aph) return;
    const BlockInfo bi = a.grid[((y << sub) >> 3) * a.gcols + ((x << sub) >> 3)];
    const size_t lbase = p ? size_t(a.aw) * a.ah + size_t(p - 1) * (a.aw / 2) * (a.ah / 2) : 0;
    const int s = bi.log2 - 3;
    const int shape = !bi.part ? s : bi.part <= 2 ? 4 + (s - 1) * 2 + (bi.part - 1) : 10 + (s - 2) * 2 + (bi.part - 3);
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
__global__ void planeSse(const uint8_t *src, int pitch, const uint8_t *buf, int w, int h, unsigned long long *out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    int d = 0;
    if (x < w && y < h) {
        d = (int)src[y * pitch + x] - (int)buf[y * w + x];
        d *= d;
    }
    for (int o = 16; o; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    if ((threadIdx.x & 31) == 0 && d) atomicAdd(out, (unsigned long long)d);
}

// ------------------------------------------------------------------------------------------------
// GPU tile entropy coding: one thread per AV1 tile.
// Inter tile entropy coding in four stages (pframe_tile.cuh): three parallel phases over superblocks (mode info and
// context maps, mode choice, symbol records), then one serial range coder per tile over the records.
struct TileJob {
    pft2::FrameSyntax fs;
    const BlockInfo *grid;
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
__device__ __host__ inline void tileOfSb(const TileJob &j, int miRow, int miCol, int &tr, int &tc) {
    tc = 0;
    while (tc + 1 < j.tileCols && j.colStart[tc + 1] <= miCol) ++tc;
    tr = 0;
    while (tr + 1 < j.tileRows && j.rowStart[tr + 1] <= miRow) ++tr;
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
__device__ __host__ inline void setupCoder(pft2::TileCoder &t, const TileJob &j, int sb, int16_t *qv) {
    const int miRow = (sb / j.sbCols) * 16, miCol = (sb % j.sbCols) * 16;
    int tr, tc;
    tileOfSb(j, miRow, miCol, tr, tc);
    t.fs = &j.fs;
    t.grid = j.grid;
    t.levels = j.levels;
    t.mi = j.mi;
    t.rowStart = j.rowStart[tr]; t.rowEnd = j.rowStart[tr + 1];
    t.colStart = j.colStart[tc]; t.colEnd = j.colStart[tc + 1];
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
    int16_t qv[1024];
    pft2::TileCoder t;
    setupCoder(t, j, sb, qv);
    t.phase = 1;
    t.entries = j.entries + size_t(sb) * 64;
    t.superblock((sb / j.sbCols) * 16, (sb % j.sbCols) * 16, (uint32_t)sb);
    if (t.st.error) atomicMax(j.error, t.st.error);
    j.entryCount[sb] = t.nEntries;
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
// One CTA of one warp per tile: the warp prefetches records into shared memory, lane 0 range-codes them with the tile's
// CDFs in shared memory.
__global__ void __launch_bounds__(32) tileCode(TileJob j) {
    __shared__ pft2::EntropyState es;
    __shared__ pft2::SymRec buf[32];
    const int t = blockIdx.x, lane = threadIdx.x;
    if (t >= j.tileCols * j.tileRows) return;
    if (*j.error) { if (!lane) j.sizes[t] = -1; return; }
    {
        const uint32_t *src = reinterpret_cast<const uint32_t *>(j.coderInit);
        uint32_t *dst = reinterpret_cast<uint32_t *>(&es);
        for (int i = lane; i < int(sizeof(es) / 4); i += 32) dst[i] = src[i];
    }
    __syncwarp();
    av1g_ec ec;
    if (!lane) av1g_ec_init(&ec, j.ecBuf + j.ecCapPerTile * t, (uint32_t)j.ecCapPerTile);
    const int tc = t % j.tileCols, tr = t / j.tileCols;
    uint16_t *base = reinterpret_cast<uint16_t *>(&es);
    for (int r = j.rowStart[tr]; r < j.rowStart[tr + 1]; r += 16)
        for (int c = j.colStart[tc]; c < j.colStart[tc + 1]; c += 16) {
            const int sb = (r / 16) * j.sbCols + c / 16;
            for (uint32_t b = 0; b < j.entryCount[sb]; b++) {
                const uint32_t first = j.sbOffset[sb * 64 + b], last = first + j.sbCount[sb * 64 + b];
                for (uint32_t k = first; k < last; k += 32) {
                    if (k + lane < last) buf[lane] = j.recs[k + lane];
                    __syncwarp();
                    if (!lane) {
                        const uint32_t m = min(32u, last - k);
                        for (uint32_t i = 0; i < m; i++) pft2::codeRecord(&ec, base, buf[i]);
                    }
                    __syncwarp();
                }
            }
        }
    if (!lane) {
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
// of +-kTfRange around the coarse vector, then the motion-compensated samples are blended with weights from the
// block error relative to the quantizer step.
constexpr int kTfMax = 8, kTfRange = 6, kTfWin = 16 + 2 * kTfRange;
struct TfArgs {
    const uint8_t *center[3];
    const uint8_t *nb[kTfMax][3];
    const Mv *coarse[kTfMax];
    int count, w, h, aw, ccols, crows;
    float step2;   // (quantizer step)^2 in sample units, times the strength
    float stepPix; // quantizer step in sample units
    uint8_t *out[3];
};
__global__ void __launch_bounds__(256) temporalFilterKernel(TfArgs a) {
    __shared__ uint8_t cur[256];
    __shared__ uint8_t win[kTfWin * kTfWin];
    __shared__ int sads[(2 * kTfRange + 1) * (2 * kTfRange + 1)];
    __shared__ float wsum[256], acc[256];
    __shared__ float cw[2][64], cacc[2][64];
    __shared__ int bestOff;
    __shared__ float blockW;
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
            cacc[p][t] = a.center[1 + p][uy * caw + ux];
        }
    }
    __syncthreads();
    const int side = 2 * kTfRange + 1;
    for (int j = 0; j < a.count; j++) {
        const Mv cm = a.coarse[j][min((by + 8) / 32, a.crows - 1) * a.ccols + min((bx + 8) / 32, a.ccols - 1)];
        const int mx = cm.col >> 3, my = cm.row >> 3;
        const uint8_t *ny = a.nb[j][0];
        for (int i = t; i < kTfWin * kTfWin; i += 256) {
            const int r = i / kTfWin, c = i % kTfWin;
            const int x = min(max(bx + mx - kTfRange + c, 0), a.w - 1), y = min(max(by + my - kTfRange + r, 0), a.h - 1);
            win[i] = ny[y * a.aw + x];
        }
        __syncthreads();
        if (t < side * side) {
            const int oy = t / side, ox = t % side;
            int s = 0;
            for (int r = 0; r < 16; r++)
                for (int c = 0; c < 16; c++) s += abs((int)cur[r * 16 + c] - (int)win[(oy + r) * kTfWin + ox + c]);
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
        const int d = (int)cur[t] - (int)win[(oy + (t >> 4)) * kTfWin + ox + (t & 15)];
        // block mean squared error
        float e = (float)(d * d);
        for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
        if ((t & 31) == 0) warpE[t >> 5] = e;
        __syncthreads();
        if (t == 0) {
            float tot = 0.f;
            for (int k = 0; k < 8; k++) tot += warpE[k];
            const float mse = tot / 256.f;
            blockW = __expf(-mse / (2.f * a.step2));
        }
        __syncthreads();
        // pixel weight: reject clear mismatches
        const float wpix = fabsf((float)d) > 3.f * a.stepPix + 2.f ? 0.f : blockW;
        acc[t] += wpix * (float)win[(oy + (t >> 4)) * kTfWin + ox + (t & 15)];
        wsum[t] += wpix;
        if (t < 64) {
            const int dx = mx - kTfRange + ox, dy = my - kTfRange + oy;   // full-pel luma displacement
            const int ux = bx / 2 + (t & 7), uy = by / 2 + (t >> 3);
            const int sx = min(max(ux + (dx >> 1), 0), cw2 - 1), sy = min(max(uy + (dy >> 1), 0), ch2 - 1);
            const int uxc = min(ux, cw2 - 1), uyc = min(uy, ch2 - 1);
            for (int p = 0; p < 2; p++) {
                const int v = a.nb[j][1 + p][sy * caw + sx];
                const int c0 = a.center[1 + p][uyc * caw + uxc];
                const float wc = fabsf((float)(c0 - v)) > 3.f * a.stepPix + 2.f ? 0.f : blockW;
                cacc[p][t] += wc * v;
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
    bool recordsOnly = false;
    cudaEvent_t staged = nullptr, done = nullptr;
    cudaStream_t fetch = nullptr;
    int nTiles = 0;
};

struct State {
    uint8_t *ds[1 + kMaxRefs] = {};
    Mv *coarse[kMaxRefs] = {};
    Cand *cand[kSizes] = {};
    uint8_t *pred[kSizes] = {}, *recon[kSizes] = {};
    int16_t *levels[kSizes] = {}, *outLevels = nullptr;
    uint8_t *recon2[kSizes] = {}, *recon3[kSizes] = {};
    int16_t *levels3[kSizes] = {};
    Cand *rcand[10] = {};
    uint8_t *rpred[10] = {}, *rrecon[10] = {};
    int16_t *rlevels[10] = {};
    int rcols[10] = {}, rrows[10] = {};
    uint8_t *rrecon2[10] = {};
    int16_t *rlevels2[10] = {};
    int16_t *levels2[kSizes] = {};
    BlockInfo *grid = nullptr, *prevGrid = nullptr;
    bool havePrev = false;
    int dw = 0, dh = 0, ccols = 0, crows = 0;
    int cols[kSizes] = {}, rows[kSizes] = {};
    int gcols = 0, grows = 0;
    bool tablesReady = false;
    void *pfScratch = nullptr;
    uint8_t *work = nullptr, *cdefOut = nullptr;
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
    int16_t *hphase = nullptr;
    uint8_t *phases[kMaxRefs] = {};
    bool profile = false;
    cudaEvent_t ev[8] = {};
    double stageMs[8] = {};
    cudaEvent_t eev[5] = {};
    double entMs[4] = {};
    int entFrames = 0;
    int frames = 0;
};

}  // namespace

Analyzer::~Analyzer() {
    auto *s = static_cast<State *>(state);
    if (!s) return;
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
    cudaFree(s->grid);
    cudaFree(s->prevGrid);
    cudaFree(s->pfScratch);
    cudaFree(s->work);
    cudaFree(s->cdefOut);
    cudaFree(s->sse);
    cudaFree(s->hphase);
    for (void *p : {(void *)s->slots, (void *)s->tileWork, (void *)s->staging, (void *)s->defaultInter, (void *)s->mi,
                    (void *)s->tileStats, (void *)s->ecBuf})
        cudaFree(p);
    for (auto &es : s->eslot) {
        cudaFree(es.grid); cudaFree(es.levels); cudaFree(es.tileOut); cudaFree(es.packed); cudaFree(es.sizes); cudaFree(es.error); cudaFree(es.tileCount); cudaFree(es.recs);
        if (es.staged) cudaEventDestroy(es.staged);
        if (es.done) cudaEventDestroy(es.done);
        if (es.fetch) cudaStreamDestroy(es.fetch);
    }
    if (s->entStream) cudaStreamDestroy(s->entStream);
    cudaFree(s->entries); cudaFree(s->entryCount); cudaFree(s->sbTotal);
    cudaFree(s->frameInit); cudaFree(s->coderInit); cudaFree(s->packOff); cudaFree(s->sbCount); cudaFree(s->sbOffset); cudaFree(s->recs);
    for (int p = 0; p < 3; p++) { cudaFree(s->ctxLevel[p]); cudaFree(s->ctxDc[p]); }
    for (auto *p : s->tfDs) cudaFree(p);
    for (auto *p : s->tfCoarse) cudaFree(p);
    for (auto *p : s->best) cudaFree(p);
    for (auto *p : s->mode) cudaFree(p);
    for (auto *p : s->phases) cudaFree(p);
    if (s->profile && s->frames) {
        const char *names[] = {"downsample+coarse", "refine", "evaluate", "decide", "commit", "download"};
        for (int i = 0; i < 6; i++) fprintf(stderr, "pframe %-18s %7.2f ms/frame\n", names[i], s->stageMs[i] / s->frames);
    }
    for (auto &e : s->ev) if (e) cudaEventDestroy(e);
    if (s->profile && s->entFrames) {
        const char *names[] = {"entropy phase 1", "entropy phase 2", "entropy phase 3", "entropy coder"};
        for (int i = 0; i < 4; i++) fprintf(stderr, "pframe %-18s %7.2f ms/frame\n", names[i], s->entMs[i] / s->entFrames);
    }
    for (auto &e : s->eev) if (e) cudaEventDestroy(e);
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
    }
    alloc(&s->outLevels, aframe * sizeof(int16_t));
    s->gcols = aw / 8; s->grows = ah / 8;
    alloc(&s->grid, sizeof(BlockInfo) * s->gcols * s->grows);
    alloc(&s->prevGrid, sizeof(BlockInfo) * s->gcols * s->grows);
    {
        av1g_frame_params fp{};
        fp.width = w; fp.height = h; fp.chroma_format = AV1G_CHROMA_420; fp.bit_depth = 8; fp.enable_cdef = 1;
        alloc((uint8_t **)&s->pfScratch, av1g_gpu_pf_scratch_size(&fp));
        alloc(&s->work, frame);
        alloc(&s->cdefOut, frame);
        alloc(&s->sse, sizeof(unsigned long long) * 64);
        alloc(&s->hphase, sizeof(int16_t) * 4 * size_t(w) * h);
        for (auto *&p : s->phases) alloc(&p, 16 * size_t(w) * h);
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
        memcpy(d.compMode, pft::Default_Comp_Mode_Cdf, sizeof d.compMode);
        memcpy(d.compRefType, pft::Default_Comp_Ref_Type_Cdf, sizeof d.compRefType);
        memcpy(d.compRef, pft::Default_Comp_Ref_Cdf, sizeof d.compRef);
        memcpy(d.compBwdRef, pft::Default_Comp_Bwd_Ref_Cdf, sizeof d.compBwdRef);
        memcpy(d.compoundMode, pft::Default_Compound_Mode_Cdf, sizeof d.compoundMode);
        memcpy(d.txfmSplit, pft::Default_Txfm_Split_Cdf, sizeof d.txfmSplit);
        if (e == cudaSuccess) e = cudaMemcpy(s->defaultInter, &d, sizeof d, cudaMemcpyHostToDevice);
    }
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(c_filter, pft::Subpel_Filters, sizeof(pft::Subpel_Filters));
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame allocation: %s", cudaGetErrorString(e));
        return -1;
    }
    s->profile = getenv("AV1GPU_PF_PROFILE") != nullptr;
    if (s->profile)
        for (auto &ev : s->ev) cudaEventCreate(&ev);
    if (s->profile)
        for (auto &ev : s->eev) cudaEventCreate(&ev);
    if (gp_upload_tables_pframe(gp_host_tables())) {
        snprintf(err, errLen, "P-frame table upload failed");
        return -1;
    }
    return 0;
}

int Analyzer::analyze(const uint8_t *const src[3], const int pitch[3], const uint8_t *const *refs, int refCount,
                      const Settings &st, uint8_t *recon, std::vector<BlockInfo> *info,
                      std::vector<int16_t> *levels, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    const auto t0 = std::chrono::steady_clock::now();
    refCount = refCount < 1 ? 1 : refCount > kMaxRefs ? kMaxRefs : refCount;
    Frame f{};
    for (int p = 0; p < 3; p++) { f.src[p] = src[p]; f.pitch[p] = pitch[p]; }
    for (int r = 0; r < refCount; r++) f.ref[r] = refs[r];
    f.w = w; f.h = h; f.aw = aw; f.ah = ah;
    const float lambda = gp_lambda<8>(st.qindex, st.lambdaMul);
    const float lambdaSad = sqrtf(lambda);
    const dim3 t16(16, 16);
    const dim3 gds((s->dw + 15) / 16, (s->dh + 15) / 16);
    auto mark = [&](int i) { if (s->profile) cudaEventRecord(s->ev[i]); };
    mark(0);
    downsample4<<<gds, t16>>>(src[0], pitch[0], w, h, s->ds[0], s->dw, s->dh);
    const int R = std::min(kCoarseMaxR, std::max(2, st.searchRange / 4));
    const int nCoarse = s->ccols * s->crows;
    for (int r = 0; r < refCount; r++) {
        downsample4<<<gds, t16>>>(refs[r], w, w, h, s->ds[1 + r], s->dw, s->dh);
        coarseSearch<<<(nCoarse + kCoarseWarps - 1) / kCoarseWarps, 32 * kCoarseWarps>>>(
            s->ds[0], s->ds[1 + r], s->dw, s->dh, R, s->ccols, s->crows, lambdaSad * 0.5f, s->coarse[r]);
    }
    mark(1);
    const BlockInfo *temporal = s->havePrev ? s->prevGrid : nullptr;
    for (int r = 0; r < refCount; r++) {
        horizontalPhases<<<dim3((w + 31) / 32, (h + 7) / 8, 4), dim3(32, 8)>>>(refs[r], w, h, s->hphase);
        verticalPhases<<<dim3((w + 31) / 32, (h + 7) / 8, 16), dim3(32, 8)>>>(s->hphase, w, h, s->phases[r]);
#define PF_REFINE(K, N)                                                                                         \
    refine<N, N><<<(s->cols[K] * s->rows[K] + RefineCfg<N, N>::per() - 1) / RefineCfg<N, N>::per(),               \
                   32 * RefineCfg<N, N>::warps>>>(f, r, s->phases[r], s->coarse[r], s->ccols, s->crows,             \
                                                  K == 3 ? nullptr : s->cand[K + 1], K == 3 ? 0 : s->cols[K + 1],    \
                                                  2 * N, temporal, s->gcols, s->cand[K], s->cols[K], s->rows[K], lambdaSad)
        PF_REFINE(3, 64);
        PF_REFINE(2, 32);
        PF_REFINE(1, 16);
        PF_REFINE(0, 8);
#define PF_RREFINE(RK, K, BW, BH)                                                                               \
    refine<BW, BH><<<(s->rcols[RK] * s->rrows[RK] + RefineCfg<BW, BH>::per() - 1) / RefineCfg<BW, BH>::per(),   \
                     32 * RefineCfg<BW, BH>::warps>>>(f, r, s->phases[r], s->coarse[r], s->ccols, s->crows,          \
                                                      s->cand[K], s->cols[K], 8 << K, temporal, s->gcols,            \
                                                      s->rcand[RK], s->rcols[RK], s->rrows[RK], lambdaSad)
        if (st.rectPartitions) {
            PF_RREFINE(0, 1, 16, 8); PF_RREFINE(1, 1, 8, 16);
            PF_RREFINE(2, 2, 32, 16); PF_RREFINE(3, 2, 16, 32);
            PF_RREFINE(4, 3, 64, 32); PF_RREFINE(5, 3, 32, 64);
            PF_RREFINE(6, 2, 32, 8); PF_RREFINE(7, 2, 8, 32);
            PF_RREFINE(8, 3, 64, 16); PF_RREFINE(9, 3, 16, 64);
        }
#undef PF_RREFINE
#undef PF_REFINE
    }
#define PF_CHOOSE(C, CO, R, BW, BH)                                                                             \
    chooseInter<BW, BH><<<(CO * R + RefineCfg<BW, BH>::per() - 1) / RefineCfg<BW, BH>::per(),                     \
                          32 * RefineCfg<BW, BH>::warps>>>(f, refCount, s->phases[0], s->phases[1], C, CO, R,        \
                                                           lambdaSad, st.compoundBias)
    PF_CHOOSE(s->cand[3], s->cols[3], s->rows[3], 64, 64);
    PF_CHOOSE(s->cand[2], s->cols[2], s->rows[2], 32, 32);
    PF_CHOOSE(s->cand[1], s->cols[1], s->rows[1], 16, 16);
    PF_CHOOSE(s->cand[0], s->cols[0], s->rows[0], 8, 8);
    if (st.rectPartitions) {
        PF_CHOOSE(s->rcand[0], s->rcols[0], s->rrows[0], 16, 8); PF_CHOOSE(s->rcand[1], s->rcols[1], s->rrows[1], 8, 16);
        PF_CHOOSE(s->rcand[2], s->rcols[2], s->rrows[2], 32, 16); PF_CHOOSE(s->rcand[3], s->rcols[3], s->rrows[3], 16, 32);
        PF_CHOOSE(s->rcand[4], s->rcols[4], s->rrows[4], 64, 32); PF_CHOOSE(s->rcand[5], s->rcols[5], s->rrows[5], 32, 64);
        PF_CHOOSE(s->rcand[6], s->rcols[6], s->rrows[6], 32, 8); PF_CHOOSE(s->rcand[7], s->rcols[7], s->rrows[7], 8, 32);
        PF_CHOOSE(s->rcand[8], s->rcols[8], s->rrows[8], 64, 16); PF_CHOOSE(s->rcand[9], s->rcols[9], s->rrows[9], 16, 64);
    }
#undef PF_CHOOSE
    mark(2);
    EvalArgs ea{};
    ea.f = f;
    ea.qindex = st.qindex;
    ea.qctx = av1g_coeff_cdf_q_ctx(st.qindex);
    ea.lambda = lambda;
    ea.rnd = getenv("AV1GPU_PF_RND") ? (float)atof(getenv("AV1GPU_PF_RND")) : st.rnd;
    ea.zbin = getenv("AV1GPU_PF_ZBIN") ? (float)atof(getenv("AV1GPU_PF_ZBIN")) : st.zbin;
    ea.mvWeight = getenv("AV1GPU_PF_MVW") ? (float)atof(getenv("AV1GPU_PF_MVW")) : 0.25f;
    ea.rdoq = st.rdoq;
#define PF_EVAL(K, N)                                                                                           \
    ea.cand = s->cand[K]; ea.cols = s->cols[K]; ea.rows = s->rows[K];                                          \
    ea.pred = s->pred[K]; ea.recon = s->recon[K]; ea.levels = s->levels[K];                                   \
    ea.recon2 = st.txSplit ? s->recon2[K] : nullptr; ea.levels2 = s->levels2[K];                               \
    ea.recon3 = s->recon3[K]; ea.levels3 = s->levels3[K];                                                      \
    evaluate<N, N><<<(s->cols[K] * s->rows[K] + EvalCfg<N, N>::warps - 1) / EvalCfg<N, N>::warps,                 \
                     32 * EvalCfg<N, N>::warps>>>(ea)
    PF_EVAL(0, 8);
    PF_EVAL(1, 16);
    PF_EVAL(2, 32);
    PF_EVAL(3, 64);
#define PF_REVAL(RK, BW, BH)                                                                                    \
    ea.cand = s->rcand[RK]; ea.cols = s->rcols[RK]; ea.rows = s->rrows[RK];                                    \
    ea.pred = s->rpred[RK]; ea.recon = s->rrecon[RK]; ea.levels = s->rlevels[RK];                              \
    ea.recon2 = st.txSplit ? s->rrecon2[RK] : nullptr; ea.levels2 = s->rlevels2[RK];                           \
    evaluate<BW, BH><<<(s->rcols[RK] * s->rrows[RK] + EvalCfg<BW, BH>::warps - 1) / EvalCfg<BW, BH>::warps,       \
                       32 * EvalCfg<BW, BH>::warps>>>(ea)
    if (st.rectPartitions) {
        PF_REVAL(0, 16, 8); PF_REVAL(1, 8, 16); PF_REVAL(2, 32, 16);
        PF_REVAL(3, 16, 32); PF_REVAL(4, 64, 32); PF_REVAL(5, 32, 64);
        PF_REVAL(6, 32, 8); PF_REVAL(7, 8, 32); PF_REVAL(8, 64, 16); PF_REVAL(9, 16, 64);
    }
#undef PF_REVAL
#undef PF_EVAL
    mark(3);
    DecideArgs da{};
    for (int k = 0; k < kSizes; k++) {
        da.cand[k] = s->cand[k]; da.cols[k] = s->cols[k]; da.best[k] = s->best[k]; da.mode[k] = s->mode[k];
    }
    for (int k = 0; k < 10; k++) { da.rcand[k] = s->rcand[k]; da.rcols[k] = s->rcols[k]; }
    da.rect = st.rectPartitions;
    da.w = w; da.h = h; da.gcols = s->gcols; da.grows = s->grows; da.grid = s->grid; da.lambda = lambda;
    // approximate partition symbol costs (bits)
    const float part[kSizes][2] = {{1.0f, 0.f}, {1.0f, 2.5f}, {1.0f, 2.5f}, {1.0f, 2.0f}};
    memcpy(da.partBits, part, sizeof part);
    for (int k = 0; k < kSizes; k++)
        decideLevel<<<(s->cols[k] * s->rows[k] + 127) / 128, 128>>>(da, k, s->rows[k]);
    decideGrid<<<dim3((s->gcols + 15) / 16, (s->grows + 15) / 16), t16>>>(da);
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
    commit<<<dim3((aw + 15) / 16, (ah + 15) / 16, 3), t16>>>(ca);
    mark(5);
    cudaError_t e = cudaGetLastError();
    const size_t gridCount = size_t(s->gcols) * s->grows;
    if (info) {
        info->resize(gridCount);
        if (e == cudaSuccess) e = cudaMemcpy(info->data(), s->grid, gridCount * sizeof(BlockInfo), cudaMemcpyDeviceToHost);
    }
    if (levels) {
        levels->resize(size_t(aw) * ah * 3 / 2);
        if (e == cudaSuccess)
            e = cudaMemcpy(levels->data(), s->outLevels, levels->size() * sizeof(int16_t), cudaMemcpyDeviceToHost);
    }
    if (e == cudaSuccess) e = cudaMemcpyAsync(s->prevGrid, s->grid, gridCount * sizeof(BlockInfo), cudaMemcpyDeviceToDevice);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame analysis: %s", cudaGetErrorString(e));
        return -1;
    }
    s->havePrev = true;
    if (s->profile) {
        mark(6);
        cudaEventSynchronize(s->ev[6]);
        for (int i = 0; i < 6; i++) {
            float ms = 0;
            cudaEventElapsedTime(&ms, s->ev[i], s->ev[i + 1]);
            s->stageMs[i] += ms;
        }
        s->frames++;
    }
    elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return 0;
}

// Entropy coding of the last analyzed frame, asynchronous: the decisions are staged into entropy slot k (on the
// analysis stream), then the tile kernel runs on its own stream, overlapping the analysis of later frames. Kernels of
// that stream run in coding order, so the reference-slot CDFs they read and write stay consistent.
int Analyzer::entropyLaunch(const EntropyIn &in, int k, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    const int miCols = 2 * ((w + 7) >> 3), miRows = 2 * ((h + 7) >> 3);
    const int sbCols = (miCols + 15) >> 4, sbRows = (miRows + 15) >> 4;
    auto maxLog2 = [](int sb) { int q = 0; while ((1 << q) < (sb < 64 ? sb : 64)) ++q; return q; };
    const int cl = std::min(in.tileColsLog2, maxLog2(sbCols)), rl = std::min(in.tileRowsLog2, maxLog2(sbRows));
    TileJob j{};
    const int wSb = (sbCols + (1 << cl) - 1) >> cl, hSb = (sbRows + (1 << rl) - 1) >> rl;
    for (int sb = 0; sb < sbCols; sb += wSb) j.colStart[j.tileCols++] = sb * 16;
    j.colStart[j.tileCols] = miCols;
    for (int sb = 0; sb < sbRows; sb += hSb) j.rowStart[j.tileRows++] = sb * 16;
    j.rowStart[j.tileRows] = miRows;
    const int nTiles = j.tileCols * j.tileRows;
    if (nTiles > s->maxTiles || k < 0 || k >= kEntropySlots) {
        snprintf(err, errLen, "too many tiles (%d) or bad entropy slot", nTiles);
        return -1;
    }
    EntropySlot &es = s->eslot[k];
    es.nTiles = nTiles;
    cudaError_t e = cudaMemcpyAsync(es.grid, s->grid, gridCount() * sizeof(BlockInfo), cudaMemcpyDeviceToDevice);
    if (e == cudaSuccess) e = cudaMemcpyAsync(es.levels, s->outLevels, levelCount() * sizeof(int16_t), cudaMemcpyDeviceToDevice);
    if (e == cudaSuccess) e = cudaEventRecord(es.staged, 0);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(s->entStream, es.staged, 0);
    j.fs.refName[0] = in.refName[0];
    j.fs.refName[1] = in.refName[1];
    j.fs.referenceSelect = in.referenceSelect;
    j.fs.txModeSelect = in.txModeSelect;
    j.fs.miRows = miRows; j.fs.miCols = miCols; j.fs.aw = aw; j.fs.ah = ah; j.fs.gcols = s->gcols;
    j.grid = es.grid;
    j.levels = es.levels;
    j.mi = s->mi;
    j.frameInit = s->frameInit;
    j.coderInit = s->coderInit;
    j.packOff = s->packOff; j.tileCount = es.tileCount; j.packed = es.recs;
    j.save = in.saveSlot >= 0 ? s->slots + in.saveSlot : nullptr;
    j.contextTile = std::min(in.contextTile, nTiles - 1);
    j.sbCols = sbCols; j.sbRows = sbRows;
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
    const int nSb = sbCols * sbRows;
    auto emark = [&](int i) { if (s->profile) cudaEventRecord(s->eev[i], st); };
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
        if (e == cudaSuccess) e = cudaEventRecord(es.done, st);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) {
            snprintf(err, errLen, "tile entropy launch: %s", cudaGetErrorString(e));
            return -1;
        }
        return 0;
    }
    tileCode<<<nTiles, 32, 0, st>>>(j);
    emark(4);
    if (s->profile) {
        cudaEventSynchronize(s->eev[4]);
        for (int i = 0; i < 4; i++) { float ms; cudaEventElapsedTime(&ms, s->eev[i], s->eev[i + 1]); s->entMs[i] += ms; }
        s->entFrames++;
    }
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

int Analyzer::download(BlockInfo *info, int16_t *levels, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    cudaError_t e = cudaMemcpy(info, s->grid, gridCount() * sizeof(BlockInfo), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(levels, s->outLevels, levelCount() * sizeof(int16_t), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        snprintf(err, errLen, "decision download: %s", cudaGetErrorString(e));
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
    memcpy(d.compMode, pft::Default_Comp_Mode_Cdf, sizeof d.compMode);
    memcpy(d.compRefType, pft::Default_Comp_Ref_Type_Cdf, sizeof d.compRefType);
    memcpy(d.compRef, pft::Default_Comp_Ref_Cdf, sizeof d.compRef);
    memcpy(d.compBwdRef, pft::Default_Comp_Bwd_Ref_Cdf, sizeof d.compBwdRef);
    memcpy(d.compoundMode, pft::Default_Compound_Mode_Cdf, sizeof d.compoundMode);
        memcpy(d.txfmSplit, pft::Default_Txfm_Split_Cdf, sizeof d.txfmSplit);
    return 0;
}

int HostEntropy::code(const EntropyIn &in, const BlockInfo *grid, const int16_t *levels, EntropyOut &out, char *err,
                      int errLen) {
    auto *s = static_cast<HostState *>(state);
    const int miCols = 2 * ((w + 7) >> 3), miRows = 2 * ((h + 7) >> 3);
    const int sbCols = (miCols + 15) >> 4, sbRows = (miRows + 15) >> 4;
    auto maxLog2 = [](int sb) { int k = 0; while ((1 << k) < (sb < 64 ? sb : 64)) ++k; return k; };
    const int cl = std::min(in.tileColsLog2, maxLog2(sbCols)), rl = std::min(in.tileRowsLog2, maxLog2(sbRows));
    const int wSb = (sbCols + (1 << cl) - 1) >> cl, hSb = (sbRows + (1 << rl) - 1) >> rl;
    TileJob j{};
    for (int sb = 0; sb < sbCols; sb += wSb) j.colStart[j.tileCols++] = sb * 16;
    j.colStart[j.tileCols] = miCols;
    for (int sb = 0; sb < sbRows; sb += hSb) j.rowStart[j.tileRows++] = sb * 16;
    j.rowStart[j.tileRows] = miRows;
    const int nTiles = j.tileCols * j.tileRows;
    pft2::FrameSyntax fs{};
    fs.refName[0] = in.refName[0];
    fs.refName[1] = in.refName[1];
    fs.referenceSelect = in.referenceSelect;
    fs.txModeSelect = in.txModeSelect;
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
    j.levels = levels;
    j.mi = s->mi.data();
    j.frameInit = &init;
    j.sbCols = sbCols; j.sbRows = sbRows;
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
    downsample4<<<gds, t16>>>(center[0], aw, w, h, s->tfDs[0], s->dw, s->dh);
    const int nCoarse = s->ccols * s->crows;
    TfArgs a{};
    for (int j = 0; j < count; j++) {
        downsample4<<<gds, t16>>>(neighbors[0][j], aw, w, h, s->tfDs[1 + j], s->dw, s->dh);
        coarseSearch<<<(nCoarse + kCoarseWarps - 1) / kCoarseWarps, 32 * kCoarseWarps>>>(
            s->tfDs[0], s->tfDs[1 + j], s->dw, s->dh, 12, s->ccols, s->crows, 2.f, s->tfCoarse[j]);
        for (int p = 0; p < 3; p++) a.nb[j][p] = neighbors[p][j];
        a.coarse[j] = s->tfCoarse[j];
    }
    for (int p = 0; p < 3; p++) { a.center[p] = center[p]; a.out[p] = out[p]; }
    a.count = count; a.w = w; a.h = h; a.aw = aw; a.ccols = s->ccols; a.crows = s->crows;
    const float step = av1g_ac_q(qindex) / 8.f;
    a.stepPix = step;
    a.step2 = std::max(1.f, step * step * strength);
    temporalFilterKernel<<<dim3((w + 15) / 16, (h + 15) / 16), 256>>>(a);
    for (int p = 0; p < 3; p++) {
        const int pw = p ? w / 2 : w, phh = p ? h / 2 : h, pitch = p ? aw / 2 : aw, pah = p ? ah / 2 : ah;
        padFrame<<<dim3((pitch + 31) / 32, (pah + 7) / 8), dim3(32, 8)>>>(out[p], pitch, pw, phh, pah);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        snprintf(err, errLen, "temporal filter: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

void Analyzer::resetTemporal() {
    if (state) static_cast<State *>(state)->havePrev = false;
}

int Analyzer::filter(const uint8_t *const src[3], const int pitch[3], uint8_t *recon, int qindex, bool cdefAllowed,
                     bool search, FilterParams &out, char *err, int errLen) {
    auto *s = static_cast<State *>(state);
    av1g_frame_params fp{};
    fp.width = w; fp.height = h; fp.chroma_format = AV1G_CHROMA_420; fp.bit_depth = 8;
    fp.enable_cdef = cdefAllowed;
    av1g_pf_frame f;
    av1g_gpu_pf_attach(&fp, s->pfScratch, &f);
    const dim3 t16(16, 16);
    buildFilterMaps<<<dim3((f.MiCols + 15) / 16, (f.MiRows + 15) / 16), t16>>>(s->grid, s->gcols, f);
    buildCdefIdx<<<(f.sbCols * f.sbRows + 63) / 64, 64>>>(s->grid, s->gcols, s->grows, f);
    const int cw = w / 2, ch = h / 2;
    const size_t ysz = size_t(w) * h, csz = size_t(cw) * ch, frame = ysz + 2 * csz;
    uint8_t *planesOf[2][3] = {{recon, recon + ysz, recon + ysz + csz},
                               {s->work, s->work + ysz, s->work + ysz + csz}};
    const int pw[3] = {w, cw, cw}, ph[3] = {h, ch, ch};
    cudaError_t e = cudaSuccess;
    auto sseOf = [&](uint8_t *const planes[3], unsigned long long res[3]) {
        cudaMemsetAsync(s->sse, 0, sizeof(unsigned long long) * 3);
        for (int p = 0; p < 3; p++)
            planeSse<<<dim3((pw[p] + 31) / 32, (ph[p] + 7) / 8), dim3(32, 8)>>>(src[p], pitch[p], planes[p], pw[p], ph[p],
                                                                               s->sse + p);
        e = cudaMemcpy(res, s->sse, sizeof(unsigned long long) * 3, cudaMemcpyDeviceToHost);
    };
    // deblocking level: libaom's inter-frame estimate, optionally refined per plane by SSE
    const int acq = av1g_ac_q(qindex);
    const int guess = std::max(0, std::min(63, (int)(((long)acq * 20723 + 1015158 + (1 << 17)) >> 18)));
    int best[3] = {guess, guess, guess};
    if (search) {
        unsigned long long bestSse[3] = {~0ull, ~0ull, ~0ull};
        const int deltas[] = {-12, -6, 0, 6, 12};
        for (int d : deltas) {
            const int L = std::max(0, std::min(63, guess + d));
            if (d && L == std::max(0, std::min(63, guess + d - (d > 0 ? 6 : -6)))) continue;
            e = cudaMemcpyAsync(s->work, recon, frame, cudaMemcpyDeviceToDevice);
            for (int p = 0; p < 3; p++) { f.plane[p] = planesOf[1][p]; f.stride[p] = pw[p]; }
            for (int k = 0; k < 4; k++) f.lfLevel[k] = L;
            if (L) av1g_gpu_deblock(&f, 0);
            unsigned long long r[3];
            sseOf(planesOf[1], r);
            for (int p = 0; p < 3; p++)
                if (r[p] < bestSse[p]) { bestSse[p] = r[p]; best[p] = L; }
        }
    }
    out.lf[0] = out.lf[1] = best[0];
    out.lf[2] = best[0] ? best[1] : 0;
    out.lf[3] = best[0] ? best[2] : 0;
    out.sharpness = 0;
    for (int p = 0; p < 3; p++) { f.plane[p] = planesOf[0][p]; f.stride[p] = pw[p]; }
    for (int k = 0; k < 4; k++) f.lfLevel[k] = out.lf[k];
    f.sharpness = 0;
    if (out.lf[0] || out.lf[1]) av1g_gpu_deblock(&f, 0);
    out.cdef = cdefAllowed;
    if (cdefAllowed) {
        out.damping = 3 + std::min(3, qindex >> 6);
        f.enableCdef = 1;
        f.cdefDamping = out.damping;
        av1g_gpu_cdef_dir(&f, 0);
        uint8_t *dst[3] = {s->cdefOut, s->cdefOut + ysz, s->cdefOut + ysz + csz};
        const int strides[3] = {w, cw, cw};
        struct Str { int pri, sec; };
        const Str cands[] = {{0, 0}, {1, 0}, {2, 1}, {3, 1}, {4, 2}, {6, 2}, {8, 2}, {10, 4}, {12, 4}};
        int by = 0, buv = 0;
        if (search) {
            unsigned long long bY = ~0ull, bUV = ~0ull;
            for (int k = 0; k < (int)(sizeof cands / sizeof cands[0]); k++) {
                f.cdefYPri[0] = cands[k].pri; f.cdefYSec[0] = cands[k].sec;
                f.cdefUVPri[0] = cands[k].pri; f.cdefUVSec[0] = cands[k].sec;
                av1g_gpu_cdef(&f, dst, strides, 0);
                unsigned long long r[3];
                sseOf(dst, r);
                if (r[0] < bY) { bY = r[0]; by = k; }
                if (r[1] + r[2] < bUV) { bUV = r[1] + r[2]; buv = k; }
            }
        } else {
            by = buv = std::min(8, qindex / 32);
        }
        out.yPri = cands[by].pri; out.ySec = cands[by].sec;
        out.uvPri = cands[buv].pri; out.uvSec = cands[buv].sec;
        f.cdefYPri[0] = out.yPri; f.cdefYSec[0] = out.ySec;
        f.cdefUVPri[0] = out.uvPri; f.cdefUVSec[0] = out.uvSec;
        av1g_gpu_cdef(&f, dst, strides, 0);
        e = cudaMemcpy(recon, s->cdefOut, frame, cudaMemcpyDeviceToDevice);
    }
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame filters: %s", cudaGetErrorString(e));
        return -1;
    }
    return 0;
}

}  // namespace pf
