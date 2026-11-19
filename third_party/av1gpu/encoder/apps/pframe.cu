// CUDA P-frame analysis (see pframe.h).
#include "pframe.h"

#include "dev_common.cuh"
#include "dev_tx.cuh"
#include "rc_inv.cuh"
#include "inter_tables.h"
#include "av1g/gpu_postfilter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_pframe)
const GpHostTables *gp_host_tables();

namespace pf {
namespace {

constexpr int kSizes = 4;       // 8, 16, 32, 64
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

template <int N>
__device__ __forceinline__ int warpSadPlane(const uint8_t *src, int pitch, const uint8_t *plane, int w, int h, int x0,
                                            int y0, int dx, int dy) {
    constexpr int L = N == 8 ? 3 : N == 16 ? 4 : N == 32 ? 5 : 6;
    const int lane = threadIdx.x & 31;
    int s = 0;
    const bool inside = x0 + dx >= 0 && y0 + dy >= 0 && x0 + dx + N <= w && y0 + dy + N <= h && x0 + N <= w && y0 + N <= h;
    if (inside) {
        const uint8_t *p = plane + size_t(y0 + dy) * w + x0 + dx;
        const uint8_t *q = src + size_t(y0) * pitch + x0;
#pragma unroll 4
        for (int i = lane; i < N * N; i += 32) {
            const int r = i >> L, c = i & (N - 1);
            s += abs((int)q[r * pitch + c] - (int)p[r * w + c]);
        }
    } else {
        for (int i = lane; i < N * N; i += 32) {
            const int r = i >> L, c = i & (N - 1);
            const int x = x0 + c, y = y0 + r;
            if (x >= w || y >= h) continue;
            const int rx = min(max(x + dx, 0), w - 1), ry = min(max(y + dy, 0), h - 1);
            s += abs((int)src[y * pitch + x] - (int)plane[size_t(ry) * w + rx]);
        }
    }
    return warp_sumi(s);
}

// Per-size motion refinement from coarse, parent, temporal and zero candidates: full-pel search, then half
// and quarter sample steps on the phase planes. ref r > 0 replaces a candidate only if cheaper.
template <int N>
struct RefineCfg {
    static constexpr int warps = N >= 64 ? 2 : N >= 32 ? 4 : 8;
};
template <int N>
__global__ void __launch_bounds__(32 * RefineCfg<N>::warps) refine(Frame f, int refIdx, const uint8_t *phases,
                                                                  const Mv *coarse, int ccols, int crows,
                                                                  const Cand *parent, int pcols,
                                                                  const BlockInfo *temporal, int tcols, Cand *out,
                                                                  int cols, int rows, float lambda) {
    constexpr int W = RefineCfg<N>::warps;
    const int warp = threadIdx.x >> 5;
    const int b = blockIdx.x * W + warp;
    if (b >= cols * rows) return;
    const int bx = b % cols, by = b / cols, x0 = bx * N, y0 = by * N;
    Cand &o = out[b];
    if (x0 >= f.w || y0 >= f.h) {
        if ((threadIdx.x & 31) == 0) { o.mvR[refIdx] = Mv{0, 0}; o.costR[refIdx] = kInf; o.bitsR[refIdx] = 0.f; }
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
    const int cx = min((x0 + N / 2) / 32, ccols - 1), cy = min((y0 + N / 2) / 32, crows - 1);
    const Mv center = coarse[cy * ccols + cx];
    add(center);
    add(Mv{0, 0});
    if (cx > 0) add(coarse[cy * ccols + cx - 1]);
    if (cx + 1 < ccols) add(coarse[cy * ccols + cx + 1]);
    if (cy > 0) add(coarse[(cy - 1) * ccols + cx]);
    if (cy + 1 < crows) add(coarse[(cy + 1) * ccols + cx]);
    Mv pred = center;
    if (parent) {
        const Cand &p = parent[(y0 / (2 * N)) * pcols + x0 / (2 * N)];
        add(p.mvR[refIdx]);
        pred = p.mvR[refIdx];
    }
    if (temporal) add(temporal[(y0 / 8) * tcols + x0 / 8].mv);
    auto costOf = [&](Mv m) {
        const int ph = (((2 * m.row) & 15) >> 2) * 4 + (((2 * m.col) & 15) >> 2);
        const int s = warpSadPlane<N>(src, pitch, phases + ph * planeSize, f.w, f.h, x0, y0, m.col >> 3, m.row >> 3);
        return (float)s + lambda * mvBits(m, pred);
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
    if ((threadIdx.x & 31) == 0) {
        o.mvR[refIdx] = bm;
        o.costR[refIdx] = bc;
        o.bitsR[refIdx] = mvBits(bm, pred);
    }
}

// Choose single reference 0, single reference 1, or their compound average by SAD + motion cost on the phase planes.
template <int N>
__global__ void __launch_bounds__(32 * RefineCfg<N>::warps) chooseInter(Frame f, int refCount, const uint8_t *phases0,
                                                                       const uint8_t *phases1, Cand *cand, int cols,
                                                                       int rows, float lambda, float compoundBias) {
    constexpr int W = RefineCfg<N>::warps;
    constexpr int L = N == 8 ? 3 : N == 16 ? 4 : N == 32 ? 5 : 6;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * W + warp;
    if (b >= cols * rows) return;
    Cand &c = cand[b];
    const int x0 = (b % cols) * N, y0 = (b / cols) * N;
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
        for (int i = lane; i < N * N; i += 32) {
            const int r = i >> L, col = i & (N - 1);
            const int x = x0 + col, y = y0 + r;
            if (x >= f.w || y >= f.h) continue;
            const int a = p0[size_t(min(max(y + dy0, 0), f.h - 1)) * f.w + min(max(x + dx0, 0), f.w - 1)];
            const int bb = p1[size_t(min(max(y + dy1, 0), f.h - 1)) * f.w + min(max(x + dx1, 0), f.w - 1)];
            sad += abs((int)f.src[0][y * f.pitch[0] + x] - ((a + bb + 1) >> 1));
        }
        sad = warp_sumi(sad);
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
// Exact inverse transform + add (DCT_DCT, square N <= 32), warp-cooperative. Halves the levels while the
// coefficients violate the conformance ranges. recon = Clip1(pred + residual).
template <int LN>
__device__ void warpInverseAdd(int16_t *q, int txSz, int dcQ, int acQ, int16_t *res, const uint8_t *pred, uint8_t *recon) {
    constexpr int N = 1 << LN, T = N > 32 ? 32 : N;   // T x T stored coefficients
    const int lane = threadIdx.x & 31;
    const int den = av1g_dq_denom(txSz);
    const int rk = av1g_row_kind(AV1G_DCT_DCT), ck = av1g_col_kind(AV1G_DCT_DCT);
    const int rowShift = AV1G_T(Transform_Row_Shift)[txSz];
    for (;;) {
        int any = 0;
        for (int i = lane; i < T * T; i += 32) any |= q[i];
        if (!__any_sync(0xffffffffu, any)) {
            for (int i = lane; i < N * N; i += 32) recon[i] = pred[i];
            __syncwarp();
            return;
        }
        uint32_t err = 0;
        for (int i = lane; i < N; i += 32) err |= rci::row<LN, 8>(i, T, T, q, dcQ, acQ, nullptr, den, false, rk, rowShift, res);
        __syncwarp();
        for (int j = lane; j < N; j += 32) err |= rci::col<LN>(j, N, ck, res);
        __syncwarp();
        if (!__any_sync(0xffffffffu, err != 0)) break;
        for (int i = lane; i < T * T; i += 32) q[i] = (int16_t)(q[i] / 2);
        __syncwarp();
    }
    for (int i = lane; i < N * N; i += 32) recon[i] = (uint8_t)min(max((int)pred[i] + res[i], 0), 255);
    __syncwarp();
}

template <int N>
struct EvalCfg {
    static constexpr int warps = N >= 64 ? 1 : N >= 32 ? 1 : N >= 16 ? 2 : 4;
};

struct EvalArgs {
    Frame f;
    Cand *cand;
    int cols, rows;
    uint8_t *pred, *recon;   // per-size frame-sized buffers (tight YUV420)
    int16_t *levels;          // per-size tile-layout levels (aligned planes)
    int qindex, qctx;
    float lambda, rnd, zbin;
    float mvWeight;
};

// Evaluate skip and coded options of every block of size N.
template <int N>
__global__ void __launch_bounds__(32 * EvalCfg<N>::warps) evaluate(EvalArgs a) {
    constexpr int W = EvalCfg<N>::warps;
    constexpr int NC = N / 2;
    constexpr bool coded = true;
    constexpr int TN = N, TQ = N > 32 ? 32 : N;   // transform size, stored coefficients per side
    __shared__ int16_t interp[W][(N + 7) * N];
    __shared__ int16_t hpS[W][N * N];
    __shared__ uint8_t predS[W][N * N + 2 * NC * NC];
    __shared__ uint8_t recS[W][coded ? N * N : 1];
    __shared__ int16_t resS[W][TN * TN];
    __shared__ float tmpS[W][TQ * (TN + 1)];
    __shared__ int16_t qS[W][TQ * TQ];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int b = blockIdx.x * W + warp;
    if (b >= a.cols * a.rows) return;
    Cand &c = a.cand[b];
    const int bx = b % a.cols, by = b / a.cols, x0 = bx * N, y0 = by * N;
    const Frame &f = a.f;
    if (x0 >= f.w || y0 >= f.h) {
        if (!lane) { c.costSkip = c.costCoded = kInf; c.nz = 0; }
        return;
    }
    const uint8_t *ref = c.ref == 1 ? f.ref[1] : f.ref[0];
    const uint8_t *ref2 = f.ref[1];
    const int cw = f.w >> 1, chh = f.h >> 1;
    const size_t ysz = size_t(f.w) * f.h, csz = size_t(cw) * chh;
    const int dcQ = av1g_dc_q(a.qindex), acQ = av1g_ac_q(a.qindex);
    float dSkip = 0.f, dCoded = 0.f, bits = 0.f;
    bool nz = false;
#pragma unroll
    for (int p = 0; p < 3; p++) {
        const int sub = p > 0, n = p ? NC : N;
        const int pw = p ? cw : f.w, ph = p ? chh : f.h;
        const int px = x0 >> sub, py = y0 >> sub;
        uint8_t *pred = predS[warp] + (p ? N * N + (p - 1) * NC * NC : 0);
        const uint8_t *refP = ref + (p ? ysz + (p - 1) * csz : 0);
        if (c.ref == 2)
            warpPredictCompound(refP, ref2 + (p ? ysz + (p - 1) * csz : 0), pw, ph, px, py, n, n, c.mv, c.mv2, sub,
                                interp[warp], hpS[warp], pred);
        else
            warpPredict(refP, pw, ph, px, py, n, n, c.mv, sub, interp[warp], pred);
        const uint8_t *src = f.src[p];
        const int pitch = f.pitch[p];
        int sse = 0;
        for (int i = lane; i < n * n; i += 32) {
            const int r = i / n, col = i - r * n;
            if (px + col >= pw || py + r >= ph) continue;
            const int d = (int)src[(py + r) * pitch + px + col] - pred[i];
            sse += d * d;
        }
        sse = warp_sumi(sse);
        dSkip += (float)sse;
        // store the prediction (skip reconstruction)
        uint8_t *outPred = a.pred + (p ? ysz + (p - 1) * csz : 0);
        for (int i = lane; i < n * n; i += 32) {
            const int r = i / n, col = i - r * n;
            if (px + col < pw && py + r < ph) outPred[(py + r) * pw + px + col] = pred[i];
        }
        if constexpr (coded) {
            const int tn = n;
            int16_t *res = resS[warp];
            float energy = 0.f;
            for (int i = lane; i < tn * tn; i += 32) {
                const int r = i / tn, col = i - r * tn;
                int d = 0;
                if (px + col < pw && py + r < ph) d = (int)src[(py + r) * pitch + px + col] - pred[i];
                res[i] = (int16_t)d;
                energy += (float)(d * d);
            }
            __syncwarp();
            const int txSz = tn == 4 ? AV1G_TX_4X4 : tn == 8 ? AV1G_TX_8X8 : tn == 16 ? AV1G_TX_16X16
                           : tn == 32 ? AV1G_TX_32X32 : AV1G_TX_64X64;
            const int tq = tn > 32 ? 32 : tn;
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
            warp_fwd_rt(res, tmpS[warp], tn, tn, txSz, AV1G_DCT_DCT);
            warp_quant_rate_rt<8>(tmpS[warp], q, e, o, tn, tn);
            bits += o.R;
            uint8_t *rec = recS[warp];
            switch (tn) {
            case 4: warpInverseAdd<2>(q, txSz, dcQ, acQ, res, pred, rec); break;
            case 8: warpInverseAdd<3>(q, txSz, dcQ, acQ, res, pred, rec); break;
            case 16: warpInverseAdd<4>(q, txSz, dcQ, acQ, res, pred, rec); break;
            case 32: warpInverseAdd<5>(q, txSz, dcQ, acQ, res, pred, rec); break;
            default: warpInverseAdd<6>(q, txSz, dcQ, acQ, res, pred, rec); break;
            }
            int any = 0;
            for (int i = lane; i < tq * tq; i += 32) any |= q[i];
            nz = nz || __any_sync(0xffffffffu, any);
            int sseC = 0;
            uint8_t *outRec = a.recon + (p ? ysz + (p - 1) * csz : 0);
            for (int i = lane; i < tn * tn; i += 32) {
                const int r = i / tn, col = i - r * tn;
                if (px + col >= pw || py + r >= ph) continue;
                const int d = (int)src[(py + r) * pitch + px + col] - rec[i];
                sseC += d * d;
                outRec[(py + r) * pw + px + col] = rec[i];
            }
            dCoded += (float)warp_sumi(sseC);
            // levels, tile layout of the aligned plane
            const int apw = p ? f.aw / 2 : f.aw;
            const size_t base = p ? size_t(f.aw) * f.ah + size_t(p - 1) * (f.aw / 2) * (f.ah / 2) : 0;
            // the tn x tn tile holds the stored tq x tq coefficients in its top-left corner (zeros elsewhere)
            int16_t *dst = a.levels + base + (size_t(py / tn) * (apw / tn) + px / tn) * tn * tn;
            for (int i = lane; i < tn * tn; i += 32) {
                const int r = i / tn, col = i - r * tn;
                dst[i] = (r < tq && col < tq) ? q[r * tq + col] : 0;
            }
            __syncwarp();
        }
    }
    if (!lane) {
        // mode + vector bits: the search cost already holds lambda_sad * mv bits; approximate in bits here
        const float modeBits = 2.f + c.mvBits * a.mvWeight;
        const float skipFlag1 = 0.5f, skipFlag0 = 1.5f;
        c.costSkip = dSkip + a.lambda * (modeBits + skipFlag1);
        c.costCoded = dCoded + a.lambda * (modeBits + skipFlag0 + bits);
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
    uint8_t *mode[kSizes];    // 0: split, 1: none + skip, 2: none + coded
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
    const bool useNone = none <= split;
    a.best[s][i] = useNone ? none : split;
    a.mode[s][i] = useNone ? (uint8_t)noneMode : 0;
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
            if (m || s == 0) {
                const Cand &c = a.cand[s][idx];
                bi.mv = c.mv;
                bi.mv2 = c.mv2;
                bi.log2 = (uint8_t)(3 + s);
                bi.skip = (uint8_t)(m != 2);
                bi.ref = c.ref;
                break;
            }
        }
    }
    a.grid[gy * a.gcols + gx] = bi;
}

// Gather the chosen reconstruction and levels.
struct CommitArgs {
    const BlockInfo *grid;
    int gcols, w, h, aw, ah;
    const uint8_t *pred[kSizes], *recon[kSizes];
    const int16_t *levels[kSizes];
    uint8_t *out;
    int16_t *outLevels;
};
template <class T>
__device__ __forceinline__ T pick4(int s, T a0, T a1, T a2, T a3) { return s == 0 ? a0 : s == 1 ? a1 : s == 2 ? a2 : a3; }
__global__ void commit(CommitArgs a) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    const int p = blockIdx.z, sub = p > 0;
    const int apw = a.aw >> sub, aph = a.ah >> sub;
    if (x >= apw || y >= aph) return;
    const BlockInfo bi = a.grid[((y << sub) >> 3) * a.gcols + ((x << sub) >> 3)];
    const size_t lbase = p ? size_t(a.aw) * a.ah + size_t(p - 1) * (a.aw / 2) * (a.ah / 2) : 0;
    const int s = bi.log2 - 3;
    int16_t level = 0;
    if (bi.log2 && !bi.skip) {
        const int lt = bi.log2 - sub, t = 1 << lt;
        const int tx = x >> lt, ty = y >> lt;
        const int16_t *lv = pick4(s, a.levels[0], a.levels[1], a.levels[2], a.levels[3]);
        level = lv[lbase + ((size_t(ty) * (apw >> lt) + tx) << (2 * lt)) + ((y - (ty << lt)) << lt) + (x - (tx << lt))];
    }
    a.outLevels[lbase + size_t(y) * apw + x] = level;
    const int pw = a.w >> sub, ph = a.h >> sub;
    if (x >= pw || y >= ph) return;
    const size_t off = (p ? size_t(a.w) * a.h + size_t(p - 1) * pw * ph : 0) + size_t(y) * pw + x;
    const uint8_t *srcBuf = bi.skip ? pick4(s, a.pred[0], a.pred[1], a.pred[2], a.pred[3])
                                    : pick4(s, a.recon[0], a.recon[1], a.recon[2], a.recon[3]);
    a.out[off] = srcBuf[off];
}

// ------------------------------------------------------------------------------------------------
// In-loop filter maps (spec Skips, LoopfilterTxSizes, cdef_idx) from the decision grid. Every transform equals
// its block, so all transform edges are block edges and the intra-frame filter rules apply unchanged.
__global__ void buildFilterMaps(const BlockInfo *grid, int gcols, av1g_pf_frame f) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= f.MiCols || r >= f.MiRows) return;
    const BlockInfo bi = grid[(r >> 1) * gcols + (c >> 1)];
    const size_t k = size_t(r) * f.MiCols + c;
    f.skip[k] = bi.skip;
    f.lfTx[0][k] = (uint8_t)(bi.log2 - 2);
    if (!(r & 1) && !(c & 1)) {
        const int cc = f.MiCols >> 1;
        const size_t kc = size_t(r >> 1) * cc + (c >> 1);
        f.lfTx[1][kc] = f.lfTx[2][kc] = (uint8_t)(bi.log2 - 3);
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

struct State {
    uint8_t *ds[1 + kMaxRefs] = {};
    Mv *coarse[kMaxRefs] = {};
    Cand *cand[kSizes] = {};
    uint8_t *pred[kSizes] = {}, *recon[kSizes] = {};
    int16_t *levels[kSizes] = {}, *outLevels = nullptr;
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
    int16_t *hphase = nullptr;
    uint8_t *phases[kMaxRefs] = {};
    bool profile = false;
    cudaEvent_t ev[8] = {};
    double stageMs[8] = {};
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
    cudaFree(s->outLevels);
    cudaFree(s->grid);
    cudaFree(s->prevGrid);
    cudaFree(s->pfScratch);
    cudaFree(s->work);
    cudaFree(s->cdefOut);
    cudaFree(s->sse);
    cudaFree(s->hphase);
    for (auto *p : s->best) cudaFree(p);
    for (auto *p : s->mode) cudaFree(p);
    for (auto *p : s->phases) cudaFree(p);
    if (s->profile && s->frames) {
        const char *names[] = {"downsample+coarse", "refine", "evaluate", "decide", "commit", "download"};
        for (int i = 0; i < 6; i++) fprintf(stderr, "pframe %-18s %7.2f ms/frame\n", names[i], s->stageMs[i] / s->frames);
    }
    for (auto &e : s->ev) if (e) cudaEventDestroy(e);
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
    if (e == cudaSuccess) e = cudaMemcpyToSymbol(c_filter, pft::Subpel_Filters, sizeof(pft::Subpel_Filters));
    if (e != cudaSuccess) {
        snprintf(err, errLen, "P-frame allocation: %s", cudaGetErrorString(e));
        return -1;
    }
    s->profile = getenv("AV1GPU_PF_PROFILE") != nullptr;
    if (s->profile)
        for (auto &ev : s->ev) cudaEventCreate(&ev);
    if (gp_upload_tables_pframe(gp_host_tables())) {
        snprintf(err, errLen, "P-frame table upload failed");
        return -1;
    }
    return 0;
}

int Analyzer::analyze(const uint8_t *const src[3], const int pitch[3], const uint8_t *const *refs, int refCount,
                      const Settings &st, uint8_t *recon, std::vector<BlockInfo> &info, std::vector<int16_t> &levels,
                      char *err, int errLen) {
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
    refine<N><<<(s->cols[K] * s->rows[K] + RefineCfg<N>::warps - 1) / RefineCfg<N>::warps,                      \
                32 * RefineCfg<N>::warps>>>(f, r, s->phases[r], s->coarse[r], s->ccols, s->crows, K == 3 ? nullptr : s->cand[K + 1], \
                                            K == 3 ? 0 : s->cols[K + 1], temporal, s->gcols, s->cand[K], s->cols[K],  \
                                            s->rows[K], lambdaSad)
        PF_REFINE(3, 64);
        PF_REFINE(2, 32);
        PF_REFINE(1, 16);
        PF_REFINE(0, 8);
#undef PF_REFINE
    }
#define PF_CHOOSE(K, N)                                                                                         \
    chooseInter<N><<<(s->cols[K] * s->rows[K] + RefineCfg<N>::warps - 1) / RefineCfg<N>::warps,                 \
                     32 * RefineCfg<N>::warps>>>(f, refCount, s->phases[0], s->phases[1], s->cand[K], s->cols[K], \
                                                 s->rows[K], lambdaSad, st.compoundBias)
    PF_CHOOSE(3, 64);
    PF_CHOOSE(2, 32);
    PF_CHOOSE(1, 16);
    PF_CHOOSE(0, 8);
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
#define PF_EVAL(K, N)                                                                                           \
    ea.cand = s->cand[K]; ea.cols = s->cols[K]; ea.rows = s->rows[K];                                          \
    ea.pred = s->pred[K]; ea.recon = s->recon[K]; ea.levels = s->levels[K];                                   \
    evaluate<N><<<(s->cols[K] * s->rows[K] + EvalCfg<N>::warps - 1) / EvalCfg<N>::warps, 32 * EvalCfg<N>::warps>>>(ea)
    PF_EVAL(0, 8);
    PF_EVAL(1, 16);
    PF_EVAL(2, 32);
    PF_EVAL(3, 64);
#undef PF_EVAL
    mark(3);
    DecideArgs da{};
    for (int k = 0; k < kSizes; k++) {
        da.cand[k] = s->cand[k]; da.cols[k] = s->cols[k]; da.best[k] = s->best[k]; da.mode[k] = s->mode[k];
    }
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
    for (int k = 0; k < kSizes; k++) ca.pred[k] = s->pred[k];
    for (int k = 0; k < kSizes; k++) { ca.recon[k] = s->recon[k]; ca.levels[k] = s->levels[k]; }
    ca.out = recon;
    ca.outLevels = s->outLevels;
    commit<<<dim3((aw + 15) / 16, (ah + 15) / 16, 3), t16>>>(ca);
    mark(5);
    cudaError_t e = cudaGetLastError();
    info.resize(size_t(s->gcols) * s->grows);
    levels.resize(size_t(aw) * ah * 3 / 2);
    if (e == cudaSuccess) e = cudaMemcpy(info.data(), s->grid, info.size() * sizeof(BlockInfo), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(levels.data(), s->outLevels, levels.size() * sizeof(int16_t), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(s->prevGrid, s->grid, info.size() * sizeof(BlockInfo), cudaMemcpyDeviceToDevice);
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
