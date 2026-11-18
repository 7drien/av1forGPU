#pragma once

#include "av1g/dsp.h"
#include "av1g/decisions.h"
#include "inter_motion.h"

namespace inter {

// Each thread transforms one block. Levels are stored in raster block
// order, with row-major coefficients within each block.
template <int N = 4>
static __global__ void interResidualKernel(
    const uint8_t *source, int pitch, const uint8_t *reference,
    int width, int height, int plane, int motionCols,
    const MotionBlock *motion, uint32_t coeffBase, int16_t *levels, av1g_txb *txBlocks, int threshold = 0, uint8_t *reconstructed = nullptr, int step = 2, int qindex = 0, const uint8_t *anchor = nullptr, bool rdoq = false, int motionBlockSize = 64) {
    const int blocksWide = (width + N - 1) / N;
    const int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= blocksWide * ((height + N - 1) / N)) return;

    const int chromaShift = plane != 0;
    const int x0 = N * (block % blocksWide);
    const int y0 = N * (block / blocksWide);
    const int motionBlock = ((y0 << chromaShift) / motionBlockSize) * motionCols
                          + ((x0 << chromaShift) / motionBlockSize);
    int mode = motion[motionBlock].mode;
    if (anchor && mode == AnchorMode) { reference = anchor; mode = 0; }
    const int dx = motionX(mode, step) / (1 << chromaShift);
    const int dy = motionY(mode, step) / (1 << chromaShift);
    int32_t residual[N*N];
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            const int x = x0 + col;
            const int y = y0 + row;
            if (x >= width || y >= height) {
                residual[row * N + col] = 0;
                continue;
            }
            const int rx = min(width - 1, max(0, x + dx));
            const int ry = min(height - 1, max(0, y + dy));
            residual[row * N + col] = int(source[y * pitch + x]) - reference[ry * width + rx];
        }
    }
    const uint32_t coeffOffset = coeffBase + N*N * block;
    int16_t *coefficients = levels + coeffOffset;
    if (qindex) {
        // Orthonormal 4x4 DCT, scaled to AV1's inverse-transform domain.
        constexpr float basis[4][4] = {
            {0.5f, 0.5f, 0.5f, 0.5f},
            {0.65328148f, 0.27059805f, -0.27059805f, -0.65328148f},
            {0.5f, -0.5f, -0.5f, 0.5f},
            {0.27059805f, -0.65328148f, 0.65328148f, -0.27059805f}
        };
        constexpr float basis8[8][8] = {
            {0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f, 0.353553391f},
            {0.490392640f, 0.415734806f, 0.277785117f, 0.097545161f, -0.097545161f, -0.277785117f, -0.415734806f, -0.490392640f},
            {0.461939766f, 0.191341716f, -0.191341716f, -0.461939766f, -0.461939766f, -0.191341716f, 0.191341716f, 0.461939766f},
            {0.415734806f, -0.097545161f, -0.490392640f, -0.277785117f, 0.277785117f, 0.490392640f, 0.097545161f, -0.415734806f},
            {0.353553391f, -0.353553391f, -0.353553391f, 0.353553391f, 0.353553391f, -0.353553391f, -0.353553391f, 0.353553391f},
            {0.277785117f, -0.490392640f, 0.097545161f, 0.415734806f, -0.415734806f, -0.097545161f, 0.490392640f, -0.277785117f},
            {0.191341716f, -0.461939766f, 0.461939766f, -0.191341716f, -0.191341716f, 0.461939766f, -0.461939766f, 0.191341716f},
            {0.097545161f, -0.277785117f, 0.415734806f, -0.490392640f, 0.490392640f, -0.415734806f, 0.277785117f, -0.097545161f}
        };
        auto weight = [&](int k, int x) { return N == 4 ? basis[k][x] : basis8[k][x]; };
        float rows[N*N]{};
        for (int y = 0; y < N; ++y)
            for (int u = 0; u < N; ++u)
                for (int x = 0; x < N; ++x)
                    rows[y*N+u] += residual[y*N+x]*weight(u,x);
        constexpr int txSize = N == 4 ? AV1G_TX_4X4 : AV1G_TX_8X8;
        const float scale = float(1 << (4 + AV1G_T(Transform_Row_Shift)[txSize])) * 2 / N;
        for (int v = 0; v < N; ++v) {
            for (int u = 0; u < N; ++u) {
                float coefficient = 0;
                for (int y = 0; y < N; ++y) coefficient += weight(v,y)*rows[y*N+u];
                const int stepQ = (u || v) ? av1g_ac_q(qindex) : av1g_dc_q(qindex);
                const float level = coefficient*scale/stepQ;
                int quantized = __float2int_rn(level);
                if (rdoq && quantized) {
                    // Compare in quantizer-step units. Zero avoids emitting the
                    // magnitude and sign, and can shorten the end of block.
                    const float magnitude = fabsf(level);
                    const int nearest = abs(quantized);
                    const float lambda = (u || v) ? 0.12f : 0.035f;
                    auto cost = [&](int candidate) {
                        const float error = magnitude - candidate;
                        const int bits = candidate ? 3 + 2*(31-__clz(unsigned(candidate))) : 0;
                        return error*error + lambda*bits;
                    };
                    int best = nearest;
                    if (cost(nearest-1) < cost(best)) best = nearest-1;
                    if (cost(0) < cost(best)) best = 0;
                    quantized = level < 0 ? -best : best;
                }
                coefficients[v*N+u] = quantized;
            }
        }
    } else if constexpr (N == 4) {
        av1g_fwht4x4(residual, coefficients);
    }
    // The bitstream still uses the lossless transform. Dropping small
    // levels is an encoder decision and makes the prediction lossy.
    if (threshold) {
        for (int i = 0; i < N*N; ++i) {
            const int limit = i == 0 ? threshold / 2 : threshold;
            if (abs(int(coefficients[i])) <= limit) coefficients[i] = 0;
        }
    }
    if (threshold || reconstructed) {
        int32_t dequant[N*N], correction[N*N];
        for (int i = 0; i < N*N; ++i) {
            const int stepQ = i ? av1g_ac_q(qindex) : av1g_dc_q(qindex);
            dequant[i] = coefficients[i] * stepQ;
        }
        uint32_t error = 0;
        if (qindex) av1g_inv_txfm2d(dequant, N == 4 ? AV1G_TX_4X4 : AV1G_TX_8X8, AV1G_DCT_DCT, correction, &error);
        else av1g_inv_wht4x4(dequant, correction, &error);
        if (error && !qindex) {
            // A pruned transform can exceed the decoder's legal residual range.
            av1g_fwht4x4(residual, coefficients);
            for (int i = 0; i < N*N; ++i) correction[i] = residual[i];
        }
        if (reconstructed) {
            for (int row = 0; row < N; ++row) {
                for (int col = 0; col < N; ++col) {
                    const int x = x0 + col, y = y0 + row;
                    if (x >= width || y >= height) continue;
                    const int i = row * N + col;
                    const int predicted = int(source[y * pitch + x]) - residual[i];
                    reconstructed[y * width + x] = av1g_clip1<8>(predicted + correction[i]);
                }
            }
        }
    }
    if (!txBlocks) return;
    const uint16_t *scan = av1g_get_scan(N == 4 ? AV1G_TX_4X4 : AV1G_TX_8X8, AV1G_DCT_DCT);
    int eob = N*N;
    while (eob && !coefficients[scan[eob - 1]]) --eob;
    av1g_txb tx{};
    tx.coeff_off = coeffOffset;
    tx.x4 = x0 / 4;
    tx.y4 = y0 / 4;
    tx.plane = plane;
    tx.tx_size = N == 4 ? AV1G_TX_4X4 : AV1G_TX_8X8;
    tx.tx_type = AV1G_DCT_DCT;
    tx.eob = eob;
    txBlocks[block] = tx;
}

} // namespace inter
