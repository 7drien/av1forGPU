// Build from repository root:
// nvcc -std=c++17 -arch=sm_86 -Ithird_party/av1gpu/av1core/include \
//   tests/check_inter_residual.cu -o build-cuda/check-inter-residual
#include "../third_party/av1gpu/encoder/apps/inter_residual.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    fprintf(stderr, "%s\n", cudaGetErrorString(status));
    exit(1);
}

int main() {
    constexpr int width = 70, height = 66, pitch = 80;
    constexpr int blocksWide = (width + 3) / 4;
    constexpr int blockCount = blocksWide * ((height + 3) / 4);
    uint8_t *source, *reference;
    MotionBlock *motion;
    int16_t *levels;
    av1g_txb *txBlocks;
    constexpr int coeffBase = 16;
    check(cudaMallocManaged(&source, pitch * height));
    check(cudaMallocManaged(&reference, width * height));
    check(cudaMallocManaged(&motion, 4 * sizeof(*motion)));
    check(cudaMallocManaged(&levels, (coeffBase + blockCount * 16) * sizeof(*levels)));
    check(cudaMallocManaged(&txBlocks, blockCount * sizeof(*txBlocks)));
    for (int i = 0; i < pitch * height; ++i) source[i] = (i * 43 + i / 19) & 255;
    for (int i = 0; i < width * height; ++i) reference[i] = (i * 19 + i / 7) & 255;
    motion[0].mode = 5;
    motion[1].mode = 1;
    motion[2].mode = 6;
    motion[3].mode = 0;
    inter::interResidualKernel<<<(blockCount + 127) / 128, 128>>>(
        source, pitch, reference, width, height, 0, 2, motion, coeffBase, levels, txBlocks);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    for (int block = 0; block < blockCount; ++block) {
        int32_t dequant[16], residual[16];
        const auto &tx = txBlocks[block];
        if (tx.coeff_off != coeffBase + block * 16 || tx.x4 != block % blocksWide ||
            tx.y4 != block / blocksWide || tx.plane != 0 ||
            tx.tx_size != AV1G_TX_4X4 || tx.tx_type != AV1G_DCT_DCT) return 1;
        const uint16_t *scan = av1g_get_scan(tx.tx_size, tx.tx_type);
        int expectedEob = 0;
        for (int i = 0; i < 16; ++i) {
            dequant[i] = levels[tx.coeff_off + i] * 4;
            if (levels[tx.coeff_off + scan[i]]) expectedEob = i + 1;
        }
        if (expectedEob != tx.eob) return 1;
        uint32_t error = 0;
        av1g_inv_wht4x4(dequant, residual, &error);
        if (error) return 1;
        for (int i = 0; i < 16; ++i) {
            const int x = 4 * (block % blocksWide) + i % 4;
            const int y = 4 * (block / blocksWide) + i / 4;
            if (x >= width || y >= height) {
                if (residual[i] != 0) return 1;
                continue;
            }
            const int mode = motion[(y / 64) * 2 + x / 64].mode;
            const int rx = av1g_clip3(0, width - 1, x + motionX(mode));
            const int ry = av1g_clip3(0, height - 1, y + motionY(mode));
            if (reference[ry * width + rx] + residual[i] != source[y * pitch + x]) return 1;
        }
    }
    check(cudaFree(txBlocks));
    check(cudaFree(levels));
    check(cudaFree(motion));
    check(cudaFree(reference));
    check(cudaFree(source));
    puts("CUDA residual round trip passed");
}
