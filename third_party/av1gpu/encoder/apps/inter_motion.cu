#include "inter_motion.h"
#include "inter_residual.cuh"

#include <algorithm>

namespace {
// One CTA per superblock. Each lane accumulates errors on Y, U and V for
// seven translations. Warp shuffles reduce each candidate's squared error.
template <bool Sampled>
__global__ void search(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                       const uint8_t *ref, const uint8_t *anchor, int w, int h, int pitch, int modeCount, int step, int blockSize, MotionBlock *out,
                       unsigned long long *sampleScores) {
    if constexpr (Sampled) step = motionStepAt(blockIdx.z);
    const int lane = threadIdx.x, x0 = blockIdx.x * blockSize, y0 = blockIdx.y * blockSize;
    const uint8_t *src[] = {y,u,v};
    unsigned sum[MotionModes] = {};
    int samples = 0;
    for (int p = 0; p < (Sampled ? 1 : 3); ++p) {
        int shift = p != 0, width = (w+shift)>>shift, height = (h+shift)>>shift;
        int bx = x0>>shift, by = y0>>shift;
        int bw = min(blockSize>>shift, width-bx), bh = min(blockSize>>shift,height-by);
        samples += bw*bh;
        constexpr int spacing = Sampled ? 4 : 1;
        const int sampleWidth = (bw + spacing - 1) / spacing;
        const int sampleHeight = (bh + spacing - 1) / spacing;
        for (int i = lane; i < sampleWidth*sampleHeight; i += blockDim.x) {
            int x = bx+(i%sampleWidth)*spacing, yy = by+(i/sampleWidth)*spacing;
            int value = src[p][yy*(pitch>>shift)+x];
            for (int m = 0; m < modeCount; ++m) {
                if (anchor && m == AnchorMode) {
                    const int diff = value - anchor[yy*width+x];
                    sum[m] += diff*diff;
                    continue;
                }
                int xx = min(width-1,max(0,x+motionX(m, step)/(1<<shift)));
                int ry = min(height-1,max(0,yy+motionY(m, step)/(1<<shift)));
                int diff = value-ref[ry*width+xx];
                sum[m] += diff*diff;
            }
        }
        ref += width*height;
        if (anchor) anchor += width*height;
    }
    __shared__ unsigned warpSums[MotionModes][4];
    for (int m = 0; m < modeCount; ++m) {
        for (int off=16;off;off/=2) sum[m] += __shfl_down_sync(0xffffffff,sum[m],off);
        if (!(lane%32)) warpSums[m][lane/32]=sum[m];
    }
    __syncthreads();
    if (!lane) {
        MotionBlock best{~0u,(unsigned)samples,0};
        for(int m=0;m<modeCount;++m) {
            unsigned e=0;
            for(int i=0;i<4;++i) e+=warpSums[m][i];
            best.candidateSse[m]=e;
            if(e<best.sse) {best.sse=e;best.mode=m;}
        }
        if constexpr (Sampled) atomicAdd(sampleScores + blockIdx.z, (unsigned long long)best.sse);
        else out[blockIdx.y*gridDim.x+blockIdx.x]=best;
    }
}
}
cudaError_t searchMotion(const uint8_t *const src[3],const uint8_t *ref,int w,int h,
                         int pitch,int modeCount,int step,MotionBlock *device,MotionBlock *host,const uint8_t *anchor,int blockSize) {
    dim3 grid((w+blockSize-1)/blockSize,(h+blockSize-1)/blockSize);
    search<false><<<grid,128>>>(src[0],src[1],src[2],ref,anchor,w,h,pitch,modeCount,step,blockSize,device,nullptr);
    auto err=cudaGetLastError();
    return err==cudaSuccess ? cudaMemcpy(host,device,grid.x*grid.y*sizeof(MotionBlock),cudaMemcpyDeviceToHost) : err;
}

cudaError_t chooseMotionStep(const uint8_t *source, const uint8_t *ref, const uint8_t *anchor,
                             int width, int height, int pitch, int modeCount,
                             unsigned long long *deviceScores, int &step) {
    unsigned long long scores[MotionStepCount];
    auto status = cudaMemsetAsync(deviceScores, 0, sizeof(scores));
    if (status != cudaSuccess) return status;
    const dim3 grid((width+63)/64, (height+63)/64, MotionStepCount);
    search<true><<<grid,128>>>(source, nullptr, nullptr, ref, anchor, width, height,
                         pitch, modeCount, 0, 64, nullptr, deviceScores);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    status = cudaMemcpy(scores, deviceScores, sizeof(scores), cudaMemcpyDeviceToHost);
    if (status == cudaSuccess)
        step = motionStepAt(std::min_element(scores, scores + MotionStepCount) - scores);
    return status;
}

namespace {
__global__ void reconstruct(const uint8_t *ref, uint8_t *output, int width, int height,
                            const MotionBlock *blocks, int step, const uint8_t *anchor, int blockSize) {
    const int shift = blockIdx.z != 0;
    const int planeWidth = width >> shift, planeHeight = height >> shift;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= planeWidth || y >= planeHeight) return;
    const int cols = (width + blockSize - 1) / blockSize;
    int mode = blocks[((y << shift) / blockSize) * cols + ((x << shift) / blockSize)].mode;
    if (anchor && mode == AnchorMode) { ref = anchor; mode = 0; }
    const int rx = min(planeWidth - 1, max(0, x + motionX(mode, step) / (1 << shift)));
    const int ry = min(planeHeight - 1, max(0, y + motionY(mode, step) / (1 << shift)));
    const int offset = shift ? width * height + (blockIdx.z - 1) * planeWidth * planeHeight : 0;
    output[offset + y * planeWidth + x] = ref[offset + ry * planeWidth + rx];
}
}

cudaError_t reconstructMotion(const uint8_t *ref, uint8_t *output, int width, int height,
                              const MotionBlock *blocks, int step, const uint8_t *anchor, int blockSize) {
    const dim3 threads(32, 8);
    const dim3 grid((width + 31) / 32, (height + 7) / 8, 3);
    reconstruct<<<grid, threads>>>(ref, output, width, height, blocks, step, anchor, blockSize);
    return cudaGetLastError();
}

namespace {
__device__ float residualRate(const int16_t *levels, int count) {
    float bits = 0;
    for (int i = 0; i < count; ++i) {
        unsigned magnitude = abs(int(levels[i]));
        if (magnitude) bits += 3 + 2 * (31 - __clz(magnitude));
    }
    return bits;
}

// Use separate reconstructions to compare the actual clipped spatial error
// for each candidate. Repack the winner into the existing luma coefficient plane.
__global__ void chooseLumaTransform(const uint8_t *source, int pitch, int width,
                                    int height, int qindex, const int16_t *largerLevels,
                                    const uint8_t *largerRecon, int16_t *levels,
                                    uint8_t *reconstruction, uint8_t *txMap) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int cols = width / 8;
    if (id >= cols * (height / 8)) return;
    const int x = (id % cols) * 8, y = (id / cols) * 8;
    const int smallStride = width / 4;
    const int first = (y/4)*smallStride + x/4;
    int sse4 = 0, sse8 = 0;
    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; ++col) {
            const int at = (y+row)*width + x+col;
            const int value = source[(y+row)*pitch+x+col];
            const int d4 = value-reconstruction[at], d8 = value-largerRecon[at];
            sse4 += d4*d4;
            sse8 += d8*d8;
        }
    }
    const int16_t *large = largerLevels + 64*id;
    float bits4 = 2;
    for (int part = 0; part < 4; ++part)
        bits4 += residualRate(levels + 16*(first + (part/2)*smallStride + part%2), 16);
    const float step = av1g_ac_q(qindex) / 8.0f;
    const float lambda = 0.3f*step*step;
    const bool useLarge = sse8 + lambda*residualRate(large, 64) < sse4 + lambda*bits4;
    txMap[id] = useLarge;
    if (!useLarge) return;
    for (int part = 0; part < 4; ++part) {
        int16_t *dst = levels + 16*(first + (part/2)*smallStride + part%2);
        for (int i = 0; i < 16; ++i) dst[i] = large[part*16+i];
    }
    for (int row = 0; row < 8; ++row)
        for (int col = 0; col < 8; ++col) {
            const int at = (y+row)*width + x+col;
            reconstruction[at] = largerRecon[at];
        }
}
} // namespace

cudaError_t prepareInterResiduals(const uint8_t *const source[3], const uint8_t *reference,
                                  int width, int height, int pitch, const MotionBlock *motion,
                                  int16_t *deviceLevels, int16_t *hostLevels, int threshold, uint8_t *reconstructed, int step, int qindex, bool luma8, int16_t *alternateLevels,
                                  uint8_t *alternateRecon, uint8_t *deviceTxMap, uint8_t *hostTxMap, const uint8_t *anchor, bool rdoq, int motionBlockSize) {
    uint32_t offset = 0;
    for (int plane = 0; plane < 3; ++plane) {
        const int shift = plane != 0;
        const int w = width >> shift, h = height >> shift;
        const int blocks = w * h / 16;
        if (luma8 && plane == 0) {
            const int count = w*h/64;
            inter::interResidualKernel<8><<<(count + 127) / 128, 128>>>(
                source[plane], pitch, reference + offset, w, h, plane,
                (width + motionBlockSize - 1) / motionBlockSize, motion, offset, deviceLevels, nullptr, 0,
                reconstructed ? reconstructed + offset : nullptr, step, qindex, anchor ? anchor + offset : nullptr, rdoq, motionBlockSize);
        } else {
            inter::interResidualKernel<4><<<(blocks + 127) / 128, 128>>>(
                source[plane], pitch >> shift, reference + offset, w, h, plane,
                (width + motionBlockSize - 1) / motionBlockSize, motion, offset, deviceLevels, nullptr, threshold, reconstructed ? reconstructed + offset : nullptr, step, qindex, anchor ? anchor + offset : nullptr, rdoq, motionBlockSize);
        }
        auto status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        if (plane == 0 && alternateLevels) {
            const int count = w*h/64;
            inter::interResidualKernel<8><<<(count + 127) / 128, 128>>>(
                source[0], pitch, reference, w, h, 0, (w + motionBlockSize - 1) / motionBlockSize, motion,
                0, alternateLevels, nullptr, 0, alternateRecon, step, qindex, anchor, rdoq, motionBlockSize);
            status = cudaGetLastError();
            if (status != cudaSuccess) return status;
            chooseLumaTransform<<<(count + 127) / 128, 128>>>(
                source[0], pitch, w, h, qindex, alternateLevels, alternateRecon,
                deviceLevels, reconstructed, deviceTxMap);
            status = cudaGetLastError();
            if (status != cudaSuccess) return status;
        }
        offset += w * h;
    }
    if (hostTxMap) {
        auto status = cudaMemcpy(hostTxMap, deviceTxMap, width*height/64, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return status;
    }
    return cudaMemcpy(hostLevels, deviceLevels, offset * sizeof(int16_t), cudaMemcpyDeviceToHost);
}
