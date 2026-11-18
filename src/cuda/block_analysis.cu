#include "cuda/block_analysis.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace av1gpu::cuda {
namespace {

constexpr int kBlockSide = 8;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (data_) cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

__global__ void blockStatisticsKernel(const std::uint8_t* plane,
                                      int width,
                                      int height,
                                      BlockStatistics* output) {
    const int lane = threadIdx.y * kBlockSide + threadIdx.x;
    const int x = static_cast<int>(blockIdx.x) * kBlockSide + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * kBlockSide + threadIdx.y;
    const bool valid = x < width && y < height;
    const std::uint32_t value = valid ? plane[y * width + x] : 0;

    std::uint32_t sum = value;
    std::uint32_t squared_sum = value * value;
    std::uint32_t pixel_count = valid ? 1 : 0;

    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(mask, sum, offset);
        squared_sum += __shfl_down_sync(mask, squared_sum, offset);
        pixel_count += __shfl_down_sync(mask, pixel_count, offset);
    }

    __shared__ BlockStatistics warp_totals[2];
    const int warp = lane >> 5;
    if ((lane & 31) == 0) {
        warp_totals[warp] = {sum, squared_sum, pixel_count};
    }
    __syncthreads();

    if (lane == 0) {
        const BlockStatistics a = warp_totals[0];
        const BlockStatistics b = warp_totals[1];
        const int blocks_wide = (width + kBlockSide - 1) / kBlockSide;
        const int output_index = static_cast<int>(blockIdx.y) * blocks_wide + blockIdx.x;
        output[output_index] = {a.sum + b.sum,
                                a.squared_sum + b.squared_sum,
                                a.pixel_count + b.pixel_count};
    }
}

}  // namespace

PlaneActivity analyzePlane8x8(Plane8 plane) {
    if (!plane.data || plane.width <= 0 || plane.height <= 0) {
        throw std::runtime_error("invalid image plane");
    }

    const int blocks_wide = (plane.width + kBlockSide - 1) / kBlockSide;
    const int blocks_high = (plane.height + kBlockSide - 1) / kBlockSide;
    const std::size_t block_count = static_cast<std::size_t>(blocks_wide) * blocks_high;

    DeviceBuffer<std::uint8_t> device_plane(static_cast<std::size_t>(plane.width) * plane.height);
    DeviceBuffer<BlockStatistics> device_stats(block_count);
    std::vector<BlockStatistics> host_stats(block_count);

    checkCuda(cudaMemcpy(device_plane.get(), plane.data,
                         static_cast<std::size_t>(plane.width) * plane.height,
                         cudaMemcpyHostToDevice), "copy input plane to GPU");

    const dim3 threads(kBlockSide, kBlockSide);
    const dim3 blocks(blocks_wide, blocks_high);
    blockStatisticsKernel<<<blocks, threads>>>(device_plane.get(), plane.width, plane.height,
                                               device_stats.get());
    checkCuda(cudaGetLastError(), "launch block statistics kernel");
    checkCuda(cudaMemcpy(host_stats.data(), device_stats.get(),
                         block_count * sizeof(BlockStatistics), cudaMemcpyDeviceToHost),
              "copy block statistics from GPU");

    double mean_sum = 0.0;
    double variance_sum = 0.0;
    for (const auto& stats : host_stats) {
        if (stats.pixel_count == 0) continue;
        const double count = stats.pixel_count;
        const double mean = stats.sum / count;
        const double variance = stats.squared_sum / count - mean * mean;
        mean_sum += mean;
        variance_sum += variance;
    }

    return {blocks_wide, blocks_high,
            mean_sum / block_count,
            variance_sum / block_count};
}

}  // namespace av1gpu::cuda
