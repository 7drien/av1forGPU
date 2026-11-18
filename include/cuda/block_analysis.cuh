#pragma once

#include "yuv420.hpp"

#include <cstdint>
#include <vector>

namespace av1gpu::cuda {

struct BlockStatistics {
    std::uint32_t sum;
    std::uint32_t squared_sum;
    std::uint32_t pixel_count;
};

struct PlaneActivity {
    int block_width;
    int block_height;
    double average_mean;
    double average_variance;
};

PlaneActivity analyzePlane8x8(Plane8 plane);

}  // namespace av1gpu::cuda
