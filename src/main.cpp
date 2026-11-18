#include "cuda/block_analysis.cuh"
#include "yuv420.hpp"

#include <cuda_runtime.h>

#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int parseDimension(const char* text, const char* name) {
    std::size_t parsed = 0;
    const std::string value(text);
    const long result = std::stol(value, &parsed);
    if (parsed != value.size() || result <= 0 || result > 65536) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<int>(result);
}

void printActivity(const char* plane_name, av1gpu::Plane8 plane) {
    const auto activity = av1gpu::cuda::analyzePlane8x8(plane);
    std::cout << "  " << plane_name << ": " << activity.block_width << "x"
              << activity.block_height << " 8x8 blocks, average mean="
              << activity.average_mean << ", average variance="
              << activity.average_variance << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: av1gpu-analyze <frame.yuv> <width> <height>\n"
                     "Analyzes one tightly packed 8-bit planar YUV420 frame.\n";
        return 2;
    }

    try {
        const int width = parseDimension(argv[2], "width");
        const int height = parseDimension(argv[3], "height");
        const auto frame = av1gpu::Yuv420Frame::loadSingleFrame(argv[1], width, height);

        int device = 0;
        cudaError_t status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("no usable CUDA device: ") +
                                     cudaGetErrorString(status));
        }
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("cannot query CUDA device: ") +
                                     cudaGetErrorString(status));
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "CUDA device: " << properties.name << " (compute "
                  << properties.major << "." << properties.minor << ")\n"
                  << "Input: " << frame.width() << "x" << frame.height()
                  << " 8-bit YUV420\n";
        printActivity("Y", frame.y());
        printActivity("U", frame.u());
        printActivity("V", frame.v());
    } catch (const std::exception& error) {
        std::cerr << "av1gpu-analyze: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
