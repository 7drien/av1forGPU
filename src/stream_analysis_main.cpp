#include "cuda/block_analysis.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int parseDimension(const char* text, const char* name) {
    std::size_t parsed = 0;
    const std::string value(text);
    const long result = std::stol(value, &parsed);
    if (parsed != value.size() || result <= 0 || result > 65536 || (result & 1)) {
        throw std::runtime_error(std::string("invalid even ") + name + ": " + value);
    }
    return static_cast<int>(result);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: av1gpu-analyze-stream <width> <height> < input.yuv\n"
                     "Reads tightly packed 8-bit planar YUV420 frames from stdin.\n";
        return 2;
    }

    try {
        const int width = parseDimension(argv[1], "width");
        const int height = parseDimension(argv[2], "height");
        const std::uint64_t y_size = static_cast<std::uint64_t>(width) * height;
        const std::uint64_t frame_size_64 = y_size + y_size / 2;
        if (frame_size_64 > std::numeric_limits<std::size_t>::max() ||
            frame_size_64 > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("frame size is too large");
        }
        const auto frame_size = static_cast<std::size_t>(frame_size_64);
        const auto y_size_bytes = static_cast<std::size_t>(y_size);
        const auto chroma_size = y_size_bytes / 4;
        std::vector<std::uint8_t> frame(frame_size);

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

        std::uint64_t frame_count = 0;
        double mean_activity = 0.0;
        double variance_activity = 0.0;
        const auto started = std::chrono::steady_clock::now();
        while (true) {
            std::cin.read(reinterpret_cast<char*>(frame.data()),
                          static_cast<std::streamsize>(frame.size()));
            const auto bytes_read = std::cin.gcount();
            if (bytes_read == 0 && std::cin.eof()) break;
            if (bytes_read != static_cast<std::streamsize>(frame.size())) {
                throw std::runtime_error("input ended in the middle of a YUV420 frame");
            }

            const av1gpu::Plane8 planes[] = {
                {frame.data(), width, height},
                {frame.data() + y_size_bytes, width / 2, height / 2},
                {frame.data() + y_size_bytes + chroma_size, width / 2, height / 2},
            };
            for (const auto& plane : planes) {
                const auto activity = av1gpu::cuda::analyzePlane8x8(plane);
                mean_activity += activity.average_mean;
                variance_activity += activity.average_variance;
            }
            ++frame_count;
        }
        const auto finished = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(finished - started).count();
        if (frame_count == 0) throw std::runtime_error("input contains no frames");

        std::cout << "CUDA device: " << properties.name << " (compute "
                  << properties.major << "." << properties.minor << ")\n"
                  << "Frames: " << frame_count << " at " << width << "x" << height << " YUV420\n"
                  << "Elapsed: " << elapsed << " s (input read + H2D + GPU analysis + D2H)\n"
                  << "Throughput: " << frame_count / elapsed << " frames/s, "
                  << (static_cast<double>(frame_count) * width * height / elapsed / 1.0e6)
                  << " Mpixels/s luma\n"
                  << "Average block mean: " << mean_activity / (frame_count * 3.0)
                  << ", average block variance: "
                  << variance_activity / (frame_count * 3.0) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "av1gpu-analyze-stream: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
