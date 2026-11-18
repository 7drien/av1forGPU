#include "yuv420.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace av1gpu {
namespace {

std::size_t checkedFrameSize(int width, int height) {
    if (width <= 0 || height <= 0 || (width & 1) || (height & 1)) {
        throw std::runtime_error("width and height must be positive even numbers");
    }
    const auto luma = static_cast<std::uint64_t>(width) * height;
    const auto total = luma + luma / 2;
    if (total > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("frame is too large for this platform");
    }
    return static_cast<std::size_t>(total);
}

}  // namespace

Yuv420Frame Yuv420Frame::loadSingleFrame(const std::string& path, int width, int height) {
    const std::size_t expected = checkedFrameSize(width, height);
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path);
    }

    const auto file_size = input.tellg();
    if (file_size < 0 || static_cast<std::uint64_t>(file_size) != expected) {
        throw std::runtime_error("input must contain exactly one tightly packed 8-bit YUV420 frame (" +
                                 std::to_string(expected) + " bytes)");
    }

    Yuv420Frame frame;
    frame.width_ = width;
    frame.height_ = height;
    frame.bytes_.resize(expected);
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(frame.bytes_.data()),
                    static_cast<std::streamsize>(frame.bytes_.size()))) {
        throw std::runtime_error("failed to read complete input frame");
    }
    return frame;
}

Plane8 Yuv420Frame::y() const {
    return {bytes_.data(), width_, height_};
}

Plane8 Yuv420Frame::u() const {
    const auto luma_size = static_cast<std::size_t>(width_) * height_;
    return {bytes_.data() + luma_size, width_ / 2, height_ / 2};
}

Plane8 Yuv420Frame::v() const {
    const auto luma_size = static_cast<std::size_t>(width_) * height_;
    const auto chroma_size = luma_size / 4;
    return {bytes_.data() + luma_size + chroma_size, width_ / 2, height_ / 2};
}

}  // namespace av1gpu
