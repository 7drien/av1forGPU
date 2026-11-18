#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace av1gpu {

struct Plane8 {
    const std::uint8_t* data;
    int width;
    int height;
};

class Yuv420Frame {
public:
    static Yuv420Frame loadSingleFrame(const std::string& path, int width, int height);

    Plane8 y() const;
    Plane8 u() const;
    Plane8 v() const;

    int width() const { return width_; }
    int height() const { return height_; }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<std::uint8_t> bytes_;
};

}  // namespace av1gpu
