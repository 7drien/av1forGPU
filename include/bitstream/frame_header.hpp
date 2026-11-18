#pragma once

#include <cstdint>
#include <vector>

namespace av1gpu::bitstream {

struct FrameHeaderConfig {
    int width;
    int height;
};

class FrameHeaderWriter {
public:
    // Emits syntax for a reduced-header key frame with one tile and lossless Q.
    // The matching tile-group OBU must be emitted separately.
    static std::vector<std::uint8_t> makeObu(const FrameHeaderConfig& config);
};

}  // namespace av1gpu::bitstream
