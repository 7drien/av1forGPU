#pragma once

#include <cstdint>
#include <vector>

namespace av1gpu::bitstream {

enum class SequenceMode {
    StillPicture,
    Video,
};

struct SequenceHeaderConfig {
    int width;
    int height;
    std::uint8_t level_index;
    SequenceMode mode;
};

class SequenceHeaderWriter {
public:
    static std::vector<std::uint8_t> makePayload(const SequenceHeaderConfig& config);
    static std::vector<std::uint8_t> makeObu(const SequenceHeaderConfig& config);
};

}  // namespace av1gpu::bitstream
