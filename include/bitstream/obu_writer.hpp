#pragma once

#include <cstdint>
#include <vector>

namespace av1gpu::bitstream {

enum class ObuType : std::uint8_t {
    Reserved = 0,
    SequenceHeader = 1,
    TemporalDelimiter = 2,
    FrameHeader = 3,
    TileGroup = 4,
    Metadata = 5,
    Frame = 6,
    RedundantFrameHeader = 7,
    TileList = 8,
    Padding = 15,
};

class ObuWriter {
public:
    static std::vector<std::uint8_t> wrap(ObuType type,
                                         const std::vector<std::uint8_t>& payload,
                                         bool has_extension = false,
                                         std::uint8_t temporal_id = 0,
                                         std::uint8_t spatial_id = 0);
};

}  // namespace av1gpu::bitstream
