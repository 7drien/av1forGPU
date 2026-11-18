#include "bitstream/obu_writer.hpp"

#include "bitstream/bit_writer.hpp"

#include <stdexcept>

namespace av1gpu::bitstream {

std::vector<std::uint8_t> ObuWriter::wrap(ObuType type,
                                          const std::vector<std::uint8_t>& payload,
                                          bool has_extension,
                                          std::uint8_t temporal_id,
                                          std::uint8_t spatial_id) {
    const auto type_value = static_cast<std::uint8_t>(type);
    if (type_value > 15) {
        throw std::invalid_argument("OBU type must fit in four bits");
    }
    if (temporal_id > 7 || spatial_id > 3) {
        throw std::invalid_argument("OBU temporal/spatial id is out of range");
    }

    BitWriter output;
    // Header byte: forbidden=0, type[4], extension flag, size flag=1, reserved=0.
    output.writeBits((static_cast<std::uint64_t>(type_value) << 3) |
                         (static_cast<std::uint64_t>(has_extension) << 2) | 0x02U,
                     8);
    if (has_extension) {
        output.writeBits((static_cast<std::uint64_t>(temporal_id) << 5) |
                             (static_cast<std::uint64_t>(spatial_id) << 3),
                         8);
    }
    output.writeUnsignedLeb128(payload.size());
    for (const std::uint8_t byte : payload) output.writeBits(byte, 8);
    return output.bytes();
}

}  // namespace av1gpu::bitstream
