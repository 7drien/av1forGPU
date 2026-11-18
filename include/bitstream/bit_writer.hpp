#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace av1gpu::bitstream {

class BitWriter {
public:
    void writeBit(bool value);
    void writeBits(std::uint64_t value, unsigned int bit_count);
    void writeUnsignedLeb128(std::uint64_t value);
    void alignToByte();

    const std::vector<std::uint8_t>& bytes() const { return bytes_; }
    std::size_t bitSize() const { return bit_size_; }
    bool byteAligned() const { return (bit_size_ & 7U) == 0; }

private:
    std::vector<std::uint8_t> bytes_;
    std::size_t bit_size_ = 0;
};

}  // namespace av1gpu::bitstream
