#include "bitstream/bit_writer.hpp"

#include <stdexcept>

namespace av1gpu::bitstream {

void BitWriter::writeBit(bool value) {
    if ((bit_size_ & 7U) == 0) {
        bytes_.push_back(0);
    }
    if (value) {
        const unsigned int bit_in_byte = 7U - static_cast<unsigned int>(bit_size_ & 7U);
        bytes_.back() |= static_cast<std::uint8_t>(1U << bit_in_byte);
    }
    ++bit_size_;
}

void BitWriter::writeBits(std::uint64_t value, unsigned int bit_count) {
    if (bit_count > 64) {
        throw std::invalid_argument("cannot write more than 64 bits at once");
    }
    for (unsigned int remaining = bit_count; remaining > 0; --remaining) {
        writeBit(((value >> (remaining - 1)) & 1U) != 0);
    }
}

void BitWriter::writeUnsignedLeb128(std::uint64_t value) {
    if (!byteAligned()) {
        throw std::logic_error("LEB128 values must start at a byte boundary");
    }
    do {
        std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7fU);
        value >>= 7;
        if (value != 0) byte |= 0x80U;
        writeBits(byte, 8);
    } while (value != 0);
}

void BitWriter::alignToByte() {
    while (!byteAligned()) writeBit(false);
}

}  // namespace av1gpu::bitstream
