#include "bitstream/frame_header.hpp"

#include "bitstream/bit_writer.hpp"
#include "bitstream/obu_writer.hpp"

#include <algorithm>
#include <stdexcept>

namespace av1gpu::bitstream {
namespace {

unsigned int ceilLog2(unsigned int value) {
    unsigned int bits = 0;
    unsigned int power = 1;
    while (power < value) {
        power <<= 1;
        ++bits;
    }
    return bits;
}

void writeSingleTileInfo(BitWriter& bits, int width, int height) {
    constexpr unsigned int kSuperblockSize = 64;
    constexpr unsigned int kMaxTileWidthSb = 64;
    constexpr unsigned int kMaxTileAreaSb = 2304;
    constexpr unsigned int kMaxTileRows = 64;
    constexpr unsigned int kMaxTileCols = 64;

    const unsigned int sb_cols = (static_cast<unsigned int>(width) + 63) / 64;
    const unsigned int sb_rows = (static_cast<unsigned int>(height) + 63) / 64;
    if (sb_cols > kMaxTileWidthSb || sb_rows > kMaxTileRows ||
        sb_cols > kMaxTileCols || sb_cols * sb_rows > kMaxTileAreaSb) {
        throw std::invalid_argument("frame dimensions exceed the one-tile MVP limits");
    }

    const unsigned int min_log2_tile_cols = ceilLog2((sb_cols + kMaxTileWidthSb - 1) /
                                                      kMaxTileWidthSb);
    const unsigned int max_log2_tile_cols = ceilLog2(std::min(sb_cols, kMaxTileCols));
    const unsigned int min_log2_tiles = std::max(
        min_log2_tile_cols,
        ceilLog2((sb_cols * sb_rows + kMaxTileAreaSb - 1) / kMaxTileAreaSb));
    const unsigned int max_log2_tile_rows = ceilLog2(std::min(sb_rows, kMaxTileRows));
    const unsigned int min_log2_tile_rows = min_log2_tiles > min_log2_tile_cols
        ? min_log2_tiles - min_log2_tile_cols : 0;

    // The supported sizes fit in one tile, but emit the spacing syntax from
    // the derived log2 bounds so this remains correct at small frame sizes.
    bits.writeBit(true);  // uniform_tile_spacing_flag
    unsigned int tile_cols_log2 = min_log2_tile_cols;
    while (tile_cols_log2 < max_log2_tile_cols) {
        bits.writeBit(false);  // increment_tile_cols_log2
        break;
    }
    unsigned int tile_rows_log2 = min_log2_tile_rows;
    while (tile_rows_log2 < max_log2_tile_rows) {
        bits.writeBit(false);  // increment_tile_rows_log2
        break;
    }
    if (tile_cols_log2 > 0 || tile_rows_log2 > 0) {
        throw std::logic_error("one-tile configuration unexpectedly requires multiple tiles");
    }
}

}  // namespace

std::vector<std::uint8_t> FrameHeaderWriter::makeObu(const FrameHeaderConfig& config) {
    if (config.width <= 0 || config.height <= 0 ||
        config.width > 65536 || config.height > 65536 ||
        (config.width & 1) || (config.height & 1)) {
        throw std::invalid_argument("frame dimensions must be positive even 4:2:0 sizes");
    }

    BitWriter bits;
    // Reduced still-picture uncompressed header implies KEY_FRAME and show_frame.
    bits.writeBit(false); // disable_cdf_update
    bits.writeBit(false); // allow_screen_content_tools
    // Frame is intra, so force_integer_mv is inferred as one.
    // frame_size_override_flag and order_hint are inferred as zero.
    // primary_ref_frame is inferred as PRIMARY_REF_NONE.
    // refresh_frame_flags is inferred as all reference slots for a shown key frame.
    // frame_size uses the dimensions from the matching sequence header; superres is off.
    bits.writeBit(false); // render_and_frame_size_different

    writeSingleTileInfo(bits, config.width, config.height);

    // Lossless quantization: no deltas, q-matrix, segmentation, or loop filters.
    bits.writeBits(0, 8); // base_q_idx
    bits.writeBit(false); // DeltaQYDc
    bits.writeBit(false); // DeltaQUDc
    bits.writeBit(false); // DeltaQUAc
    bits.writeBit(false); // using_qmatrix
    bits.writeBit(false); // segmentation_enabled
    // delta_q_present, delta_lf_present, loop filter, CDEF and LR are inferred off.
    // TxMode is inferred as ONLY_4X4 because CodedLossless is true.
    // Frame is intra, so reference mode, skip mode and global motion are inferred.
    bits.writeBit(true); // reduced_tx_set

    bits.writeBit(true); // trailing_one_bit
    bits.alignToByte();
    return ObuWriter::wrap(ObuType::FrameHeader, bits.bytes());
}

}  // namespace av1gpu::bitstream
