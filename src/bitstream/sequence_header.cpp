#include "bitstream/sequence_header.hpp"

#include "bitstream/bit_writer.hpp"
#include "bitstream/obu_writer.hpp"

#include <stdexcept>

namespace av1gpu::bitstream {
namespace {

unsigned int bitsRequired(unsigned int value) {
    unsigned int count = 0;
    do {
        ++count;
        value >>= 1;
    } while (value != 0);
    return count;
}

void validate(const SequenceHeaderConfig& config) {
    constexpr int kMaxDimension = 65536;
    if (config.width <= 0 || config.height <= 0 ||
        config.width > kMaxDimension || config.height > kMaxDimension ||
        (config.width & 1) || (config.height & 1)) {
        throw std::invalid_argument("4:2:0 dimensions must be positive, even, and at most 65536");
    }
    if (config.level_index > 23) {
        throw std::invalid_argument("AV1 level index must be in the range 0..23");
    }
}

}  // namespace

std::vector<std::uint8_t> SequenceHeaderWriter::makePayload(
    const SequenceHeaderConfig& config) {
    validate(config);
    const unsigned int width_minus_one = static_cast<unsigned int>(config.width - 1);
    const unsigned int height_minus_one = static_cast<unsigned int>(config.height - 1);
    const unsigned int width_bits = bitsRequired(width_minus_one);
    const unsigned int height_bits = bitsRequired(height_minus_one);

    BitWriter bits;

    const bool still_picture = config.mode == SequenceMode::StillPicture;
    const bool reduced_header = still_picture;

    // Main profile. The reduced header is only valid for a single still picture.
    bits.writeBits(0, 3);  // seq_profile
    bits.writeBit(still_picture);
    bits.writeBit(reduced_header);
    if (reduced_header) {
        bits.writeBits(config.level_index, 5);  // seq_level_idx[0]
    } else {
        bits.writeBit(false);  // timing_info_present_flag
        bits.writeBit(false);  // initial_display_delay_present_flag
        bits.writeBits(0, 5);  // operating_points_cnt_minus_1
        bits.writeBits(0, 12); // operating_point_idc[0]
        bits.writeBits(config.level_index, 5); // seq_level_idx[0]
        if (config.level_index > 7) bits.writeBit(false); // seq_tier[0]
    }

    bits.writeBits(width_bits - 1, 4);  // frame_width_bits_minus_1
    bits.writeBits(height_bits - 1, 4); // frame_height_bits_minus_1
    bits.writeBits(width_minus_one, width_bits);
    bits.writeBits(height_minus_one, height_bits);

    if (!reduced_header) bits.writeBit(false); // frame_id_numbers_present_flag

    // Use 64x64 superblocks and disable optional tools for the first syntax subset.
    bits.writeBit(false);  // use_128x128_superblock
    bits.writeBit(false);  // enable_filter_intra
    bits.writeBit(false);  // enable_intra_edge_filter
    if (!reduced_header) {
        bits.writeBit(false); // enable_interintra_compound
        bits.writeBit(false); // enable_masked_compound
        bits.writeBit(false); // enable_warped_motion
        bits.writeBit(false); // enable_dual_filter
        bits.writeBit(false); // enable_order_hint
        bits.writeBit(false); // seq_choose_screen_content_tools
        bits.writeBit(false); // seq_force_screen_content_tools
    }
    bits.writeBit(false);  // enable_superres
    bits.writeBit(false);  // enable_cdef
    bits.writeBit(false);  // enable_restoration

    // 8-bit non-monochrome 4:2:0, unspecified color description, limited range.
    bits.writeBit(false);  // high_bitdepth
    bits.writeBit(false);  // mono_chrome
    bits.writeBit(false);  // color_description_present_flag
    bits.writeBit(false);  // color_range
    bits.writeBits(0, 2);  // chroma_sample_position: unknown
    bits.writeBit(false);  // separate_uv_delta_q
    bits.writeBit(false);  // film_grain_params_present

    bits.writeBit(true);   // trailing_one_bit
    bits.alignToByte();
    return bits.bytes();
}

std::vector<std::uint8_t> SequenceHeaderWriter::makeObu(
    const SequenceHeaderConfig& config) {
    return ObuWriter::wrap(ObuType::SequenceHeader, makePayload(config));
}

}  // namespace av1gpu::bitstream
