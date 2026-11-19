// Host side of the CUDA inter frames: frame header and OBU assembly around the tiles coded on the GPU
// (pframe_tile.cuh, Analyzer::entropy).
#pragma once
#include "internal.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pfw {

enum { INTRA_FRAME = 0, LAST_FRAME = 1, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME, BWDREF_FRAME, ALTREF2_FRAME,
       ALTREF_FRAME };

struct FrameParams {
    int qindex = 100;
    int refSlot[7] = {0, 0, 0, 0, 0, 0, 0};   // ref_frame_idx for LAST .. ALTREF
    int refreshFlags = 1;
    bool referenceSelect = false;   // compound prediction allowed
    bool txModeSelect = false;      // TX_MODE_SELECT (coded blocks may split their luma transform)
    bool reducedTxSet = true;
    bool persistent = false;        // error_resilient_mode = 0 and CDFs from primary_ref_frame
    bool primary = false;           // primary_ref_frame = LAST (else NONE)
    int grain = 0;
    uint16_t grainSeed = 0;
    // in-loop filters (the reconstruction already includes them)
    int lfLevel[4] = {0, 0, 0, 0};
    int lfSharpness = 0;
    bool cdef = false;   // sequence enable_cdef; one CDEF strength is coded
    int cdefDamping = 3, cdefY = 0, cdefYSec = 0, cdefUV = 0, cdefUVSec = 0;
    bool showFrame = true;
    bool temporalDelimiter = true;
    int tileColsLog2 = 0, tileRowsLog2 = 0, contextTile = 0;
};

// Uniform tile layout (spec 5.9.15, uniform_tile_spacing_flag = 1) in mode-info units.
struct TileGrid {
    int colsLog2 = 0, rowsLog2 = 0;
    std::vector<int> colStart, rowStart;   // one entry more than the tile count
    int cols() const { return int(colStart.size()) - 1; }
    int rows() const { return int(rowStart.size()) - 1; }
};
inline int maxTileLog2(int sb) {
    int k = 0;
    while ((1 << k) < std::min(sb, 64)) ++k;
    return k;
}
inline TileGrid tileGrid(int miCols, int miRows, int colsLog2, int rowsLog2) {
    TileGrid g;
    const int sbCols = (miCols + 15) >> 4, sbRows = (miRows + 15) >> 4;
    g.colsLog2 = std::min(colsLog2, maxTileLog2(sbCols));
    g.rowsLog2 = std::min(rowsLog2, maxTileLog2(sbRows));
    const int wSb = (sbCols + (1 << g.colsLog2) - 1) >> g.colsLog2;
    for (int s = 0; s < sbCols; s += wSb) g.colStart.push_back(s * 16);
    g.colStart.push_back(miCols);
    const int hSb = (sbRows + (1 << g.rowsLog2) - 1) >> g.rowsLog2;
    for (int s = 0; s < sbRows; s += hSb) g.rowStart.push_back(s * 16);
    g.rowStart.push_back(miRows);
    return g;
}

inline void header(av1g::BitWriter &b, const FrameParams &fp, int miCols, int miRows, int tileBytes) {
    b.bit(0);              // show_existing_frame
    b.put(1, 2);           // frame_type INTER_FRAME
    b.bit(fp.showFrame);   // show_frame
    if (!fp.showFrame) b.bit(1);   // showable_frame
    b.bit(!fp.persistent);         // error_resilient_mode
    b.bit(0);                      // disable_cdf_update
    b.bit(0);                      // allow_screen_content_tools
    b.bit(0);                      // frame_size_override_flag
    if (fp.persistent) b.put(fp.primary ? 0 : 7, 3);   // primary_ref_frame (LAST or NONE)
    b.put(fp.refreshFlags, 8);
    for (int r = 0; r < 7; ++r) b.put(fp.refSlot[r], 3);
    b.bit(0);      // render_and_frame_size_different
    b.bit(0);      // allow_high_precision_mv
    b.bit(0);      // is_filter_switchable
    b.put(0, 2);   // interpolation_filter EIGHTTAP
    b.bit(0);      // is_motion_mode_switchable
    b.bit(0);      // disable_frame_end_update_cdf
    // tile_info
    const int sbCols = (miCols + 15) >> 4, sbRows = (miRows + 15) >> 4;
    const TileGrid g = tileGrid(miCols, miRows, fp.tileColsLog2, fp.tileRowsLog2);
    b.bit(1);   // uniform_tile_spacing_flag
    for (int i = 0; i < g.colsLog2; ++i) b.bit(1);
    if (g.colsLog2 < maxTileLog2(sbCols)) b.bit(0);
    for (int i = 0; i < g.rowsLog2; ++i) b.bit(1);
    if (g.rowsLog2 < maxTileLog2(sbRows)) b.bit(0);
    if (g.colsLog2 || g.rowsLog2) {
        b.put(fp.contextTile, g.colsLog2 + g.rowsLog2);   // context_update_tile_id
        b.put(tileBytes - 1, 2);                         // tile_size_bytes_minus_1
    }
    // quantization_params
    b.put(fp.qindex, 8);
    b.bit(0); b.bit(0); b.bit(0);   // DeltaQYDc, DeltaQUDc, DeltaQUAc
    b.bit(0);                       // using_qmatrix
    b.bit(0);                       // segmentation_enabled
    b.bit(0);                       // delta_q_present
    // loop_filter_params
    b.put(fp.lfLevel[0], 6);
    b.put(fp.lfLevel[1], 6);
    if (fp.lfLevel[0] || fp.lfLevel[1]) {
        b.put(fp.lfLevel[2], 6);
        b.put(fp.lfLevel[3], 6);
    }
    b.put(fp.lfSharpness, 3);
    b.bit(0);   // loop_filter_delta_enabled
    if (fp.cdef) {
        b.put(fp.cdefDamping - 3, 2);
        b.put(0, 2);   // cdef_bits
        b.put(fp.cdefY, 4);
        b.put(fp.cdefYSec == 4 ? 3 : fp.cdefYSec, 2);
        b.put(fp.cdefUV, 4);
        b.put(fp.cdefUVSec == 4 ? 3 : fp.cdefUVSec, 2);
    }
    b.bit(fp.txModeSelect);      // tx_mode_select (else TX_MODE_LARGEST)
    b.bit(fp.referenceSelect);   // reference_select
    b.bit(fp.reducedTxSet);      // reduced_tx_set
    for (int r = 0; r < 7; ++r) b.bit(0);   // is_global
    av1g::write_video_grain(b, fp.grain, fp.grainSeed, true);
}

// OBU_FRAME (with an optional temporal delimiter) from the header parameters and the coded tiles in raster order.
inline std::vector<uint8_t> frameObu(const FrameParams &fp, int miCols, int miRows,
                                     const std::vector<std::vector<uint8_t>> &tiles) {
    size_t largest = 1;
    for (size_t i = 0; i + 1 < tiles.size(); ++i) largest = std::max(largest, tiles[i].size());
    int tileBytes = 1;
    while (tileBytes < 4 && (largest - 1) >> (8 * tileBytes)) ++tileBytes;
    av1g::BitWriter b;
    header(b, fp, miCols, miRows, tileBytes);
    b.byte_align();
    std::vector<uint8_t> payload(b.buf.begin(), b.buf.end());
    if (tiles.size() > 1) payload.push_back(0);   // tile_start_and_end_present_flag = 0, byte_alignment()
    for (size_t i = 0; i < tiles.size(); ++i) {
        if (tiles[i].empty()) throw std::runtime_error("empty tile");
        if (i + 1 < tiles.size())
            for (int k = 0; k < tileBytes; ++k) payload.push_back(uint8_t((tiles[i].size() - 1) >> (8 * k)));
        payload.insert(payload.end(), tiles[i].begin(), tiles[i].end());
    }
    std::vector<uint8_t> obu;
    if (fp.temporalDelimiter) obu = {0x12, 0};
    obu.push_back(0x32);   // OBU_FRAME with size field
    av1g::put_leb128(obu, payload.size());
    obu.insert(obu.end(), payload.begin(), payload.end());
    return obu;
}

// show_existing_frame of a non-key frame (OBU_FRAME_HEADER with trailing bits).
inline std::vector<uint8_t> showExistingObu(int slot, bool temporalDelimiter) {
    av1g::BitWriter b;
    b.bit(1);
    b.put(slot, 3);
    b.bit(1);   // trailing_bits()
    b.byte_align();
    std::vector<uint8_t> obu;
    if (temporalDelimiter) obu = {0x12, 0};
    obu.push_back(0x1A);
    av1g::put_leb128(obu, b.buf.size());
    obu.insert(obu.end(), b.buf.begin(), b.buf.end());
    return obu;
}

}  // namespace pfw
