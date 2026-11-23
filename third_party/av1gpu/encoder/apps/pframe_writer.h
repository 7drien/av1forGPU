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
    bool switchable = false;        // is_filter_switchable (else EIGHTTAP)
    bool skipModePresent = false;   // skip_mode_present (when skip_mode_params allows it)
    int lrType[3] = {0, 0, 0};
    // order hints (enable_order_hint = 1 when orderHintBits > 0)
    int orderHintBits = 0;
    int orderHint = 0;
    int slotHint[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // RefOrderHint of each reference slot before this frame
    bool useRefFrameMvs = false;      // lr_type per plane (0 NONE, 2 WIENER); 64-sample units
    bool persistent = false;        // error_resilient_mode = 0 and CDFs from primary_ref_frame
    bool primary = false;           // primary_ref_frame = LAST (else NONE)
    int grain = 0;
    uint16_t grainSeed = 0;
    // in-loop filters (the reconstruction already includes them)
    int lfLevel[4] = {0, 0, 0, 0};
    int lfSharpness = 0;
    bool cdef = false;   // sequence enable_cdef; one CDEF strength is coded
    int cdefDamping = 3, cdefY = 0, cdefYSec = 0, cdefUV = 0, cdefUVSec = 0;
    int cdefBits = 0;              // cdef_bits; with cdefBits > 0 the presets below are coded (index 0 = the fields above)
    int cdefPreset[8][4] = {};     // y pri, y sec, uv pri, uv sec
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

inline int relDist(int a, int b, int bits) {   // get_relative_dist
    if (!bits) return 0;
    int diff = a - b;
    const int m = 1 << (bits - 1);
    return (diff & (m - 1)) - (diff & m);
}
// skip_mode_params(): whether skip_mode_present is coded, and SkipModeFrame[0..1] (AV1 reference names)
inline bool skipModeFrames(const FrameParams &fp, int &f0, int &f1) {
    if (!fp.referenceSelect || !fp.orderHintBits) return false;
    const int bits = fp.orderHintBits;
    int fwd = -1, bwd = -1, fwdHint = 0, bwdHint = 0;
    for (int i = 0; i < 7; i++) {
        const int h = fp.slotHint[fp.refSlot[i]];
        if (relDist(h, fp.orderHint, bits) < 0) {
            if (fwd < 0 || relDist(h, fwdHint, bits) > 0) { fwd = i; fwdHint = h; }
        } else if (relDist(h, fp.orderHint, bits) > 0) {
            if (bwd < 0 || relDist(h, bwdHint, bits) < 0) { bwd = i; bwdHint = h; }
        }
    }
    if (fwd < 0) return false;
    if (bwd >= 0) { f0 = 1 + std::min(fwd, bwd); f1 = 1 + std::max(fwd, bwd); return true; }
    int fwd2 = -1, fwd2Hint = 0;
    for (int i = 0; i < 7; i++) {
        const int h = fp.slotHint[fp.refSlot[i]];
        if (relDist(h, fwdHint, bits) < 0 && (fwd2 < 0 || relDist(h, fwd2Hint, bits) > 0)) { fwd2 = i; fwd2Hint = h; }
    }
    if (fwd2 < 0) return false;
    f0 = 1 + std::min(fwd, fwd2); f1 = 1 + std::max(fwd, fwd2);
    return true;
}
inline bool skipModeAllowed(const FrameParams &fp) {
    if (!fp.referenceSelect || !fp.orderHintBits) return false;
    const int bits = fp.orderHintBits;
    int fwd = -1, bwd = -1, fwdHint = 0, bwdHint = 0;
    for (int i = 0; i < 7; i++) {
        const int h = fp.slotHint[fp.refSlot[i]];
        if (relDist(h, fp.orderHint, bits) < 0) {
            if (fwd < 0 || relDist(h, fwdHint, bits) > 0) { fwd = i; fwdHint = h; }
        } else if (relDist(h, fp.orderHint, bits) > 0) {
            if (bwd < 0 || relDist(h, bwdHint, bits) < 0) { bwd = i; bwdHint = h; }
        }
    }
    if (fwd < 0) return false;
    if (bwd >= 0) return true;
    for (int i = 0; i < 7; i++)
        if (relDist(fp.slotHint[fp.refSlot[i]], fwdHint, bits) < 0) return true;
    return false;
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
    if (fp.orderHintBits) b.put(fp.orderHint, fp.orderHintBits);   // order_hint
    if (fp.persistent) b.put(fp.primary ? 0 : 7, 3);   // primary_ref_frame (LAST or NONE)
    b.put(fp.refreshFlags, 8);
    if (!fp.persistent && fp.orderHintBits)
        for (int i = 0; i < 8; ++i) b.put(fp.slotHint[i], fp.orderHintBits);   // ref_order_hint
    if (fp.orderHintBits) b.bit(0);                                            // frame_refs_short_signaling
    for (int r = 0; r < 7; ++r) b.put(fp.refSlot[r], 3);
    b.bit(0);      // render_and_frame_size_different
    b.bit(0);      // allow_high_precision_mv
    b.bit(fp.switchable);           // is_filter_switchable
    if (!fp.switchable) b.put(0, 2);   // interpolation_filter EIGHTTAP
    b.bit(0);      // is_motion_mode_switchable
    if (fp.persistent && fp.orderHintBits) b.bit(fp.useRefFrameMvs);   // use_ref_frame_mvs
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
        b.put(fp.cdefBits, 2);   // cdef_bits
        for (int i = 0; i < (1 << fp.cdefBits); i++) {
            const int *c = fp.cdefBits ? fp.cdefPreset[i] : nullptr;
            const int yp = c ? c[0] : fp.cdefY, ys = c ? c[1] : fp.cdefYSec, up = c ? c[2] : fp.cdefUV, us = c ? c[3] : fp.cdefUVSec;
            b.put(yp, 4);
            b.put(ys == 4 ? 3 : ys, 2);
            b.put(up, 4);
            b.put(us == 4 ? 3 : us, 2);
        }
    }
    // lr_params (enable_restoration = 1 in the sequence header)
    for (int p = 0; p < 3; ++p) b.put(fp.lrType[p], 2);
    if (fp.lrType[0] || fp.lrType[1] || fp.lrType[2]) {
        b.bit(0);                                     // lr_unit_shift: LoopRestorationSize 64
        if (fp.lrType[1] || fp.lrType[2]) b.bit(0);   // lr_uv_shift
    }
    b.bit(fp.txModeSelect);      // tx_mode_select (else TX_MODE_LARGEST)
    b.bit(fp.referenceSelect);   // reference_select
    if (skipModeAllowed(fp)) b.bit(fp.skipModePresent);   // skip_mode_present
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
