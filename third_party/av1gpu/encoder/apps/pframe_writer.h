// AV1 inter-frame syntax for the CUDA P-frame decisions (pframe.h).
//
// One tile per frame, 64x64 superblocks, square partitions, single-reference inter blocks with per-block
// motion vectors (NEWMV / NEARESTMV / NEARMV / GLOBALMV chosen by coded size), TX_MODE_LARGEST, DCT_DCT
// residuals. The motion vector candidate list follows the AV1 specification, section 7.10.2 (spatial
// candidates only: use_ref_frame_mvs is 0 because enable_order_hint is 0).
#pragma once
#include "pframe.h"
#include "inter_tables.h"
#include "inter_packet.h"
#include "internal.h"
#include "av1g/entenc.h"
#include "av1g/symstream.h"
#include "av1g/cdf_defaults.h"
#include "av1g/cdfctx.h"
#include "av1g/dsp.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace pfw {

using inter::emitCoeffPayload;   // shared coefficient syntax (inter_packet.h)

// AV1 reference frame names
enum { INTRA_FRAME = 0, LAST_FRAME = 1, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME, BWDREF_FRAME, ALTREF2_FRAME, ALTREF_FRAME };
enum { NEARESTMV = 13, NEARMV, GLOBALMV, NEWMV };

// Inter-specific adaptive CDFs (spec format: increasing, 32768 terminator, adaptation counter).
struct InterCdfs {
    uint16_t isInter[4][3];
    uint16_t singleRef[3][6][3];
    uint16_t newMv[6][3], zeroMv[2][3], refMv[6][3], drl[3][3];
    uint16_t mvJoint[5];
    uint16_t mvSign[2][3], mvClass[2][12], class0Bit[2][3], class0Fr[2][2][5], mvFr[2][5], mvBit[2][10][3];
    uint16_t txSet3[4][3];
    void init() {
        memcpy(isInter, pft::Default_Is_Inter_Cdf, sizeof isInter);
        memcpy(singleRef, pft::Default_Single_Ref_Cdf, sizeof singleRef);
        memcpy(newMv, pft::Default_New_Mv_Cdf, sizeof newMv);
        memcpy(zeroMv, pft::Default_Zero_Mv_Cdf, sizeof zeroMv);
        memcpy(refMv, pft::Default_Ref_Mv_Cdf, sizeof refMv);
        memcpy(drl, pft::Default_Drl_Mode_Cdf, sizeof drl);
        memcpy(mvJoint, pft::Default_Mv_Joint_Cdf, sizeof mvJoint);
        for (int c = 0; c < 2; c++) {
            memcpy(mvSign[c], pft::Default_Mv_Sign_Cdf, sizeof mvSign[c]);
            memcpy(mvClass[c], pft::Default_Mv_Class_Cdf[c], sizeof mvClass[c]);
            memcpy(class0Bit[c], pft::Default_Mv_Class0_Bit_Cdf, sizeof class0Bit[c]);
            memcpy(class0Fr[c], pft::Default_Mv_Class0_Fr_Cdf[c], sizeof class0Fr[c]);
            memcpy(mvFr[c], pft::Default_Mv_Fr_Cdf[c], sizeof mvFr[c]);
            memcpy(mvBit[c], pft::Default_Mv_Bit_Cdf, sizeof mvBit[c]);
        }
        memcpy(txSet3, pft::Default_Inter_Tx_Type_Set3_Cdf, sizeof txSet3);
    }
};

// Entropy state carried between frames through primary_ref_frame (decoders clear the adaptation counters
// when they save frame-end CDFs).
struct EntropyState {
    av1g_cdfs cdf{};
    InterCdfs inter{};
    bool ready = false;
};
inline void clearCounters(uint16_t *v, size_t n) {
    for (size_t i = 0; i + 1 < n; ++i)
        if (v[i] == 32768) v[++i] = 0;
}
inline void clearCounters(EntropyState &s) {
    clearCounters(reinterpret_cast<uint16_t *>(&s.cdf), sizeof(s.cdf) / 2);
    clearCounters(reinterpret_cast<uint16_t *>(&s.inter), sizeof(s.inter) / 2);
}

inline float symCost(const uint16_t *cdf, int s) {
    const int p = cdf[s] - (s ? cdf[s - 1] : 0);
    return -std::log2(std::max(p, 1) / 32768.0f);
}

struct FrameParams {
    int qindex = 100;
    int refSlot[7] = {0, 0, 0, 1, 0, 0, 2};   // ref_frame_idx for LAST .. ALTREF
    int refreshFlags = 1;
    int refName[2] = {LAST_FRAME, GOLDEN_FRAME};   // AV1 reference used by BlockInfo::ref 0 and 1
    bool persistent = false;                         // carry CDFs (error_resilient_mode = 0)
    int grain = 0;
    uint16_t grainSeed = 0;
    // in-loop filters (stored in the frame header; the reconstruction must already include them)
    int lfLevel[4] = {0, 0, 0, 0};
    int lfSharpness = 0;
    bool cdef = false;   // sequence enable_cdef; when set, one CDEF strength is coded
    int cdefDamping = 0, cdefY = 0, cdefYSec = 0, cdefUV = 0, cdefUVSec = 0;
    bool showFrame = true;
    bool showExisting = false;
    bool temporalDelimiter = true;
    int showExistingSlot = 0;
};

class Writer {
public:
    Writer(int width, int height, int alignedWidth, int alignedHeight)
        : w(width), h(height), aw(alignedWidth), ah(alignedHeight), miCols(2 * ((width + 7) >> 3)),
          miRows(2 * ((height + 7) >> 3)) {
        const size_t n = size_t(miCols) * miRows;
        mi.resize(n);
    }

    std::vector<uint8_t> frame(const FrameParams &fp, const std::vector<pf::BlockInfo> &grid,
                               const std::vector<int16_t> &levels, EntropyState &state) {
        if (fp.showExisting) {
            // show_existing_frame of a non-key frame: header only
            av1g::BitWriter b;
            b.bit(1);
            b.put(fp.showExistingSlot, 3);
            b.byte_align();
            std::vector<uint8_t> obu = {0x12, 0, 0x1A};   // temporal delimiter, OBU_FRAME_HEADER
            av1g::put_leb128(obu, b.buf.size());
            obu.insert(obu.end(), b.buf.begin(), b.buf.end());
            return obu;
        }
        const bool primary = fp.persistent && state.ready;
        if (!primary) {
            state = EntropyState{};
            av1g_cdfs_init(&state.cdf, fp.qindex);
            state.inter.init();
        }
        cdf = &state.cdf;
        ic = &state.inter;
        q = fp.qindex;
        params = &fp;
        av1g::BitWriter b;
        header(b, fp, primary);
        b.byte_align();
        const size_t capacity = size_t(aw) * ah * 4 + 4096;
        std::unique_ptr<uint16_t[]> scratch(new uint16_t[capacity]);
        av1g_ec_init(&ec, scratch.get(), capacity);
        tile(grid, levels);
        std::vector<uint8_t> data(ec.offs + 8);
        const int n = av1g_ec_done(&ec, data.data());
        if (n < 0) throw std::runtime_error("inter tile overflow");
        std::vector<uint8_t> obu;
        if (fp.temporalDelimiter) obu = {0x12, 0};
        obu.push_back(0x32);   // OBU_FRAME with size field
        av1g::put_leb128(obu, b.buf.size() + n);
        obu.insert(obu.end(), b.buf.begin(), b.buf.end());
        obu.insert(obu.end(), data.begin(), data.begin() + n);
        clearCounters(state);
        state.ready = true;
        return obu;
    }

    // statistics of the last frame
    int modeCount[4] = {};   // NEAREST, NEAR, GLOBAL, NEW

private:
    struct MiInfo {
        uint8_t size;      // log2 of the block size in pixels (3..6)
        uint8_t skip, isInter, mode;
        int8_t ref[2];
        pf::Mv mv[2];
        bool written;
    };
    int w, h, aw, ah, miCols, miRows;
    std::vector<MiInfo> mi;
    av1g_cdfs *cdf = nullptr;
    InterCdfs *ic = nullptr;
    int q = 0;
    const FrameParams *params = nullptr;
    av1g_ec ec{};
    std::vector<uint8_t> aboveLevel[3], aboveDc[3], leftLevel[3], leftDc[3];

    MiInfo &at(int r, int c) { return mi[size_t(r) * miCols + c]; }
    bool inside(int r, int c) const { return r >= 0 && c >= 0 && r < miRows && c < miCols; }

    void header(av1g::BitWriter &b, const FrameParams &fp, bool primary) {
        b.bit(0);              // show_existing_frame
        b.put(1, 2);           // frame_type INTER_FRAME
        b.bit(fp.showFrame);   // show_frame
        if (!fp.showFrame) b.bit(1);   // showable_frame
        b.bit(!fp.persistent);         // error_resilient_mode
        b.bit(0);                      // disable_cdf_update
        b.bit(0);                      // allow_screen_content_tools
        b.bit(0);                      // frame_size_override_flag
        if (fp.persistent) b.put(primary ? 0 : 7, 3);   // primary_ref_frame (LAST)
        b.put(fp.refreshFlags, 8);
        for (int r = 0; r < 7; ++r) b.put(fp.refSlot[r], 3);
        b.bit(0);   // render_and_frame_size_different
        b.bit(0);   // allow_high_precision_mv
        b.bit(0);   // is_filter_switchable
        b.put(0, 2);   // interpolation_filter EIGHTTAP
        b.bit(0);   // is_motion_mode_switchable
        b.bit(0);   // disable_frame_end_update_cdf
        // tile_info: uniform spacing, one tile
        const int sbCols = (miCols + 15) >> 4, sbRows = (miRows + 15) >> 4;
        b.bit(1);
        int maxLog2 = 0;
        while ((1 << maxLog2) < std::min(sbCols, 64)) maxLog2++;
        if (maxLog2 > 0) b.bit(0);
        int maxLog2R = 0;
        while ((1 << maxLog2R) < std::min(sbRows, 64)) maxLog2R++;
        if (maxLog2R > 0) b.bit(0);
        // quantization_params
        b.put(q, 8);
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
            b.put(fp.cdefDamping, 2);
            b.put(0, 2);   // cdef_bits
            b.put(fp.cdefY, 4);
            b.put(fp.cdefYSec, 2);
            b.put(fp.cdefUV, 4);
            b.put(fp.cdefUVSec, 2);
        }
        b.bit(0);   // tx_mode_select = 0: TX_MODE_LARGEST
        b.bit(0);   // reference_select
        b.bit(1);   // reduced_tx_set
        for (int r = 0; r < 7; ++r) b.bit(0);   // is_global
        av1g::write_video_grain(b, fp.grain, fp.grainSeed, true);
    }

    // ---- motion vector prediction (spec 7.10.2) ----
    struct Stack {
        int num = 0, newCount = 0;
        pf::Mv mv[pft::MAX_REF_MV_STACK_SIZE];
        int weight[pft::MAX_REF_MV_STACK_SIZE];
        bool foundMatch = false;
        int closeMatches = 0, totalMatches = 0;
        int newMvCtx = 0, refMvCtx = 0, zeroMvCtx = 0;
        int drlCtx[pft::MAX_REF_MV_STACK_SIZE];
    };
    static bool hasNewmv(int mode) { return mode == NEWMV; }
    static pf::Mv lowerPrecision(pf::Mv m) {
        if (m.row & 1) m.row += m.row > 0 ? -1 : 1;
        if (m.col & 1) m.col += m.col > 0 ? -1 : 1;
        return m;
    }
    void addRefMv(Stack &st, int mvRow, int mvCol, int refFrame, int weight) {
        const MiInfo &cand = at(mvRow, mvCol);
        if (!cand.isInter) return;
        for (int list = 0; list < 2; ++list) {
            if (cand.ref[list] != refFrame) continue;
            const pf::Mv candMv = lowerPrecision(cand.mv[list]);
            if (hasNewmv(cand.mode)) st.newCount++;
            st.foundMatch = true;
            int idx = 0;
            for (; idx < st.num; ++idx)
                if (st.mv[idx] == candMv) break;
            if (idx < st.num) st.weight[idx] += weight;
            else if (st.num < pft::MAX_REF_MV_STACK_SIZE) {
                st.mv[st.num] = candMv;
                st.weight[st.num] = weight;
                st.num++;
            }
        }
    }
    void scanRow(Stack &st, int miRow, int miCol, int bw4, int deltaRow, int refFrame) {
        const int end4 = std::min(std::min(bw4, miCols - miCol), 16);
        int deltaCol = 0;
        const bool useStep16 = bw4 >= 16;
        if (std::abs(deltaRow) > 1) {
            deltaRow += miRow & 1;
            deltaCol = 1 - (miCol & 1);
        }
        for (int i = 0; i < end4;) {
            const int r = miRow + deltaRow, c = miCol + deltaCol + i;
            if (!inside(r, c)) break;
            int len = std::min(bw4, 1 << (at(r, c).size - 2));
            if (std::abs(deltaRow) > 1) len = std::max(2, len);
            if (useStep16) len = std::max(4, len);
            addRefMv(st, r, c, refFrame, len * 2);
            i += len;
        }
    }
    void scanCol(Stack &st, int miRow, int miCol, int bh4, int deltaCol, int refFrame) {
        const int end4 = std::min(std::min(bh4, miRows - miRow), 16);
        int deltaRow = 0;
        const bool useStep16 = bh4 >= 16;
        if (std::abs(deltaCol) > 1) {
            deltaRow = 1 - (miRow & 1);
            deltaCol += miCol & 1;
        }
        for (int i = 0; i < end4;) {
            const int r = miRow + deltaRow + i, c = miCol + deltaCol;
            if (!inside(r, c)) break;
            int len = std::min(bh4, 1 << (at(r, c).size - 2));
            if (std::abs(deltaCol) > 1) len = std::max(2, len);
            if (useStep16) len = std::max(4, len);
            addRefMv(st, r, c, refFrame, len * 2);
            i += len;
        }
    }
    void scanPoint(Stack &st, int miRow, int miCol, int deltaRow, int deltaCol, int refFrame) {
        const int r = miRow + deltaRow, c = miCol + deltaCol;
        if (inside(r, c) && at(r, c).written) addRefMv(st, r, c, refFrame, 4);
    }
    static void sortStack(Stack &st, int start, int end) {
        while (end > start) {
            int newEnd = start;
            for (int idx = start + 1; idx < end; ++idx) {
                if (st.weight[idx - 1] < st.weight[idx]) {
                    std::swap(st.weight[idx - 1], st.weight[idx]);
                    std::swap(st.mv[idx - 1], st.mv[idx]);
                    newEnd = idx;
                }
            }
            end = newEnd;
        }
    }
    void findMvStack(Stack &st, int miRow, int miCol, int size, int refFrame) {
        const int bw4 = 1 << (size - 2), bh4 = bw4;
        st = Stack{};
        const pf::Mv globalMv{0, 0};
        st.foundMatch = false;
        scanRow(st, miRow, miCol, bw4, -1, refFrame);
        bool foundAbove = st.foundMatch;
        st.foundMatch = false;
        scanCol(st, miRow, miCol, bh4, -1, refFrame);
        bool foundLeft = st.foundMatch;
        st.foundMatch = false;
        if (std::max(bw4, bh4) <= 16) scanPoint(st, miRow, miCol, -1, bw4, refFrame);
        if (st.foundMatch) foundAbove = true;
        st.closeMatches = foundAbove + foundLeft;
        const int numNearest = st.num, numNew = st.newCount;
        for (int i = 0; i < numNearest; ++i) st.weight[i] += pft::REF_CAT_LEVEL;
        st.zeroMvCtx = 0;
        st.foundMatch = false;
        scanPoint(st, miRow, miCol, -1, -1, refFrame);
        if (st.foundMatch) foundAbove = true;
        st.foundMatch = false;
        scanRow(st, miRow, miCol, bw4, -3, refFrame);
        if (st.foundMatch) foundAbove = true;
        st.foundMatch = false;
        scanCol(st, miRow, miCol, bh4, -3, refFrame);
        if (st.foundMatch) foundLeft = true;
        st.foundMatch = false;
        if (bh4 > 1) scanRow(st, miRow, miCol, bw4, -5, refFrame);
        if (st.foundMatch) foundAbove = true;
        st.foundMatch = false;
        if (bw4 > 1) scanCol(st, miRow, miCol, bh4, -5, refFrame);
        if (st.foundMatch) foundLeft = true;
        st.totalMatches = foundAbove + foundLeft;
        sortStack(st, 0, numNearest);
        sortStack(st, numNearest, st.num);
        if (st.num < 2) extraSearch(st, miRow, miCol, bw4, bh4, refFrame);
        for (int idx = st.num; idx < 2; ++idx) st.mv[idx] = globalMv;
        // context and clamping
        for (int idx = 0; idx < st.num; ++idx) {
            int z = 0;
            if (idx + 1 < st.num) {
                const int w0 = st.weight[idx], w1 = st.weight[idx + 1];
                if (w0 >= pft::REF_CAT_LEVEL) {
                    if (w1 < pft::REF_CAT_LEVEL) z = 1;
                } else {
                    z = 2;
                }
            }
            st.drlCtx[idx] = z;
        }
        const int bw = 4 * bw4, bh = 4 * bh4;
        for (int idx = 0; idx < st.num; ++idx) {
            const int top = -((miRow * 4) * 8), bottom = ((miRows - bh4 - miRow) * 4) * 8;
            const int left = -((miCol * 4) * 8), right = ((miCols - bw4 - miCol) * 4) * 8;
            const int br = pft::MV_BORDER + bh * 8, bc = pft::MV_BORDER + bw * 8;
            st.mv[idx].row = (int16_t)std::min(std::max((int)st.mv[idx].row, top - br), bottom + br);
            st.mv[idx].col = (int16_t)std::min(std::max((int)st.mv[idx].col, left - bc), right + bc);
        }
        if (st.closeMatches == 0) {
            st.newMvCtx = std::min(st.totalMatches, 1);
            st.refMvCtx = st.totalMatches;
        } else if (st.closeMatches == 1) {
            st.newMvCtx = 3 - std::min(numNew, 1);
            st.refMvCtx = 2 + st.totalMatches;
        } else {
            st.newMvCtx = 5 - std::min(numNew, 1);
            st.refMvCtx = 5;
        }
    }
    void extraSearch(Stack &st, int miRow, int miCol, int bw4, int bh4, int refFrame) {
        int w4 = std::min(16, bw4), h4 = std::min(16, bh4);
        w4 = std::min(w4, miCols - miCol);
        h4 = std::min(h4, miRows - miRow);
        const int num4x4 = std::min(w4, h4);
        for (int pass = 0; pass < 2; ++pass) {
            int idx = 0;
            while (idx < num4x4 && st.num < 2) {
                int r, c;
                if (pass == 0) { r = miRow - 1; c = miCol + idx; }
                else { r = miRow + idx; c = miCol - 1; }
                if (!inside(r, c)) break;
                const MiInfo &cand = at(r, c);
                for (int list = 0; list < 2; ++list) {
                    const int candRef = cand.ref[list];
                    if (candRef > INTRA_FRAME) {
                        // RefFrameSignBias is 0 for every reference (enable_order_hint = 0)
                        const pf::Mv candMv = cand.mv[list];
                        int k = 0;
                        for (; k < st.num; ++k)
                            if (candMv == st.mv[k]) break;
                        if (k == st.num) {
                            st.mv[k] = candMv;
                            st.weight[k] = 2;
                            st.num++;
                        }
                    }
                }
                idx += pass == 0 ? (1 << (cand.size - 2)) : (1 << (cand.size - 2));
            }
        }
        (void)refFrame;
    }

    // ---- symbol helpers ----
    void sym(uint16_t *c, int s, int n) { av1g_ec_symbol(&ec, s, c, n); }
    void bool_(int v) { av1g_ec_bool(&ec, v); }

    float mvCompCost(int comp, int d) const {
        const int sign = d < 0, mag = std::abs(d);
        float bits = symCost(ic->mvSign[comp], sign);
        const int v = mag - 1;
        if (v < 16) {
            bits += symCost(ic->mvClass[comp], 0);
            const int bit = v >> 3, fr = (v >> 1) & 3;
            bits += symCost(ic->class0Bit[comp], bit) + symCost(ic->class0Fr[comp][bit], fr);
        } else {
            int c = 1;
            while (c < 10 && v >= (1 << (c + 4))) c++;
            bits += symCost(ic->mvClass[comp], c);
            const int rem = v - (1 << (c + 3));
            const int d2 = rem >> 3, fr = (rem >> 1) & 3;
            for (int i = 0; i < c; ++i) bits += symCost(ic->mvBit[comp][i], (d2 >> i) & 1);
            bits += symCost(ic->mvFr[comp], fr);
        }
        return bits;
    }
    float mvCost(pf::Mv mv, pf::Mv pred) const {
        const int dr = mv.row - pred.row, dc = mv.col - pred.col;
        const int joint = (dr != 0) * 2 + (dc != 0);
        float bits = symCost(ic->mvJoint, joint);
        if (dr) bits += mvCompCost(0, dr);
        if (dc) bits += mvCompCost(1, dc);
        return bits;
    }
    void writeMvComp(int comp, int d) {
        const int sign = d < 0, mag = std::abs(d);
        sym(ic->mvSign[comp], sign, 2);
        const int v = mag - 1;
        if (v < 16) {
            sym(ic->mvClass[comp], 0, 11);
            const int bit = v >> 3, fr = (v >> 1) & 3;
            sym(ic->class0Bit[comp], bit, 2);
            sym(ic->class0Fr[comp][bit], fr, 4);
        } else {
            int c = 1;
            while (c < 10 && v >= (1 << (c + 4))) c++;
            sym(ic->mvClass[comp], c, 11);
            const int rem = v - (1 << (c + 3));
            const int d2 = rem >> 3, fr = (rem >> 1) & 3;
            for (int i = 0; i < c; ++i) sym(ic->mvBit[comp][i], (d2 >> i) & 1, 2);
            sym(ic->mvFr[comp], fr, 4);
        }
    }
    void writeMv(pf::Mv mv, pf::Mv pred) {
        const int dr = mv.row - pred.row, dc = mv.col - pred.col;
        const int joint = (dr != 0) * 2 + (dc != 0);
        sym(ic->mvJoint, joint, 4);
        if (dr) writeMvComp(0, dr);
        if (dc) writeMvComp(1, dc);
    }

    int countRefs(int frameType, bool availU, bool availL, const MiInfo *above, const MiInfo *left) const {
        int c = 0;
        if (availU) c += (above->ref[0] == frameType) + (above->ref[1] == frameType);
        if (availL) c += (left->ref[0] == frameType) + (left->ref[1] == frameType);
        return c;
    }
    static int refCountCtx(int a, int b) { return a < b ? 0 : a == b ? 1 : 2; }

    void writeRefFrame(int refFrame, bool availU, bool availL, const MiInfo *above, const MiInfo *left) {
        auto cnt = [&](int f) { return countRefs(f, availU, availL, above, left); };
        const int fwd = cnt(LAST_FRAME) + cnt(LAST2_FRAME) + cnt(LAST3_FRAME) + cnt(GOLDEN_FRAME);
        const int bwd = cnt(BWDREF_FRAME) + cnt(ALTREF2_FRAME) + cnt(ALTREF_FRAME);
        const int p1 = refFrame >= BWDREF_FRAME;
        sym(ic->singleRef[refCountCtx(fwd, bwd)][0], p1, 2);
        if (p1) {
            const int p2 = refFrame == ALTREF_FRAME;
            sym(ic->singleRef[refCountCtx(cnt(BWDREF_FRAME) + cnt(ALTREF2_FRAME), cnt(ALTREF_FRAME))][1], p2, 2);
            if (!p2) sym(ic->singleRef[refCountCtx(cnt(BWDREF_FRAME), cnt(ALTREF2_FRAME))][5], refFrame == ALTREF2_FRAME, 2);
        } else {
            const int p3 = refFrame == LAST3_FRAME || refFrame == GOLDEN_FRAME;
            sym(ic->singleRef[refCountCtx(cnt(LAST_FRAME) + cnt(LAST2_FRAME), cnt(LAST3_FRAME) + cnt(GOLDEN_FRAME))][2], p3, 2);
            if (p3) sym(ic->singleRef[refCountCtx(cnt(LAST3_FRAME), cnt(GOLDEN_FRAME))][4], refFrame == GOLDEN_FRAME, 2);
            else sym(ic->singleRef[refCountCtx(cnt(LAST_FRAME), cnt(LAST2_FRAME))][3], refFrame == LAST2_FRAME, 2);
        }
    }

    // ---- residual ----
    void coeffs(int plane, int x4, int y4, int txSz, int bsizePixels, const int16_t *qv) {
        const int sub = plane > 0;
        const int tw = AV1G_T(Tx_Width)[txSz], th = AV1G_T(Tx_Height)[txSz];
        const int w4 = tw >> 2, h4 = th >> 2;
        const int maxX4 = miCols >> sub, maxY4 = miRows >> sub;
        uint8_t *aL = aboveLevel[plane].data(), *aD = aboveDc[plane].data();
        uint8_t *lL = leftLevel[plane].data(), *lD = leftDc[plane].data();
        const int txSzCtx = (AV1G_T(Tx_Size_Sqr)[txSz] + AV1G_T(Tx_Size_Sqr_Up)[txSz] + 1) >> 1;
        int ctx;
        if (plane == 0) {
            ctx = 0;   // the transform covers the whole block
        } else {
            int above = 0, left = 0;
            for (int i = 0; i < w4; i++) if (x4 + i < maxX4) above |= aL[x4 + i] | aD[x4 + i];
            for (int i = 0; i < h4; i++) if (y4 + i < maxY4) left |= lL[y4 + i] | lD[y4 + i];
            ctx = (above != 0) + (left != 0) + 7;
            const int bp = bsizePixels >> sub;
            if (bp * bp > tw * th) ctx += 3;
        }
        const uint16_t *scan = av1g_get_scan(txSz, AV1G_DCT_DCT);
        const int n = std::min(tw, 32) * std::min(th, 32);
        int eob = n;
        while (eob && !qv[scan[eob - 1]]) --eob;
        sym(cdf->txb_skip[txSzCtx][ctx], eob == 0, 2);
        int culLevel = 0, dcCategory = 0;
        if (eob) {
            if (plane == 0) sym(ic->txSet3[AV1G_T(Tx_Size_Sqr)[txSz]], 1, 2);   // DCT_DCT in TX_SET_INTER_3
            int dcSign = 0;
            for (int k = 0; k < w4; k++)
                if (x4 + k < maxX4) dcSign += aD[x4 + k] == 1 ? -1 : aD[x4 + k] == 2 ? 1 : 0;
            for (int k = 0; k < h4; k++)
                if (y4 + k < maxY4) dcSign += lD[y4 + k] == 1 ? -1 : lD[y4 + k] == 2 ? 1 : 0;
            struct Sink {
                av1g_ec &coder;
                av1g_cdfs &tables;
                void sym(uint32_t id, int symbol, int count) {
                    av1g_ec_symbol(&coder, symbol, reinterpret_cast<uint16_t *>(&tables) + id, count);
                }
                void bit(int value) { av1g_ec_bool(&coder, value); }
            } sink{ec, *cdf};
            emitCoeffPayload(sink, qv, eob, txSz, AV1G_DCT_DCT, txSzCtx, plane > 0, dcSign);
            for (int i = 0; i < n; ++i) culLevel += std::abs(qv[i]);
            culLevel = std::min(63, culLevel);
            if (qv[0]) dcCategory = qv[0] < 0 ? 1 : 2;
        }
        for (int i = 0; i < w4; i++) { aL[x4 + i] = (uint8_t)culLevel; aD[x4 + i] = (uint8_t)dcCategory; }
        for (int i = 0; i < h4; i++) { lL[y4 + i] = (uint8_t)culLevel; lD[y4 + i] = (uint8_t)dcCategory; }
    }

    // ---- block / partition ----
    void block(int miRow, int miCol, int size, const pf::BlockInfo &bi, const std::vector<int16_t> &levels,
               const pf::Analyzer *) {
        const int bw4 = 1 << (size - 2), bh4 = bw4;
        const bool availU = inside(miRow - 1, miCol), availL = inside(miRow, miCol - 1);
        const MiInfo *above = availU ? &at(miRow - 1, miCol) : nullptr;
        const MiInfo *left = availL ? &at(miRow, miCol - 1) : nullptr;
        // skip
        int skipCtx = (availU ? above->skip : 0) + (availL ? left->skip : 0);
        const int skip = bi.skip;
        sym(cdf->skip[skipCtx], skip, 2);
        // is_inter
        const bool aboveIntra = availU && !above->isInter, leftIntra = availL && !left->isInter;
        int interCtx;
        if (availU && availL) interCtx = (leftIntra && aboveIntra) ? 3 : (leftIntra || aboveIntra);
        else if (availU || availL) interCtx = 2 * (availU ? aboveIntra : leftIntra);
        else interCtx = 0;
        sym(ic->isInter[interCtx], 1, 2);
        const int refFrame = params->refName[bi.ref];
        writeRefFrame(refFrame, availU, availL, above, left);
        Stack st;
        findMvStack(st, miRow, miCol, size, refFrame);
        // choose the cheapest mode that reproduces the vector
        const pf::Mv mv = bi.mv;
        float best = 1e30f;
        int bestMode = NEWMV, bestIdx = 0;
        const float newmv0 = symCost(ic->newMv[st.newMvCtx], 0), newmv1 = symCost(ic->newMv[st.newMvCtx], 1);
        if (mv.row == 0 && mv.col == 0) {
            const float c = newmv1 + symCost(ic->zeroMv[st.zeroMvCtx], 0);
            if (c < best) { best = c; bestMode = GLOBALMV; }
        }
        const float zero1 = symCost(ic->zeroMv[st.zeroMvCtx], 1);
        if (mv == st.mv[0]) {
            const float c = newmv1 + zero1 + symCost(ic->refMv[st.refMvCtx], 0);
            if (c < best) { best = c; bestMode = NEARESTMV; }
        }
        for (int k = 1; k <= 3; ++k) {
            if (k > 1 && st.num <= k) break;
            if (!(mv == st.mv[k])) continue;
            float c = newmv1 + zero1 + symCost(ic->refMv[st.refMvCtx], 1);
            for (int idx = 1; idx < 3; ++idx) {
                if (st.num > idx + 1) {
                    const int d = idx < k ? 1 : 0;
                    c += symCost(ic->drl[st.drlCtx[idx]], d);
                    if (!d) break;
                }
            }
            if (c < best) { best = c; bestMode = NEARMV; bestIdx = k; }
        }
        for (int k = 0; k <= 2; ++k) {
            if (k > 0 && st.num <= k) break;
            float c = newmv0;
            for (int idx = 0; idx < 2; ++idx) {
                if (st.num > idx + 1) {
                    const int d = idx < k ? 1 : 0;
                    c += symCost(ic->drl[st.drlCtx[idx]], d);
                    if (!d) break;
                }
            }
            const pf::Mv pred = st.mv[st.num <= 1 ? 0 : k];
            c += mvCost(mv, pred);
            if (c < best) { best = c; bestMode = NEWMV; bestIdx = k; }
        }
        // write the mode
        if (bestMode == NEWMV) {
            sym(ic->newMv[st.newMvCtx], 0, 2);
            modeCount[3]++;
        } else {
            sym(ic->newMv[st.newMvCtx], 1, 2);
            if (bestMode == GLOBALMV) {
                sym(ic->zeroMv[st.zeroMvCtx], 0, 2);
                modeCount[2]++;
            } else {
                sym(ic->zeroMv[st.zeroMvCtx], 1, 2);
                sym(ic->refMv[st.refMvCtx], bestMode == NEARMV, 2);
                modeCount[bestMode == NEARMV]++;
            }
        }
        if (bestMode == NEWMV) {
            for (int idx = 0; idx < 2; ++idx) {
                if (st.num > idx + 1) {
                    const int d = idx < bestIdx ? 1 : 0;
                    sym(ic->drl[st.drlCtx[idx]], d, 2);
                    if (!d) break;
                }
            }
            writeMv(mv, st.mv[st.num <= 1 ? 0 : bestIdx]);
        } else if (bestMode == NEARMV) {
            for (int idx = 1; idx < 3; ++idx) {
                if (st.num > idx + 1) {
                    const int d = idx < bestIdx ? 1 : 0;
                    sym(ic->drl[st.drlCtx[idx]], d, 2);
                    if (!d) break;
                }
            }
        }
        // store mode info
        for (int r = miRow; r < std::min(miRow + bh4, miRows); ++r)
            for (int c = miCol; c < std::min(miCol + bw4, miCols); ++c) {
                MiInfo &m = at(r, c);
                m.size = (uint8_t)size;
                m.skip = (uint8_t)skip;
                m.isInter = 1;
                m.mode = (uint8_t)bestMode;
                m.ref[0] = (int8_t)refFrame;
                m.ref[1] = -1;
                m.mv[0] = mv;
                m.mv[1] = pf::Mv{0, 0};
                m.written = true;
            }
        // residual
        if (skip) {
            for (int p = 0; p < 3; ++p) {
                const int sub = p > 0;
                for (int i = miCol >> sub; i < ((miCol + bw4) >> sub); i++) aboveLevel[p][i] = aboveDc[p][i] = 0;
                for (int i = miRow >> sub; i < ((miRow + bh4) >> sub); i++) leftLevel[p][i] = leftDc[p][i] = 0;
            }
            return;
        }
        const int bsize = 4 * bw4;
        for (int p = 0; p < 3; ++p) {
            const int sub = p > 0;
            const int t = bsize >> sub;
            const int txSz = t == 4 ? AV1G_TX_4X4 : t == 8 ? AV1G_TX_8X8 : t == 16 ? AV1G_TX_16X16 : AV1G_TX_32X32;
            const int px = (miCol * 4) >> sub, py = (miRow * 4) >> sub;
            const int pw = p ? aw / 2 : aw;
            const size_t base = p ? size_t(aw) * ah + size_t(p - 1) * (aw / 2) * (ah / 2) : 0;
            int16_t qv[1024];
            for (int r = 0; r < t; ++r)
                for (int c = 0; c < t; ++c) qv[r * t + c] = levels[base + size_t(py + r) * pw + px + c];
            coeffs(p, px >> 2, py >> 2, txSz, bsize, qv);
        }
    }

    void partition(int miRow, int miCol, int size, const std::vector<pf::BlockInfo> &grid,
                   const std::vector<int16_t> &levels) {
        if (miRow >= miRows || miCol >= miCols) return;
        const int num4x4 = 1 << (size - 2), half = num4x4 >> 1;
        const bool hasRows = miRow + half < miRows, hasCols = miCol + half < miCols;
        const int gcols = aw / 8;
        const pf::BlockInfo &bi = grid[size_t(miRow / 2) * gcols + miCol / 2];
        const bool split = bi.log2 < size;
        if (size > 3 || true) {
            const int ctxAbove = inside(miRow - 1, miCol) && at(miRow - 1, miCol).size < size;
            const int ctxLeft = inside(miRow, miCol - 1) && at(miRow, miCol - 1).size < size;
            const int pcx = ctxLeft * 2 + ctxAbove;
            uint16_t *pc = size == 6 ? cdf->partition_w64[pcx] : size == 5 ? cdf->partition_w32[pcx]
                         : size == 4 ? cdf->partition_w16[pcx] : cdf->partition_w8[pcx];
            if (hasRows && hasCols) {
                sym(pc, split ? AV1G_PARTITION_SPLIT : AV1G_PARTITION_NONE, size == 3 ? 4 : 10);
            } else if (hasCols || hasRows) {
                if (!split) throw std::runtime_error("edge block must split");
                uint16_t t[2] = {uint16_t(32768 - av1g_split_or_psum(pc, hasRows)), 32768};
                av1g_ec_encode_cdf(&ec, 1, t, 2);
            } else if (!split) {
                throw std::runtime_error("edge block must split");
            }
        }
        if (split) {
            if (size == 3) throw std::runtime_error("8x8 blocks cannot split");
            const int q4 = half;
            partition(miRow, miCol, size - 1, grid, levels);
            partition(miRow, miCol + q4, size - 1, grid, levels);
            partition(miRow + q4, miCol, size - 1, grid, levels);
            partition(miRow + q4, miCol + q4, size - 1, grid, levels);
            return;
        }
        if (bi.log2 != size) throw std::runtime_error("partition grid mismatch");
        block(miRow, miCol, size, bi, levels, nullptr);
    }

    void tile(const std::vector<pf::BlockInfo> &grid, const std::vector<int16_t> &levels) {
        for (auto &m : mi) m = MiInfo{};
        for (int p = 0; p < 3; ++p) {
            aboveLevel[p].assign(miCols + 32, 0);
            aboveDc[p].assign(miCols + 32, 0);
            leftLevel[p].assign(miRows + 32, 0);
            leftDc[p].assign(miRows + 32, 0);
        }
        for (auto &m : modeCount) m = 0;
        for (int sbr = 0; sbr < miRows; sbr += 16) {
            for (int p = 0; p < 3; ++p) {
                std::fill(leftLevel[p].begin(), leftLevel[p].end(), 0);
                std::fill(leftDc[p].begin(), leftDc[p].end(), 0);
            }
            for (int sbc = 0; sbc < miCols; sbc += 16) partition(sbr, sbc, 6, grid, levels);
        }
    }
};

}  // namespace pfw
