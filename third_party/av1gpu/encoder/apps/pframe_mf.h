// Temporal motion vector prediction on the host: the motion field saved with each reference slot (spec 7.19) and its
// projection into MotionFieldMvs for the current frame (spec 7.9), exactly as decoders compute them. The projection
// writes scattered positions in a fixed order (last writer wins), so it stays a sequential host loop (~0.1 ms at 1080p).
#pragma once
#include "pframe.h"
#include "pframe_writer.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace pfm {

constexpr int kInvalid = -32768;

struct SlotField {
    bool inter = false;          // RefFrameType is an inter frame (key frames are never projected)
    int savedHint[8] = {};       // SavedOrderHints[slot][ref]
    std::vector<int8_t> ref;     // SavedRefFrames per 8x8 (h8 x w8), -1 NONE
    std::vector<pf::Mv> mv;      // SavedMvs per 8x8
};

struct MotionField {
    int w8 = 0, h8 = 0, bits = 7;
    SlotField slot[8];
    std::vector<pf::Mv> out;     // MotionFieldMvs[8][h8][w8]

    void init(int miCols, int miRows, int orderHintBits) {
        w8 = miCols >> 1; h8 = miRows >> 1; bits = orderHintBits;
        out.assign(size_t(8) * w8 * h8, pf::Mv{0, 0});
    }
    void keyFrame() { for (auto &s : slot) s.inter = false; }

    // Motion field motion vector storage (7.19) of the frame just coded into slot outSlot. grid: decisions (gcols
    // per row); refName: AV1 reference of BlockInfo::ref 0, 1 and 3; orderHints[ref]: OrderHints of the frame.
    void store(int outSlot, const pf::BlockInfo *grid, int gcols, const int refName[3], const int orderHints[8],
               int orderHint) {
        SlotField &s = slot[outSlot];
        s.inter = true;
        for (int r = 0; r < 8; ++r) s.savedHint[r] = orderHints[r];
        s.ref.assign(size_t(w8) * h8, -1);
        s.mv.assign(size_t(w8) * h8, pf::Mv{0, 0});
        for (int y = 0; y < h8; ++y)
            for (int x = 0; x < w8; ++x) {
                const pf::BlockInfo &b = grid[size_t(y) * gcols + x];
                const int refs[2] = {refName[b.ref == 1 ? 1 : b.ref == 3 ? 2 : 0], b.ref == 2 ? refName[1] : -1};
                const pf::Mv mvs[2] = {b.mv, b.mv2};
                for (int list = 0; list < 2; ++list) {
                    const int r = refs[list];
                    if (r <= 0) continue;
                    if (pfw::relDist(orderHints[r], orderHint, bits) >= 0) continue;
                    if (std::abs(mvs[list].row) > (1 << 12) - 1 || std::abs(mvs[list].col) > (1 << 12) - 1) continue;
                    s.ref[size_t(y) * w8 + x] = (int8_t)r;
                    s.mv[size_t(y) * w8 + x] = mvs[list];
                }
            }
    }

    static pf::Mv projectMv(pf::Mv mv, int num, int den) {   // get_mv_projection
        static const int divMult[32] = {0,    16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
                                        1489, 1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
                                        744,  712,   682,  655,  630,  606,  585,  564,  546,  528};
        const int d = den < 31 ? den : 31, n = num < -31 ? -31 : num > 31 ? 31 : num;
        auto one = [&](int v) {
            const long long x = (long long)v * n * divMult[d];
            const long long r = x >= 0 ? (x + (1 << 13)) >> 14 : -((-x + (1 << 13)) >> 14);
            return (int16_t)(r < -(1 << 14) + 1 ? -(1 << 14) + 1 : r > (1 << 14) - 1 ? (1 << 14) - 1 : r);
        };
        return pf::Mv{one(mv.row), one(mv.col)};
    }
    static int project(int v8, int delta, int dstSign, int max8, int maxOff8, bool &valid) {
        const int base8 = (v8 >> 3) << 3;
        const int off8 = delta >= 0 ? delta >> 6 : -((-delta) >> 6);
        v8 += dstSign * off8;
        if (v8 < 0 || v8 >= max8 || v8 < base8 - maxOff8 || v8 >= base8 + 8 + maxOff8) valid = false;
        return v8;
    }
    // Projection process (7.9.2) of reference src (1..7); returns whether the source was usable.
    bool projectFrom(int src, int dstSign, const int refSlot[7], const int orderHints[8], int orderHint) {
        const SlotField &s = slot[refSlot[src - 1]];
        if (!s.inter || s.ref.empty()) return false;
        for (int y8 = 0; y8 < h8; ++y8)
            for (int x8 = 0; x8 < w8; ++x8) {
                const int srcRef = s.ref[size_t(y8) * w8 + x8];
                if (srcRef <= 0) continue;
                const int refToCur = pfw::relDist(orderHints[src], orderHint, bits);
                const int refOffset = pfw::relDist(orderHints[src], s.savedHint[srcRef], bits);
                if (!(std::abs(refToCur) <= 31 && std::abs(refOffset) <= 31 && refOffset > 0)) continue;
                const pf::Mv mv = s.mv[size_t(y8) * w8 + x8];
                const pf::Mv proj = projectMv(mv, refToCur * dstSign, refOffset);
                bool valid = true;
                const int py = project(y8, proj.row, dstSign, h8, 0, valid);
                const int px = project(x8, proj.col, dstSign, w8, 8, valid);
                if (!valid) continue;
                for (int dst = 1; dst <= 7; ++dst) {
                    const int refToDst = pfw::relDist(orderHint, orderHints[dst], bits);
                    out[(size_t(dst) * h8 + py) * w8 + px] = projectMv(mv, refToDst, refOffset);
                }
            }
        return true;
    }
    // Motion field estimation (7.9.1) for the current frame.
    const pf::Mv *estimate(const int refSlot[7], const int orderHints[8], int orderHint) {
        for (auto &m : out) m = pf::Mv{(int16_t)kInvalid, (int16_t)kInvalid};
        const int lastIdx = refSlot[0];
        const int curGold = orderHints[4];
        const bool useLast = !slot[lastIdx].inter ? true : slot[lastIdx].savedHint[7] != curGold;
        if (useLast) projectFrom(1, -1, refSlot, orderHints, orderHint);
        int refStamp = 3 - 2;
        if (pfw::relDist(orderHints[5], orderHint, bits) > 0 && projectFrom(5, 1, refSlot, orderHints, orderHint)) --refStamp;
        if (pfw::relDist(orderHints[6], orderHint, bits) > 0 && projectFrom(6, 1, refSlot, orderHints, orderHint)) --refStamp;
        if (pfw::relDist(orderHints[7], orderHint, bits) > 0 && refStamp >= 0 &&
            projectFrom(7, 1, refSlot, orderHints, orderHint))
            --refStamp;
        if (refStamp >= 0) projectFrom(2, -1, refSlot, orderHints, orderHint);
        return out.data();
    }
};

}  // namespace pfm
