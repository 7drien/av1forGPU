// AV1 inter tile syntax for the CUDA P-frame decisions (pframe.h), host + device.
//
// Square partitions 8x8..64x64, single-reference or LAST + ALTREF compound inter blocks with per-block motion
// vectors, TX_MODE_LARGEST, DCT_DCT residuals. Motion vector candidates follow the AV1 specification, section
// 7.10.2 (spatial candidates only: use_ref_frame_mvs is 0 because enable_order_hint is 0; global motion is the
// identity; RefFrameSignBias is 0). Every tile is coded independently, so one thread codes one tile.
#pragma once
#include "pframe.h"
#include "av1g/hd.h"
#include "av1g/entenc.h"
#include "av1g/symstream.h"
#include "av1g/cdfctx.h"
#include "av1g/dsp.h"

namespace pft2 {

enum { INTRA_FRAME = 0, LAST_FRAME = 1, LAST2_FRAME, LAST3_FRAME, GOLDEN_FRAME, BWDREF_FRAME, ALTREF2_FRAME,
       ALTREF_FRAME };
enum { NEARESTMV = 14, NEARMV, GLOBALMV, NEWMV, NEAREST_NEARESTMV, NEAR_NEARMV, NEAREST_NEWMV, NEW_NEARESTMV,
       NEAR_NEWMV, NEW_NEARMV, GLOBAL_GLOBALMV, NEW_NEWMV };
constexpr int kStack = 8;          // MAX_REF_MV_STACK_SIZE
constexpr int kRefCatLevel = 640;  // REF_CAT_LEVEL
constexpr int kMvBorder = 128;     // MV_BORDER

// Inter-specific adaptive CDFs (spec format: increasing, 32768 terminator, adaptation counter).
struct InterCdfs {
    uint16_t isInter[4][3];
    uint16_t singleRef[3][6][3];
    uint16_t newMv[6][3], zeroMv[2][3], refMv[6][3], drl[3][3];
    uint16_t mvJoint[5];
    uint16_t mvSign[2][3], mvClass[2][12], class0Bit[2][3], class0Fr[2][2][5], mvFr[2][5], mvBit[2][10][3];
    uint16_t txSet3[4][3];
    uint16_t compMode[5][3], compRefType[5][3], compRef[3][3][3], compBwdRef[3][2][3], compoundMode[8][9];
    uint16_t txfmSplit[21][3];
};
// All CDFs of a frame: what a reference slot saves at the end of a frame.
struct EntropyState {
    av1g_cdfs cdf;
    InterCdfs inter;
};

// Decoders clear every CDF's adaptation counter (the entry after the final 32768) when they save frame-end CDFs.
AV1G_HDI void clearCounters(uint16_t *v, int n) {
    for (int i = 0; i + 1 < n; ++i)
        if (v[i] == 32768) v[++i] = 0;
}

struct MiInfo {
    uint8_t size;      // log2 of the block size in pixels (3..6)
    uint8_t skip, isInter, mode;
    int8_t ref[2];
    pf::Mv mv[2];
    uint8_t written;
    uint8_t txLog2;    // log2 of the luma transform width (InterTxSizes)
};

struct TileStats {
    float bits[4];     // partition + mode info, motion vectors, luma coefficients, chroma coefficients
    int modeCount[4];  // NEAREST, NEAR, GLOBAL, NEW (first list)
    int compound;
    int error;         // nonzero: the decisions could not be coded
};

struct FrameSyntax {
    int refName[2];       // AV1 reference of BlockInfo::ref 0 and 1
    int referenceSelect;
    int txModeSelect;     // TX_MODE_SELECT: coded blocks signal txfm_split
    int miRows, miCols, aw, ah, gcols;
};

AV1G_HDI float symCost(const uint16_t *cdf, int s) {
    const int p = cdf[s] - (s ? cdf[s - 1] : 0);
#ifdef __CUDA_ARCH__
    return -log2f((float)(p > 1 ? p : 1) * (1.f / 32768.f));
#else
    return -std::log2((float)(p > 1 ? p : 1) * (1.f / 32768.f));
#endif
}
AV1G_HDI int imin(int a, int b) { return a < b ? a : b; }
AV1G_HDI int imax(int a, int b) { return a > b ? a : b; }
AV1G_HDI int iabs(int a) { return a < 0 ? -a : a; }
AV1G_HDI bool mvEq(pf::Mv a, pf::Mv b) { return a.row == b.row && a.col == b.col; }

// Coefficient syntax shared with the keyframe coder (cdf ids are offsets in av1g_cdfs).
struct CoeffSinkT {
    av1g_ec *coder;
    av1g_cdfs *tables;
    AV1G_HDI void sym(uint32_t id, int symbol, int count) {
        av1g_ec_symbol(coder, symbol, reinterpret_cast<uint16_t *>(tables) + id, count);
    }
    AV1G_HDI void bit(int value) { av1g_ec_bool(coder, value); }
};

struct TileCoder {
    // frame
    const FrameSyntax *fs;
    const pf::BlockInfo *grid;
    const int16_t *levels;   // spatial layout of the aligned planes
    MiInfo *mi;              // frame-sized mode info (this tile writes its own area)
    // tile
    int rowStart, rowEnd, colStart, colEnd;   // mode-info units
    av1g_cdfs *cdf;
    InterCdfs *ic;
    av1g_ec ec;
    uint8_t *aboveLevel[3], *aboveDc[3], *leftLevel[3], *leftDc[3];   // absolute 4x4 indices
    TileStats st;

    AV1G_HDI float tell() const {
#ifdef __CUDA_ARCH__
        return ec.offs * 8.f + ec.cnt + 16.f - log2f((float)ec.rng);
#else
        return ec.offs * 8.f + ec.cnt + 16.f - std::log2((float)ec.rng);
#endif
    }
    AV1G_HDI MiInfo &at(int r, int c) { return mi[r * fs->miCols + c]; }
    AV1G_HDI bool inside(int r, int c) const { return r >= rowStart && c >= colStart && r < rowEnd && c < colEnd; }
    AV1G_HDI void sym(uint16_t *c, int s, int n) { av1g_ec_symbol(&ec, s, c, n); }

    // ---- motion vector prediction ----
    struct Stack {
        int num, newCount;
        pf::Mv mv[kStack + 1][2];
        int weight[kStack + 1];
        bool foundMatch;
        int closeMatches, totalMatches;
        int newMvCtx, refMvCtx, zeroMvCtx;
        int drlCtx[kStack + 1];
    };
    struct RefPair { int r0, r1; bool compound; };
    AV1G_HDI static bool hasNewmv(int mode) {
        return mode == NEWMV || mode == NEW_NEWMV || mode == NEAR_NEWMV || mode == NEW_NEARMV ||
               mode == NEAREST_NEWMV || mode == NEW_NEARESTMV;
    }
    AV1G_HDI static pf::Mv lowerPrecision(pf::Mv m) {
        if (m.row & 1) m.row += m.row > 0 ? -1 : 1;
        if (m.col & 1) m.col += m.col > 0 ? -1 : 1;
        return m;
    }
    AV1G_HDI void addRefMv(Stack &s, int mvRow, int mvCol, const RefPair &rp, int weight) {
        const MiInfo &cand = at(mvRow, mvCol);
        if (!cand.isInter) return;
        if (!rp.compound) {
            for (int list = 0; list < 2; ++list) {
                if (cand.ref[list] != rp.r0) continue;
                const pf::Mv candMv = lowerPrecision(cand.mv[list]);
                if (hasNewmv(cand.mode)) s.newCount++;
                s.foundMatch = true;
                int idx = 0;
                for (; idx < s.num; ++idx)
                    if (mvEq(s.mv[idx][0], candMv)) break;
                if (idx < s.num) s.weight[idx] += weight;
                else if (s.num < kStack) {
                    s.mv[s.num][0] = candMv;
                    s.weight[s.num] = weight;
                    s.num++;
                }
            }
        } else if (cand.ref[0] == rp.r0 && cand.ref[1] == rp.r1) {
            const pf::Mv c0 = lowerPrecision(cand.mv[0]), c1 = lowerPrecision(cand.mv[1]);
            s.foundMatch = true;
            int idx = 0;
            for (; idx < s.num; ++idx)
                if (mvEq(s.mv[idx][0], c0) && mvEq(s.mv[idx][1], c1)) break;
            if (idx < s.num) s.weight[idx] += weight;
            else if (s.num < kStack) {
                s.mv[s.num][0] = c0;
                s.mv[s.num][1] = c1;
                s.weight[s.num] = weight;
                s.num++;
            }
            if (hasNewmv(cand.mode)) s.newCount++;
        }
    }
    AV1G_HDI void scanRow(Stack &s, int miRow, int miCol, int bw4, int deltaRow, const RefPair &rp) {
        const int end4 = imin(imin(bw4, fs->miCols - miCol), 16);
        int deltaCol = 0;
        const bool useStep16 = bw4 >= 16;
        if (iabs(deltaRow) > 1) {
            deltaRow += miRow & 1;
            deltaCol = 1 - (miCol & 1);
        }
        for (int i = 0; i < end4;) {
            const int r = miRow + deltaRow, c = miCol + deltaCol + i;
            if (!inside(r, c)) break;
            int len = imin(bw4, 1 << (at(r, c).size - 2));
            if (iabs(deltaRow) > 1) len = imax(2, len);
            if (useStep16) len = imax(4, len);
            addRefMv(s, r, c, rp, len * 2);
            i += len;
        }
    }
    AV1G_HDI void scanCol(Stack &s, int miRow, int miCol, int bh4, int deltaCol, const RefPair &rp) {
        const int end4 = imin(imin(bh4, fs->miRows - miRow), 16);
        int deltaRow = 0;
        const bool useStep16 = bh4 >= 16;
        if (iabs(deltaCol) > 1) {
            deltaRow = 1 - (miRow & 1);
            deltaCol += miCol & 1;
        }
        for (int i = 0; i < end4;) {
            const int r = miRow + deltaRow + i, c = miCol + deltaCol;
            if (!inside(r, c)) break;
            int len = imin(bh4, 1 << (at(r, c).size - 2));
            if (iabs(deltaCol) > 1) len = imax(2, len);
            if (useStep16) len = imax(4, len);
            addRefMv(s, r, c, rp, len * 2);
            i += len;
        }
    }
    AV1G_HDI void scanPoint(Stack &s, int miRow, int miCol, int deltaRow, int deltaCol, const RefPair &rp) {
        const int r = miRow + deltaRow, c = miCol + deltaCol;
        if (inside(r, c) && at(r, c).written) addRefMv(s, r, c, rp, 4);
    }
    AV1G_HDI static void sortStack(Stack &s, int start, int end) {
        while (end > start) {
            int newEnd = start;
            for (int idx = start + 1; idx < end; ++idx) {
                if (s.weight[idx - 1] < s.weight[idx]) {
                    const int tw = s.weight[idx - 1]; s.weight[idx - 1] = s.weight[idx]; s.weight[idx] = tw;
                    for (int l = 0; l < 2; ++l) {
                        const pf::Mv tm = s.mv[idx - 1][l]; s.mv[idx - 1][l] = s.mv[idx][l]; s.mv[idx][l] = tm;
                    }
                    newEnd = idx;
                }
            }
            end = newEnd;
        }
    }
    AV1G_HDI void extraSearch(Stack &s, int miRow, int miCol, int bw4, int bh4, const RefPair &rp) {
        pf::Mv idMvs[2][2], diffMvs[2][2];
        int idCount[2] = {0, 0}, diffCount[2] = {0, 0};
        int w4 = imin(16, bw4), h4 = imin(16, bh4);
        w4 = imin(w4, fs->miCols - miCol);
        h4 = imin(h4, fs->miRows - miRow);
        const int num4x4 = imin(w4, h4);
        const int refs[2] = {rp.r0, rp.r1};
        for (int pass = 0; pass < 2; ++pass) {
            int idx = 0;
            while (idx < num4x4 && s.num < 2) {
                int r, c;
                if (pass == 0) { r = miRow - 1; c = miCol + idx; }
                else { r = miRow + idx; c = miCol - 1; }
                if (!inside(r, c)) break;
                const MiInfo &cand = at(r, c);
                for (int candList = 0; candList < 2; ++candList) {
                    const int candRef = cand.ref[candList];
                    if (candRef <= INTRA_FRAME) continue;
                    const pf::Mv candMv = cand.mv[candList];
                    if (rp.compound) {
                        for (int list = 0; list < 2; ++list) {
                            if (candRef == refs[list] && idCount[list] < 2) idMvs[list][idCount[list]++] = candMv;
                            else if (diffCount[list] < 2) diffMvs[list][diffCount[list]++] = candMv;
                        }
                    } else {
                        int k = 0;
                        for (; k < s.num; ++k)
                            if (mvEq(candMv, s.mv[k][0])) break;
                        if (k == s.num) {
                            s.mv[k][0] = candMv;
                            s.weight[k] = 2;
                            s.num++;
                        }
                    }
                }
                idx += 1 << (cand.size - 2);
            }
        }
        if (rp.compound) {
            pf::Mv combined[2][2];
            for (int list = 0; list < 2; ++list) {
                int n = 0;
                for (int i = 0; i < idCount[list]; ++i) combined[n++][list] = idMvs[list][i];
                for (int i = 0; i < diffCount[list] && n < 2; ++i) combined[n++][list] = diffMvs[list][i];
                while (n < 2) combined[n++][list] = pf::Mv{0, 0};
            }
            if (s.num == 1) {
                const int k = (mvEq(combined[0][0], s.mv[0][0]) && mvEq(combined[0][1], s.mv[0][1])) ? 1 : 0;
                s.mv[1][0] = combined[k][0];
                s.mv[1][1] = combined[k][1];
                s.weight[1] = 2;
                s.num = 2;
            } else {
                for (int i = 0; i < 2; ++i) {
                    s.mv[s.num][0] = combined[i][0];
                    s.mv[s.num][1] = combined[i][1];
                    s.weight[s.num] = 2;
                    s.num++;
                }
            }
        } else {
            for (int k = s.num; k < 2; ++k) s.mv[k][0] = pf::Mv{0, 0};
        }
    }
    AV1G_HDI void findMvStack(Stack &s, int miRow, int miCol, int size, const RefPair &rp) {
        const int bw4 = 1 << (size - 2), bh4 = bw4;
        s.num = 0; s.newCount = 0;
        s.foundMatch = false;
        scanRow(s, miRow, miCol, bw4, -1, rp);
        bool foundAbove = s.foundMatch;
        s.foundMatch = false;
        scanCol(s, miRow, miCol, bh4, -1, rp);
        bool foundLeft = s.foundMatch;
        s.foundMatch = false;
        if (imax(bw4, bh4) <= 16) scanPoint(s, miRow, miCol, -1, bw4, rp);
        if (s.foundMatch) foundAbove = true;
        s.closeMatches = foundAbove + foundLeft;
        const int numNearest = s.num, numNew = s.newCount;
        for (int i = 0; i < numNearest; ++i) s.weight[i] += kRefCatLevel;
        s.zeroMvCtx = 0;
        s.foundMatch = false;
        scanPoint(s, miRow, miCol, -1, -1, rp);
        if (s.foundMatch) foundAbove = true;
        s.foundMatch = false;
        scanRow(s, miRow, miCol, bw4, -3, rp);
        if (s.foundMatch) foundAbove = true;
        s.foundMatch = false;
        scanCol(s, miRow, miCol, bh4, -3, rp);
        if (s.foundMatch) foundLeft = true;
        s.foundMatch = false;
        if (bh4 > 1) scanRow(s, miRow, miCol, bw4, -5, rp);
        if (s.foundMatch) foundAbove = true;
        s.foundMatch = false;
        if (bw4 > 1) scanCol(s, miRow, miCol, bh4, -5, rp);
        if (s.foundMatch) foundLeft = true;
        s.totalMatches = foundAbove + foundLeft;
        sortStack(s, 0, numNearest);
        sortStack(s, numNearest, s.num);
        if (s.num < 2) extraSearch(s, miRow, miCol, bw4, bh4, rp);
        for (int idx = 0; idx < s.num; ++idx) {
            int z = 0;
            if (idx + 1 < s.num) {
                const int w0 = s.weight[idx], w1 = s.weight[idx + 1];
                if (w0 >= kRefCatLevel) {
                    if (w1 < kRefCatLevel) z = 1;
                } else {
                    z = 2;
                }
            }
            s.drlCtx[idx] = z;
        }
        const int bw = 4 * bw4, bh = 4 * bh4;
        const int top = -((miRow * 4) * 8), bottom = ((fs->miRows - bh4 - miRow) * 4) * 8;
        const int left = -((miCol * 4) * 8), right = ((fs->miCols - bw4 - miCol) * 4) * 8;
        const int br = kMvBorder + bh * 8, bc = kMvBorder + bw * 8;
        for (int list = 0; list < 1 + rp.compound; ++list)
            for (int idx = 0; idx < s.num; ++idx) {
                pf::Mv &m = s.mv[idx][list];
                m.row = (int16_t)imin(imax((int)m.row, top - br), bottom + br);
                m.col = (int16_t)imin(imax((int)m.col, left - bc), right + bc);
            }
        if (s.closeMatches == 0) {
            s.newMvCtx = imin(s.totalMatches, 1);
            s.refMvCtx = s.totalMatches;
        } else if (s.closeMatches == 1) {
            s.newMvCtx = 3 - imin(numNew, 1);
            s.refMvCtx = 2 + s.totalMatches;
        } else {
            s.newMvCtx = 5 - imin(numNew, 1);
            s.refMvCtx = 5;
        }
    }

    // ---- motion vectors ----
    AV1G_HDI static int mvClassOf(int v, int &rem) {   // v = |diff| - 1 (1/8 units, even)
        if (v < 16) { rem = v; return 0; }
        int c = 1;
        while (c < 10 && v >= (1 << (c + 4))) c++;
        rem = v - (1 << (c + 3));
        return c;
    }
    AV1G_HDI float mvCompCost(int comp, int d) const {
        const int sign = d < 0, mag = iabs(d);
        float bits = symCost(ic->mvSign[comp], sign);
        int rem;
        const int c = mvClassOf(mag - 1, rem);
        bits += symCost(ic->mvClass[comp], c);
        if (!c) {
            const int bit = rem >> 3, fr = (rem >> 1) & 3;
            bits += symCost(ic->class0Bit[comp], bit) + symCost(ic->class0Fr[comp][bit], fr);
        } else {
            const int d2 = rem >> 3, fr = (rem >> 1) & 3;
            for (int i = 0; i < c; ++i) bits += symCost(ic->mvBit[comp][i], (d2 >> i) & 1);
            bits += symCost(ic->mvFr[comp], fr);
        }
        return bits;
    }
    AV1G_HDI float mvCost(pf::Mv mv, pf::Mv pred) const {
        const int dr = mv.row - pred.row, dc = mv.col - pred.col;
        float bits = symCost(ic->mvJoint, (dr != 0) * 2 + (dc != 0));
        if (dr) bits += mvCompCost(0, dr);
        if (dc) bits += mvCompCost(1, dc);
        return bits;
    }
    AV1G_HDI void writeMvComp(int comp, int d) {
        const int sign = d < 0, mag = iabs(d);
        sym(ic->mvSign[comp], sign, 2);
        int rem;
        const int c = mvClassOf(mag - 1, rem);
        sym(ic->mvClass[comp], c, 11);
        if (!c) {
            const int bit = rem >> 3, fr = (rem >> 1) & 3;
            sym(ic->class0Bit[comp], bit, 2);
            sym(ic->class0Fr[comp][bit], fr, 4);
        } else {
            const int d2 = rem >> 3, fr = (rem >> 1) & 3;
            for (int i = 0; i < c; ++i) sym(ic->mvBit[comp][i], (d2 >> i) & 1, 2);
            sym(ic->mvFr[comp], fr, 4);
        }
    }
    AV1G_HDI void writeMv(pf::Mv mv, pf::Mv pred) {
        const int dr = mv.row - pred.row, dc = mv.col - pred.col;
        sym(ic->mvJoint, (dr != 0) * 2 + (dc != 0), 4);
        if (dr) writeMvComp(0, dr);
        if (dc) writeMvComp(1, dc);
    }

    // ---- reference contexts ----
    AV1G_HDI int countRefs(int frameType, bool availU, bool availL, const MiInfo *above, const MiInfo *left) const {
        int c = 0;
        if (availU) c += (above->ref[0] == frameType) + (above->ref[1] == frameType);
        if (availL) c += (left->ref[0] == frameType) + (left->ref[1] == frameType);
        return c;
    }
    AV1G_HDI static int refCountCtx(int a, int b) { return a < b ? 0 : a == b ? 1 : 2; }
    AV1G_HDI static bool checkBackward(int r) { return r >= BWDREF_FRAME && r <= ALTREF_FRAME; }
    AV1G_HDI static bool sameDir(int a, int b) { return (a >= BWDREF_FRAME) == (b >= BWDREF_FRAME); }
    AV1G_HDI int compModeCtx(bool availU, bool availL, const MiInfo *above, const MiInfo *left) const {
        const int a0 = availU ? above->ref[0] : INTRA_FRAME, l0 = availL ? left->ref[0] : INTRA_FRAME;
        const bool aSingle = !availU || above->ref[1] <= INTRA_FRAME, lSingle = !availL || left->ref[1] <= INTRA_FRAME;
        const bool aIntra = a0 <= INTRA_FRAME, lIntra = l0 <= INTRA_FRAME;
        if (availU && availL) {
            if (aSingle && lSingle) return checkBackward(a0) ^ checkBackward(l0);
            if (aSingle) return 2 + (checkBackward(a0) || aIntra);
            if (lSingle) return 2 + (checkBackward(l0) || lIntra);
            return 4;
        }
        if (availU) return aSingle ? checkBackward(a0) : 3;
        if (availL) return lSingle ? checkBackward(l0) : 3;
        return 1;
    }
    AV1G_HDI int compRefTypeCtx(bool availU, bool availL, const MiInfo *above, const MiInfo *left) const {
        const int a0 = availU ? above->ref[0] : INTRA_FRAME, a1 = availU ? above->ref[1] : -1;
        const int l0 = availL ? left->ref[0] : INTRA_FRAME, l1 = availL ? left->ref[1] : -1;
        const bool aIntra = a0 <= INTRA_FRAME, lIntra = l0 <= INTRA_FRAME;
        const bool aSingle = a1 <= INTRA_FRAME, lSingle = l1 <= INTRA_FRAME;
        const bool aComp = availU && !aIntra && !aSingle, lComp = availL && !lIntra && !lSingle;
        const bool aUni = aComp && sameDir(a0, a1), lUni = lComp && sameDir(l0, l1);
        if (availU && !aIntra && availL && !lIntra) {
            const int samedir = sameDir(a0, l0);
            if (!aComp && !lComp) return 1 + 2 * samedir;
            if (!aComp) return lUni ? 3 + samedir : 1;
            if (!lComp) return aUni ? 3 + samedir : 1;
            if (!aUni && !lUni) return 0;
            if (!aUni || !lUni) return 2;
            return 3 + ((a0 == BWDREF_FRAME) == (l0 == BWDREF_FRAME));
        }
        if (availU && availL) {
            if (aComp) return 1 + 2 * aUni;
            if (lComp) return 1 + 2 * lUni;
            return 2;
        }
        if (aComp) return 4 * aUni;
        if (lComp) return 4 * lUni;
        return 2;
    }
    AV1G_HDI void writeSingleRef(int refFrame, bool availU, bool availL, const MiInfo *above, const MiInfo *left) {
#define PF_CNT(f) countRefs(f, availU, availL, above, left)
        const int fwd = PF_CNT(LAST_FRAME) + PF_CNT(LAST2_FRAME) + PF_CNT(LAST3_FRAME) + PF_CNT(GOLDEN_FRAME);
        const int bwd = PF_CNT(BWDREF_FRAME) + PF_CNT(ALTREF2_FRAME) + PF_CNT(ALTREF_FRAME);
        const int p1 = refFrame >= BWDREF_FRAME;
        sym(ic->singleRef[refCountCtx(fwd, bwd)][0], p1, 2);
        if (p1) {
            const int p2 = refFrame == ALTREF_FRAME;
            sym(ic->singleRef[refCountCtx(PF_CNT(BWDREF_FRAME) + PF_CNT(ALTREF2_FRAME), PF_CNT(ALTREF_FRAME))][1], p2, 2);
            if (!p2) sym(ic->singleRef[refCountCtx(PF_CNT(BWDREF_FRAME), PF_CNT(ALTREF2_FRAME))][5], refFrame == ALTREF2_FRAME, 2);
        } else {
            const int p3 = refFrame == LAST3_FRAME || refFrame == GOLDEN_FRAME;
            sym(ic->singleRef[refCountCtx(PF_CNT(LAST_FRAME) + PF_CNT(LAST2_FRAME), PF_CNT(LAST3_FRAME) + PF_CNT(GOLDEN_FRAME))][2], p3, 2);
            if (p3) sym(ic->singleRef[refCountCtx(PF_CNT(LAST3_FRAME), PF_CNT(GOLDEN_FRAME))][4], refFrame == GOLDEN_FRAME, 2);
            else sym(ic->singleRef[refCountCtx(PF_CNT(LAST_FRAME), PF_CNT(LAST2_FRAME))][3], refFrame == LAST2_FRAME, 2);
        }
    }

    // ---- modes ----
    AV1G_HDI static int modeOfList(int yMode, int list) {   // get_mode()
        if (yMode < NEAREST_NEARESTMV) return yMode;
        if (list == 0) {
            if (yMode == NEW_NEWMV || yMode == NEW_NEARESTMV || yMode == NEW_NEARMV) return NEWMV;
            if (yMode == NEAREST_NEARESTMV || yMode == NEAREST_NEWMV) return NEARESTMV;
            if (yMode == NEAR_NEARMV || yMode == NEAR_NEWMV) return NEARMV;
            return GLOBALMV;
        }
        if (yMode == NEW_NEWMV || yMode == NEAREST_NEWMV || yMode == NEAR_NEWMV) return NEWMV;
        if (yMode == NEAREST_NEARESTMV || yMode == NEW_NEARESTMV) return NEARESTMV;
        if (yMode == NEAR_NEARMV || yMode == NEW_NEARMV) return NEARMV;
        return GLOBALMV;
    }
    AV1G_HDI static bool isNewType(int m) { return m == NEWMV || m == NEW_NEWMV; }
    AV1G_HDI static bool hasNearmv(int m) { return m == NEARMV || m == NEAR_NEARMV || m == NEAR_NEWMV || m == NEW_NEARMV; }
    AV1G_HDI static int compoundModeCtx(const Stack &s) {
        const int map[3][5] = {{0, 1, 1, 1, 1}, {1, 2, 3, 4, 4}, {4, 4, 5, 6, 7}};
        return map[s.refMvCtx >> 1][imin(s.newMvCtx, 4)];
    }
    // Cost of coding (yMode, refMvIdx) for target vectors; < 0 when it cannot reproduce them.
    AV1G_HDI float modeCost(const Stack &s, bool compound, int yMode, int refMvIdx, const pf::Mv *target,
                            pf::Mv *pred) const {
        float c = 0.f;
        if (compound) {
            c += symCost(ic->compoundMode[compoundModeCtx(s)], yMode - NEAREST_NEARESTMV);
        } else if (yMode == NEWMV) {
            c += symCost(ic->newMv[s.newMvCtx], 0);
        } else {
            c += symCost(ic->newMv[s.newMvCtx], 1);
            if (yMode == GLOBALMV) c += symCost(ic->zeroMv[s.zeroMvCtx], 0);
            else c += symCost(ic->zeroMv[s.zeroMvCtx], 1) + symCost(ic->refMv[s.refMvCtx], yMode == NEARMV);
        }
        if (isNewType(yMode)) {
            for (int idx = 0; idx < 2; ++idx)
                if (s.num > idx + 1) {
                    const int d = idx < refMvIdx;
                    c += symCost(ic->drl[s.drlCtx[idx]], d);
                    if (!d) break;
                }
        } else if (hasNearmv(yMode)) {
            for (int idx = 1; idx < 3; ++idx)
                if (s.num > idx + 1) {
                    const int d = idx < refMvIdx;
                    c += symCost(ic->drl[s.drlCtx[idx]], d);
                    if (!d) break;
                }
        }
        for (int i = 0; i < 1 + compound; ++i) {
            const int cm = modeOfList(yMode, i);
            pf::Mv p{0, 0};
            if (cm != GLOBALMV) {
                int pos = cm == NEARESTMV ? 0 : refMvIdx;
                if (cm == NEWMV && s.num <= 1) pos = 0;
                p = s.mv[pos][i];
            }
            pred[i] = p;
            if (cm == NEWMV) c += mvCost(target[i], p);
            else if (!mvEq(target[i], p)) return -1.f;
        }
        return c;
    }

    // ---- residual ----
    AV1G_HDI void coeffs(int plane, int x4, int y4, int txSz, int bsizePixels, const int16_t *qv) {
        const int sub = plane > 0;
        const int tw = AV1G_T(Tx_Width)[txSz], th = AV1G_T(Tx_Height)[txSz];
        const int w4 = tw >> 2, h4 = th >> 2;
        const int maxX4 = fs->miCols >> sub, maxY4 = fs->miRows >> sub;
        uint8_t *aL = aboveLevel[plane], *aD = aboveDc[plane];
        uint8_t *lL = leftLevel[plane], *lD = leftDc[plane];
        const int txSzCtx = (AV1G_T(Tx_Size_Sqr)[txSz] + AV1G_T(Tx_Size_Sqr_Up)[txSz] + 1) >> 1;
        int ctx;
        if (plane == 0) {
            if (tw == bsizePixels) {
                ctx = 0;   // the transform covers the whole block
            } else {
                int top = 0, left = 0;
                for (int k = 0; k < w4; k++) if (x4 + k < maxX4 && aL[x4 + k] > top) top = aL[x4 + k];
                for (int k = 0; k < h4; k++) if (y4 + k < maxY4 && lL[y4 + k] > left) left = lL[y4 + k];
                const int mx = imax(top, left), mn = imin(top, left);
                if (!top && !left) ctx = 1;
                else if (!top || !left) ctx = 2 + (mx > 3);
                else if (mx <= 3) ctx = 4;
                else if (mn <= 3) ctx = 5;
                else ctx = 6;
            }
        } else {
            int above = 0, left = 0;
            for (int i = 0; i < w4; i++) if (x4 + i < maxX4) above |= aL[x4 + i] | aD[x4 + i];
            for (int i = 0; i < h4; i++) if (y4 + i < maxY4) left |= lL[y4 + i] | lD[y4 + i];
            ctx = (above != 0) + (left != 0) + 7;
            const int bp = bsizePixels >> sub;
            if (bp * bp > tw * th) ctx += 3;
        }
        const uint16_t *scan = av1g_get_scan(txSz, AV1G_DCT_DCT);
        const int n = imin(tw, 32) * imin(th, 32);
        int eob = n;
        while (eob && !qv[scan[eob - 1]]) --eob;
        sym(cdf->txb_skip[txSzCtx][ctx], eob == 0, 2);
        int culLevel = 0, dcCategory = 0;
        if (eob) {
            // DCT_DCT in TX_SET_INTER_3 (64-point transforms use TX_SET_DCTONLY: nothing coded)
            if (plane == 0 && txSz != AV1G_TX_64X64) sym(ic->txSet3[AV1G_T(Tx_Size_Sqr)[txSz]], 1, 2);
            int dcSign = 0;
            for (int k = 0; k < w4; k++)
                if (x4 + k < maxX4) dcSign += aD[x4 + k] == 1 ? -1 : aD[x4 + k] == 2 ? 1 : 0;
            for (int k = 0; k < h4; k++)
                if (y4 + k < maxY4) dcSign += lD[y4 + k] == 1 ? -1 : lD[y4 + k] == 2 ? 1 : 0;
            CoeffSinkT sink{&ec, cdf};
            emitCoeffPayload(sink, qv, eob, txSz, AV1G_DCT_DCT, txSzCtx, plane > 0, dcSign);
            for (int i = 0; i < n; ++i) culLevel += iabs(qv[i]);
            culLevel = imin(63, culLevel);
            if (qv[0]) dcCategory = qv[0] < 0 ? 1 : 2;
        }
        for (int i = 0; i < w4; i++) { aL[x4 + i] = (uint8_t)culLevel; aD[x4 + i] = (uint8_t)dcCategory; }
        for (int i = 0; i < h4; i++) { lL[y4 + i] = (uint8_t)culLevel; lD[y4 + i] = (uint8_t)dcCategory; }
    }

    // ---- block / partition ----
    AV1G_HDI void block(int miRow, int miCol, int size, const pf::BlockInfo &bi) {
        const int bw4 = 1 << (size - 2), bh4 = bw4;
        const bool availU = inside(miRow - 1, miCol), availL = inside(miRow, miCol - 1);
        const MiInfo *above = availU ? &at(miRow - 1, miCol) : nullptr;
        const MiInfo *left = availL ? &at(miRow, miCol - 1) : nullptr;
        const int skipCtx = (availU ? above->skip : 0) + (availL ? left->skip : 0);
        const int skip = bi.skip;
        float t0 = tell();
        sym(cdf->skip[skipCtx], skip, 2);
        const bool aboveIntra = availU && !above->isInter, leftIntra = availL && !left->isInter;
        int interCtx;
        if (availU && availL) interCtx = (leftIntra && aboveIntra) ? 3 : (leftIntra || aboveIntra);
        else if (availU || availL) interCtx = 2 * (availU ? aboveIntra : leftIntra);
        else interCtx = 0;
        sym(ic->isInter[interCtx], 1, 2);
        const bool compound = bi.ref == 2;
        RefPair rp{fs->refName[bi.ref == 1 ? 1 : 0], compound ? fs->refName[1] : -1, compound};
        if (fs->referenceSelect) sym(ic->compMode[compModeCtx(availU, availL, above, left)], compound, 2);
        if (compound) {
            if (rp.r0 != LAST_FRAME || rp.r1 != ALTREF_FRAME) { st.error = 1; return; }
            sym(ic->compRefType[compRefTypeCtx(availU, availL, above, left)], 1, 2);   // BIDIR_COMP_REFERENCE
            sym(ic->compRef[refCountCtx(PF_CNT(LAST_FRAME) + PF_CNT(LAST2_FRAME), PF_CNT(LAST3_FRAME) + PF_CNT(GOLDEN_FRAME))][0], 0, 2);
            sym(ic->compRef[refCountCtx(PF_CNT(LAST_FRAME), PF_CNT(LAST2_FRAME))][1], 0, 2);   // LAST_FRAME
            sym(ic->compBwdRef[refCountCtx(PF_CNT(BWDREF_FRAME) + PF_CNT(ALTREF2_FRAME), PF_CNT(ALTREF_FRAME))][0], 1, 2);
        } else {
            writeSingleRef(rp.r0, availU, availL, above, left);
        }
#undef PF_CNT
        Stack s;
        findMvStack(s, miRow, miCol, size, rp);
        const pf::Mv target[2] = {bi.mv, bi.mv2};
        float best = 1e30f;
        int bestMode = -1, bestIdx = 0;
        pf::Mv bestPred[2] = {}, pred[2];
        const int first = compound ? NEAREST_NEARESTMV : NEARESTMV, last = compound ? NEW_NEWMV : NEWMV;
        for (int mode = first; mode <= last; ++mode) {
            int k0 = 0, k1 = 0;
            if (isNewType(mode)) { k0 = 0; k1 = 2; }
            else if (hasNearmv(mode)) { k0 = 1; k1 = 3; }
            for (int k = k0; k <= k1; ++k) {
                if (k > k0 && s.num <= k) break;
                const float c = modeCost(s, compound, mode, k, target, pred);
                if (c >= 0.f && c < best) {
                    best = c; bestMode = mode; bestIdx = k;
                    bestPred[0] = pred[0]; bestPred[1] = pred[1];
                }
            }
        }
        if (bestMode < 0) { st.error = 2; return; }
        if (compound) {
            sym(ic->compoundMode[compoundModeCtx(s)], bestMode - NEAREST_NEARESTMV, 8);
        } else if (bestMode == NEWMV) {
            sym(ic->newMv[s.newMvCtx], 0, 2);
        } else {
            sym(ic->newMv[s.newMvCtx], 1, 2);
            if (bestMode == GLOBALMV) {
                sym(ic->zeroMv[s.zeroMvCtx], 0, 2);
            } else {
                sym(ic->zeroMv[s.zeroMvCtx], 1, 2);
                sym(ic->refMv[s.refMvCtx], bestMode == NEARMV, 2);
            }
        }
        if (isNewType(bestMode)) {
            for (int idx = 0; idx < 2; ++idx)
                if (s.num > idx + 1) {
                    const int d = idx < bestIdx;
                    sym(ic->drl[s.drlCtx[idx]], d, 2);
                    if (!d) break;
                }
        } else if (hasNearmv(bestMode)) {
            for (int idx = 1; idx < 3; ++idx)
                if (s.num > idx + 1) {
                    const int d = idx < bestIdx;
                    sym(ic->drl[s.drlCtx[idx]], d, 2);
                    if (!d) break;
                }
        }
        float t1 = tell();
        st.bits[0] += t1 - t0;
        for (int i = 0; i < 1 + compound; ++i)
            if (modeOfList(bestMode, i) == NEWMV) writeMv(target[i], bestPred[i]);
        t0 = tell();
        st.bits[1] += t0 - t1;
        const int m0 = modeOfList(bestMode, 0);
        st.modeCount[m0 == NEARESTMV ? 0 : m0 == NEARMV ? 1 : m0 == GLOBALMV ? 2 : 3]++;
        st.compound += compound;
        for (int r = miRow; r < imin(miRow + bh4, fs->miRows); ++r)
            for (int c = miCol; c < imin(miCol + bw4, fs->miCols); ++c) {
                MiInfo &m = at(r, c);
                m.size = (uint8_t)size;
                m.skip = (uint8_t)skip;
                m.isInter = 1;
                m.mode = (uint8_t)bestMode;
                m.ref[0] = (int8_t)rp.r0;
                m.ref[1] = (int8_t)(compound ? rp.r1 : -1);
                m.mv[0] = target[0];
                m.mv[1] = compound ? target[1] : pf::Mv{0, 0};
                m.written = 1;
                m.txLog2 = (uint8_t)(skip ? size : size - (bi.txSplit ? 1 : 0));
            }
        const bool split = !skip && bi.txSplit;
        if (fs->txModeSelect && !skip) {
            const float tt = tell();
            const int maxTx = size - 2;   // TX_8X8 = 1 .. TX_64X64 = 4 (square transform of the block)
            auto txCtx = [&](int row, int col, int txLog2) {
                int aw2, lh2;
                if (row == miRow) {
                    if (!inside(row - 1, col)) aw2 = 64;
                    else { const MiInfo &n = at(row - 1, col); aw2 = n.skip && n.isInter ? 1 << n.size : 1 << n.txLog2; }
                } else aw2 = 1 << at(row - 1, col).txLog2;
                if (col == miCol) {
                    if (!inside(row, col - 1)) lh2 = 64;
                    else { const MiInfo &n = at(row, col - 1); lh2 = n.skip && n.isInter ? 1 << n.size : 1 << n.txLog2; }
                } else lh2 = 1 << at(row, col - 1).txLog2;
                const int txSz = txLog2 - 2;
                return (txSz != maxTx) * 3 + (4 - maxTx) * 6 + (aw2 < (1 << txLog2)) + (lh2 < (1 << txLog2));
            };
            sym(ic->txfmSplit[txCtx(miRow, miCol, size)], split, 2);
            if (split && size - 1 > 2) {   // children of 4x4 code nothing; depth 1 children code txfm_split = 0
                const int half = 1 << (size - 3);
                for (int k = 0; k < 4; k++) {
                    const int r = miRow + (k >> 1) * half, c = miCol + (k & 1) * half;
                    if (r >= fs->miRows || c >= fs->miCols) continue;
                    sym(ic->txfmSplit[txCtx(r, c, size - 1)], 0, 2);
                }
            }
            st.bits[0] += tell() - tt;
        }
        if (skip) {
            for (int p = 0; p < 3; ++p) {
                const int sub = p > 0;
                for (int i = miCol >> sub; i < ((miCol + bw4) >> sub); i++) aboveLevel[p][i] = aboveDc[p][i] = 0;
                for (int i = miRow >> sub; i < ((miRow + bh4) >> sub); i++) leftLevel[p][i] = leftDc[p][i] = 0;
            }
            return;
        }
        const int bsize = 4 * bw4;
        if (split) {
            // luma: four half-size transforms in raster order (transform_tree)
            const int t = bsize >> 1;
            const int txSz = t == 4 ? AV1G_TX_4X4 : t == 8 ? AV1G_TX_8X8 : t == 16 ? AV1G_TX_16X16 : AV1G_TX_32X32;
            for (int k = 0; k < 4; ++k) {
                const int px = miCol * 4 + (k & 1) * t, py = miRow * 4 + (k >> 1) * t;
                if (px >= fs->miCols * 4 || py >= fs->miRows * 4) continue;
                int16_t qv[1024];
                for (int r = 0; r < t; ++r)
                    for (int c = 0; c < t; ++c) qv[r * t + c] = levels[size_t(py + r) * fs->aw + px + c];
                const float c0 = tell();
                coeffs(0, px >> 2, py >> 2, txSz, bsize, qv);
                st.bits[2] += tell() - c0;
            }
        }
        for (int p = split ? 1 : 0; p < 3; ++p) {
            const int sub = p > 0;
            const int t = bsize >> sub;
            const int txSz = t == 4 ? AV1G_TX_4X4 : t == 8 ? AV1G_TX_8X8 : t == 16 ? AV1G_TX_16X16
                           : t == 32 ? AV1G_TX_32X32 : AV1G_TX_64X64;
            const int tq = imin(t, 32);
            const int px = (miCol * 4) >> sub, py = (miRow * 4) >> sub;
            const int pw = p ? fs->aw / 2 : fs->aw;
            const size_t base = p ? size_t(fs->aw) * fs->ah + size_t(p - 1) * (fs->aw / 2) * (fs->ah / 2) : 0;
            int16_t qv[1024];
            for (int r = 0; r < tq; ++r)
                for (int c = 0; c < tq; ++c) qv[r * tq + c] = levels[base + size_t(py + r) * pw + px + c];
            const float c0 = tell();
            coeffs(p, px >> 2, py >> 2, txSz, bsize, qv);
            st.bits[p ? 3 : 2] += tell() - c0;
        }
    }

    // Recursive partition walk (no recursion on the device: explicit stack of quadrants).
    AV1G_HDI void superblock(int sbRow, int sbCol) {
        struct Node { int r, c, size, k; };
        Node stack[8];
        int top = 0;
        stack[0] = Node{sbRow, sbCol, 6, -1};
        while (top >= 0) {
            Node &n = stack[top];
            if (n.k >= 0) {   // visiting children
                if (++n.k == 4) { --top; continue; }
                const int half = 1 << (n.size - 3);
                stack[top + 1] = Node{n.r + (n.k >> 1) * half, n.c + (n.k & 1) * half, n.size - 1, -1};
                ++top;
                continue;
            }
            const int miRow = n.r, miCol = n.c, size = n.size;
            if (miRow >= fs->miRows || miCol >= fs->miCols) { --top; continue; }
            const int half = 1 << (size - 3);
            const bool hasRows = miRow + half < fs->miRows, hasCols = miCol + half < fs->miCols;
            const pf::BlockInfo &bi = grid[(miRow >> 1) * fs->gcols + (miCol >> 1)];
            const bool split = bi.log2 < size;
            const float t0 = tell();
            const int ctxAbove = inside(miRow - 1, miCol) && at(miRow - 1, miCol).size < size;
            const int ctxLeft = inside(miRow, miCol - 1) && at(miRow, miCol - 1).size < size;
            const int pcx = ctxLeft * 2 + ctxAbove;
            uint16_t *pc = size == 6 ? cdf->partition_w64[pcx] : size == 5 ? cdf->partition_w32[pcx]
                         : size == 4 ? cdf->partition_w16[pcx] : cdf->partition_w8[pcx];
            if (hasRows && hasCols) {
                sym(pc, split ? AV1G_PARTITION_SPLIT : AV1G_PARTITION_NONE, size == 3 ? 4 : 10);
            } else if (hasCols || hasRows) {
                if (!split) { st.error = 3; return; }
                uint16_t t[2] = {uint16_t(32768 - av1g_split_or_psum(pc, hasRows)), 32768};
                av1g_ec_encode_cdf(&ec, 1, t, 2);
            } else if (!split) {
                st.error = 3;
                return;
            }
            st.bits[0] += tell() - t0;
            if (split) {
                if (size == 3) { st.error = 4; return; }
                n.k = -1;
                n.k = 0;
                stack[top + 1] = Node{miRow, miCol, size - 1, -1};
                ++top;
                continue;
            }
            if (bi.log2 != size) { st.error = 5; return; }
            block(miRow, miCol, size, bi);
            if (st.error) return;
            --top;
        }
    }

    AV1G_HDI void tile() {
        for (int r = rowStart; r < rowEnd; ++r)
            for (int c = colStart; c < colEnd; ++c) mi[r * fs->miCols + c] = MiInfo{};
        for (int p = 0; p < 3; ++p) {
            const int sub = p > 0;
            for (int i = colStart >> sub; i < ((colEnd + sub) >> sub) + 16; ++i) aboveLevel[p][i] = aboveDc[p][i] = 0;
        }
        for (int sbr = rowStart; sbr < rowEnd; sbr += 16) {
            for (int p = 0; p < 3; ++p) {
                const int sub = p > 0;
                for (int i = sbr >> sub; i < ((sbr + 16) >> sub) + 16; ++i) leftLevel[p][i] = leftDc[p][i] = 0;
            }
            for (int sbc = colStart; sbc < colEnd; sbc += 16) {
                superblock(sbr, sbc);
                if (st.error) return;
            }
        }
    }
};

}  // namespace pft2
