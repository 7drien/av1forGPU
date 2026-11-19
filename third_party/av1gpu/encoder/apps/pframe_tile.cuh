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
    uint16_t txSet1[2][17], txSet2[13];
    uint16_t compMode[5][3], compRefType[5][3], compRef[3][3][3], compBwdRef[3][2][3], compoundMode[8][9];
    uint16_t txfmSplit[21][3];
};
// All CDFs of a frame: what a reference slot saves at the end of a frame.
struct EntropyState {
    av1g_cdfs cdf;
    InterCdfs inter;
};

// Inter transform sets (spec 5.11.48 get_tx_set, is_inter = 1) and the position of a type in its set's
// inter_tx_type alphabet (inverse of Tx_Type_Inter_Inv_Set1/2/3).
AV1G_HDI int interTxSet(int txSz, int reduced) {
    const int up = AV1G_T(Tx_Size_Sqr_Up)[txSz];
    if (up > AV1G_TX_32X32) return AV1G_TX_SET_DCTONLY;
    if (reduced || up == AV1G_TX_32X32) return AV1G_TX_SET_INTER_3;
    if (AV1G_T(Tx_Size_Sqr)[txSz] == AV1G_TX_16X16) return AV1G_TX_SET_INTER_2;
    return AV1G_TX_SET_INTER_1;
}
AV1G_HDI int interTxCount(int set) { return set == AV1G_TX_SET_INTER_1 ? 16 : set == AV1G_TX_SET_INTER_2 ? 12 : set ? 2 : 1; }
AV1G_HDI int interTxInv(int set, int i) {
    const uint8_t s1[16] = {9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8};
    const uint8_t s2[12] = {9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8};
    return set == AV1G_TX_SET_INTER_1 ? s1[i] : set == AV1G_TX_SET_INTER_2 ? s2[i] : set ? (i ? 0 : 9) : 0;
}
AV1G_HDI int interTxIndex(int set, int t) {
    for (int i = 0; i < interTxCount(set); i++)
        if (interTxInv(set, i) == t) return i;
    return -1;
}
// Inter chroma type: the luma type at the same position when the chroma transform's set allows it (7.13.3).
AV1G_HDI int interChromaType(int lumaType, int uvTxSz, int reduced) {
    const int set = interTxSet(uvTxSz, reduced);
    return AV1G_T(Tx_Type_In_Set_Inter)[set][lumaType] ? lumaType : AV1G_DCT_DCT;
}

// Decoders clear every CDF's adaptation counter (the entry after the final 32768) when they save frame-end CDFs.
AV1G_HDI void clearCounters(uint16_t *v, int n) {
    for (int i = 0; i + 1 < n; ++i)
        if (v[i] == 32768) v[++i] = 0;
}

struct MiInfo {
    uint8_t wLog2, hLog2;   // log2 of the block width and height in pixels (3..6)
    uint8_t skip, isInter, mode;
    int8_t ref[2];
    pf::Mv mv[2];
    uint8_t refIdx;    // RefMvIdx of the chosen mode
    uint8_t txW, txH;  // log2 of the luma transform width and height (InterTxSizes)
    uint32_t order;    // decode order: superblock index << 8 | block index in the superblock
};

// One recorded symbol (phase 3), replayed by the range coder in decode order.
//   type 0: adaptive symbol, cdf = uint16 offset in EntropyState; 1: equiprobable bit;
//   2: split_or_horz / split_or_vert from the partition cdf at `cdf` (sym = vertical flag), always 1;
//   3: n equiprobable bits, first bit in the most significant of the n low bits of sym.
struct SymRec {
    uint16_t cdf;
    uint8_t sym;
    uint8_t kind;   // n | type << 5
};
AV1G_HDI void codeRecord(av1g_ec *ec, uint16_t *base, SymRec r) {
    const int type = r.kind >> 5, n = r.kind & 31;
    if (type == 0) av1g_ec_symbol(ec, r.sym, base + r.cdf, n);
    else if (type == 1) av1g_ec_bool(ec, r.sym);
    else if (type == 3) {
        for (int i = n - 1; i >= 0; i--) av1g_ec_bool(ec, (r.sym >> i) & 1);
    } else {
        uint16_t t[2] = {uint16_t(32768 - av1g_split_or_psum(base + r.cdf, r.sym)), 32768};
        av1g_ec_encode_cdf(ec, 1, t, 2);
    }
}

// A block in decode order with the partition nodes whose symbols precede it (phase 1 output).
struct PartNode { int16_t r, c; uint8_t size, part; };
struct BlockEntry {
    int16_t r, c;
    uint8_t lw, lh, nNodes, pad;
    uint32_t order;
    PartNode nodes[4];
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
    int reducedTxSet;     // reduced_tx_set
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
    SymRec *recs;
    uint32_t *n;
    uint32_t cap;
    int *run;   // bits in the open literal run (the last record), 0 when closed
    AV1G_HDI void sym(uint32_t id, int symbol, int count) {
        if (recs && *n < cap) recs[*n] = SymRec{(uint16_t)id, (uint8_t)symbol, (uint8_t)count};
        ++*n;
        *run = 0;
    }
    AV1G_HDI void bit(int value) {
        if (*run > 0 && *run < 8) {   // append to the open run
            if (recs && *n - 1 < cap) {
                SymRec &r = recs[*n - 1];
                r.sym = (uint8_t)((r.sym << 1) | value);
                r.kind = (uint8_t)((3 << 5) | (*run + 1));
            }
            ++*run;
            return;
        }
        if (recs && *n < cap) recs[*n] = SymRec{0, (uint8_t)value, (uint8_t)((3 << 5) | 1)};
        ++*n;
        *run = 1;
    }
};

struct TileCoder {
    // frame
    const FrameSyntax *fs;
    const pf::BlockInfo *grid;
    const int16_t *levels;   // spatial layout of the aligned planes
    MiInfo *mi;              // frame-sized mode info (this tile writes its own area)
    // tile
    int rowStart, rowEnd, colStart, colEnd;   // mode-info units
    // CDFs at the start of the frame: symbol cdf ids are offsets in this state, and the mode choice uses its costs
    av1g_cdfs *cdf;
    InterCdfs *ic;
    const uint16_t *base;    // the EntropyState holding cdf / ic
    uint8_t *ctxLevel[3], *ctxDc[3];   // per plane 4x4 maps: culLevel / dcCategory of the covering transform
    int ctxStride[3];
    int16_t *qv;             // scratch: 1024 levels of one transform block
    int phase;               // 1: fill mode info and context maps, 2: choose modes, 3: emit symbols
    SymRec *recs;            // phase 3: output (null: count only)
    uint32_t nrec, recCap;   // records written, capacity of recs
    int run;                 // open literal run length (see CoeffSinkT)
    uint32_t sbOrder, blockIndex, curOrder;
    BlockEntry *entries;     // phase 1: decode-order block list of the superblock (up to 64)
    uint32_t nEntries;
    PartNode pending[4];
    int nPending;
    TileStats st;

    AV1G_HDI float tell() const { return 0.f; }
    AV1G_HDI MiInfo &at(int r, int c) { return mi[r * fs->miCols + c]; }
    AV1G_HDI bool inside(int r, int c) const { return r >= rowStart && c >= colStart && r < rowEnd && c < colEnd; }
    AV1G_HDI void sym(const uint16_t *c, int s, int n) {
        if (phase != 3) return;
        if (recs && nrec < recCap) recs[nrec] = SymRec{(uint16_t)(c - base), (uint8_t)s, (uint8_t)n};
        ++nrec;
        run = 0;
    }
    AV1G_HDI void symSplitOr(const uint16_t *c, int vertical) {
        if (phase != 3) return;
        if (recs && nrec < recCap) recs[nrec] = SymRec{(uint16_t)(c - base), (uint8_t)vertical, (uint8_t)(2 << 5)};
        ++nrec;
        run = 0;
    }

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
            int len = imin(bw4, 1 << (at(r, c).wLog2 - 2));
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
            int len = imin(bh4, 1 << (at(r, c).hLog2 - 2));
            if (iabs(deltaCol) > 1) len = imax(2, len);
            if (useStep16) len = imax(4, len);
            addRefMv(s, r, c, rp, len * 2);
            i += len;
        }
    }
    AV1G_HDI void scanPoint(Stack &s, int miRow, int miCol, int deltaRow, int deltaCol, const RefPair &rp) {
        const int r = miRow + deltaRow, c = miCol + deltaCol;
        if (inside(r, c) && at(r, c).order < curOrder) addRefMv(s, r, c, rp, 4);
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
                idx += 1 << ((pass == 0 ? cand.wLog2 : cand.hLog2) - 2);
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
    AV1G_HDI void findMvStack(Stack &s, int miRow, int miCol, int bw4, int bh4, const RefPair &rp) {
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
    // Coefficient context of plane 4x4 position (x, y): the transform covering it, 0 outside the tile (the decoder
    // clears the above context at the tile start and the left context at each superblock row of the tile).
    AV1G_HDI int ctxAt(const uint8_t *map, int plane, int x, int y) const {
        const int sub = plane > 0;
        if (x < (colStart >> sub) || y < (rowStart >> sub)) return 0;
        return map[y * ctxStride[plane] + x];
    }
    AV1G_HDI static int txSizeOfLog2(int lw, int lh) {
        for (int t = 0; t < 19; t++)
            if (AV1G_T(Tx_Width_Log2)[t] == lw && AV1G_T(Tx_Height_Log2)[t] == lh) return t;
        return 0;
    }
    // Load the stored levels of a transform (tqh x tqw, row-major) at plane sample position (px, py).
    AV1G_HDI void loadLevels(int plane, int px, int py, int tqw, int tqh) {
        const int pw = plane ? fs->aw / 2 : fs->aw;
        const size_t base0 = plane ? size_t(fs->aw) * fs->ah + size_t(plane - 1) * (fs->aw / 2) * (fs->ah / 2) : 0;
        for (int r = 0; r < tqh; ++r)
            for (int c = 0; c < tqw; ++c) qv[r * tqw + c] = levels[base0 + size_t(py + r) * pw + px + c];
    }
    // Phase 1: record the transform's culLevel / dcCategory in the context maps.
    AV1G_HDI void txContext(int plane, int x4, int y4, int txSz) {
        const int tw = AV1G_T(Tx_Width)[txSz], th = AV1G_T(Tx_Height)[txSz];
        const int n = imin(tw, 32) * imin(th, 32);
        int cul = 0;
        for (int i = 0; i < n; ++i) cul += iabs(qv[i]);
        cul = imin(63, cul);
        const int dc = qv[0] ? (qv[0] < 0 ? 1 : 2) : 0;
        for (int r = 0; r < (th >> 2); ++r)
            for (int c = 0; c < (tw >> 2); ++c) {
                ctxLevel[plane][(y4 + r) * ctxStride[plane] + x4 + c] = (uint8_t)cul;
                ctxDc[plane][(y4 + r) * ctxStride[plane] + x4 + c] = (uint8_t)dc;
            }
    }
    // Phase 4: culLevel / dcCategory of the transform at plane sample position (px, py), read from the levels.
    AV1G_HDI void txContextDirect(int plane, int px, int py, int txSz) {
        const int tw = AV1G_T(Tx_Width)[txSz], th = AV1G_T(Tx_Height)[txSz];
        const int tqw = imin(tw, 32), tqh = imin(th, 32);
        const int pw = plane ? fs->aw / 2 : fs->aw;
        const int16_t *lv = levels + (plane ? size_t(fs->aw) * fs->ah + size_t(plane - 1) * (fs->aw / 2) * (fs->ah / 2) : 0)
                          + size_t(py) * pw + px;
        int cul = 0;
        for (int r = 0; r < tqh && cul < 63; ++r)
            for (int c = 0; c < tqw; ++c) cul += iabs(lv[size_t(r) * pw + c]);
        cul = imin(63, cul);
        const int dc = lv[0] ? (lv[0] < 0 ? 1 : 2) : 0;
        const int x4 = px >> 2, y4 = py >> 2;
        for (int r = 0; r < (th >> 2); ++r)
            for (int c = 0; c < (tw >> 2); ++c) {
                ctxLevel[plane][(y4 + r) * ctxStride[plane] + x4 + c] = (uint8_t)cul;
                ctxDc[plane][(y4 + r) * ctxStride[plane] + x4 + c] = (uint8_t)dc;
            }
    }
    template <class F>
    AV1G_HDI void forTransformAreas(int miRow, int miCol, int lw, int lh, int depth, F f) {
        const bool split = depth > 0;
        if (split && lw != lh) {
            const int cw = imin(lw, 6) - (lw > lh), ch = imin(lh, 6) - (lh > lw), txSz = txSizeOfLog2(cw, ch);
            for (int k = 0; k < 2; ++k) {
                const int px = miCol * 4 + (lw > lh ? k << cw : 0), py = miRow * 4 + (lh > lw ? k << ch : 0);
                if (px >= fs->miCols * 4 || py >= fs->miRows * 4) continue;
                f(0, px, py, txSz);
            }
        } else
        if (split) {
            const int t = 1 << (lw - depth), txSz = txSizeOfLog2(lw - depth, lw - depth), per = 1 << depth;
            for (int k = 0; k < per * per; ++k) {
                const int px = miCol * 4 + treeX(k, depth) * t, py = miRow * 4 + treeY(k, depth) * t;
                if (px >= fs->miCols * 4 || py >= fs->miRows * 4) continue;
                f(0, px, py, txSz);
            }
        }
        for (int p = split ? 1 : 0; p < 3; ++p) {
            const int sub = p > 0;
            f(p, (miCol * 4) >> sub, (miRow * 4) >> sub, txSizeOfLog2(imin(lw, 6) - sub, imin(lh, 6) - sub));
        }
    }
    // Phase 3: coeffs() syntax of one transform.
    AV1G_HDI void coeffs(int plane, int x4, int y4, int txSz, int bwPix, int bhPix) {
        const int sub = plane > 0;
        const int tw = AV1G_T(Tx_Width)[txSz], th = AV1G_T(Tx_Height)[txSz];
        const int w4 = tw >> 2, h4 = th >> 2;
        const int maxX4 = fs->miCols >> sub, maxY4 = fs->miRows >> sub;
        const uint8_t *L = ctxLevel[plane], *D = ctxDc[plane];
        const int txSzCtx = (AV1G_T(Tx_Size_Sqr)[txSz] + AV1G_T(Tx_Size_Sqr_Up)[txSz] + 1) >> 1;
        int ctx;
        if (plane == 0) {
            if (tw == bwPix && th == bhPix) {
                ctx = 0;   // the transform covers the whole block
            } else {
                int top = 0, left = 0;
                for (int k = 0; k < w4; k++) if (x4 + k < maxX4) top = imax(top, ctxAt(L, plane, x4 + k, y4 - 1));
                for (int k = 0; k < h4; k++) if (y4 + k < maxY4) left = imax(left, ctxAt(L, plane, x4 - 1, y4 + k));
                const int mx = imax(top, left), mn = imin(top, left);
                if (!top && !left) ctx = 1;
                else if (!top || !left) ctx = 2 + (mx > 3);
                else if (mx <= 3) ctx = 4;
                else if (mn <= 3) ctx = 5;
                else ctx = 6;
            }
        } else {
            int above = 0, left = 0;
            for (int i = 0; i < w4; i++)
                if (x4 + i < maxX4) above |= ctxAt(L, plane, x4 + i, y4 - 1) | ctxAt(D, plane, x4 + i, y4 - 1);
            for (int i = 0; i < h4; i++)
                if (y4 + i < maxY4) left |= ctxAt(L, plane, x4 - 1, y4 + i) | ctxAt(D, plane, x4 - 1, y4 + i);
            ctx = (above != 0) + (left != 0) + 7;
            if ((bwPix >> sub) * (bhPix >> sub) > tw * th) ctx += 3;
        }
        // luma: the type chosen for this transform; inter chroma: the type of the luma 4x4 at the same position
        // (spec 7.13.3 get_tx_type; every type we use is in TX_SET_INTER_3, which chroma sizes also allow)
        const int lx4 = plane ? x4 << 1 : x4, ly4 = plane ? y4 << 1 : y4;
        int txType = (grid[(ly4 >> 1) * fs->gcols + (lx4 >> 1)].txTypes >> (4 * ((ly4 & 1) * 2 + (lx4 & 1)))) & 15;
        if (plane) txType = interChromaType(txType, txSz, fs->reducedTxSet);
        const uint16_t *scan = av1g_get_scan(txSz, txType);
        const int n = imin(tw, 32) * imin(th, 32);
        int eob = n;
        while (eob && !qv[scan[eob - 1]]) --eob;
        sym(cdf->txb_skip[txSzCtx][ctx], eob == 0, 2);
        if (!eob) return;
        // DCT_DCT in TX_SET_INTER_3 (64-point transforms use TX_SET_DCTONLY: nothing coded)
        if (plane == 0) {
            const int set = interTxSet(txSz, fs->reducedTxSet), sq = AV1G_T(Tx_Size_Sqr)[txSz];
            const int idx = interTxIndex(set, txType);
            if (set == AV1G_TX_SET_INTER_1) sym(ic->txSet1[sq], idx, 16);
            else if (set == AV1G_TX_SET_INTER_2) sym(ic->txSet2, idx, 12);
            else if (set == AV1G_TX_SET_INTER_3) sym(ic->txSet3[sq], idx, 2);
        }
        int dcSign = 0;
        for (int k = 0; k < w4; k++)
            if (x4 + k < maxX4) { const int v = ctxAt(D, plane, x4 + k, y4 - 1); dcSign += v == 1 ? -1 : v == 2 ? 1 : 0; }
        for (int k = 0; k < h4; k++)
            if (y4 + k < maxY4) { const int v = ctxAt(D, plane, x4 - 1, y4 + k); dcSign += v == 1 ? -1 : v == 2 ? 1 : 0; }
        CoeffSinkT sink{recs, &nrec, recCap, &run};
        emitCoeffPayload(sink, qv, eob, txSz, txType, txSzCtx, plane > 0, dcSign);
    }
    // Walk the transforms of a coded block: f(plane, x4, y4, txSz) with the levels loaded into qv.
    // transform_tree order of the k-th leaf at uniform depth: quadrants recursively (depth 2: z-order)
    AV1G_HDI static int treeX(int k, int depth) { return depth == 1 ? (k & 1) : ((k >> 2) & 1) * 2 + (k & 1); }
    AV1G_HDI static int treeY(int k, int depth) { return depth == 1 ? (k >> 1) : ((k >> 3) & 1) * 2 + ((k >> 1) & 1); }
    template <class F>
    AV1G_HDI void forTransforms(int miRow, int miCol, int lw, int lh, int depth, F f) {
        const bool split = depth > 0;
        if (split && lw != lh) {
            const int cw = imin(lw, 6) - (lw > lh), ch = imin(lh, 6) - (lh > lw), txSz = txSizeOfLog2(cw, ch);
            for (int k = 0; k < 2; ++k) {
                const int px = miCol * 4 + (lw > lh ? k << cw : 0), py = miRow * 4 + (lh > lw ? k << ch : 0);
                if (px >= fs->miCols * 4 || py >= fs->miRows * 4) continue;
                loadLevels(0, px, py, imin(1 << cw, 32), imin(1 << ch, 32));
                f(0, px >> 2, py >> 2, txSz);
            }
        } else
        if (split) {
            const int t = 1 << (lw - depth), txSz = txSizeOfLog2(lw - depth, lw - depth), per = 1 << depth;
            for (int k = 0; k < per * per; ++k) {
                const int px = miCol * 4 + treeX(k, depth) * t, py = miRow * 4 + treeY(k, depth) * t;
                if (px >= fs->miCols * 4 || py >= fs->miRows * 4) continue;
                loadLevels(0, px, py, t, t);
                f(0, px >> 2, py >> 2, txSz);
            }
        }
        for (int p = split ? 1 : 0; p < 3; ++p) {
            const int sub = p > 0;
            const int tlw = imin(lw, 6) - sub, tlh = imin(lh, 6) - sub;
            const int txSz = txSizeOfLog2(tlw, tlh);
            const int px = (miCol * 4) >> sub, py = (miRow * 4) >> sub;
            loadLevels(p, px, py, imin(1 << tlw, 32), imin(1 << tlh, 32));
            f(p, px >> 2, py >> 2, txSz);
        }
    }

    // ---- block ----
    AV1G_HDI void block(int miRow, int miCol, int lw, int lh, const pf::BlockInfo &bi) {
        const int bw4 = 1 << (lw - 2), bh4 = 1 << (lh - 2);
        const int size = lw > lh ? lw : lh;   // log2 of the larger side
        const int skip = bi.skip;
        const bool compound = bi.ref == 2;
        const int depth = skip ? 0 : bi.txSplit;   // luma transform split depth (0..2)
        const bool split = depth > 0;
        RefPair rp{fs->refName[bi.ref == 1 ? 1 : 0], compound ? fs->refName[1] : -1, compound};
        if (phase == 1) {   // block list only
            curOrder = sbOrder | blockIndex++;
            BlockEntry &e = entries[nEntries++];
            e.r = (int16_t)miRow; e.c = (int16_t)miCol; e.lw = (uint8_t)lw; e.lh = (uint8_t)lh;
            e.order = curOrder;
            e.nNodes = (uint8_t)nPending;
            for (int k = 0; k < nPending; k++) e.nodes[k] = pending[k];
            nPending = 0;
            return;
        }
        if (phase == 4) {   // mode info and coefficient context maps
            for (int r = miRow; r < imin(miRow + bh4, fs->miRows); ++r)
                for (int c = miCol; c < imin(miCol + bw4, fs->miCols); ++c) {
                    MiInfo &m = at(r, c);
                    m.wLog2 = (uint8_t)lw;
                    m.hLog2 = (uint8_t)lh;
                    m.skip = (uint8_t)skip;
                    m.isInter = 1;
                    m.mode = 0;
                    m.refIdx = 0;
                    m.ref[0] = (int8_t)rp.r0;
                    m.ref[1] = (int8_t)(compound ? rp.r1 : -1);
                    m.mv[0] = bi.mv;
                    m.mv[1] = compound ? bi.mv2 : pf::Mv{0, 0};
                    m.txW = (uint8_t)(imin(lw, 6) - (lw == lh ? depth : (depth && lw > lh)));
                    m.txH = (uint8_t)(imin(lh, 6) - (lw == lh ? depth : (depth && lh > lw)));
                    m.order = curOrder;
                }
            if (!skip) forTransformAreas(miRow, miCol, lw, lh, depth, [&](int p, int px, int py, int t) { txContextDirect(p, px, py, t); });
            return;
        }
        const pf::Mv target[2] = {bi.mv, bi.mv2};
        Stack s;
        findMvStack(s, miRow, miCol, bw4, bh4, rp);
        if (phase == 2) {
            // Mode choice from the frame-start CDFs, with the NEWMV context of a neighbourhood without NEWMV blocks:
            // it then depends only on decisions known before any mode is chosen, so blocks choose in parallel.
            Stack sc = s;
            sc.newMvCtx = sc.closeMatches == 0 ? imin(sc.totalMatches, 1) : sc.closeMatches == 1 ? 3 : 5;
            float best = 1e30f;
            int bestMode = -1, bestIdx = 0;
            pf::Mv pred[2];
            const int first = compound ? NEAREST_NEARESTMV : NEARESTMV, last = compound ? NEW_NEWMV : NEWMV;
            for (int mode = first; mode <= last; ++mode) {
                int k0 = 0, k1 = 0;
                if (isNewType(mode)) { k0 = 0; k1 = 2; }
                else if (hasNearmv(mode)) { k0 = 1; k1 = 3; }
                for (int k = k0; k <= k1; ++k) {
                    if (k > k0 && sc.num <= k) break;
                    const float c = modeCost(sc, compound, mode, k, target, pred);
                    if (c >= 0.f && c < best) { best = c; bestMode = mode; bestIdx = k; }
                }
            }
            if (bestMode < 0) { st.error = 2; return; }
            for (int r = miRow; r < imin(miRow + bh4, fs->miRows); ++r)
                for (int c = miCol; c < imin(miCol + bw4, fs->miCols); ++c) {
                    at(r, c).mode = (uint8_t)bestMode;
                    at(r, c).refIdx = (uint8_t)bestIdx;
                }
            const int m0 = modeOfList(bestMode, 0);
            st.modeCount[m0 == NEARESTMV ? 0 : m0 == NEARMV ? 1 : m0 == GLOBALMV ? 2 : 3]++;
            st.compound += compound;
            return;
        }
        // phase 3: symbols
        const bool availU = inside(miRow - 1, miCol), availL = inside(miRow, miCol - 1);
        const MiInfo *above = availU ? &at(miRow - 1, miCol) : nullptr;
        const MiInfo *left = availL ? &at(miRow, miCol - 1) : nullptr;
        const int skipCtx = (availU ? above->skip : 0) + (availL ? left->skip : 0);
        sym(cdf->skip[skipCtx], skip, 2);
        const bool aboveIntra = availU && !above->isInter, leftIntra = availL && !left->isInter;
        int interCtx;
        if (availU && availL) interCtx = (leftIntra && aboveIntra) ? 3 : (leftIntra || aboveIntra);
        else if (availU || availL) interCtx = 2 * (availU ? aboveIntra : leftIntra);
        else interCtx = 0;
        sym(ic->isInter[interCtx], 1, 2);
        if (fs->referenceSelect) sym(ic->compMode[compModeCtx(availU, availL, above, left)], compound, 2);
#define PF_CNT2(f) countRefs(f, availU, availL, above, left)
        if (compound) {
            if (rp.r0 != LAST_FRAME || rp.r1 != ALTREF_FRAME) { st.error = 1; return; }
            sym(ic->compRefType[compRefTypeCtx(availU, availL, above, left)], 1, 2);   // BIDIR_COMP_REFERENCE
            sym(ic->compRef[refCountCtx(PF_CNT2(LAST_FRAME) + PF_CNT2(LAST2_FRAME), PF_CNT2(LAST3_FRAME) + PF_CNT2(GOLDEN_FRAME))][0], 0, 2);
            sym(ic->compRef[refCountCtx(PF_CNT2(LAST_FRAME), PF_CNT2(LAST2_FRAME))][1], 0, 2);   // LAST_FRAME
            sym(ic->compBwdRef[refCountCtx(PF_CNT2(BWDREF_FRAME) + PF_CNT2(ALTREF2_FRAME), PF_CNT2(ALTREF_FRAME))][0], 1, 2);
        } else {
            writeSingleRef(rp.r0, availU, availL, above, left);
        }
#undef PF_CNT2
        const MiInfo &self = at(miRow, miCol);
        const int bestMode = self.mode, bestIdx = self.refIdx;
        pf::Mv bestPred[2];
        if (modeCost(s, compound, bestMode, bestIdx, target, bestPred) < 0.f) { st.error = 8; return; }
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
        for (int i = 0; i < 1 + compound; ++i)
            if (modeOfList(bestMode, i) == NEWMV) writeMv(target[i], bestPred[i]);
        if (fs->txModeSelect && !skip) {
            const int maxTx = imin(size, 6) - 2;   // square find_tx_size(Min(64, Max(bw, bh)))
            auto txCtx = [&](int row, int col, int tw2, int th2) {
                int aw2, lh2;
                if (row == miRow) {
                    if (!inside(row - 1, col)) aw2 = 64;
                    else { const MiInfo &n = at(row - 1, col); aw2 = n.skip && n.isInter ? 1 << n.wLog2 : 1 << n.txW; }
                } else aw2 = 1 << at(row - 1, col).txW;
                if (col == miCol) {
                    if (!inside(row, col - 1)) lh2 = 64;
                    else { const MiInfo &n = at(row, col - 1); lh2 = n.skip && n.isInter ? 1 << n.hLog2 : 1 << n.txH; }
                } else lh2 = 1 << at(row, col - 1).txH;
                const int sqrUp = imax(tw2, th2) - 2;
                return (sqrUp != maxTx) * 3 + (4 - maxTx) * 6 + (aw2 < (1 << tw2)) + (lh2 < (1 << th2));
            };
            sym(ic->txfmSplit[txCtx(miRow, miCol, imin(lw, 6), imin(lh, 6))], split, 2);
            if (split && lw != lh) {
                // rectangular: two children with the longer side halved, coding txfm_split = 0 unless 4x4
                const int cw = imin(lw, 6) - (lw > lh), ch = imin(lh, 6) - (lh > lw);
                for (int k = 0; k < 2; k++) {
                    const int r = miRow + (lh > lw ? k * (1 << (ch - 2)) : 0), c = miCol + (lw > lh ? k * (1 << (cw - 2)) : 0);
                    if (r >= fs->miRows || c >= fs->miCols) continue;
                    if (!(cw == 2 && ch == 2)) sym(ic->txfmSplit[txCtx(r, c, cw, ch)], 0, 2);
                }
            } else if (split && size - 1 > 2) {   // depth 1 children (not 4x4) code txfm_split; depth 2 leaves code nothing
                const int half = 1 << (size - 3);
                for (int k = 0; k < 4; k++) {
                    const int r = miRow + (k >> 1) * half, c = miCol + (k & 1) * half;
                    if (r >= fs->miRows || c >= fs->miCols) continue;
                    sym(ic->txfmSplit[txCtx(r, c, size - 1, size - 1)], depth == 2, 2);
                }
            }
        }
        if (skip) return;
        forTransforms(miRow, miCol, lw, lh, depth, [&](int p, int x4, int y4, int t) { coeffs(p, x4, y4, t, 4 * bw4, 4 * bh4); });
    }

    // Phase 3: the partition symbol of a node.
    AV1G_HDI void emitNode(const PartNode &nd) {
        const int miRow = nd.r, miCol = nd.c, size = nd.size, half = 1 << (size - 3);
        const bool hasRows = miRow + half < fs->miRows, hasCols = miCol + half < fs->miCols;
        const int ctxAbove = inside(miRow - 1, miCol) && at(miRow - 1, miCol).wLog2 < size;
        const int ctxLeft = inside(miRow, miCol - 1) && at(miRow, miCol - 1).hLog2 < size;
        const int pcx = ctxLeft * 2 + ctxAbove;
        const uint16_t *pc = size == 6 ? cdf->partition_w64[pcx] : size == 5 ? cdf->partition_w32[pcx]
                           : size == 4 ? cdf->partition_w16[pcx] : cdf->partition_w8[pcx];
        if (hasRows && hasCols) sym(pc, nd.part, size == 3 ? 4 : 10);
        else symSplitOr(pc, hasRows);
    }
    // Phases 2 and 3 for one block of the list (blocks are independent in these phases).
    AV1G_HDI void runEntry(const BlockEntry &e) {
        curOrder = e.order;
        run = 0;
        if (phase == 3)
            for (int k = 0; k < e.nNodes; k++) emitNode(e.nodes[k]);
        block(e.r, e.c, e.lw, e.lh, grid[(e.r >> 1) * fs->gcols + (e.c >> 1)]);
        if (phase == 3 && recs && nrec > recCap) st.error = 9;   // record region overflow
    }
    // One superblock in decode order (explicit stack of quadrants; no recursion on the device).
    AV1G_HDI void superblock(int sbRow, int sbCol, uint32_t sbIndex) {
        sbOrder = sbIndex << 8;
        blockIndex = 0;
        nEntries = 0;
        nPending = 0;
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
            const bool abNode = bi.ab && bi.abLog2 == size;
            const bool split = !abNode && bi.log2 < size;
            const int part = abNode ? bi.ab : split ? AV1G_PARTITION_SPLIT : bi.part == 1 ? AV1G_PARTITION_HORZ
                           : bi.part == 2 ? AV1G_PARTITION_VERT : bi.part == 3 ? AV1G_PARTITION_HORZ_4
                           : bi.part == 4 ? AV1G_PARTITION_VERT_4 : AV1G_PARTITION_NONE;
            if (phase == 1 && (hasRows || hasCols) && nPending < 4)
                pending[nPending++] = PartNode{(int16_t)miRow, (int16_t)miCol, (uint8_t)size, (uint8_t)part};
            if (!(hasRows && hasCols) && !split) { st.error = 3; return; }
            if (split) {
                if (size == 3) { st.error = 4; return; }
                n.k = 0;
                stack[top + 1] = Node{miRow, miCol, size - 1, -1};
                ++top;
                continue;
            }
            if (abNode) {
                if (!(hasRows && hasCols)) { st.error = 7; return; }
                auto at2 = [&](int r, int c) -> const pf::BlockInfo & { return grid[(r >> 1) * fs->gcols + (c >> 1)]; };
                const int sq = size - 1;
                if (part == AV1G_PARTITION_HORZ_A) {
                    block(miRow, miCol, sq, sq, at2(miRow, miCol));
                    if (!st.error) block(miRow, miCol + half, sq, sq, at2(miRow, miCol + half));
                    if (!st.error) block(miRow + half, miCol, size, size - 1, at2(miRow + half, miCol));
                } else if (part == AV1G_PARTITION_HORZ_B) {
                    block(miRow, miCol, size, size - 1, at2(miRow, miCol));
                    if (!st.error) block(miRow + half, miCol, sq, sq, at2(miRow + half, miCol));
                    if (!st.error) block(miRow + half, miCol + half, sq, sq, at2(miRow + half, miCol + half));
                } else if (part == AV1G_PARTITION_VERT_A) {
                    block(miRow, miCol, sq, sq, at2(miRow, miCol));
                    if (!st.error) block(miRow + half, miCol, sq, sq, at2(miRow + half, miCol));
                    if (!st.error) block(miRow, miCol + half, size - 1, size, at2(miRow, miCol + half));
                } else {
                    block(miRow, miCol, size - 1, size, at2(miRow, miCol));
                    if (!st.error) block(miRow, miCol + half, sq, sq, at2(miRow, miCol + half));
                    if (!st.error) block(miRow + half, miCol + half, sq, sq, at2(miRow + half, miCol + half));
                }
                if (st.error) return;
                --top;
                continue;
            }
            if (bi.log2 != size) { st.error = 5; return; }
            if (part == AV1G_PARTITION_HORZ) {
                block(miRow, miCol, size, size - 1, bi);
                if (!st.error) block(miRow + half, miCol, size, size - 1, grid[((miRow + half) >> 1) * fs->gcols + (miCol >> 1)]);
            } else if (part == AV1G_PARTITION_VERT) {
                block(miRow, miCol, size - 1, size, bi);
                if (!st.error) block(miRow, miCol + half, size - 1, size, grid[(miRow >> 1) * fs->gcols + ((miCol + half) >> 1)]);
            } else if (part == AV1G_PARTITION_HORZ_4 || part == AV1G_PARTITION_VERT_4) {
                const int q = half >> 1;   // quarter of the node in mode-info units
                for (int k = 0; k < 4 && !st.error; k++) {
                    const int r = miRow + (part == AV1G_PARTITION_HORZ_4 ? k * q : 0);
                    const int c = miCol + (part == AV1G_PARTITION_VERT_4 ? k * q : 0);
                    if (r >= fs->miRows || c >= fs->miCols) break;
                    block(r, c, part == AV1G_PARTITION_HORZ_4 ? size : size - 2, part == AV1G_PARTITION_HORZ_4 ? size - 2 : size,
                          grid[(r >> 1) * fs->gcols + (c >> 1)]);
                }
            } else {
                block(miRow, miCol, size, size, bi);
            }
            if (st.error) return;
            --top;
        }
    }
};

}  // namespace pft2
