#pragma once
#include "inter_motion.h"
#include "internal.h"
#include "av1g/entenc.h"
#include "av1g/symstream.h"
#include "av1g/cdf_defaults.h"
#include "av1g/cdfctx.h"
#include "av1g/dsp.h"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <memory>

namespace inter {
using std::min;
inline av1g_cdfs coeffLayout{};
#define CID(field) (static_cast<uint32_t>(&coeffLayout.field[0] - reinterpret_cast<const uint16_t *>(&coeffLayout)))
#include "gpu_coeff_payload.cuh"
#undef CID

struct CoeffSink {
    av1g_ec &coder;
    av1g_cdfs &tables;
    void sym(uint32_t id, int symbol, int count) {
        av1g_ec_symbol(&coder, symbol, reinterpret_cast<uint16_t *>(&tables) + id, count);
    }
    void bit(int value) { av1g_ec_bool(&coder, value); }
};

// Rank a segment relative to the spatial predictor (AV1 neg_deinterleave inverse).
inline int segmentSymbol(int mode, int pred, int modeCount) {
    const int d=mode-pred;
    if (std::abs(d)<=std::min(pred,modeCount-1-pred)) return 2*std::abs(d)-(d>0);
    return pred<modeCount/2 ? mode : modeCount-1-mode;
}
inline void translation(av1g::BitWriter &b,int pixels,int previousPixels = 0) {
    // Signed subexp with the previous global parameter as predictor. AV1's
    // translation precision here is quarter-pixel because high precision MVs
    // are disabled. All supported displacements fit the ordinary subexp path.
    const int current = pixels * 4, previous = previousPixels * 4;
    const int difference = current - previous;
    const int value = difference < 0 ? -2*difference-1 : 2*difference;
    // At most 16 pixels. The value always fits before the final ns() branch.
    int base = 0;
    for (int bits = 3; ; bits += base != 8) {
        const int span = 1 << bits;
        const bool more = value >= base + span;
        b.bit(more);
        if (!more) {
            b.put(value - base, bits);
            break;
        }
        base += span;
    }
}
struct EntropyState {
    av1g_cdfs cdf{};
    uint16_t seg[3][9] = {
        {5622,7893,16093,18233,27809,28373,32533,32768},
        {14274,18230,22557,24935,29980,30851,32344,32768},
        {27527,28487,28723,28890,32397,32647,32679,32768}};
    uint16_t splitTx[3][3] = {{28165,32768,0},{22401,32768,0},{16088,32768,0}};
    uint16_t interTxType[3] = {16384,32768,0};
    uint16_t interTxType8[3] = {4167,32768,0};
    bool ready = false;
    int previousStep = 0;
};
// Decoders clear every CDF's adaptation counter (the entry after the final
// 32768) when frame-end CDFs are saved to a reference slot. Mirror that so the
// next frame, which loads them through primary_ref_frame, adapts identically.
inline void clearCounters(uint16_t *v, size_t n) {
    for (size_t i = 0; i + 1 < n; ++i)
        if (v[i] == 32768) v[++i] = 0;
}
inline void resetCounters(EntropyState &s) {
    clearCounters(reinterpret_cast<uint16_t *>(&s.cdf), sizeof(s.cdf)/2);
    clearCounters(&s.seg[0][0], 27);
    clearCounters(&s.splitTx[0][0], 9);
    clearCounters(s.interTxType, 3);
    clearCounters(s.interTxType8, 3);
}
inline std::vector<uint8_t> packet(int w,int h,const std::vector<MotionBlock>& blocks,int modeCount,int grain,uint16_t seed,bool rolling, const int16_t *levels = nullptr, bool skipEmpty = false, int step = 2, int qindex = 0, bool luma8 = false, const uint8_t *txMap = nullptr, bool anchor = false, bool refreshAnchor = false, int motionBlockSize = 64, EntropyState *persistent = nullptr) {
    const int cols=(w+63)/64, rows=(h+63)/64;
    const int motionCols=(w+motionBlockSize-1)/motionBlockSize;
    EntropyState local;
    EntropyState &state = persistent ? *persistent : local;
    const bool primary = persistent && state.ready;
    if (!primary) state = EntropyState{};
    av1g::BitWriter b;
    b.bit(0); b.put(1,2); b.bit(1); b.bit(!persistent); // shown INTER_FRAME
    b.bit(!levels); b.bit(0); b.bit(0); // adapt CDFs on residual frames
    if (persistent) b.put(primary ? 0 : 7,3); // use rolling LAST as primary reference
    b.put(rolling ? (refreshAnchor ? 3 : 1) : 0,8); // refresh rolling slot and optionally anchor
    for(int r=0;r<7;++r) b.put(rolling ? (anchor && r == AnchorMode ? 1 : 0) : r,3);
    b.bit(0); b.bit(0); // render size, high precision MV
    b.bit(0); b.put(0,2); b.bit(0); // fixed eight-tap interpolation, simple translation
    if (levels) b.bit(0); // disable_frame_end_update_cdf = 0: save adapted CDFs
    b.bit(1); if(cols>1)b.bit(0); if(rows>1)b.bit(0); // single uniform tile
    b.put(qindex,8); b.put(0,4); // base Q, three delta flags, qmatrix
    b.bit(1); // segmentation, fresh map and features
    if (primary) { b.bit(1); b.bit(0); b.bit(1); } // update map, no temporal prediction, update features
    for(int s=0;s<8;++s) {
        b.put(0,5); b.bit(s<modeCount); if(s<modeCount)b.put(s+1,3); // forced reference
        b.bit(s<modeCount && !levels); b.bit(s<modeCount && levels); // force GLOBALMV for residual mode
    }
    if (qindex) {
        b.bit(0);      // delta_q_present
        b.put(0, 16);  // deblock levels, sharpness, delta disabled
        b.put(0, 6);   // lr_params: lr_type RESTORE_NONE for Y, U, V (video sequence headers enable restoration)
        b.bit(!luma8); // TX_MODE_SELECT for 4x4 luma, TX_MODE_LARGEST for 8x8
    }
    b.bit(0); b.bit(1); // single reference mode, reduced tx set
    for(int m=0;m<7;++m) {
        bool translated=m>0 && m<modeCount && !(anchor && m == AnchorMode);
        b.bit(translated);
        if(translated) {
            b.bit(0); b.bit(1);
            translation(b,motionY(m, step), primary ? motionY(m, state.previousStep) : 0);
            translation(b,motionX(m, step), primary ? motionX(m, state.previousStep) : 0);
        }
    }
    av1g::write_video_grain(b,grain,seed,true);
    b.byte_align();
    auto &seg = state.seg;
    const size_t capacity = blocks.size()*16 + (cols+rows)*256 + 16 + (levels ? size_t(w)*h*8 : 0);
    // The range coder writes each precarry entry before reading it back.
    std::unique_ptr<uint16_t[]> scratch(new uint16_t[capacity]);
    av1g_ec ec;
    av1g_ec_init(&ec, scratch.get(), capacity);
    // Edge partitions split until PARTITION_NONE is legal. Maintain neighbor
    // sizes and segment IDs on the decoder's 4x4 mode-info grid.
    const int miCols=2*((w+7)/8), miRows=2*((h+7)/8);
    std::vector<uint8_t> logSize(miCols*miRows), ids(miCols*miRows);
    auto &cdf = state.cdf;
    if (levels && !primary) av1g_cdfs_init(&cdf, qindex);
    CoeffSink sink{ec, cdf};
    std::vector<uint8_t> neighbors[3];
    if (levels) {
        neighbors[0].resize(size_t(w)*h/16);
        neighbors[1].resize(size_t(w)*h/64);
        neighbors[2].resize(size_t(w)*h/64);
    }
    std::vector<uint8_t> skipped(skipEmpty ? size_t(miCols)*miRows : 0);
    auto &splitTx = state.splitTx;
    auto &interTxType = state.interTxType;
    auto &interTxType8 = state.interTxType8;
    auto emitResidual = [&](int x, int y, int side) {
        bool skip = skipEmpty;
        size_t base = 0;
        for (int plane = 0; skip && plane < 3; ++plane) {
            if (luma8 && plane == 0) {
                const auto *q = levels + (size_t(y/2)*(w/8) + x/2)*64;
                skip = std::all_of(q, q+64, [](int16_t v) { return v == 0; });
                base += neighbors[0].size()*16;
                continue;
            }
            const int shift = plane != 0;
            const int stride = (w >> shift)/4;
            const int columns = std::min((x+side) >> shift, stride) - (x >> shift);
            for (int yy = y >> shift; skip && yy < std::min((y+side) >> shift, (h >> shift)/4); ++yy) {
                const auto *row = levels + base + size_t(yy*stride + (x >> shift))*16;
                skip = std::all_of(row, row + 16*columns, [](int16_t q) { return q == 0; });
            }
            base += neighbors[plane].size()*16;
        }
        int skipContext = 0;
        if (skipEmpty) {
            const int at = y*miCols + x;
            skipContext = (x ? skipped[at-1] : 0) + (y ? skipped[at-miCols] : 0);
            if (skip) {
                for (int yy = y; yy < std::min(y+side, miRows); ++yy)
                    std::fill_n(skipped.begin() + yy*miCols + x, std::min(side, miCols-x), 1);
            }
        }
        av1g_ec_symbol(&ec, skip, cdf.skip[skipContext], 2);
        if (skip) return;
        const int blockIndex = (y/2)*(w/8) + x/2;
        const bool large = luma8 || (txMap && txMap[blockIndex]);
        if (qindex && !luma8) {
            const int at = y*miCols + x;
            int context = (x && !skipped[at-1] && (!txMap || !txMap[blockIndex-1])) +
                          (y && !skipped[at-miCols] && (!txMap || !txMap[blockIndex-w/8]));
            av1g_ec_symbol(&ec, !large, splitTx[context], 2);
        }
        size_t planeBase = 0;
        for (int plane = 0; plane < 3; ++plane) {
            if (large && plane == 0) {
                const int stride = w/4;
                const int at = y*stride + x;
                auto &map = neighbors[0];
                const int16_t *q = levels + size_t(blockIndex)*64;
                int16_t packed[64];
                if (txMap) {
                    for (int i = 0; i < 4; ++i)
                        std::copy_n(levels + size_t(at + (i/2)*stride + i%2)*16, 16, packed+16*i);
                    q = packed;
                }
                const auto *scan = av1g_get_scan(AV1G_TX_8X8, AV1G_DCT_DCT);
                int eob = 64;
                while (eob && !q[scan[eob-1]]) --eob;
                av1g_ec_symbol(&ec, eob == 0, cdf.txb_skip[1][0], 2);
                int sum = 0;
                if (eob) {
                    av1g_ec_symbol(&ec, 1, interTxType8, 2);
                    auto sign = [](int v) { return v == 1 ? -1 : v == 2 ? 1 : 0; };
                    int dcContext = 0;
                    if (y) dcContext += sign(map[at-stride]>>6) + sign(map[at-stride+1]>>6);
                    if (x) dcContext += sign(map[at-1]>>6) + sign(map[at+stride-1]>>6);
                    emitCoeffPayload(sink, q, eob, AV1G_TX_8X8, AV1G_DCT_DCT, 1, 0, dcContext);
                    for (int i = 0; i < 64; ++i) sum += std::abs(q[i]);
                }
                const uint8_t context = std::min(sum, 63) | ((q[0]<0 ? 1 : q[0]>0 ? 2 : 0)<<6);
                map[at] = map[at+1] = map[at+stride] = map[at+stride+1] = context;
                planeBase += map.size()*16;
                continue;
            }
            const int shift = plane != 0;
            const int stride = (w >> shift)/4;
            auto &map = neighbors[plane];
            for (int yy = y >> shift; yy < std::min((y+side) >> shift, (h >> shift)/4); ++yy) {
                for (int xx = x >> shift; xx < std::min((x+side) >> shift, stride); ++xx) {
                    const int at = yy*stride + xx;
                    const int top = yy ? map[at-stride] : 0;
                    const int left = xx ? map[at-1] : 0;
                    const int tl = top & 63, ll = left & 63;
                    int ctx;
                    if (plane) ctx = (qindex ? 7 : 10) + (top != 0) + (left != 0);
                    else if (!tl && !ll) ctx = 1;
                    else if (!tl || !ll) ctx = 2 + (std::max(tl,ll)>3);
                    else if (std::max(tl,ll)<=3) ctx = 4;
                    else ctx = std::min(tl,ll)<=3 ? 5 : 6;
                    const int16_t *q = levels + planeBase + size_t(at)*16;
                    int eob = 16;
                    const auto *scan = av1g_get_scan(AV1G_TX_4X4, AV1G_DCT_DCT);
                    while (eob && !q[scan[eob-1]]) --eob;
                    av1g_ec_symbol(&ec, eob == 0, cdf.txb_skip[0][ctx], 2);
                    int sum = 0;
                    if (eob) {
                        if (qindex && plane == 0) av1g_ec_symbol(&ec, 1, interTxType, 2);
                        auto sign = [](int v) { return v == 1 ? -1 : v == 2 ? 1 : 0; };
                        emitCoeffPayload(sink, q, eob, AV1G_TX_4X4, AV1G_DCT_DCT, 0, shift,
                                         sign(top>>6) + sign(left>>6));
                        for (int i = 0; i < 16; ++i) sum += std::abs(q[i]);
                    }
                    map[at] = std::min(sum, 63) | ((q[0]<0 ? 1 : q[0]>0 ? 2 : 0)<<6);
                }
            }
            planeBase += map.size()*16;
        }
    };
    auto node = [&](auto &&self,int x,int y,int log,int mode) -> void {
        if(x>=miCols || y>=miRows) return;
        int side=1<<log, half=side/2;
        int at=y*miCols+x;
        if(log) {
            int pcx=(y && logSize[at-miCols]<log) + 2*(x && logSize[at-1]<log);
            const uint16_t *pc=log==4 ? AV1G_T(Default_Partition_W64_Cdf)[pcx] :
                               log==3 ? AV1G_T(Default_Partition_W32_Cdf)[pcx] :
                               log==2 ? AV1G_T(Default_Partition_W16_Cdf)[pcx] :
                                        AV1G_T(Default_Partition_W8_Cdf)[pcx];
            bool hasRows=y+half<miRows, hasCols=x+half<miCols;
            uint16_t *adaptive = !levels ? nullptr : log == 4 ? cdf.partition_w64[pcx] :
                                 log == 3 ? cdf.partition_w32[pcx] :
                                 log == 2 ? cdf.partition_w16[pcx] : cdf.partition_w8[pcx];
            if (adaptive) pc = adaptive;
            const bool splitNode = (qindex && log > 1) || !hasRows || !hasCols;
            if (hasRows && hasCols) {
                const int count = log == 1 ? 4 : 10;
                const int partition = splitNode ? AV1G_PARTITION_SPLIT : AV1G_PARTITION_NONE;
                if (adaptive) av1g_ec_symbol(&ec, partition, adaptive, count);
                else av1g_ec_encode_cdf(&ec, partition, pc, count);
            } else if (hasRows || hasCols) {
                uint16_t split[] = {uint16_t(32768-av1g_split_or_psum(pc,hasRows)),32768};
                av1g_ec_encode_cdf(&ec,1,split,2);
            }
            if (splitNode) {
                for(int dy : {0,half}) for(int dx : {0,half}) self(self,x+dx,y+dy,log-1,mode);
                return;
            }
        }
        mode = blocks[((y*4)/motionBlockSize)*motionCols + (x*4)/motionBlockSize].mode;
        int u=y?ids[at-miCols]:-1, l=x?ids[at-1]:-1;
        int ul=x&&y?ids[at-miCols-1]:-1;
        int ctx=ul<0?0:(ul==u&&ul==l?2:int(ul==u||ul==l||u==l));
        int pred=u<0?std::max(0,l):(l<0||ul==u?u:l);
        const int symbol = segmentSymbol(mode, pred, modeCount);
        if (levels) av1g_ec_symbol(&ec, symbol, seg[ctx], 8);
        else av1g_ec_encode_cdf(&ec, symbol, seg[ctx], 8);
        if (levels) emitResidual(x, y, side);
        for(int row=y;row<std::min(y+side,miRows);++row) {
            int begin=row*miCols+x, end=row*miCols+std::min(x+side,miCols);
            std::fill(logSize.begin()+begin,logSize.begin()+end,log);
            std::fill(ids.begin()+begin,ids.begin()+end,mode);
        }
    };
    for(int y=0;y<rows;++y) for(int x=0;x<cols;++x)
        node(node,x*16,y*16,4,blocks[((y*64)/motionBlockSize)*motionCols + (x*64)/motionBlockSize].mode);
    std::vector<uint8_t> tile(ec.offs + 4);
    int n=av1g_ec_done(&ec,tile.data());
    if(n<0) throw std::runtime_error("inter tile overflow");
    std::vector<uint8_t> obu={0x12,0,0x32};
    av1g::put_leb128(obu,b.buf.size()+n);
    obu.insert(obu.end(),b.buf.begin(),b.buf.end());
    obu.insert(obu.end(),tile.begin(),tile.begin()+n);
    state.previousStep = step;
    state.ready = true;
    resetCounters(state);
    return obu;
}
}
