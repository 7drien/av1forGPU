#pragma once

// The coefficient payload is shared by intra and inter transforms. The caller
// signals skip and any transform type, and supplies the neighboring DC context.
template <class Output>
AV1G_HDI void emitCoeffPayload(Output &o, const int16_t *q, int eob, int txSz,
                                 int txType, int txSzCtx, int ptype, int dcSign) {
    const uint16_t *scan = av1g_get_scan(txSz, txType);
    const int txClass = av1g_tx_class(txType);
    const int eobMultisize = min((int)AV1G_T(Tx_Width_Log2)[txSz], 5) + min((int)AV1G_T(Tx_Height_Log2)[txSz], 5) - 4;
    int eobPt;
    if (eob <= 2) eobPt = eob;
    else { eobPt = 2; int v = eob - 1; while (v > 1) { v >>= 1; eobPt++; } }
    {
        const int ectx = txClass == AV1G_TX_CLASS_2D ? 0 : 1;
        uint32_t c;
        switch (eobMultisize) {
        case 0: c = CID(eob_pt_16[ptype][ectx]); break;
        case 1: c = CID(eob_pt_32[ptype][ectx]); break;
        case 2: c = CID(eob_pt_64[ptype][ectx]); break;
        case 3: c = CID(eob_pt_128[ptype][ectx]); break;
        case 4: c = CID(eob_pt_256[ptype][ectx]); break;
        case 5: c = CID(eob_pt_512[ptype]); break;
        default: c = CID(eob_pt_1024[ptype]); break;
        }
        o.sym(c, eobPt - 1, eobMultisize + 5);
    }
    if (eobPt >= 3) {
        const int rem = eob - ((1 << (eobPt - 2)) + 1);
        o.sym(CID(eob_extra[txSzCtx][ptype][eobPt - 3]), (rem >> (eobPt - 3)) & 1, 2);
        for (int i = 1; i < eobPt - 2; i++) o.bit((rem >> (eobPt - 3 - i)) & 1);
    }
    const int adj = AV1G_T(Adjusted_Tx_Size)[txSz];
    const int bwl = AV1G_T(Tx_Width_Log2)[adj];
    const int width = 1 << bwl, height = AV1G_T(Tx_Height)[adj];
    // levels of the context neighbours: they are right of / below the coefficient, so later in the scan and
    // already coded (in reverse scan order), and past eob they are 0 - their own coefficients, read from q
    // (no per-thread level map: 1 KB of local memory per thread)
    auto lev = [&](int i) -> int { const int v = q[i] < 0 ? -q[i] : q[i]; return v < 15 ? v : 15; };
    for (int c = eob - 1; c >= 0; c--) {
        const int pos = scan[c];
        const int v = q[pos] < 0 ? -q[pos] : q[pos];
        const int level = v < 15 ? v : 15;
        const int row = pos >> bwl, col = pos - (row << bwl);
        if (c == eob - 1) {
            int bctx = c == 0 ? 0 : c <= (height << bwl) / 8 ? 1 : c <= (height << bwl) / 4 ? 2 : 3;
            o.sym(CID(coeff_base_eob[txSzCtx][ptype][bctx]), min(level, 3) - 1, 3);
        } else {
            int mag = 0;
            for (int idx = 0; idx < AV1G_SIG_REF_DIFF_OFFSET_NUM; idx++) {
                const int rr = row + AV1G_T(Sig_Ref_Diff_Offset)[txClass][idx][0];
                const int cc = col + AV1G_T(Sig_Ref_Diff_Offset)[txClass][idx][1];
                if (rr >= 0 && cc >= 0 && rr < height && cc < width) mag += min(lev((rr << bwl) + cc), 3);
            }
            int bctx = min((mag + 1) >> 1, 4);
            if (txClass == AV1G_TX_CLASS_2D) {
                if (row == 0 && col == 0) bctx = 0;
                else bctx += AV1G_T(Coeff_Base_Ctx_Offset)[txSz][min(row, 4)][min(col, 4)];
            } else {
                const int id = txClass == AV1G_TX_CLASS_VERT ? row : col;
                bctx += AV1G_T(Coeff_Base_Pos_Ctx_Offset)[min(id, 2)];
            }
            o.sym(CID(coeff_base[txSzCtx][ptype][bctx]), min(level, 3), 4);
        }
        if (level > AV1G_NUM_BASE_LEVELS) {
            int mag = 0;
            for (int idx = 0; idx < 3; idx++) {
                const int rr = row + AV1G_T(Mag_Ref_Offset_With_Tx_Class)[txClass][idx][0];
                const int cc = col + AV1G_T(Mag_Ref_Offset_With_Tx_Class)[txClass][idx][1];
                if (rr >= 0 && cc >= 0 && rr < height && cc < width) mag += lev(rr * width + cc);
            }
            mag = min((mag + 1) >> 1, 6);
            int brctx;
            if (pos == 0) brctx = mag;
            else if (txClass == AV1G_TX_CLASS_2D) brctx = (row < 2 && col < 2) ? mag + 7 : mag + 14;
            else if (txClass == AV1G_TX_CLASS_HORIZ) brctx = col == 0 ? mag + 7 : mag + 14;
            else brctx = row == 0 ? mag + 7 : mag + 14;
            const int brsz = min(txSzCtx, (int)AV1G_TX_32X32);
            int rem = level - 3;
            for (int idx = 0; idx < AV1G_COEFF_BASE_RANGE / (AV1G_BR_CDF_SIZE - 1); idx++) {
                const int kk = min(rem, 3);
                o.sym(CID(coeff_br[brsz][ptype][brctx]), kk, AV1G_BR_CDF_SIZE);
                rem -= kk;
                if (kk < 3) break;
            }
        }
    }
    for (int c = 0; c < eob; c++) {
        const int pos = scan[c];
        const int v = q[pos];
        if (!v) continue;
        const int sign = v < 0, a = sign ? -v : v;
        if (c == 0) {
            o.sym(CID(dc_sign[ptype][dcSign < 0 ? 1 : dcSign > 0 ? 2 : 0]), sign, 2);
        } else {
            o.bit(sign);
        }
        if (a > AV1G_NUM_BASE_LEVELS + AV1G_COEFF_BASE_RANGE) {
            const uint32_t xg = (uint32_t)(a - (AV1G_NUM_BASE_LEVELS + AV1G_COEFF_BASE_RANGE));
            int len = 0;
            while ((xg >> len) > 1) len++;
            for (int i = 0; i < len; i++) o.bit(0);
            o.bit(1);
            for (int i = len - 1; i >= 0; i--) o.bit((xg >> i) & 1);
        }
    }
}

