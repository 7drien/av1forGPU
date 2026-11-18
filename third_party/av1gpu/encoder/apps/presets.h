#pragma once
#include "pipeline.h"
#include <cstring>

// speed presets: the speed knobs only (base = the GpConfig defaults)
inline bool apply_preset(GpConfig &cfg, const char *v) {
    const GpConfig d;   // defaults
    // reset to base first, so a preset fully defines the speed knobs
    cfg.P.shapeMask = d.P.shapeMask; cfg.P.K = d.P.K; cfg.P.Krecheck = d.P.Krecheck; cfg.P.txSearch = d.P.txSearch;
    cfg.P.txSearchCL = d.P.txSearchCL; cfg.P.pruneFlat = d.P.pruneFlat; cfg.P.screenSub = d.P.screenSub; cfg.P.uvK = d.P.uvK;
    cfg.P.deltas = d.P.deltas; cfg.P.uvDeltas = d.P.uvDeltas;
    cfg.P.filterIntra = d.P.filterIntra; cfg.P.txAllCL = d.P.txAllCL; cfg.P.txSizeSearch = d.P.txSizeSearch;
    cfg.tileMinMp = d.tileMinMp; cfg.tileColsLog2 = d.tileColsLog2; cfg.tileRowsLog2 = d.tileRowsLog2;
    // the decision-changing speed switches (staged analysis: --topdown 1 --prune-41 10; open-loop shortlists:
    // --uv-shortlist 3 --tx-shortlist 4): on in base and turbo, the two presets of the defaults, where they were measured
    // (v0.13.0: turbo +10 % MP/s for +0.3 % BD-rate, base +43 % for +0.4 %); off in the others.  Explicit options
    // override every preset, wherever they stand (see parse)
    cfg.an = d.an; cfg.P.uvShortlist = 0; cfg.P.txShortlist = 0;
    if (!strcmp(v, "base") || !strcmp(v, "turbo")) {
        cfg.an.topdown = 1; cfg.an.prune41 = 10.f;
        cfg.P.uvShortlist = 3; cfg.P.txShortlist = 4;
    }
    if (!strcmp(v, "base")) return true;
    if (!strcmp(v, "slow")) {
        // max quality (round 4): closed-loop tx-size search (largest vs one split), every tx type for
        // every re-checked candidate, K 6 / Krecheck 6, filter intra, deblocking-level + CDEF search
        cfg.P.txSizeSearch = 1; cfg.P.txAllCL = 1; cfg.P.K = 6; cfg.P.Krecheck = 6; cfg.P.filterIntra = 1;
        cfg.fLf = 2; cfg.fCdef = 2;
        return true;
    }
    if (!strcmp(v, "fast")) {          // no 4:1 shapes, K 2, no tx search in the analysis
        cfg.P.shapeMask = 0x1FFF; cfg.P.K = 2; cfg.P.Krecheck = 2; cfg.P.txSearch = 0;
        return true;
    }
    if (!strcmp(v, "faster")) {        // + flat pruning, single closed-loop candidate
        cfg.P.shapeMask = 0x1FFF; cfg.P.K = 2; cfg.P.Krecheck = 1; cfg.P.txSearch = 0; cfg.P.pruneFlat = 1.0f;
        return true;
    }
    if (!strcmp(v, "turbo") || !strcmp(v, "turbo2") || !strcmp(v, "turbo3")) {
        // faster + no 32x64/64x32 shapes, 2x-subsampled SATD screening for >= 16x16, 4x4 tiles for
        // images >= 2 MP (shorter wavefront / entropy critical paths), K 1
        cfg.P.shapeMask = 0x13FF; cfg.P.K = 1; cfg.P.Krecheck = 1; cfg.P.txSearch = 0; cfg.P.pruneFlat = 1.0f;
        cfg.P.screenSub = 2;
        cfg.tileMinMp = 2.f; cfg.tileColsLog2 = 2; cfg.tileRowsLog2 = 2;
        if (strcmp(v, "turbo")) {     // turbo2: + no 4x4/4x8/8x4, no closed-loop tx search, 1 chroma RD candidate
            cfg.P.shapeMask = 0x13F8; cfg.P.txSearchCL = 0; cfg.P.uvK = 1;
        }
        if (!strcmp(v, "turbo3")) {   // + no luma / UV angle deltas
            cfg.P.deltas = 0; cfg.P.uvDeltas = 0;
        }
        return true;
    }
    return false;
}

