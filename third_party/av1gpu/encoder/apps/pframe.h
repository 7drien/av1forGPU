// CUDA P-frame analysis: per-block motion search, partition and residual decisions, exact reconstruction.
//
// Every inter block predicts from one reference frame with its own quarter-sample motion vector. The
// device chooses square partitions from 64x64 down to 8x8, a transform equal to the block size (TX_MODE_LARGEST;
// 64x64 blocks are coded only as skip), quantized DCT residuals, and writes the decoder-exact reconstruction.
// The host writes the AV1 syntax for these decisions (pframe_writer.h).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <vector>

namespace pf {

struct Mv {
    int16_t row, col;   // 1/8 luma sample units; always even (allow_high_precision_mv = 0)
};
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool operator==(Mv a, Mv b) { return a.row == b.row && a.col == b.col; }

// Decision for each 8x8 luma unit (raster order, ceil(W/8) x ceil(H/8)).
struct BlockInfo {
    Mv mv;          // vector of the first (or only) reference
    Mv mv2;         // vector of the second reference of a compound block
    uint8_t log2;   // log2 of the block size containing this unit (3..6)
    uint8_t skip;   // 1: no residual coded
    uint8_t ref;    // 0: reference 0 only, 1: reference 1 only, 2: compound average of both, 3: reference 2 only
    uint8_t txSplit;   // 1: luma uses four half-size transforms (coded blocks)
    uint8_t part;      // partition of the log2 node: 0 NONE, 1 HORZ (two n x n/2 blocks), 2 VERT (two n/2 x n),
                       // 3 HORZ_4, 4 VERT_4
    uint8_t ab;        // AV1 PARTITION_HORZ_A .. VERT_B when the block belongs to a three-way node, else 0
    uint8_t abLog2;    // log2 size of that node
    uint8_t filter;    // interp_filter: 0 EIGHTTAP, 1 SMOOTH, 2 SHARP
    uint8_t intra;     // 1: intra block (square, one transform of the block size, DCT_DCT, UV_DC_PRED chroma)
    uint8_t yMode;     // intra luma mode (DC, V, H, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH)
    uint8_t uvMode;    // intra chroma mode (same set, UV numbering)
    int8_t yAngle;     // intra luma angle_delta (-3..3)
    int8_t cflU, cflV; // CfL alphas (uvMode UV_CFL_PRED)
    // loop restoration of the 64x64 luma unit whose top-left 8x8 this is (set only there): lrUse 1 = Wiener with
    // LrWiener coefficients lrCoef[pass][0..2] (pass 0 vertical, 1 horizontal)
    int8_t lrUse;
    int8_t lrCoef[2][3];
    uint16_t txTypes;  // AV1 luma transform type of the transform covering each 4x4 (nibble 2 * dy + dx); 0 DCT_DCT
};

struct Settings {
    int qindex = 100;
    float lambdaMul = 0.5f;
    int searchRange = 64;    // full-pel radius of the zero-centred quarter-resolution search (at most 192; the 1/16-resolution
                             // stage covers +-768)
    float rnd = 0.46f, zbin = 0.70f;   // quantizer rounding and dead zone (five-clip sweeps)
    float compoundBias = 1.0f;   // extra bits charged to compound prediction in the search
    bool txSplit = true;
    bool blocks8 = true;         // 8x8 level and 16x8 / 8x16 blocks (--pf-no-8x8: only where a 16x16 node must split)
    bool rect4 = true;           // 4:1 shapes (HORZ_4 / VERT_4: 32x8 .. 16x64) with rectPartitions (--pf-no-rect4)
    bool rectPartitions = true;  // search HORZ / VERT partitions of 16x16 .. 64x64 nodes         // search four half-size luma transforms per coded block (TX_MODE_SELECT)
    int txTypes = 3;             // inter transform types: 0 DCT_DCT only, 1 IDTX / DCT_DCT (reduced_tx_set),
                                 // 2 the full inter sets (reduced_tx_set = 0) with a 7-type search, 3 every type
    bool switchableFilter = true;   // per-block interpolation filter (is_filter_switchable; --pf-no-switchable)
    int splitTypes = 0;          // split luma transforms: 0 same type search, 1 seven types, 2 DCT_DCT only (--pf-fast)
    float pruneMargin = 2.f;     // rectangular search skipped unless the square's coded cost < margin * skip cost
    bool pruneRect = true;       // skip HORZ / VERT / 4:1 searches of nodes whose square evaluation chose skip
    bool splitPrune = false;     // no split-transform search for blocks whose whole transform is all zero
    float modeBits = 2.f;        // bits charged per block for its mode info besides the vectors (evaluation)
    float squarePrune = 0.f;     // > 0: skip the square children of nodes that clearly prefer skip (margin as pruneMargin);
                                 // the app enables it (1) for moving content only (static content gains from children)
    bool rdoq = false;           // greedy rate-distortion optimized quantization (--pf-rdoq)
    bool modeRefine = true;      // recheck motion vectors on the final partition
    int modeRefinePasses = 6;    // passes of that check (each against the neighbours' vectors of the previous one)
    bool intra = true;           // intra blocks in the inter frame (square blocks, decided after mode refinement)
    float intraBits = 6.f;       // extra mode bits charged to an intra block in that decision
    bool intraRect = true;       // intra blocks of rectangular shapes too (22-36% of the encode time, 2-7% of size)
    bool intraDirectional = true;   // D113 / D135 / D157 intra modes (requires enable_intra_edge_filter = 0)
};

// In-loop filter parameters chosen for a P frame (the reconstruction already includes them).
struct FilterParams {
    int lf[4] = {0, 0, 0, 0};   // Y vertical, Y horizontal, U, V
    int sharpness = 0;
    bool cdef = false;
    int damping = 3;             // CdefDamping (3..6)
    int yPri = 0, ySec = 0, uvPri = 0, uvSec = 0;   // secondary strengths 0, 1, 2, 4
    bool lrAllowed = false;      // input: try luma Wiener loop restoration
    bool cdefPerSb = false;      // input: choose up to 8 presets and one per 64x64 superblock (cdef_bits > 0)
    int cdefBits = 0;            // output: cdef_bits; presets below (index 0 = yPri / ySec / uvPri / uvSec)
    int cdefPreset[8][4] = {};   // y pri, y sec, uv pri, uv sec
    std::vector<int8_t> cdefIdx; // output: per 64x64 superblock (raster), -1 without coded blocks
    bool lrLuma = false;         // output: some luma unit uses it (FrameRestorationType[0] = RESTORE_WIENER)
};

// GPU tile entropy coding of the last analyzed frame (pframe_tile.cuh).
struct EntropyIn {
    int refName[3] = {1, 7, 6};   // AV1 reference (LAST, ALTREF, ALTREF2) of BlockInfo::ref 0, 1 and 3
    bool referenceSelect = false;
    bool txModeSelect = false;
    bool reducedTxSet = true;
    bool lrLuma = false;        // luma Wiener loop restoration (parameters in the grid)
    int signBias[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // RefFrameSignBias by reference frame
    bool useTmvp = false;                // use_ref_frame_mvs
    const Mv *motionField = nullptr;     // HostEntropy's copy of MotionFieldMvs[8][MiRows / 2][MiCols / 2]
    bool switchableFilter = false;
    bool skipMode = false;      // skip_mode_present (the app sets it only when SkipModeFrame = LAST, ALTREF)
    bool filterIntra = false;   // enable_filter_intra of the sequence header (intra DC blocks <= 32x32 code the flag)
    int cdefBits = 0;           // cdef_bits (per-superblock CDEF presets)
    const int8_t *cdefIdx = nullptr;   // host copy of the per-64x64 presets (CPU tile coder; the GPU uses its own)
    int qindex = 100;
    int primarySlot = -1;       // slot whose saved CDFs initialize the frame, or -1 for the defaults
    int saveSlot = -1;          // slot that receives the frame-end CDFs, or -1
    int tileColsLog2 = 0, tileRowsLog2 = 0;
    int contextTile = 0;        // tile whose final CDFs are the frame-end CDFs
    bool recordsOnly = false;   // GPU: model the symbols only (fetch with entropyFetchRecords, code on the host)
};
struct EntropyOut {
    std::vector<std::vector<uint8_t>> tiles;
    float bits[4] = {};          // partition + mode info, motion vectors, luma coefficients, chroma coefficients
    int modes[4] = {};           // NEAREST, NEAR, GLOBAL, NEW (first list)
    int compound = 0;
};

// CPU tile entropy coding with the same tile coder as the GPU path (pframe_tile.cuh). It owns host copies of the
// reference-slot CDFs, so it can run on a worker thread while the GPU analyzes later frames.
class HostEntropy {
public:
    ~HostEntropy();
    int init(int width, int height, char *err, int errLen);
    // grid / levels: host copies of the decisions and levels of the frame.
    int code(const EntropyIn &in, const BlockInfo *grid, const int16_t *levels, EntropyOut &out, char *err, int errLen);
    // Range-code symbol records modeled on the GPU (in.primarySlot / saveSlot refer to this object's slots).
    int codeRecords(const EntropyIn &in, const std::vector<uint32_t> &tileCounts, const std::vector<uint8_t> &records,
                    EntropyOut &out, char *err, int errLen);
private:
    int w = 0, h = 0, aw = 0, ah = 0;
    void *state = nullptr;
};

// Reference frames are tightly packed 8-bit YUV420 (W x H luma, then (W/2) x (H/2) U and V).
class Analyzer {
public:
    ~Analyzer();
    int init(int width, int height, char *err, int errLen);
    cudaStream_t stream() const;
    // src: device planes with pitches; refs: device reference frames (refs[0] = LAST); recon: output frame.
    // With syntax.useTmvp, call projectMotion after queuing coarse search and upload its result
    // on the compute stream for both motion refinement passes and GPU entropy staging.
    using ProjectMotion = std::function<const Mv *()>;
    int analyze(const uint8_t *const src[3], const int pitch[3], const uint8_t *const *refs, int refCount,
                const Settings &s, const EntropyIn &syntax, uint8_t *recon, const ProjectMotion &projectMotion,
                char *err, int errLen);
    // Enqueue copies into pinned buffers. The caller waits on done before reading or releasing them.
    int download(BlockInfo *info, int16_t *levels, cudaEvent_t done, char *err, int errLen);
    // Queue an early download of the decisions used by the host motion-field storage.
    int stageMotionGrid(char *err, int errLen);
    const BlockInfo *motionGrid(char *err, int errLen);
    int gridCols() const { return aw / 8; }
    size_t gridCount() const { return size_t(aw / 8) * (ah / 8); }
    size_t levelCount() const { return size_t(aw) * ah * 3 / 2; }
    // Deblocking and CDEF of the last analyzed frame's reconstruction (in place), choosing levels by SSE.
    int filter(const uint8_t *const src[3], const int pitch[3], uint8_t *recon, int qindex, bool cdefAllowed,
               FilterParams &out, char *err, int errLen);
    // Forget the temporal motion field (after a keyframe).
    void resetTemporal();
    // Motion-compensated temporal filter of a padded source frame (planes with pitches aw, aw/2, aw/2) from up to 8
    // neighbouring padded frames; out receives the filtered padded planes. strength scales the blending weights.
    int temporalFilter(const uint8_t *const center[3], const uint8_t *const *const neighbors[3], int count, int qindex,
                       float strength, uint8_t *const out[3], char *err, int errLen);
    // Estimated noise standard deviation of a luma plane (libaom estimator; blocking), -1 if undetermined.
    double noiseSigma(const uint8_t *luma, int pitch, char *err, int errLen);
    // Code the tiles of the last analyzed frame on the GPU (blocking).
    int entropy(const EntropyIn &in, EntropyOut &out, char *err, int errLen);
    // Asynchronous form: launch into entropy slot k (0 .. kEntropySlots-1) on the entropy stream, fetch later (any
    // thread). Launches must be issued in coding order.
    static constexpr int kEntropySlots = 3;
    int entropyLaunch(const EntropyIn &in, int k, char *err, int errLen);
    int entropyFetch(int k, EntropyOut &out, char *err, int errLen);
    // Records-only launches: per-tile record counts and the records (pf::HostEntropy::codeRecords codes them).
    int entropyFetchRecords(int k, std::vector<uint32_t> &tileCounts, std::vector<uint8_t> &records, char *err,
                            int errLen);
    int width() const { return w; }
    int height() const { return h; }
    int alignedWidth() const { return aw; }
    int alignedHeight() const { return ah; }
    // Coefficient (natural order index r * T + c) of the transform block of size T at (x, y) of plane p:
    // levels[planeOffset(p) + (y + r) * planeWidth(p) + x + c].
    size_t planeOffset(int p) const { return p == 0 ? 0 : size_t(aw) * ah + size_t(p - 1) * (aw / 2) * (ah / 2); }
    int planeWidth(int p) const { return p ? aw / 2 : aw; }
    double lastMs() const { return elapsedMs; }

private:
    int w = 0, h = 0, aw = 0, ah = 0;
    void *state = nullptr;
    double elapsedMs = 0;
};

}  // namespace pf
