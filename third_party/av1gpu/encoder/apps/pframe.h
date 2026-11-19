// CUDA P-frame analysis: per-block motion search, partition and residual decisions, exact reconstruction.
//
// Every inter block predicts from one reference frame with its own quarter-sample motion vector. The
// device chooses square partitions from 64x64 down to 8x8, a transform equal to the block size (TX_MODE_LARGEST;
// 64x64 blocks are coded only as skip), quantized DCT residuals, and writes the decoder-exact reconstruction.
// The host writes the AV1 syntax for these decisions (pframe_writer.h).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
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
    uint8_t ref;    // 0: reference 0 only, 1: reference 1 only, 2: compound average of both
    uint8_t pad;
};

struct Settings {
    int qindex = 100;
    float lambdaMul = 1.0f;
    int searchRange = 64;    // full-pel coarse search radius
    float rnd = 0.33f, zbin = 0.70f;
    float compoundBias = 1.0f;   // extra bits charged to compound prediction in the search
};

// In-loop filter parameters chosen for a P frame (the reconstruction already includes them).
struct FilterParams {
    int lf[4] = {0, 0, 0, 0};   // Y vertical, Y horizontal, U, V
    int sharpness = 0;
    bool cdef = false;
    int damping = 3;             // CdefDamping (3..6)
    int yPri = 0, ySec = 0, uvPri = 0, uvSec = 0;   // secondary strengths 0, 1, 2, 4
};

// Reference frames are tightly packed 8-bit YUV420 (W x H luma, then (W/2) x (H/2) U and V).
class Analyzer {
public:
    ~Analyzer();
    int init(int width, int height, char *err, int errLen);
    // src: device planes with pitches; refs: device reference frames (refs[0] = LAST); recon: output frame.
    // Fills info (8x8 grid) and levels (spatial layout, see levelsAt) on the host.
    int analyze(const uint8_t *const src[3], const int pitch[3], const uint8_t *const *refs, int refCount,
                const Settings &s, uint8_t *recon, std::vector<BlockInfo> &info, std::vector<int16_t> &levels,
                char *err, int errLen);
    // Deblocking and CDEF of the last analyzed frame's reconstruction (in place), choosing levels by SSE.
    int filter(const uint8_t *const src[3], const int pitch[3], uint8_t *recon, int qindex, bool cdefAllowed,
               bool search, FilterParams &out, char *err, int errLen);
    // Forget the temporal motion field (after a keyframe).
    void resetTemporal();
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
