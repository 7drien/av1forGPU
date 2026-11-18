#pragma once
#include <cuda_runtime.h>
#include <cstdint>

constexpr int MotionModes = 7;
constexpr int AnchorMode = 5;
constexpr int MotionStepCount = 4;
__host__ __device__ inline int motionStepAt(int index) { return 2 << index; }
cudaError_t chooseMotionStep(const uint8_t *source, const uint8_t *ref, const uint8_t *anchor,
                             int width, int height, int pitch, int modeCount,
                             unsigned long long *deviceScores, int &step);

struct MotionBlock { unsigned sse, samples, mode; unsigned candidateSse[MotionModes]; };
// Whole-pixel translations are even so chroma predictions also land on samples.
__host__ __device__ inline int motionX(int mode, int step = 2) { return (mode == 1 || mode == 5) * step - (mode == 2 || mode == 6) * step; }
__host__ __device__ inline int motionY(int mode, int step = 2) { return (mode == 3 || mode == 5) * step - (mode == 4 || mode == 6) * step; }
cudaError_t searchMotion(const uint8_t *const src[3], const uint8_t *ref, int w, int h,
                         int pitch, int modeCount, int step, MotionBlock *device, MotionBlock *host, const uint8_t *anchor = nullptr, int blockSize = 64);
cudaError_t reconstructMotion(const uint8_t *ref, uint8_t *output, int width, int height,
                              const MotionBlock *blocks, int step, const uint8_t *anchor = nullptr, int blockSize = 64);
cudaError_t prepareInterResiduals(const uint8_t *const source[3], const uint8_t *reference,
                                  int width, int height, int pitch, const MotionBlock *motion,
                                  int16_t *deviceLevels, int16_t *hostLevels, int threshold, uint8_t *reconstructed, int step, int qindex, bool luma8, int16_t *alternateLevels = nullptr,
                                  uint8_t *alternateRecon = nullptr, uint8_t *deviceTxMap = nullptr,
                                  uint8_t *hostTxMap = nullptr, const uint8_t *anchor = nullptr, bool rdoq = false, int motionBlockSize = 64);
