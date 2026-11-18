cmake_minimum_required(VERSION 3.24)
project(av1forGPU LANGUAGES C CXX)

# CUDA software encoder backend used as the starting point for video support.
# It performs AV1 mode decisions, reconstruction, filtering, and entropy coding
# on CUDA cores; output support is currently still-image AVIF.
add_subdirectory(third_party/av1gpu)
