NVCC ?= nvcc
CUDA_ARCH ?= sm_86
BUILD_DIR ?= build
TARGET := $(BUILD_DIR)/av1gpu-analyze
SOURCES := src/main.cpp src/yuv420.cpp src/bitstream/bit_writer.cpp \
	src/bitstream/obu_writer.cpp src/bitstream/sequence_header.cpp \
	src/cuda/block_analysis.cu
SEQHDR_TARGET := $(BUILD_DIR)/av1gpu-seqhdr
SEQHDR_SOURCES := src/sequence_header_main.cpp src/bitstream/bit_writer.cpp \
	src/bitstream/obu_writer.cpp src/bitstream/sequence_header.cpp
STREAM_TARGET := $(BUILD_DIR)/av1gpu-analyze-stream
STREAM_SOURCES := src/stream_analysis_main.cpp src/cuda/block_analysis.cu
FRAMEHDR_TARGET := $(BUILD_DIR)/av1gpu-framehdr
FRAMEHDR_SOURCES := src/frame_header_main.cpp src/bitstream/bit_writer.cpp \
	src/bitstream/obu_writer.cpp src/bitstream/frame_header.cpp

.PHONY: all clean

all: $(TARGET) $(SEQHDR_TARGET) $(STREAM_TARGET) $(FRAMEHDR_TARGET)

$(TARGET): $(SOURCES) include/yuv420.hpp include/cuda/block_analysis.cuh \
	include/bitstream/bit_writer.hpp include/bitstream/obu_writer.hpp \
	include/bitstream/sequence_header.hpp
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) -Iinclude $(SOURCES) -o $@

$(SEQHDR_TARGET): $(SEQHDR_SOURCES) include/bitstream/bit_writer.hpp \
	include/bitstream/obu_writer.hpp include/bitstream/sequence_header.hpp
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) -Iinclude $(SEQHDR_SOURCES) -o $@

$(STREAM_TARGET): $(STREAM_SOURCES) include/cuda/block_analysis.cuh include/yuv420.hpp
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) -Iinclude $(STREAM_SOURCES) -o $@

$(FRAMEHDR_TARGET): $(FRAMEHDR_SOURCES) include/bitstream/bit_writer.hpp \
	include/bitstream/obu_writer.hpp include/bitstream/frame_header.hpp
	@mkdir -p $(BUILD_DIR)
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) -Iinclude $(FRAMEHDR_SOURCES) -o $@

clean:
	rm -rf $(BUILD_DIR)
