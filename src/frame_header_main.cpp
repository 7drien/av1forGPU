#include "bitstream/frame_header.hpp"

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

unsigned long parseUnsigned(const char* text, const char* name) {
    std::size_t parsed = 0;
    const std::string value(text);
    const auto result = std::stoul(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: av1gpu-framehdr <width> <height> <output.obu>\n";
        return 2;
    }

    try {
        const auto width = parseUnsigned(argv[1], "width");
        const auto height = parseUnsigned(argv[2], "height");
        if (width > 65536 || height > 65536) {
            throw std::runtime_error("dimensions are out of range");
        }
        const auto obu = av1gpu::bitstream::FrameHeaderWriter::makeObu(
            {static_cast<int>(width), static_cast<int>(height)});
        std::ofstream output(argv[3], std::ios::binary);
        if (!output || !output.write(reinterpret_cast<const char*>(obu.data()),
                                     static_cast<std::streamsize>(obu.size()))) {
            throw std::runtime_error(std::string("cannot write output: ") + argv[3]);
        }
        std::cout << "Wrote reduced still-picture frame header for " << width << "x"
                  << height << " (one tile, lossless syntax) to " << argv[3] << '\n';
    } catch (const std::exception& error) {
        std::cerr << "av1gpu-framehdr: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
