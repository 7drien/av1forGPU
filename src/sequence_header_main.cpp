#include "bitstream/sequence_header.hpp"

#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

unsigned long parseUnsigned(const char* text, const char* name) {
    std::size_t parsed = 0;
    const std::string value(text);
    const unsigned long result = std::stoul(value, &parsed);
    if (parsed != value.size()) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + value);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: av1gpu-seqhdr <still|video> <width> <height> "
                     "<level-index> <output.obu>\n";
        return 2;
    }

    try {
        const std::string mode(argv[1]);
        if (mode != "still" && mode != "video") {
            throw std::runtime_error("mode must be 'still' or 'video'");
        }
        const auto width = parseUnsigned(argv[2], "width");
        const auto height = parseUnsigned(argv[3], "height");
        const auto level = parseUnsigned(argv[4], "level index");
        if (width > 65536 || height > 65536 || level > 23) {
            throw std::runtime_error("dimension or level value is out of range");
        }
        const auto obu = av1gpu::bitstream::SequenceHeaderWriter::makeObu(
            {static_cast<int>(width), static_cast<int>(height),
             static_cast<std::uint8_t>(level),
             mode == "still" ? av1gpu::bitstream::SequenceMode::StillPicture
                              : av1gpu::bitstream::SequenceMode::Video});

        std::ofstream output(argv[5], std::ios::binary);
        if (!output || !output.write(reinterpret_cast<const char*>(obu.data()),
                                     static_cast<std::streamsize>(obu.size()))) {
            throw std::runtime_error(std::string("cannot write output: ") + argv[5]);
        }
        std::cout << "Wrote a " << mode << " AV1 sequence-header OBU for "
                  << width << "x" << height << " (level index " << level << ") to "
                  << argv[5] << '\n';
    } catch (const std::exception& error) {
        std::cerr << "av1gpu-seqhdr: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
