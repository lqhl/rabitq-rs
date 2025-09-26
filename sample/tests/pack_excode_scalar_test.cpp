#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#define RABITQ_FORCE_SCALAR
#include "quantization/pack_excode.hpp"

namespace {
size_t unpack_rabitqplus_code(const uint8_t* compact, size_t dim, size_t bits, std::vector<uint8_t>& output) {
    const uint8_t* cursor = compact;
    std::fill(output.begin(), output.end(), 0);

    switch (bits) {
        case 1: {
            for (size_t j = 0; j < dim; j += 16) {
                uint16_t word;
                std::memcpy(&word, cursor, sizeof(uint16_t));
                cursor += sizeof(uint16_t);
                for (size_t lane = 0; lane < 16; ++lane) {
                    output[j + lane] = static_cast<uint8_t>((word >> lane) & 0x1);
                }
            }
            break;
        }
        case 2: {
            for (size_t j = 0; j < dim; j += 16) {
                for (size_t lane = 0; lane < 4; ++lane) {
                    uint8_t byte = cursor[lane];
                    output[j + lane] = byte & 0x3;
                    output[j + lane + 4] = (byte >> 2) & 0x3;
                    output[j + lane + 8] = (byte >> 4) & 0x3;
                    output[j + lane + 12] = (byte >> 6) & 0x3;
                }
                cursor += 4;
            }
            break;
        }
        case 3: {
            for (size_t j = 0; j < dim; j += 64) {
                for (size_t lane = 0; lane < 16; ++lane) {
                    uint8_t byte = cursor[lane];
                    output[j + lane] = byte & 0x3;
                    output[j + lane + 16] = (byte >> 2) & 0x3;
                    output[j + lane + 32] = (byte >> 4) & 0x3;
                    output[j + lane + 48] = (byte >> 6) & 0x3;
                }
                cursor += 16;

                uint64_t top_bits;
                std::memcpy(&top_bits, cursor, sizeof(uint64_t));
                cursor += sizeof(uint64_t);
                for (size_t lane = 0; lane < 64; ++lane) {
                    output[j + lane] |= static_cast<uint8_t>(((top_bits >> lane) & 0x1) << 2);
                }
            }
            break;
        }
        case 4: {
            for (size_t j = 0; j < dim; j += 16) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    uint8_t byte = cursor[lane];
                    output[j + lane] = byte & 0xF;
                    output[j + lane + 8] = (byte >> 4) & 0xF;
                }
                cursor += 8;
            }
            break;
        }
        case 5: {
            for (size_t j = 0; j < dim; j += 64) {
                for (size_t lane = 0; lane < 16; ++lane) {
                    uint8_t byte = cursor[lane];
                    output[j + lane] = byte & 0xF;
                    output[j + lane + 16] = (byte >> 4) & 0xF;
                }
                for (size_t lane = 0; lane < 16; ++lane) {
                    uint8_t byte = cursor[16 + lane];
                    output[j + lane + 32] = byte & 0xF;
                    output[j + lane + 48] = (byte >> 4) & 0xF;
                }
                cursor += 32;

                uint64_t top_bits;
                std::memcpy(&top_bits, cursor, sizeof(uint64_t));
                cursor += sizeof(uint64_t);
                for (size_t lane = 0; lane < 64; ++lane) {
                    output[j + lane] |= static_cast<uint8_t>(((top_bits >> lane) & 0x1) << 4);
                }
            }
            break;
        }
        case 6: {
            for (size_t j = 0; j < dim; j += 16) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    uint8_t byte = cursor[lane];
                    output[j + lane] = byte & 0xF;
                    output[j + lane + 8] = (byte >> 4) & 0xF;
                }
                cursor += 8;
                for (size_t lane = 0; lane < 4; ++lane) {
                    uint8_t byte = cursor[lane];
                    for (size_t group = 0; group < 4; ++group) {
                        size_t idx = j + lane + group * 4;
                        output[idx] |= static_cast<uint8_t>(((byte >> (group * 2)) & 0x3) << 4);
                    }
                }
                cursor += 4;
            }
            break;
        }
        case 7: {
            for (size_t j = 0; j < dim; j += 64) {
                for (size_t lane = 0; lane < 16; ++lane) {
                    uint8_t byte = cursor[lane];
                    output[j + lane] = byte & 0x3F;
                    output[j + lane + 48] = byte >> 6;
                }
                for (size_t lane = 0; lane < 16; ++lane) {
                    uint8_t byte = cursor[16 + lane];
                    output[j + lane + 16] = byte & 0x3F;
                    output[j + lane + 48] |= static_cast<uint8_t>((byte >> 6) << 2);
                }
                for (size_t lane = 0; lane < 16; ++lane) {
                    uint8_t byte = cursor[32 + lane];
                    output[j + lane + 32] = byte & 0x3F;
                    output[j + lane + 48] |= static_cast<uint8_t>((byte >> 6) << 4);
                }
                cursor += 48;

                uint64_t top_bits;
                std::memcpy(&top_bits, cursor, sizeof(uint64_t));
                cursor += sizeof(uint64_t);
                for (size_t lane = 0; lane < 64; ++lane) {
                    output[j + lane] |= static_cast<uint8_t>(((top_bits >> lane) & 0x1) << 6);
                }
            }
            break;
        }
        case 8: {
            std::memcpy(output.data(), cursor, dim);
            cursor += dim;
            break;
        }
        default:
            throw std::runtime_error("unsupported ex_bits in unpack_rabitqplus_code");
    }

    return static_cast<size_t>(cursor - compact);
}
}  // namespace

int main() {
    std::vector<std::pair<size_t, size_t>> cases = {
        {1, 32}, {2, 32}, {3, 64}, {4, 32}, {5, 64}, {6, 32}, {7, 64}, {8, 64},
    };

    std::mt19937 rng(12345);
    bool all_ok = true;

    for (const auto& [bits, dim] : cases) {
        std::vector<uint8_t> raw(dim);
        uint32_t max_val = (bits == 8) ? 0xFFu : ((1u << bits) - 1u);
        std::uniform_int_distribution<uint32_t> dist(0, max_val);
        for (auto& value : raw) {
            value = static_cast<uint8_t>(dist(rng));
        }

        std::vector<uint8_t> compact(dim * 2, 0xAA);
        std::vector<uint8_t> decoded(dim, 0);

        rabitqlib::quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
            raw.data(), compact.data(), dim, bits
        );
        unpack_rabitqplus_code(compact.data(), dim, bits, decoded);

        if (decoded != raw) {
            std::cerr << "Mismatch detected for ex_bits=" << bits << "\n";
            all_ok = false;
        }
    }

    if (!all_ok) {
        std::cerr << "Pack/unpack validation failed" << std::endl;
        return 1;
    }

    std::cout << "All scalar pack_excode cases passed" << std::endl;
    return 0;
}
