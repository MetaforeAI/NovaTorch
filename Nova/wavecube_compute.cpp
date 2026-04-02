/*
 * WaveCube GPU Compute - Direct Nova integration
 * Synthesizes 64³ voxel grid from 1024² FFT wavetable using compute shader
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include "./Nova.h"

// SMFTEngine friend class for accessing private NovaCore
class SMFTEngine {
public:
    static NovaCore* getCore(Nova* nova) {
        return nova->_architect;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: wavecube_compute <input_wavetable.raw> <output_volume.raw>\n";
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::cout << "WaveCube GPU Compute\n";
    std::cout << "  Input: " << input_file << "\n";
    std::cout << "  Output: " << output_file << "\n";

    // Initialize Nova in compute-only mode
    std::cout << "  Initializing Nova compute context...\n";

    NovaConfig config = {
        .name = "WaveCube Compute",
        .screen = {64, 64},  // Minimal extent for compute-only
        .debug_level = "INFO",
        .dimensions = "3D",
        .camera_type = "fixed",
        .compute = true
    };

    try {
        Nova* nova = new Nova(config);
        NovaCore* core = SMFTEngine::getCore(nova);

        if (!nova->initialized) {
            std::cerr << "  ERROR: Nova initialization failed\n";
            delete nova;
            return 2;
        }

        std::cout << "  ✅ Nova compute context initialized\n";

        // Load input wavetable (1024×1024×4 float32 RGBA)
        std::cout << "  Loading wavetable: " << input_file << "\n";
        std::ifstream input(input_file, std::ios::binary);
        if (!input) {
            std::cerr << "  ERROR: Cannot open input file\n";
            delete nova;
            return 1;
        }

        const size_t wavetable_size = 1024 * 1024 * 4 * sizeof(float);
        std::vector<float> wavetable_data(1024 * 1024 * 4);
        input.read(reinterpret_cast<char*>(wavetable_data.data()), wavetable_size);
        input.close();

        std::cout << "  ✅ Loaded " << (wavetable_size / (1024*1024)) << " MB wavetable\n";

        // For now, return code 2 to signal CPU fallback
        // Full GPU pipeline implementation requires:
        // 1. VMA buffer creation for wavetable (1024×1024×4 RGBA)
        // 2. VMA buffer creation for wavecube (64×64×64×4 RGBA)
        // 3. Load SPIR-V shader (shaders/wavecube_synthesis.spv)
        // 4. Create descriptor sets (uniform buffer + 2D input + 3D output)
        // 5. Create compute pipeline
        // 6. Dispatch compute shader (16×16×16 workgroups with 4×4×4 local size)
        // 7. Copy result back to CPU buffer

        std::cout << "  ⚠️  Full GPU compute pipeline integration in progress\n";
        std::cout << "  Using CPU fallback for now (exit code 2)\n";

        delete nova;
        return 2;

    } catch (const std::exception& e) {
        std::cerr << "  ERROR: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
