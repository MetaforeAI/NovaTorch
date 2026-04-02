#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct NovaPipelineInfo {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout desc_layout;
    uint32_t num_buffers;        // number of storage buffer bindings
    uint32_t push_constant_size; // size of push constant block in bytes
};

class NovaPipelineCache {
public:
    void init(VkDevice device);
    void shutdown();

    /// Get or create a pipeline for a named kernel.
    /// @param kernel_name   shader name (without .spv extension)
    /// @param num_buffers   how many storage buffer bindings the shader expects
    /// @param push_constant_size  size of push constants in bytes (0 if none)
    const NovaPipelineInfo& get(
        const std::string& kernel_name,
        uint32_t num_buffers,
        uint32_t push_constant_size);

private:
    VkDevice device_ = VK_NULL_HANDLE;
    std::unordered_map<std::string, NovaPipelineInfo> cache_;

    std::vector<uint32_t> loadSPIRV(const std::string& kernel_name);
    NovaPipelineInfo createPipeline(
        const std::string& kernel_name,
        uint32_t num_buffers,
        uint32_t push_constant_size);
};
