#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

class NovaDescriptorPool {
public:
    /// Normal pool (eager path): sets are reset between batches.
    void init(VkDevice device, uint32_t max_sets = 1024);

    /// UPDATE_AFTER_BIND pool (compiled path): sets persist for
    /// command buffer reuse. Bindings can be updated after recording.
    void initUAB(VkDevice device, uint32_t max_sets = 1024);

    void shutdown();

    /// Allocate a descriptor set from the pool.
    VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    /// Reset all descriptor sets back to the pool (eager path only).
    void reset();

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
};
