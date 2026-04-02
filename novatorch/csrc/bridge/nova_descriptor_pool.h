#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

class NovaDescriptorPool {
public:
    void init(VkDevice device, uint32_t max_sets = 1024);
    void shutdown();

    /// Allocate a descriptor set from the pool.
    /// Throws on failure (e.g., pool exhausted).
    VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    /// Reset all descriptor sets back to the pool (call once per frame/batch).
    void reset();

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
};
