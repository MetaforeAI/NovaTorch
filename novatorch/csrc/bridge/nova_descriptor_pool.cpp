#include "nova_descriptor_pool.h"

#include <stdexcept>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void NovaDescriptorPool::init(VkDevice device, uint32_t max_sets) {
    device_ = device;

    // One pool size entry: storage buffers.
    // Allow up to 8 descriptors per set (covers ops with many tensor args).
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = max_sets * 8;

    VkDescriptorPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = max_sets;
    pool_ci.poolSizeCount = 1;
    pool_ci.pPoolSizes = &pool_size;

    VkResult result = vkCreateDescriptorPool(
        device_, &pool_ci, nullptr, &pool_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "NovaDescriptorPool: vkCreateDescriptorPool failed");
    }
}

void NovaDescriptorPool::shutdown() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Allocation and reset
// ---------------------------------------------------------------------------

VkDescriptorSet NovaDescriptorPool::allocate(VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult result = vkAllocateDescriptorSets(device_, &alloc_info, &set);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "NovaDescriptorPool: vkAllocateDescriptorSets failed "
            "(pool may be exhausted)");
    }

    return set;
}

void NovaDescriptorPool::reset() {
    VkResult result = vkResetDescriptorPool(device_, pool_, 0);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "NovaDescriptorPool: vkResetDescriptorPool failed");
    }
}
