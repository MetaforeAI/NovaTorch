#pragma once
#include <vulkan/vulkan.h>
#include <functional>
#include <memory>

/**
 * RAII Wrappers for Vulkan Resources
 *
 * These wrappers ensure that resources are automatically cleaned up
 * in the correct order, and only if they were successfully created.
 *
 * Design Principle: "If you create it, register how to destroy it"
 */

namespace NovaRAII {

// Generic RAII wrapper for Vulkan handles
template<typename HandleType>
class VulkanHandle {
public:
    VulkanHandle() : handle(VK_NULL_HANDLE), cleanup(nullptr) {}

    VulkanHandle(HandleType h, std::function<void(HandleType)> cleanup_fn)
        : handle(h), cleanup(cleanup_fn) {}

    ~VulkanHandle() {
        if (handle != VK_NULL_HANDLE && cleanup) {
            cleanup(handle);
        }
    }

    // Move semantics
    VulkanHandle(VulkanHandle&& other) noexcept
        : handle(other.handle), cleanup(std::move(other.cleanup)) {
        other.handle = VK_NULL_HANDLE;
        other.cleanup = nullptr;
    }

    VulkanHandle& operator=(VulkanHandle&& other) noexcept {
        if (this != &other) {
            // Clean up current resource
            if (handle != VK_NULL_HANDLE && cleanup) {
                cleanup(handle);
            }
            // Take ownership
            handle = other.handle;
            cleanup = std::move(other.cleanup);
            other.handle = VK_NULL_HANDLE;
            other.cleanup = nullptr;
        }
        return *this;
    }

    // No copy
    VulkanHandle(const VulkanHandle&) = delete;
    VulkanHandle& operator=(const VulkanHandle&) = delete;

    // Access
    HandleType get() const { return handle; }
    HandleType* ptr() { return &handle; }
    operator HandleType() const { return handle; }
    explicit operator bool() const { return handle != VK_NULL_HANDLE; }

    // Release ownership without cleanup
    HandleType release() {
        HandleType h = handle;
        handle = VK_NULL_HANDLE;
        cleanup = nullptr;
        return h;
    }

private:
    HandleType handle;
    std::function<void(HandleType)> cleanup;
};

// Specific typedefs for common Vulkan types
using Instance = VulkanHandle<VkInstance>;
using Device = VulkanHandle<VkDevice>;
using Surface = VulkanHandle<VkSurfaceKHR>;
using Swapchain = VulkanHandle<VkSwapchainKHR>;
using CommandPool = VulkanHandle<VkCommandPool>;
using Fence = VulkanHandle<VkFence>;
using Semaphore = VulkanHandle<VkSemaphore>;
using RenderPass = VulkanHandle<VkRenderPass>;
using Pipeline = VulkanHandle<VkPipeline>;
using PipelineLayout = VulkanHandle<VkPipelineLayout>;
using DescriptorPool = VulkanHandle<VkDescriptorPool>;
using DescriptorSetLayout = VulkanHandle<VkDescriptorSetLayout>;
using Buffer = VulkanHandle<VkBuffer>;
using Image = VulkanHandle<VkImage>;
using ImageView = VulkanHandle<VkImageView>;
using Framebuffer = VulkanHandle<VkFramebuffer>;

// Helper function to create RAII-wrapped fence
inline Fence createFence(VkDevice device, VkFenceCreateFlags flags = 0) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = flags;

    VkFence fence;
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        return Fence(); // Return empty handle on failure
    }

    return Fence(fence, [device](VkFence f) {
        vkDestroyFence(device, f, nullptr);
    });
}

// Helper function to create RAII-wrapped semaphore
inline Semaphore createSemaphore(VkDevice device) {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphore semaphore;
    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
        return Semaphore(); // Return empty handle on failure
    }

    return Semaphore(semaphore, [device](VkSemaphore s) {
        vkDestroySemaphore(device, s, nullptr);
    });
}

// Helper function to create RAII-wrapped command pool
inline CommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    poolInfo.flags = flags;

    VkCommandPool pool;
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &pool) != VK_SUCCESS) {
        return CommandPool(); // Return empty handle on failure
    }

    return CommandPool(pool, [device](VkCommandPool p) {
        vkDestroyCommandPool(device, p, nullptr);
    });
}

} // namespace NovaRAII
