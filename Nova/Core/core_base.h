#pragma once
#include "./modules/atomic/atomic.h"
#include "./components/resource_registry.h"
#include "./components/vk_memory.h"
#include "./components/lexicon.h"
#include <functional>

/**
 * NovaCore - Base class for all Nova modes (Compute and Graphics)
 *
 * Provides shared Vulkan resources:
 * - Instance, physical device, logical device
 * - VMA allocator
 * - Queue infrastructure (compute, transfer)
 * - Immediate submission context
 * - Resource registry for cleanup
 *
 * Derived classes:
 * - NovaCompute: Compute-only mode (no surface/swapchain)
 * - NovaGraphics: Traditional rendering pipeline
 */
class NovaCore {
protected:
    // Core Vulkan resources (shared by all modes)
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice logical_device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    // Queue infrastructure
    struct {
        VkQueue compute = VK_NULL_HANDLE;
        VkQueue transfer = VK_NULL_HANDLE;
        QueueFamilyIndices indices;
        std::vector<VkQueueFamilyProperties> families;
        std::vector<std::vector<float>> priorities;
        DeletionQueue deletion;  // Legacy deletion queue
    } queues;

    // Immediate submission context (properly initialized in Phase 1)
    struct {
        VkFence fence = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandPool pool = VK_NULL_HANDLE;
    } immediate;

    // Command pools (shared)
    VkCommandPool compute_pool = VK_NULL_HANDLE;
    VkCommandPool transfer_pool = VK_NULL_HANDLE;

    // Resource cleanup registry
    NovaRAII::ResourceRegistry resource_registry;

    // Window extent (may be unused in compute mode)
    VkExtent2D window_extent;

    // Protected constructor (only derived classes can instantiate)
    NovaCore(const std::string& debug_level);

    // Virtual destructor (required for polymorphism)
    virtual ~NovaCore();

    // Shared initialization methods
    void createVulkanInstance(bool need_surface_extensions);
    void createPhysicalDevice(bool need_presentation, VkSurfaceKHR surface = VK_NULL_HANDLE);
    void createLogicalDevice(bool need_swapchain_extension);
    void createImmediateContext();
    void createSharedCommandPools();

    // Helper methods
    bool checkValidationLayerSupport();
    void getQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface, bool need_presentation);
    bool deviceProvisioned(VkPhysicalDevice device, VkSurfaceKHR surface, bool need_swapchain);
    void setQueueFamilyProperties(unsigned int i, VkSurfaceKHR surface, bool need_presentation);

    void _blankContext();
    void setWindowExtent(VkExtent2D extent);

public:
    // Public API (shared by all modes)

    /**
     * Submit commands immediately and wait for completion
     * Uses immediate context initialized in Phase 1
     */
    void immediateSubmit(std::function<void(VkCommandBuffer)>&& func);

    /**
     * Create buffer using VMA (ephemeral or persistent)
     */
    Buffer_T createEphemeralBuffer(size_t size, VkBufferUsageFlags flags, VmaMemoryUsage usage);

    // Resource access getters
    VmaAllocator getAllocator() const { return allocator; }
    VkQueue getComputeQueue() const { return queues.compute; }
    VkQueue getComputeQueue(uint32_t index) const;
    VkQueue getTransferQueue() const { return queues.transfer; }
    VkDevice getDevice() const { return logical_device; }
    VkInstance getInstance() const { return instance; }
    VkPhysicalDevice getPhysicalDevice() const { return physical_device; }
    VkCommandPool getComputePool() const { return compute_pool; }
    uint32_t getComputeQueueFamily() const { return queues.indices.compute_family.value(); }
    uint32_t getComputeQueueCount() const { return queues.families[queues.indices.compute_family.value()].queueCount; }
};
