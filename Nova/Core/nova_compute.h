#pragma once
#include "./core_base.h"
#include <mutex>
#include <thread>
#include <unordered_map>

/**
 * NovaCompute - Compute-only mode
 *
 * Lightweight Vulkan context for GPU compute operations.
 * Supports per-thread command buffer recording for concurrent use
 * by PyTorch's autograd engine (which runs backward ops on a
 * separate thread).
 *
 * Thread model:
 *   - Each thread gets its own VkCommandPool + VkCommandBuffer + VkFence.
 *   - Recording into command buffers is lock-free (no mutex).
 *   - vkQueueSubmit is serialized via a lightweight submit mutex.
 *   - Fence waits are per-thread (block only the calling thread).
 */
class NovaCompute : public NovaCore {
public:
    /// Per-thread Vulkan resources. Each thread that dispatches GPU work
    /// gets its own isolated set — no cross-thread contention during recording.
    struct ThreadResources {
        VkCommandPool pool   = VK_NULL_HANDLE;
        VkCommandBuffer cmd  = VK_NULL_HANDLE;
        VkFence fence        = VK_NULL_HANDLE;
        bool fence_submitted = false;
    };

    NovaCompute(const std::string& debug_level);
    ~NovaCompute() override;

    // -----------------------------------------------------------------
    // Per-thread resource management
    // -----------------------------------------------------------------

    /// Get or lazily create Vulkan resources for the calling thread.
    ThreadResources& getThreadResources();

    /// Submit a pre-recorded command buffer to the compute queue.
    /// Acquires submit_mutex_ only for the vkQueueSubmit call itself.
    void submitToQueue(VkCommandBuffer cmd, VkFence fence);

    /// Block until a fence signals. No mutex held — thread-safe.
    void waitForFence(VkFence fence);

    /// Destroy resources for the calling thread.
    void releaseThreadResources();

    /// Destroy resources for ALL threads (shutdown).
    void releaseAllThreadResources();

    // -----------------------------------------------------------------
    // Legacy synchronous API (uses calling thread's resources)
    // -----------------------------------------------------------------

    /// Record + submit + wait, all in one call.
    void executeCompute(std::function<void(VkCommandBuffer)>&& func);

    /// Wait for all device operations to complete.
    void waitIdle();

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    VkDevice getDevice() const { return logical_device; }
    VkPhysicalDevice getPhysicalDevice() const { return physical_device; }
    VmaAllocator getAllocator() const { return allocator; }

private:
    // Serializes vkQueueSubmit calls only — NOT held during recording or wait.
    std::mutex submit_mutex_;

    // Per-thread resources, tracked for cleanup at shutdown.
    std::mutex thread_map_mutex_;
    std::unordered_map<std::thread::id,
                       std::unique_ptr<ThreadResources>> thread_resources_;

    ThreadResources* createThreadResources();
};
