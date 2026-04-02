#pragma once
#include "./core_base.h"
#include <mutex>

/**
 * NovaCompute - Compute-only mode
 *
 * Lightweight Vulkan context for GPU compute operations:
 * - No surface/swapchain
 * - No graphics pipeline
 * - Minimal overhead
 *
 * Use for:
 * - FFT computations (Logos encoding)
 * - Triplanar projections
 * - General GPU compute tasks
 */
class NovaCompute : public NovaCore {
private:
    // Serializes all command buffer access — held from submit through wait.
    // submitCompute() acquires, waitCompute() releases.
    // executeCompute() holds for full duration.
    std::mutex compute_mutex_;
    std::unique_lock<std::mutex> submit_lock_;  // Held between submit and wait

    // Compute-specific resources
    VkCommandBuffer compute_cmd = VK_NULL_HANDLE;
    VkFence compute_fence = VK_NULL_HANDLE;

public:
    /**
     * Constructor - Initialize compute-only mode
     * @param debug_level Logging level (INFO, DEBUG, VERBOSE, etc.)
     */
    NovaCompute(const std::string& debug_level);

    /**
     * Destructor - Cleanup compute resources
     */
    ~NovaCompute() override;

    /**
     * Execute compute commands and wait for completion
     * @param func Lambda receiving VkCommandBuffer for recording commands
     */
    void executeCompute(std::function<void(VkCommandBuffer)>&& func);

    /**
     * Submit compute commands without waiting (async)
     * @param func Lambda receiving VkCommandBuffer for recording commands
     */
    void submitCompute(std::function<void(VkCommandBuffer)>&& func);

    /**
     * Non-blocking check if submitted compute work is complete
     */
    bool isComputeComplete() const;

    /**
     * Block until submitted compute work completes
     */
    void waitCompute();

    /**
     * Wait for all device operations to complete
     */
    void waitIdle();

    /**
     * Get the compute command buffer (for manual recording)
     */
    VkCommandBuffer getComputeCommandBuffer() const { return compute_cmd; }

    /**
     * Begin batch recording: acquires compute mutex, resets fence and
     * command buffer, begins recording. The mutex is held until
     * endBatch() or endBatchAsync()+waitCompute() releases it.
     */
    void beginBatch();

    /**
     * End batch: ends command buffer, submits to compute queue,
     * waits for completion, releases the compute mutex.
     */
    void endBatch();

    /**
     * End batch async: ends command buffer and submits without waiting.
     * The compute mutex remains held until waitCompute() is called.
     */
    void endBatchAsync();
};
