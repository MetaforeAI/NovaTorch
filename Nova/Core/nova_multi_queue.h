#pragma once
#include "./nova_compute.h"
#include <vulkan/vulkan.h>
#include <functional>
#include <cstdint>

namespace Nova {

/**
 * NovaMultiQueue - Exposes multiple compute queues for concurrent execution.
 *
 * The RX 6800 XT provides 4 compute queues in Queue Family 1.
 * This class manages independent command pools, command buffers, fences,
 * and semaphores for each queue, enabling concurrent hemisphere execution
 * without mutex contention.
 *
 * Usage:
 *   NovaMultiQueue mq;
 *   mq.init(compute);
 *
 *   // Concurrent hemisphere execution
 *   mq.submit(0, [&](VkCommandBuffer cmd) { recordLeftHemisphere(cmd); });
 *   mq.submit(1, [&](VkCommandBuffer cmd) { recordRightHemisphere(cmd); });
 *   mq.waitAll();
 *
 *   // With inter-queue synchronization
 *   mq.submitAndSignal(0, mq.semaphore(0));
 *   mq.submitAndWait(1, mq.semaphore(0));
 */
class NovaMultiQueue {
public:
    static constexpr uint32_t MAX_QUEUES = 4;
    static constexpr uint32_t MAX_SEMAPHORES = 8;

    void init(NovaCompute& compute);
    void shutdown();

    // Submit work to a specific queue
    void submit(uint32_t queue_id, std::function<void(VkCommandBuffer)>&& func);

    // Submit and signal a semaphore (for inter-queue sync)
    void submitAndSignal(uint32_t queue_id, VkSemaphore signal);

    // Submit and wait on a semaphore before executing
    void submitAndWait(uint32_t queue_id, VkSemaphore wait,
                       VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Submit with both wait and signal semaphores
    void submitFull(uint32_t queue_id,
                    VkSemaphore wait, VkPipelineStageFlags wait_stage,
                    VkSemaphore signal);

    // Wait for all queues or a specific queue
    void waitAll();
    void waitQueue(uint32_t queue_id);

    // Access
    VkQueue queue(uint32_t id) const;
    VkCommandBuffer cmd(uint32_t id) const;
    VkFence fence(uint32_t id) const;
    VkSemaphore semaphore(uint32_t id) const;
    uint32_t queue_count() const { return queue_count_; }

    // Begin/end recording for a specific queue (for batch-style usage)
    void beginRecord(uint32_t queue_id);
    void endRecord(uint32_t queue_id);

private:
    NovaCompute* compute_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;

    uint32_t queue_count_ = 0;
    uint32_t queue_family_ = 0;

    VkQueue queues_[MAX_QUEUES] = {};
    VkCommandPool pools_[MAX_QUEUES] = {};
    VkCommandBuffer cmds_[MAX_QUEUES] = {};
    VkFence fences_[MAX_QUEUES] = {};
    VkSemaphore semaphores_[MAX_SEMAPHORES] = {};
    uint32_t semaphore_count_ = 0;

    void resetAndBegin(uint32_t queue_id);
    void endAndSubmit(uint32_t queue_id, const VkSubmitInfo& submit_info);
};

} // namespace Nova
