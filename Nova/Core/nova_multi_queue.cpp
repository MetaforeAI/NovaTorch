#include "./nova_multi_queue.h"
#include "./components/logger.h"
#include <algorithm>

namespace Nova {

// ---------------------------------------------------------------------------
// init / shutdown
// ---------------------------------------------------------------------------

void NovaMultiQueue::init(NovaCompute& compute)
{
    compute_ = &compute;
    device_  = compute.getDevice();
    queue_family_ = compute.getComputeQueueFamily();

    // Clamp to hardware queue count and our compile-time limit
    uint32_t hw_count = compute.getComputeQueueCount();
    queue_count_ = std::min(hw_count, MAX_QUEUES);
    report(LOGGER::INFO,
           "NovaMultiQueue - init: family=%u  hw_queues=%u  using=%u",
           queue_family_, hw_count, queue_count_);

    // Retrieve queue handles
    for (uint32_t i = 0; i < queue_count_; ++i) {
        vkGetDeviceQueue(device_, queue_family_, i, &queues_[i]);
    }

    // One command pool per queue (separate pools = thread-safe recording)
    for (uint32_t i = 0; i < queue_count_; ++i) {
        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_
        };
        VK_TRY(vkCreateCommandPool(device_, &pool_info, nullptr, &pools_[i]));
    }

    // Allocate one command buffer per pool
    for (uint32_t i = 0; i < queue_count_; ++i) {
        VkCommandBufferAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pools_[i],
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        VK_TRY(vkAllocateCommandBuffers(device_, &alloc_info, &cmds_[i]));
    }

    // Create fences (start signaled so first wait succeeds)
    for (uint32_t i = 0; i < queue_count_; ++i) {
        VkFenceCreateInfo fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
        VK_TRY(vkCreateFence(device_, &fence_info, nullptr, &fences_[i]));
    }

    // Create semaphores for inter-queue synchronization
    semaphore_count_ = MAX_SEMAPHORES;
    for (uint32_t i = 0; i < semaphore_count_; ++i) {
        VkSemaphoreCreateInfo sem_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        };
        VK_TRY(vkCreateSemaphore(device_, &sem_info, nullptr, &semaphores_[i]));
    }

    report(LOGGER::INFO, "NovaMultiQueue - init complete: %u queues, %u semaphores",
           queue_count_, semaphore_count_);
}

void NovaMultiQueue::shutdown()
{
    if (device_ == VK_NULL_HANDLE) return;

    vkDeviceWaitIdle(device_);

    for (uint32_t i = semaphore_count_; i > 0; --i) {
        vkDestroySemaphore(device_, semaphores_[i - 1], nullptr);
        semaphores_[i - 1] = VK_NULL_HANDLE;
    }
    semaphore_count_ = 0;

    for (uint32_t i = queue_count_; i > 0; --i) {
        vkDestroyFence(device_, fences_[i - 1], nullptr);
        fences_[i - 1] = VK_NULL_HANDLE;
    }

    // Destroying a pool frees its command buffers
    for (uint32_t i = queue_count_; i > 0; --i) {
        vkDestroyCommandPool(device_, pools_[i - 1], nullptr);
        pools_[i - 1] = VK_NULL_HANDLE;
        cmds_[i - 1] = VK_NULL_HANDLE;
    }

    queue_count_ = 0;
    device_ = VK_NULL_HANDLE;
    compute_ = nullptr;

    report(LOGGER::INFO, "NovaMultiQueue - shutdown complete");
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void NovaMultiQueue::resetAndBegin(uint32_t queue_id)
{
    VK_TRY(vkWaitForFences(device_, 1, &fences_[queue_id], VK_TRUE, UINT64_MAX));
    VK_TRY(vkResetFences(device_, 1, &fences_[queue_id]));
    VK_TRY(vkResetCommandBuffer(cmds_[queue_id], 0));

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_TRY(vkBeginCommandBuffer(cmds_[queue_id], &begin_info));
}

void NovaMultiQueue::endAndSubmit(uint32_t queue_id, const VkSubmitInfo& submit_info)
{
    VK_TRY(vkEndCommandBuffer(cmds_[queue_id]));
    VK_TRY(vkQueueSubmit(queues_[queue_id], 1, &submit_info, fences_[queue_id]));
}

// ---------------------------------------------------------------------------
// Submit variants
// ---------------------------------------------------------------------------

void NovaMultiQueue::submit(uint32_t queue_id, std::function<void(VkCommandBuffer)>&& func)
{
    resetAndBegin(queue_id);
    func(cmds_[queue_id]);

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmds_[queue_id]
    };
    endAndSubmit(queue_id, submit_info);
}

void NovaMultiQueue::submitAndSignal(uint32_t queue_id, VkSemaphore signal)
{
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmds_[queue_id],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal
    };
    endAndSubmit(queue_id, submit_info);
}

void NovaMultiQueue::submitAndWait(uint32_t queue_id, VkSemaphore wait,
                                   VkPipelineStageFlags wait_stage)
{
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmds_[queue_id]
    };
    endAndSubmit(queue_id, submit_info);
}

void NovaMultiQueue::submitFull(uint32_t queue_id,
                                VkSemaphore wait, VkPipelineStageFlags wait_stage,
                                VkSemaphore signal)
{
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmds_[queue_id],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal
    };
    endAndSubmit(queue_id, submit_info);
}

// ---------------------------------------------------------------------------
// Wait
// ---------------------------------------------------------------------------

void NovaMultiQueue::waitAll()
{
    if (queue_count_ == 0) return;
    VK_TRY(vkWaitForFences(device_, queue_count_, fences_, VK_TRUE, UINT64_MAX));
}

void NovaMultiQueue::waitQueue(uint32_t queue_id)
{
    VK_TRY(vkWaitForFences(device_, 1, &fences_[queue_id], VK_TRUE, UINT64_MAX));
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

VkQueue NovaMultiQueue::queue(uint32_t id) const { return queues_[id]; }
VkCommandBuffer NovaMultiQueue::cmd(uint32_t id) const { return cmds_[id]; }
VkFence NovaMultiQueue::fence(uint32_t id) const { return fences_[id]; }
VkSemaphore NovaMultiQueue::semaphore(uint32_t id) const { return semaphores_[id]; }

// ---------------------------------------------------------------------------
// Manual recording (batch-style)
// ---------------------------------------------------------------------------

void NovaMultiQueue::beginRecord(uint32_t queue_id)
{
    VK_TRY(vkWaitForFences(device_, 1, &fences_[queue_id], VK_TRUE, UINT64_MAX));
    VK_TRY(vkResetFences(device_, 1, &fences_[queue_id]));
    VK_TRY(vkResetCommandBuffer(cmds_[queue_id], 0));

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_TRY(vkBeginCommandBuffer(cmds_[queue_id], &begin_info));
}

void NovaMultiQueue::endRecord(uint32_t queue_id)
{
    VK_TRY(vkEndCommandBuffer(cmds_[queue_id]));
}

} // namespace Nova
