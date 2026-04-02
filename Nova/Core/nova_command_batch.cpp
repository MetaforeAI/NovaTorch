#include "nova_command_batch.h"

NovaCommandBatch::NovaCommandBatch(NovaCompute& compute)
    : compute_(compute) {}

NovaCommandBatch::~NovaCommandBatch() {
    if (recording_) {
        submit();
    }
}

void NovaCommandBatch::begin() {
    auto& tr = compute_.getThreadResources();

    // Wait for any previous submission before reuse
    if (tr.fence_submitted) {
        compute_.waitForFence(tr.fence);
        tr.fence_submitted = false;
    }

    VkDevice dev = compute_.getDevice();
    VK_TRY(vkResetFences(dev, 1, &tr.fence));
    VK_TRY(vkResetCommandBuffer(tr.cmd, 0));

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_TRY(vkBeginCommandBuffer(tr.cmd, &begin_info));

    recording_ = true;
    dispatch_count_ = 0;
}

void NovaCommandBatch::dispatch(
    VkPipeline pipeline, VkPipelineLayout layout,
    VkDescriptorSet desc_set,
    uint32_t gx, uint32_t gy, uint32_t gz)
{
    VkCommandBuffer cmd = compute_.getThreadResources().cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                            0, 1, &desc_set, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, gz);
    dispatch_count_++;
}

void NovaCommandBatch::dispatch(
    VkPipeline pipeline, VkPipelineLayout layout,
    VkDescriptorSet desc_set,
    const void* push_data, uint32_t push_size,
    uint32_t gx, uint32_t gy, uint32_t gz)
{
    VkCommandBuffer cmd = compute_.getThreadResources().cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, push_size, push_data);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                            0, 1, &desc_set, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, gz);
    dispatch_count_++;
}

void NovaCommandBatch::barrier() {
    VkCommandBuffer cmd = compute_.getThreadResources().cmd;
    VkMemoryBarrier mem_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mem_barrier, 0, nullptr, 0, nullptr);
}

void NovaCommandBatch::copy(
    VkBuffer src, VkBuffer dst,
    VkDeviceSize src_off, VkDeviceSize dst_off,
    VkDeviceSize size)
{
    VkCommandBuffer cmd = compute_.getThreadResources().cmd;
    VkBufferCopy region = { src_off, dst_off, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

void NovaCommandBatch::fill(
    VkBuffer buffer, VkDeviceSize offset,
    VkDeviceSize size, uint32_t value)
{
    VkCommandBuffer cmd = compute_.getThreadResources().cmd;
    vkCmdFillBuffer(cmd, buffer, offset, size, value);
}

void NovaCommandBatch::submit() {
    auto& tr = compute_.getThreadResources();
    VK_TRY(vkEndCommandBuffer(tr.cmd));
    compute_.submitToQueue(tr.cmd, tr.fence);
    compute_.waitForFence(tr.fence);
    tr.fence_submitted = false;
    recording_ = false;
}

void NovaCommandBatch::submitAsync() {
    auto& tr = compute_.getThreadResources();
    VK_TRY(vkEndCommandBuffer(tr.cmd));
    compute_.submitToQueue(tr.cmd, tr.fence);
    tr.fence_submitted = true;
    recording_ = false;
}

bool NovaCommandBatch::isComplete() const {
    auto& tr = compute_.getThreadResources();
    return vkGetFenceStatus(compute_.getDevice(), tr.fence) == VK_SUCCESS;
}

void NovaCommandBatch::wait() {
    auto& tr = compute_.getThreadResources();
    compute_.waitForFence(tr.fence);
    tr.fence_submitted = false;
}
