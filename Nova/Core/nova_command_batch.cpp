#include "nova_command_batch.h"

NovaCommandBatch::NovaCommandBatch(NovaCompute& compute)
    : compute_(compute) {}

NovaCommandBatch::~NovaCommandBatch() {
    if (recording_) {
        // Force-submit if destroyed while recording to avoid
        // leaving the command buffer in an inconsistent state
        // and to release the compute mutex.
        submit();
    }
}

void NovaCommandBatch::begin() {
    compute_.beginBatch();
    recording_ = true;
    dispatch_count_ = 0;
}

void NovaCommandBatch::dispatch(
    VkPipeline pipeline, VkPipelineLayout layout,
    VkDescriptorSet desc_set,
    uint32_t gx, uint32_t gy, uint32_t gz)
{
    VkCommandBuffer cmd = compute_.getComputeCommandBuffer();
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
    VkCommandBuffer cmd = compute_.getComputeCommandBuffer();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, push_size, push_data);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                            0, 1, &desc_set, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, gz);
    dispatch_count_++;
}

void NovaCommandBatch::barrier() {
    VkCommandBuffer cmd = compute_.getComputeCommandBuffer();

    VkMemoryBarrier mem_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };

    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &mem_barrier,
        0, nullptr,
        0, nullptr);
}

void NovaCommandBatch::copy(
    VkBuffer src, VkBuffer dst,
    VkDeviceSize src_off, VkDeviceSize dst_off,
    VkDeviceSize size)
{
    VkCommandBuffer cmd = compute_.getComputeCommandBuffer();
    VkBufferCopy region = { src_off, dst_off, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
}

void NovaCommandBatch::fill(
    VkBuffer buffer, VkDeviceSize offset,
    VkDeviceSize size, uint32_t value)
{
    VkCommandBuffer cmd = compute_.getComputeCommandBuffer();
    vkCmdFillBuffer(cmd, buffer, offset, size, value);
}

void NovaCommandBatch::submit() {
    compute_.endBatch();
    recording_ = false;
}

void NovaCommandBatch::submitAsync() {
    compute_.endBatchAsync();
    recording_ = false;
}

bool NovaCommandBatch::isComplete() const {
    return compute_.isComputeComplete();
}

void NovaCommandBatch::wait() {
    compute_.waitCompute();
}
