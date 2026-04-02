#include "nova_batch_context.h"
#include "nova_pipeline_cache.h"

NovaBatchContext::NovaBatchContext() = default;

NovaBatchContext& NovaBatchContext::instance() {
    thread_local static NovaBatchContext ctx;
    return ctx;
}

// -------------------------------------------------------------------------
// Batch lifecycle
// -------------------------------------------------------------------------

void NovaBatchContext::beginBatch() {
    auto& compute = NovaContext::instance().compute();
    resources_ = &compute.getThreadResources();

    // Lazy-init per-thread descriptor pool
    if (!desc_pool_) {
        desc_pool_ = std::make_unique<NovaDescriptorPool>();
        desc_pool_->init(NovaContext::instance().device());
    }

    // Wait for any previous submission on this thread to complete
    // before reusing the command buffer and descriptor pool.
    if (resources_->fence_submitted) {
        compute.waitForFence(resources_->fence);
        resources_->fence_submitted = false;

        // Reset descriptor pool now that GPU is done with previous batch
        desc_pool_->reset();
    }

    // Reset fence and begin recording into this thread's command buffer
    VkDevice dev = NovaContext::instance().device();
    vkResetFences(dev, 1, &resources_->fence);
    vkResetCommandBuffer(resources_->cmd, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(resources_->cmd, &begin);

    recording_ = true;
    dispatch_count_ = 0;
}

void NovaBatchContext::flush() {
    if (!recording_ || dispatch_count_ == 0) return;

    // End recording
    vkEndCommandBuffer(resources_->cmd);

    // Submit — only acquires submit_mutex_ for the vkQueueSubmit call
    auto& compute = NovaContext::instance().compute();
    compute.submitToQueue(resources_->cmd, resources_->fence);
    resources_->fence_submitted = true;

    // Wait for GPU completion on this thread (no mutex held)
    compute.waitForFence(resources_->fence);
    resources_->fence_submitted = false;

    // Reset descriptor pool for next batch
    desc_pool_->reset();

    // Release retained tensors — GPU is done, buffers are safe to free
    retained_.clear();

    recording_ = false;
    dispatch_count_ = 0;
}

// -------------------------------------------------------------------------
// Dispatch recording
// -------------------------------------------------------------------------

void NovaBatchContext::recordDispatch(
    const std::string& kernel_name,
    uint32_t num_buffers,
    uint32_t push_constant_size,
    const void* push_data,
    const VkBuffer* buffers,
    const VkDeviceSize* buffer_sizes,
    uint32_t groups_x,
    uint32_t groups_y,
    uint32_t groups_z,
    std::initializer_list<at::Tensor> retain)
{
    if (!enabled_) {
        // Fallback: synchronous dispatch via executeCompute
        auto& pi = getPipelineCache().get(
            kernel_name, num_buffers, push_constant_size);
        VkDescriptorSet desc = getDescriptorPool().allocate(pi.desc_layout);

        std::vector<VkDescriptorBufferInfo> buf_infos(num_buffers);
        std::vector<VkWriteDescriptorSet> writes(num_buffers);
        for (uint32_t i = 0; i < num_buffers; ++i) {
            buf_infos[i] = {buffers[i], 0, buffer_sizes[i]};
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(
            NovaContext::instance().device(),
            num_buffers, writes.data(), 0, nullptr);

        NovaContext::instance().executeSync([&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pi.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pi.layout, 0, 1, &desc, 0, nullptr);
            if (push_constant_size > 0)
                vkCmdPushConstants(cmd, pi.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, push_constant_size, push_data);
            vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
        });
        return;
    }

    if (!recording_) beginBatch();

    // Pipeline lookup (thread-safe: pipelines are immutable after creation)
    auto& pi = getPipelineCache().get(
        kernel_name, num_buffers, push_constant_size);

    // Allocate from PER-THREAD descriptor pool (no mutex needed)
    VkDescriptorSet desc = desc_pool_->allocate(pi.desc_layout);

    // Update descriptors (host-side operation, no mutex needed)
    std::vector<VkDescriptorBufferInfo> buf_infos(num_buffers);
    std::vector<VkWriteDescriptorSet> writes(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        buf_infos[i] = {buffers[i], 0, buffer_sizes[i]};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = desc;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(
        NovaContext::instance().device(),
        num_buffers, writes.data(), 0, nullptr);

    // Record into PER-THREAD command buffer (no mutex needed)
    VkCommandBuffer cmd = resources_->cmd;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pi.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pi.layout, 0, 1, &desc, 0, nullptr);
    if (push_constant_size > 0)
        vkCmdPushConstants(cmd, pi.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_constant_size, push_data);
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);

    // Retain tensors to prevent use-after-free during batch lifetime
    for (const auto& t : retain) {
        retained_.push_back(t);
    }

    // Compute-to-compute barrier between dispatches
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    dispatch_count_++;
}

// -------------------------------------------------------------------------
// State queries
// -------------------------------------------------------------------------

bool NovaBatchContext::hasPending() const {
    return recording_ && dispatch_count_ > 0;
}

void NovaBatchContext::setEnabled(bool enabled) {
    if (!enabled && recording_) flush();
    enabled_ = enabled;
}

bool NovaBatchContext::isEnabled() const {
    return enabled_;
}
