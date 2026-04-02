#include "nova_batch_context.h"

NovaBatchContext::NovaBatchContext() = default;

NovaBatchContext& NovaBatchContext::instance() {
    thread_local static NovaBatchContext ctx;
    return ctx;
}

void NovaBatchContext::recordDispatch(
    const std::string& kernel_name,
    uint32_t num_buffers,
    uint32_t push_constant_size,
    const void* push_data,
    const VkBuffer* buffers,
    const VkDeviceSize* buffer_sizes,
    uint32_t groups_x,
    uint32_t groups_y,
    uint32_t groups_z)
{
    if (!enabled_) {
        dispatchSync(kernel_name, num_buffers, push_constant_size,
                     push_data, buffers, buffer_sizes,
                     groups_x, groups_y, groups_z);
        return;
    }

    if (!recording_) {
        beginBatch();
    }

    // Allocate pipeline and descriptor set (host-side operations)
    auto& pi = getPipelineCache().get(
        kernel_name, num_buffers, push_constant_size);
    VkDescriptorSet desc = getDescriptorPool().allocate(pi.desc_layout);

    // Update descriptor set with buffer bindings
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

    // Record dispatch into the batch command buffer
    if (push_constant_size > 0) {
        batch_->dispatch(pi.pipeline, pi.layout, desc,
                         push_data, push_constant_size,
                         groups_x, groups_y, groups_z);
    } else {
        batch_->dispatch(pi.pipeline, pi.layout, desc,
                         groups_x, groups_y, groups_z);
    }

    // Insert compute-to-compute barrier for data dependencies
    batch_->barrier();

    dispatch_count_++;
}

void NovaBatchContext::flush() {
    if (!recording_ || dispatch_count_ == 0) {
        return;
    }

    // Submit all recorded commands, wait for GPU completion
    batch_->submit();

    // Reset descriptor pool now that GPU is done
    getDescriptorPool().reset();

    // Reset state for next batch
    recording_ = false;
    dispatch_count_ = 0;
    batch_.reset();
}

bool NovaBatchContext::hasPending() const {
    return recording_ && dispatch_count_ > 0;
}

void NovaBatchContext::setEnabled(bool enabled) {
    // Flush any pending work before changing mode
    if (!enabled && recording_) {
        flush();
    }
    enabled_ = enabled;
}

bool NovaBatchContext::isEnabled() const {
    return enabled_;
}

void NovaBatchContext::beginBatch() {
    batch_ = std::make_unique<NovaCommandBatch>(
        NovaContext::instance().compute());
    batch_->begin();
    recording_ = true;
    dispatch_count_ = 0;
}

void NovaBatchContext::dispatchSync(
    const std::string& kernel_name,
    uint32_t num_buffers,
    uint32_t push_constant_size,
    const void* push_data,
    const VkBuffer* buffers,
    const VkDeviceSize* buffer_sizes,
    uint32_t groups_x,
    uint32_t groups_y,
    uint32_t groups_z)
{
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
        vkCmdBindPipeline(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pi.pipeline);
        vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pi.layout, 0, 1, &desc, 0, nullptr);
        if (push_constant_size > 0) {
            vkCmdPushConstants(
                cmd, pi.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                0, push_constant_size, push_data);
        }
        vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    });
}
