#include "nova_batch_context.h"
#include "nova_compiled_graph.h"
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

    // Clear pending write tracking for next batch
    pending_writes_.clear();

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

    // In capture mode: record into external command buffer
    // In normal mode: record into batch context's command buffer
    if (!capturing_ && !recording_) beginBatch();

    // Pipeline lookup (thread-safe: pipelines are immutable after creation)
    auto& pi = getPipelineCache().get(
        kernel_name, num_buffers, push_constant_size);

    // Allocate from active descriptor pool.
    // During capture: use the plan's dedicated UAB pool.
    // Normal mode: use per-thread pool.
    // Both use the same UAB layout (desc_layout == desc_layout_uab now).
    NovaDescriptorPool* pool = (capturing_ && active_desc_pool_)
        ? active_desc_pool_ : desc_pool_.get();
    VkDescriptorSet desc = pool->allocate(pi.desc_layout);

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

    VkCommandBuffer cmd = capturing_ ? capture_cmd_ : resources_->cmd;

    // --- Dependency-aware barriers ---
    // Convention: last buffer is the output (write), others are inputs (reads).
    // Only insert barriers for buffers this dispatch actually depends on.
    // Independent dispatches (different buffers) execute concurrently on GPU.
    VkBuffer write_buf = buffers[num_buffers - 1];

    // Check if any input buffer has a pending write → need barrier
    std::vector<VkBufferMemoryBarrier> buf_barriers;
    for (uint32_t i = 0; i + 1 < num_buffers; ++i) {
        if (pending_writes_.count(buffers[i])) {
            VkBufferMemoryBarrier bb{};
            bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bb.buffer = buffers[i];
            bb.offset = 0;
            bb.size = VK_WHOLE_SIZE;
            buf_barriers.push_back(bb);
            pending_writes_.erase(buffers[i]);
        }
    }
    // Write-after-write: if output buffer has pending write
    if (pending_writes_.count(write_buf)) {
        VkBufferMemoryBarrier bb{};
        bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = write_buf;
        bb.offset = 0;
        bb.size = VK_WHOLE_SIZE;
        buf_barriers.push_back(bb);
        pending_writes_.erase(write_buf);
    }

    if (!buf_barriers.empty()) {
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            static_cast<uint32_t>(buf_barriers.size()),
            buf_barriers.data(),
            0, nullptr);
    }

    // Record dispatch
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pi.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pi.layout, 0, 1, &desc, 0, nullptr);
    if (push_constant_size > 0)
        vkCmdPushConstants(cmd, pi.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_constant_size, push_data);
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);

    // Mark output buffer as having a pending write
    pending_writes_.insert(write_buf);

    // Track descriptor→buffer mapping during capture for replay rebinding
    if (capturing_) {
        DescriptorMapping mapping;
        mapping.set = desc;
        mapping.pipeline = &pi;
        mapping.num_buffers = num_buffers;
        mapping.bound_buffers.resize(num_buffers);
        mapping.bound_sizes.resize(num_buffers);
        for (uint32_t i = 0; i < num_buffers; ++i) {
            mapping.bound_buffers[i] = buffers[i];
            mapping.bound_sizes[i] = buffer_sizes[i];
        }
        capture_mappings_.push_back(std::move(mapping));
    }

    // Retain tensors to prevent use-after-free during batch lifetime
    for (const auto& t : retain) {
        retained_.push_back(t);
    }

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

// -------------------------------------------------------------------------
// Capture mode: record into an external command buffer
// -------------------------------------------------------------------------

void NovaBatchContext::beginCapture(VkCommandBuffer external_cmd) {
    // Flush any pending normal batch first
    if (recording_) flush();

    // Ensure we have a descriptor pool
    if (!desc_pool_) {
        desc_pool_ = std::make_unique<NovaDescriptorPool>();
        desc_pool_->init(NovaContext::instance().device());
    }

    capture_cmd_ = external_cmd;
    capturing_ = true;
    dispatch_count_ = 0;
    pending_writes_.clear();
}

std::unique_ptr<NovaDescriptorPool> NovaBatchContext::endCapture() {
    capturing_ = false;
    capture_cmd_ = VK_NULL_HANDLE;
    dispatch_count_ = 0;
    pending_writes_.clear();

    return std::move(desc_pool_);
}

std::vector<DescriptorMapping> NovaBatchContext::endCaptureWithMap(
    std::vector<at::Tensor>* out_retained) {
    capturing_ = false;
    capture_cmd_ = VK_NULL_HANDLE;
    dispatch_count_ = 0;
    pending_writes_.clear();

    // Transfer retained tensors to caller — keeps VkBuffers alive
    // for the lifetime of the compiled plan's command buffer.
    if (out_retained) {
        *out_retained = std::move(retained_);
    }
    retained_.clear();

    return std::move(capture_mappings_);
}

NovaDescriptorPool* NovaBatchContext::swapDescPool(NovaDescriptorPool* new_pool) {
    NovaDescriptorPool* old = active_desc_pool_;
    active_desc_pool_ = new_pool;
    return old;
}
