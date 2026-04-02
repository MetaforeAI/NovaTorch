#include "nova_compiled_graph.h"
#include "nova_context.h"
#include "nova_batch_context.h"
#include "nova_storage.h"

#include <nova_compute.h>
#include <unordered_set>

std::vector<at::Tensor> executeCompiledGraph(
    NovaCompiledGraph& plan,
    const std::vector<at::Tensor>& inputs)
{
    // 1. Flush any pending eager-mode dispatches
    NovaBatchContext::instance().flush();

    // 2. Build buffer table: [input buffers | intermediate buffers]
    const size_t total_slots = plan.num_inputs + plan.intermediates.size();
    std::vector<VkBuffer> buffer_table(total_slots);

    for (uint32_t i = 0; i < plan.num_inputs; ++i) {
        buffer_table[i] = novatorch::getNovaBuffer(inputs[i]);
    }
    for (size_t i = 0; i < plan.intermediates.size(); ++i) {
        buffer_table[plan.num_inputs + i] =
            novatorch::getNovaBuffer(plan.intermediates[i]);
    }

    // 3. Get per-thread Vulkan resources
    auto& compute = NovaContext::instance().compute();
    auto& res = compute.getThreadResources();

    if (res.fence_submitted) {
        compute.waitForFence(res.fence);
        res.fence_submitted = false;
    }

    VkDevice dev = NovaContext::instance().device();
    vkResetFences(dev, 1, &res.fence);
    vkResetCommandBuffer(res.cmd, 0);

    // 4. Reset per-plan descriptor pool
    plan.desc_pool->reset();

    // 5. Begin command buffer
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(res.cmd, &begin);

    // 6. Dependency-aware dispatch recording
    //
    // Track which buffers have pending writes (not yet barriered).
    // A step needs a barrier IFF it reads a buffer with a pending write.
    // After a barrier, clear the pending writes for the barriered buffers.
    //
    // This allows independent dispatches (different buffers) to execute
    // concurrently on the GPU's multiple compute units.

    // Set of buffer slots with pending writes (dispatched but not barriered)
    std::unordered_set<uint32_t> pending_writes;

    for (const auto& step : plan.steps) {
        // Determine which buffers this step reads vs writes.
        // Convention: last buffer_index is the output (write).
        // All others are inputs (reads).
        uint32_t write_slot = step.buffer_indices.back();

        // Check if any input buffer has a pending write → need barrier
        bool needs_barrier = false;
        for (uint32_t i = 0; i + 1 < step.num_buffers; ++i) {
            uint32_t read_slot = step.buffer_indices[i];
            if (pending_writes.count(read_slot)) {
                needs_barrier = true;
                break;
            }
        }
        // Also check write-after-write: if we're writing to a buffer
        // that has a pending write, we need a barrier too
        if (pending_writes.count(write_slot)) {
            needs_barrier = true;
        }

        if (needs_barrier) {
            // Insert barrier for all pending writes that this step depends on.
            // Use per-buffer barriers for maximum concurrency.
            std::vector<VkBufferMemoryBarrier> buf_barriers;
            for (uint32_t i = 0; i < step.num_buffers; ++i) {
                uint32_t slot = step.buffer_indices[i];
                if (pending_writes.count(slot)) {
                    VkBufferMemoryBarrier bb{};
                    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    bb.dstAccessMask = (i + 1 < step.num_buffers)
                        ? VK_ACCESS_SHADER_READ_BIT    // input: read
                        : VK_ACCESS_SHADER_WRITE_BIT;  // output: write
                    bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bb.buffer = buffer_table[slot];
                    bb.offset = 0;
                    bb.size = VK_WHOLE_SIZE;
                    buf_barriers.push_back(bb);
                    pending_writes.erase(slot);
                }
            }
            if (!buf_barriers.empty()) {
                vkCmdPipelineBarrier(res.cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,  // no global memory barriers
                    static_cast<uint32_t>(buf_barriers.size()),
                    buf_barriers.data(),
                    0, nullptr);
            }
        }

        // Allocate and update descriptor set
        VkDescriptorSet desc = plan.desc_pool->allocate(
            step.pipeline->desc_layout);

        std::vector<VkDescriptorBufferInfo> buf_infos(step.num_buffers);
        std::vector<VkWriteDescriptorSet> writes(step.num_buffers);
        for (uint32_t i = 0; i < step.num_buffers; ++i) {
            buf_infos[i] = {
                buffer_table[step.buffer_indices[i]],
                0,
                step.buffer_sizes[i]
            };
            writes[i] = {};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = desc;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buf_infos[i];
        }
        vkUpdateDescriptorSets(dev, step.num_buffers, writes.data(),
                               0, nullptr);

        // Record dispatch
        vkCmdBindPipeline(res.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          step.pipeline->pipeline);
        vkCmdBindDescriptorSets(res.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                step.pipeline->layout, 0, 1, &desc,
                                0, nullptr);
        if (step.push_constant_size > 0) {
            vkCmdPushConstants(res.cmd, step.pipeline->layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               step.push_constant_size,
                               step.push_data.data());
        }
        vkCmdDispatch(res.cmd, step.groups_x, step.groups_y, step.groups_z);

        // Mark this step's output buffer as having a pending write
        pending_writes.insert(write_slot);
    }

    // 7. End + submit + wait
    vkEndCommandBuffer(res.cmd);
    compute.submitToQueue(res.cmd, res.fence);
    compute.waitForFence(res.fence);

    // 8. Return output tensors
    std::vector<at::Tensor> outputs;
    outputs.reserve(plan.num_outputs);
    for (uint32_t idx : plan.output_intermediate_indices) {
        outputs.push_back(plan.intermediates[idx]);
    }
    return outputs;
}
