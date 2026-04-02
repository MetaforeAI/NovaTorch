#include "nova_compiled_graph.h"
#include "nova_context.h"
#include "nova_batch_context.h"
#include "nova_storage.h"

#include <nova_compute.h>

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

    // 4. Reset per-plan descriptor pool (safe — fence confirmed GPU done)
    plan.desc_pool->reset();

    // 5. Begin command buffer
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(res.cmd, &begin);

    // 6. Record all dispatches
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    for (const auto& step : plan.steps) {
        // Allocate descriptor set from per-plan pool
        VkDescriptorSet desc = plan.desc_pool->allocate(
            step.pipeline->desc_layout);

        // Update descriptors with current buffer handles
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

        // Barrier between dispatches
        vkCmdPipelineBarrier(res.cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
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
