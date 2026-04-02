#include "nova_ops.h"

void dispatchCompute(
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
