#include "nova_ops.h"
#include "nova_batch_context.h"

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
    NovaBatchContext::instance().recordDispatch(
        kernel_name, num_buffers, push_constant_size, push_data,
        buffers, buffer_sizes, groups_x, groups_y, groups_z);
}
