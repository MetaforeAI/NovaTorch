#pragma once

/// Common header pulled in by every nova_ops_*.cpp translation unit.

#include <ATen/ATen.h>
#include <vulkan/vulkan.h>

#include "nova_context.h"
#include "nova_allocator.h"
#include "nova_storage.h"
#include "nova_pipeline_cache.h"
#include "nova_descriptor_pool.h"

#include <string>
#include <vector>
#include <cstdint>

// Singleton accessors — defined in nova_register.cpp.
NovaPipelineCache& getPipelineCache();
NovaDescriptorPool& getDescriptorPool();

// ---------------------------------------------------------------------------
// dispatchCompute — common shader dispatch helper
// Declared here, defined in nova_ops_dispatch.cpp
// ---------------------------------------------------------------------------

/// Dispatch a compute shader.
/// @p retain — tensors whose device buffers are bound in this dispatch.
///             The batch context holds references to them until the batch
///             is flushed and the GPU completes, preventing use-after-free
///             if Python GC destroys a tensor while the batch is pending.
void dispatchCompute(
    const std::string& kernel_name,
    uint32_t num_buffers,
    uint32_t push_constant_size,
    const void* push_data,
    const VkBuffer* buffers,
    const VkDeviceSize* buffer_sizes,
    uint32_t groups_x,
    uint32_t groups_y = 1,
    uint32_t groups_z = 1,
    std::initializer_list<at::Tensor> retain = {});
