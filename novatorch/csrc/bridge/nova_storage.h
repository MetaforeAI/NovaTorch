#pragma once

#include <ATen/ATen.h>
#include <vulkan/vulkan.h>
#include "nova_allocator.h"

namespace novatorch {

/// Extract the Allocation metadata from a Nova-backed tensor.
NovaAllocator::Allocation* getNovaAllocation(const at::Tensor& tensor);

/// Get the VkBuffer backing a Nova tensor.
VkBuffer getNovaBuffer(const at::Tensor& tensor);

/// Flush host writes so they are visible to the device (non-coherent memory).
void flushNovaBuffer(const at::Tensor& tensor);

/// Invalidate caches before host reads (non-coherent memory).
void invalidateNovaBuffer(const at::Tensor& tensor);

} // namespace novatorch
