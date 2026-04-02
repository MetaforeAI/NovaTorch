#pragma once

#include <ATen/ATen.h>
#include <vulkan/vulkan.h>
#include "nova_allocator.h"
#include "nova_staging_pool.h"

#include <functional>

namespace novatorch {

/// Extract the Allocation metadata from a Nova-backed tensor.
NovaAllocator::Allocation* getNovaAllocation(const at::Tensor& tensor);

/// Get the device-local VkBuffer for GPU shader binding.
VkBuffer getNovaBuffer(const at::Tensor& tensor);

// -------------------------------------------------------------------------
// Staging transfer helpers
//
// These manage the staging pool lifecycle and vkCmdCopyBuffer transfers.
// They flush any pending command batch before executing.
// -------------------------------------------------------------------------

/// Upload: copy @p nbytes from CPU @p src into the tensor's device buffer.
void uploadToDevice(const at::Tensor& tensor, const void* src, size_t nbytes);

/// Download: copy @p nbytes from the tensor's device buffer into CPU @p dst.
void downloadFromDevice(const at::Tensor& tensor, void* dst, size_t nbytes);

/// Acquire staging, download device→staging, call @p fn with (const ptr, size),
/// release staging.  For CPU-path ops that need to READ device data.
void withStagingRead(const at::Tensor& tensor,
                     std::function<void(const void*, size_t)> fn);

/// Acquire staging, call @p fn with (writable ptr, size) to fill staging,
/// upload staging→device, release staging.  For CPU-path ops that WRITE.
void withStagingWrite(const at::Tensor& tensor,
                      std::function<void(void*, size_t)> fn);

/// Acquire staging, download device→staging, call @p fn with (ptr, size)
/// for read+write, upload staging→device, release staging.
void withStagingReadWrite(const at::Tensor& tensor,
                          std::function<void(void*, size_t)> fn);

/// Device-to-device copy (no staging).
void copyDeviceToDevice(const at::Tensor& src, const at::Tensor& dst,
                        size_t nbytes);

} // namespace novatorch
