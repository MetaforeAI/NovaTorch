#pragma once

#include "nova_ops.h"
#include "nova_descriptor_pool.h"
#include "nova_context.h"
#include <nova_compute.h>

#include <memory>
#include <string>
#include <cstdint>
#include <unordered_set>

/// Thread-local batch context for automatic GPU dispatch coalescing.
///
/// Each thread (main thread, autograd thread, etc.) gets its own
/// independent batch context with its own VkCommandBuffer, VkFence,
/// and VkDescriptorPool. Recording is lock-free — only vkQueueSubmit
/// is serialized across threads.
///
/// Usage: dispatchCompute() → recordDispatch() → auto-batches.
///        Sync points call flush() to submit and wait.
class NovaBatchContext {
public:
    static NovaBatchContext& instance();

    void recordDispatch(
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

    void flush();
    bool hasPending() const;
    void setEnabled(bool enabled);
    bool isEnabled() const;

private:
    NovaBatchContext();

    void beginBatch();

    // Per-thread Vulkan resources (from NovaCompute)
    NovaCompute::ThreadResources* resources_ = nullptr;

    // Per-thread descriptor pool
    std::unique_ptr<NovaDescriptorPool> desc_pool_;

    bool recording_     = false;
    bool enabled_       = true;
    uint32_t dispatch_count_ = 0;

    // Tensors retained by the current batch. Prevents use-after-free:
    // if Python GC destroys a tensor while the batch is pending, the
    // at::Tensor refcount keeps the underlying VkBuffer alive until
    // flush() completes and the GPU is done with the command buffer.
    std::vector<at::Tensor> retained_;

    // Dependency-aware barriers: track which VkBuffers have pending writes
    // (dispatched but not yet barriered). A dispatch only gets a barrier
    // if it reads or writes a buffer with a pending write. Independent
    // dispatches (different buffers) execute concurrently on the GPU.
    std::unordered_set<VkBuffer> pending_writes_;
};
