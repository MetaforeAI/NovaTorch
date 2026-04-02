#pragma once

#include "nova_ops.h"
#include "nova_descriptor_pool.h"
#include "nova_pipeline_cache.h"
#include "nova_context.h"
#include <nova_compute.h>

#include <memory>
#include <string>
#include <cstdint>
#include <unordered_set>

// Forward declaration — full definition in nova_compiled_graph.h
struct DescriptorMapping;

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

    /// Begin recording into an EXTERNAL command buffer (for capture mode).
    /// The batch context records dispatches into this cmd buffer instead of
    /// its own. Call endCapture() when done to stop redirecting.
    void beginCapture(VkCommandBuffer external_cmd);

    /// Stop capturing. Returns the descriptor pool with all allocated sets.
    std::unique_ptr<NovaDescriptorPool> endCapture();

    /// Stop capturing and return descriptor→buffer mappings + retained tensors.
    /// The retained tensors keep all VkBuffers alive that the command buffer references.
    std::vector<DescriptorMapping> endCaptureWithMap(
        std::vector<at::Tensor>* out_retained = nullptr);

    /// Swap the descriptor pool used for allocation. Returns the old pool.
    /// Used during capture to switch to a UAB pool.
    NovaDescriptorPool* swapDescPool(NovaDescriptorPool* new_pool);

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

    // Dependency-aware barriers
    std::unordered_set<VkBuffer> pending_writes_;

    // Capture mode: record into an external command buffer
    VkCommandBuffer capture_cmd_ = VK_NULL_HANDLE;
    bool capturing_ = false;

    // Descriptor tracking during capture
    std::vector<DescriptorMapping> capture_mappings_;

    // Swappable descriptor pool pointer (for UAB during capture)
    NovaDescriptorPool* active_desc_pool_ = nullptr;
};
