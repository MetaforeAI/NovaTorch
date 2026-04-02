#pragma once

#include "nova_ops.h"
#include <nova_command_batch.h>
#include <memory>
#include <string>
#include <cstdint>

/// Thread-local batch context for automatic command batching.
///
/// Instead of submit+wait per GPU dispatch (~20us overhead each),
/// records multiple dispatches into a single VkCommandBuffer and
/// submits once at sync points (.item(), .cpu(), explicit flush).
///
/// For N sequential dispatches, this eliminates ~N*20us of overhead.
class NovaBatchContext {
public:
    /// Thread-local singleton accessor.
    static NovaBatchContext& instance();

    /// Record a compute dispatch into the current batch.
    /// Auto-starts a new batch if none is active.
    /// If batching is disabled, falls through to synchronous dispatch.
    void recordDispatch(
        const std::string& kernel_name,
        uint32_t num_buffers,
        uint32_t push_constant_size,
        const void* push_data,
        const VkBuffer* buffers,
        const VkDeviceSize* buffer_sizes,
        uint32_t groups_x,
        uint32_t groups_y = 1,
        uint32_t groups_z = 1);

    /// Submit the current batch and wait for GPU completion.
    /// Called at sync points: .item(), .cpu(), backward ops that
    /// read mapped memory, explicit user flush.
    /// No-op if no dispatches are pending.
    void flush();

    /// Check if there are pending (unsubmitted) dispatches.
    bool hasPending() const;

    /// Enable or disable automatic batching (default: enabled).
    void setEnabled(bool enabled);

    /// Query whether batching is currently enabled.
    bool isEnabled() const;

private:
    NovaBatchContext();

    void beginBatch();
    void dispatchSync(
        const std::string& kernel_name,
        uint32_t num_buffers,
        uint32_t push_constant_size,
        const void* push_data,
        const VkBuffer* buffers,
        const VkDeviceSize* buffer_sizes,
        uint32_t groups_x,
        uint32_t groups_y,
        uint32_t groups_z);

    std::unique_ptr<NovaCommandBatch> batch_;
    bool recording_ = false;
    bool enabled_ = true;
    uint32_t dispatch_count_ = 0;
};
