#pragma once

#include <vulkan/vulkan.h>
#include "vk_memory.h"

#include <cstddef>
#include <mutex>
#include <vector>

/// Shared pool of reusable HOST_VISIBLE staging buffers for CPU↔GPU transfers.
///
/// Staging buffers are acquired on demand and returned after the transfer
/// completes.  The pool keeps freed buffers in size-bucketed free lists so
/// subsequent acquires avoid VMA allocation overhead.
class NovaStagingPool {
public:
    struct StagingBuffer {
        VkBuffer buffer   = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        void* ptr         = nullptr;   // persistently mapped host pointer
        size_t capacity   = 0;         // allocated size (>= requested)
    };

    static NovaStagingPool& instance();

    void init(VmaAllocator vma);

    /// Acquire a staging buffer with at least @p nbytes capacity.
    StagingBuffer acquire(size_t nbytes);

    /// Return a staging buffer to the pool for later reuse.
    void release(StagingBuffer buf);

    /// Destroy every pooled buffer.  Called during shutdown.
    void destroyAll();

private:
    NovaStagingPool() = default;

    StagingBuffer createBuffer(size_t nbytes);

    static constexpr size_t kNumBuckets = 10;
    /// Bucket boundaries: 4K, 16K, 64K, 256K, 1M, 4M, 16M, 64M, 256M, unbounded
    static size_t bucketIndex(size_t nbytes);
    static size_t bucketSize(size_t bucket);

    VmaAllocator vma_ = VK_NULL_HANDLE;
    std::mutex mu_;
    std::vector<StagingBuffer> free_[kNumBuckets];
};
