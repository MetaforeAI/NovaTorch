#pragma once

#include <vulkan/vulkan.h>
#include "vk_memory.h"

#include <c10/core/Allocator.h>
#include <cstddef>
#include <mutex>
#include <unordered_set>

/// PyTorch c10::Allocator backed by VMA (Vulkan Memory Allocator).
///
/// Every allocation creates a VkBuffer with host-visible, persistently-mapped
/// storage.  The mapped pointer is returned as the data pointer so PyTorch CPU
/// code can read/write the memory directly, while the underlying VkBuffer is
/// available for Vulkan compute dispatch.
///
/// All live allocations are tracked so they can be force-released at shutdown
/// (before VMA is destroyed) to avoid assertion failures from PyTorch tensors
/// that outlive the Vulkan context.
class NovaAllocator : public c10::Allocator {
public:
    /// Per-allocation bookkeeping stored as the DataPtr context.
    struct Allocation {
        VkBuffer buffer;
        VmaAllocation vma_alloc;
        void* mapped_ptr;
        size_t size;
        VmaAllocator allocator; // cached for cleanup without singleton lookup
    };

    static NovaAllocator* getInstance();

    c10::DataPtr allocate(size_t nbytes) override;
    c10::DeleterFnPtr raw_deleter() const override;
    void copy_data(void* dest, const void* src, std::size_t count) const override {
        default_copy_data(dest, src, count);
    }

    /// Force-release all tracked VMA allocations.
    /// Must be called BEFORE vmaDestroyAllocator to prevent assertion failures.
    /// After this call, any subsequent deleter invocations for previously
    /// tracked allocations become no-ops.
    void releaseAll();

private:
    NovaAllocator() = default;

    static void deleter(void* ctx);

    // Track all live allocations for force-release at shutdown.
    std::mutex mu_;
    std::unordered_set<Allocation*> live_;
    bool released_ = false; // true after releaseAll() has been called
};
