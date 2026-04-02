#pragma once

#include <vulkan/vulkan.h>
#include "vk_memory.h"

#include <c10/core/Allocator.h>
#include <cstddef>
#include <mutex>
#include <unordered_set>

/// PyTorch c10::Allocator backed by VMA (Vulkan Memory Allocator).
///
/// Every allocation creates a DEVICE_LOCAL VkBuffer in VRAM for GPU compute.
/// Host access goes through a shared staging pool (NovaStagingPool) — no
/// per-tensor host-visible mapping.
///
/// DataPtr::data() returns an opaque pointer (Allocation*) that is NOT
/// CPU-dereferenceable, matching the CUDA model where data_ptr() returns
/// a device pointer.
///
/// All live allocations are tracked so they can be force-released at shutdown
/// (before VMA is destroyed) to avoid assertion failures from PyTorch tensors
/// that outlive the Vulkan context.
class NovaAllocator : public c10::Allocator {
public:
    /// Per-allocation bookkeeping stored as the DataPtr context.
    ///
    /// The buffer lives in DEVICE_LOCAL VRAM.  There is no per-tensor
    /// staging buffer — staging comes from the shared NovaStagingPool
    /// on demand for CPU↔GPU transfers.
    struct Allocation {
        VkBuffer buffer;          // DEVICE_LOCAL (VRAM)
        VmaAllocation vma_alloc;
        size_t size;
        VmaAllocator allocator;   // cached for cleanup without singleton lookup
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
