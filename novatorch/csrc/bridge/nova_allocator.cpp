#include "nova_allocator.h"
#include "nova_context.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

NovaAllocator* NovaAllocator::getInstance() {
    static NovaAllocator instance;
    return &instance;
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

c10::DataPtr NovaAllocator::allocate(size_t nbytes) {
    if (nbytes == 0) {
        return c10::DataPtr(
            nullptr,
            nullptr,
            &NovaAllocator::deleter,
            c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    VmaAllocator vma = NovaContext::instance().allocator();

    // Buffer creation info
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = static_cast<VkDeviceSize>(nbytes);
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                      | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // VMA allocation info: host-visible, persistently mapped
    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                   | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation vma_alloc = VK_NULL_HANDLE;
    VmaAllocationInfo alloc_info{};

    VkResult result = vmaCreateBuffer(
        vma, &buffer_info, &alloc_ci, &buffer, &vma_alloc, &alloc_info);

    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "NovaAllocator::allocate – vmaCreateBuffer failed (VkResult "
            + std::to_string(static_cast<int>(result))
            + ", requested " + std::to_string(nbytes) + " bytes)");
    }

    // Heap-allocated bookkeeping – freed in deleter() or releaseAll()
    auto* alloc = new Allocation{
        buffer,
        vma_alloc,
        alloc_info.pMappedData,
        nbytes,
        vma
    };

    // Track the allocation
    {
        std::lock_guard<std::mutex> lock(mu_);
        live_.insert(alloc);
    }

    return c10::DataPtr(
        alloc->mapped_ptr,
        static_cast<void*>(alloc),
        &NovaAllocator::deleter,
        c10::Device(c10::DeviceType::PrivateUse1, 0));
}

// ---------------------------------------------------------------------------
// Deleter
// ---------------------------------------------------------------------------

void NovaAllocator::deleter(void* ctx) {
    if (!ctx) {
        return;
    }
    auto* alloc = static_cast<Allocation*>(ctx);
    auto* self = NovaAllocator::getInstance();

    bool need_vma_destroy = false;
    {
        std::lock_guard<std::mutex> lock(self->mu_);

        if (self->released_) {
            // releaseAll() already destroyed the VMA buffer for this
            // allocation. Just free the bookkeeping struct.
            delete alloc;
            return;
        }

        // Normal path: remove from tracking, then destroy VMA buffer.
        self->live_.erase(alloc);
        need_vma_destroy = true;
    }

    if (need_vma_destroy) {
        vmaDestroyBuffer(alloc->allocator, alloc->buffer, alloc->vma_alloc);
    }
    delete alloc;
}

// ---------------------------------------------------------------------------
// releaseAll – force-free all tracked VMA allocations
// ---------------------------------------------------------------------------

void NovaAllocator::releaseAll() {
    std::lock_guard<std::mutex> lock(mu_);

    for (auto* alloc : live_) {
        // Destroy only the VMA buffer. Do NOT delete the Allocation struct —
        // PyTorch DataPtrs still hold pointers to them and will invoke
        // deleter() later. The deleter checks released_ and just frees the
        // struct without touching VMA.
        vmaDestroyBuffer(alloc->allocator, alloc->buffer, alloc->vma_alloc);
    }
    live_.clear();
    released_ = true;
}

// ---------------------------------------------------------------------------
// raw_deleter
// ---------------------------------------------------------------------------

c10::DeleterFnPtr NovaAllocator::raw_deleter() const {
    return &NovaAllocator::deleter;
}
