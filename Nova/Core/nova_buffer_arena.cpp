#include "nova_buffer_arena.h"
#include <cassert>
#include <cstdio>

NovaBufferArena::~NovaBufferArena() {
    destroy();
}

void NovaBufferArena::init(VmaAllocator allocator, VkDevice device) {
    allocator_ = allocator;
    device_ = device;
}

NovaBufferArena::SlotID NovaBufferArena::requestSlot(uint32_t size_bytes) {
    assert(!finalized_ && "Cannot request slots after finalize()");

    // Align size to ALIGNMENT boundary
    size_bytes = (size_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    Slot slot;
    slot.offset = static_cast<uint32_t>(total_size_);
    slot.size = size_bytes;
    total_size_ += size_bytes;

    SlotID id;
    id.id = static_cast<uint32_t>(slots_.size());
    slots_.push_back(slot);
    return id;
}

bool NovaBufferArena::finalize() {
    if (finalized_) return true;
    if (total_size_ == 0) return false;

    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = total_size_,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
               | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
               | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                     | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_result = {};
    VkResult result = vmaCreateBuffer(
        allocator_, &buf_info, &alloc_info,
        &backing_buffer_, &backing_alloc_, &alloc_result);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
            "[NovaBufferArena] VMA alloc failed: %luMB requested, "
            "VkResult=%d\n",
            static_cast<unsigned long>(total_size_ / (1024 * 1024)),
            result);
        return false;
    }

    mapped_ptr_ = alloc_result.pMappedData;
    finalized_ = true;

    std::fprintf(stderr,
        "[NovaBufferArena] Allocated %luMB for %u slots\n",
        static_cast<unsigned long>(total_size_ / (1024 * 1024)),
        static_cast<uint32_t>(slots_.size()));
    return true;
}

VkDeviceSize NovaBufferArena::offset(SlotID slot) const {
    assert(slot.id < slots_.size());
    return slots_[slot.id].offset;
}

uint32_t NovaBufferArena::size(SlotID slot) const {
    assert(slot.id < slots_.size());
    return slots_[slot.id].size;
}

void NovaBufferArena::upload(SlotID slot, const float* data, uint32_t count) {
    assert(finalized_ && mapped_ptr_);
    const auto& s = slots_[slot.id];
    assert(count * sizeof(float) <= s.size);

    void* dst = static_cast<char*>(mapped_ptr_) + s.offset;
    std::memcpy(dst, data, count * sizeof(float));
    vmaFlushAllocation(allocator_, backing_alloc_,
                       s.offset, count * sizeof(float));
}

void NovaBufferArena::download(SlotID slot, float* data, uint32_t count) {
    assert(finalized_ && mapped_ptr_);
    const auto& s = slots_[slot.id];
    assert(count * sizeof(float) <= s.size);

    vmaInvalidateAllocation(allocator_, backing_alloc_,
                            s.offset, count * sizeof(float));
    const void* src = static_cast<const char*>(mapped_ptr_) + s.offset;
    std::memcpy(data, src, count * sizeof(float));
}

void NovaBufferArena::zero(SlotID slot) {
    assert(finalized_ && mapped_ptr_);
    const auto& s = slots_[slot.id];

    void* dst = static_cast<char*>(mapped_ptr_) + s.offset;
    std::memset(dst, 0, s.size);
    vmaFlushAllocation(allocator_, backing_alloc_, s.offset, s.size);
}

void NovaBufferArena::recordCopy(VkCommandBuffer cmd, SlotID src, SlotID dst,
                                  uint32_t size_bytes) {
    VkBufferCopy region = {};
    region.srcOffset = slots_[src.id].offset;
    region.dstOffset = slots_[dst.id].offset;
    region.size = size_bytes;
    vkCmdCopyBuffer(cmd, backing_buffer_, backing_buffer_, 1, &region);
}

void NovaBufferArena::destroy() {
    if (backing_buffer_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, backing_buffer_, backing_alloc_);
    }
    backing_buffer_ = VK_NULL_HANDLE;
    backing_alloc_ = VK_NULL_HANDLE;
    mapped_ptr_ = nullptr;
    slots_.clear();
    total_size_ = 0;
    finalized_ = false;
}
