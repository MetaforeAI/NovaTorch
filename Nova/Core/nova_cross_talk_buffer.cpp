#include "nova_cross_talk_buffer.h"
#include "components/vk_memory.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <mutex>

namespace Nova {

void NovaCrossTalkBuffer::init(VmaAllocator allocator, uint32_t left_trees,
                               uint32_t right_trees, uint32_t signal_dim) {
    allocator_ = allocator;
    left_trees_ = left_trees;
    right_trees_ = right_trees;
    signal_dim_ = signal_dim;

    const VkDeviceSize stride = signal_dim * sizeof(float);
    left_signal_section_  = 0;
    right_signal_section_ = left_trees * stride;
    left_field_section_   = right_signal_section_ + right_trees * stride;
    right_field_section_  = left_field_section_ + left_trees * stride;
    frame_size_           = right_field_section_ + right_trees * stride;
    total_size_           = frame_size_ * 2;

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
        &buffer_, &allocation_, &alloc_result);

    if (result != VK_SUCCESS) {
        std::fprintf(stderr,
            "[NovaCrossTalkBuffer] VMA alloc failed: %lu bytes, VkResult=%d\n",
            static_cast<unsigned long>(total_size_), result);
        return;
    }

    mapped_ = static_cast<uint8_t*>(alloc_result.pMappedData);
    std::memset(mapped_, 0, total_size_);
    read_frame_.store(0, std::memory_order_relaxed);

    std::fprintf(stderr,
        "[NovaCrossTalkBuffer] Allocated %lu bytes (2x%lu) — L=%u R=%u dim=%u\n",
        static_cast<unsigned long>(total_size_),
        static_cast<unsigned long>(frame_size_),
        left_trees_, right_trees_, signal_dim_);
}

void NovaCrossTalkBuffer::shutdown() {
    if (buffer_ != VK_NULL_HANDLE && allocator_ != nullptr) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
    buffer_ = VK_NULL_HANDLE;
    allocation_ = nullptr;
    mapped_ = nullptr;
    frame_size_ = 0;
    total_size_ = 0;
}

void NovaCrossTalkBuffer::swapFrames() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    uint32_t old = read_frame_.load(std::memory_order_acquire);
    read_frame_.store(1 - old, std::memory_order_release);
}

// --- CPU write access (write frame) ---

void NovaCrossTalkBuffer::writeLeftSignal(uint32_t tree_id, const float* data) {
    assert(tree_id < left_trees_);
    std::lock_guard<std::mutex> lock(write_mutex_);
    VkDeviceSize off = frameOffset(writeFrame()) + left_signal_section_
                     + tree_id * signal_dim_ * sizeof(float);
    std::memcpy(ptrAt(off), data, signal_dim_ * sizeof(float));
}

void NovaCrossTalkBuffer::writeRightSignal(uint32_t tree_id, const float* data) {
    assert(tree_id < right_trees_);
    std::lock_guard<std::mutex> lock(write_mutex_);
    VkDeviceSize off = frameOffset(writeFrame()) + right_signal_section_
                     + tree_id * signal_dim_ * sizeof(float);
    std::memcpy(ptrAt(off), data, signal_dim_ * sizeof(float));
}

// --- CPU read access (read frame) ---

const float* NovaCrossTalkBuffer::readLeftField(uint32_t tree_id) const {
    assert(tree_id < left_trees_);
    uint32_t rf = read_frame_.load(std::memory_order_acquire);
    VkDeviceSize off = frameOffset(rf) + left_field_section_
                     + tree_id * signal_dim_ * sizeof(float);
    return cptrAt(off);
}

const float* NovaCrossTalkBuffer::readRightField(uint32_t tree_id) const {
    assert(tree_id < right_trees_);
    uint32_t rf = read_frame_.load(std::memory_order_acquire);
    VkDeviceSize off = frameOffset(rf) + right_field_section_
                     + tree_id * signal_dim_ * sizeof(float);
    return cptrAt(off);
}

const float* NovaCrossTalkBuffer::readLeftSignal(uint32_t tree_id) const {
    assert(tree_id < left_trees_);
    uint32_t rf = read_frame_.load(std::memory_order_acquire);
    VkDeviceSize off = frameOffset(rf) + left_signal_section_
                     + tree_id * signal_dim_ * sizeof(float);
    return cptrAt(off);
}

const float* NovaCrossTalkBuffer::readRightSignal(uint32_t tree_id) const {
    assert(tree_id < right_trees_);
    uint32_t rf = read_frame_.load(std::memory_order_acquire);
    VkDeviceSize off = frameOffset(rf) + right_signal_section_
                     + tree_id * signal_dim_ * sizeof(float);
    return cptrAt(off);
}

// --- MFN enrichment (additive blend into write frame fields) ---

void NovaCrossTalkBuffer::enrichLeftField(uint32_t tree_id, const float* data,
                                          float scale) {
    assert(tree_id < left_trees_);
    std::lock_guard<std::mutex> lock(write_mutex_);
    VkDeviceSize off = frameOffset(writeFrame()) + left_field_section_
                     + tree_id * signal_dim_ * sizeof(float);
    float* dst = ptrAt(off);
    for (uint32_t i = 0; i < signal_dim_; ++i) {
        dst[i] = data[i] * scale;
    }
}

void NovaCrossTalkBuffer::enrichRightField(uint32_t tree_id, const float* data,
                                           float scale) {
    assert(tree_id < right_trees_);
    std::lock_guard<std::mutex> lock(write_mutex_);
    VkDeviceSize off = frameOffset(writeFrame()) + right_field_section_
                     + tree_id * signal_dim_ * sizeof(float);
    float* dst = ptrAt(off);
    for (uint32_t i = 0; i < signal_dim_; ++i) {
        dst[i] = data[i] * scale;
    }
}

// --- GPU offset queries ---

VkDeviceSize NovaCrossTalkBuffer::frameOffset(uint32_t frame) const {
    assert(frame < 2);
    return frame * frame_size_;
}

VkDeviceSize NovaCrossTalkBuffer::leftSignalOffset(uint32_t frame,
                                                   uint32_t tree_id) const {
    assert(frame < 2 && tree_id < left_trees_);
    return frameOffset(frame) + left_signal_section_
         + tree_id * signal_dim_ * sizeof(float);
}

VkDeviceSize NovaCrossTalkBuffer::rightSignalOffset(uint32_t frame,
                                                    uint32_t tree_id) const {
    assert(frame < 2 && tree_id < right_trees_);
    return frameOffset(frame) + right_signal_section_
         + tree_id * signal_dim_ * sizeof(float);
}

VkDeviceSize NovaCrossTalkBuffer::leftFieldOffset(uint32_t frame,
                                                  uint32_t tree_id) const {
    assert(frame < 2 && tree_id < left_trees_);
    return frameOffset(frame) + left_field_section_
         + tree_id * signal_dim_ * sizeof(float);
}

VkDeviceSize NovaCrossTalkBuffer::rightFieldOffset(uint32_t frame,
                                                   uint32_t tree_id) const {
    assert(frame < 2 && tree_id < right_trees_);
    return frameOffset(frame) + right_field_section_
         + tree_id * signal_dim_ * sizeof(float);
}

// --- Flush / Invalidate ---

void NovaCrossTalkBuffer::flushWrite() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    VkDeviceSize off = frameOffset(writeFrame());
    vmaFlushAllocation(allocator_, allocation_, off, frame_size_);
}

void NovaCrossTalkBuffer::invalidateRead() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    uint32_t rf = read_frame_.load(std::memory_order_acquire);
    VkDeviceSize off = frameOffset(rf);
    vmaInvalidateAllocation(allocator_, allocation_, off, frame_size_);
}

// --- Private helpers ---

float* NovaCrossTalkBuffer::ptrAt(VkDeviceSize offset) const {
    return reinterpret_cast<float*>(mapped_ + offset);
}

const float* NovaCrossTalkBuffer::cptrAt(VkDeviceSize offset) const {
    return reinterpret_cast<const float*>(mapped_ + offset);
}

} // namespace Nova
