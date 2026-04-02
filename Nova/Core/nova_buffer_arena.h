#pragma once
#include "./components/vk_memory.h"
#include <vector>
#include <cstdint>
#include <cstring>

class NovaBufferArena {
public:
    struct SlotID { uint32_t id; };

    NovaBufferArena() = default;
    ~NovaBufferArena();

    // Non-copyable, non-movable (owns GPU resources)
    NovaBufferArena(const NovaBufferArena&) = delete;
    NovaBufferArena& operator=(const NovaBufferArena&) = delete;

    // Initialize with VMA allocator and device
    void init(VmaAllocator allocator, VkDevice device);

    // Planning phase: request buffer slots of specific sizes.
    // Call repeatedly before finalize().
    SlotID requestSlot(uint32_t size_bytes);

    // Finalize: perform the single VMA allocation for all requested slots.
    // After this, no more requestSlot() calls are allowed.
    bool finalize();

    // Access slot buffer info (for descriptor binding)
    VkBuffer buffer() const { return backing_buffer_; }
    VkDeviceSize offset(SlotID slot) const;
    uint32_t size(SlotID slot) const;

    // Data transfer (uses persistently mapped memory)
    void upload(SlotID slot, const float* data, uint32_t count);
    void download(SlotID slot, float* data, uint32_t count);
    void zero(SlotID slot);

    // Record a copy between slots into a command buffer
    void recordCopy(VkCommandBuffer cmd, SlotID src, SlotID dst,
                    uint32_t size_bytes);

    // Stats
    uint64_t totalBytes() const { return total_size_; }
    uint32_t slotCount() const { return static_cast<uint32_t>(slots_.size()); }
    bool isFinalized() const { return finalized_; }

    // Cleanup (idempotent)
    void destroy();

private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    struct Slot {
        uint32_t offset;  // byte offset in backing buffer
        uint32_t size;    // byte size (aligned)
    };
    std::vector<Slot> slots_;

    VkBuffer backing_buffer_ = VK_NULL_HANDLE;
    VmaAllocation backing_alloc_ = VK_NULL_HANDLE;
    void* mapped_ptr_ = nullptr;
    uint64_t total_size_ = 0;
    bool finalized_ = false;

    static constexpr uint32_t ALIGNMENT = 256;
};
