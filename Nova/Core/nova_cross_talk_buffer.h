#pragma once
#include <vulkan/vulkan.h>
#include <atomic>
#include <cstdint>
#include <mutex>

// Forward declare VMA types to avoid pulling in the massive header
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

namespace Nova {

class NovaCrossTalkBuffer {
public:
    void init(VmaAllocator allocator, uint32_t left_trees, uint32_t right_trees,
              uint32_t signal_dim);
    void shutdown();

    // Frame management — call once per training step
    void swapFrames();
    uint32_t readFrame() const { return read_frame_.load(std::memory_order_acquire); }
    uint32_t writeFrame() const { return 1 - read_frame_.load(std::memory_order_acquire); }

    // CPU write access (for trees depositing signals)
    void writeLeftSignal(uint32_t tree_id, const float* data);
    void writeRightSignal(uint32_t tree_id, const float* data);

    // CPU read access (for trees reading cross-talk fields)
    const float* readLeftField(uint32_t tree_id) const;
    const float* readRightField(uint32_t tree_id) const;

    // CPU read access to raw signals (for MFN sync thread)
    const float* readLeftSignal(uint32_t tree_id) const;
    const float* readRightSignal(uint32_t tree_id) const;

    // CPU write to fields (for MFN enrichment from background thread)
    void enrichLeftField(uint32_t tree_id, const float* data, float scale);
    void enrichRightField(uint32_t tree_id, const float* data, float scale);

    // GPU access — buffer handle and offsets for shader binding
    VkBuffer buffer() const { return buffer_; }
    VkDeviceSize frameOffset(uint32_t frame) const;
    VkDeviceSize leftSignalOffset(uint32_t frame, uint32_t tree_id) const;
    VkDeviceSize rightSignalOffset(uint32_t frame, uint32_t tree_id) const;
    VkDeviceSize leftFieldOffset(uint32_t frame, uint32_t tree_id) const;
    VkDeviceSize rightFieldOffset(uint32_t frame, uint32_t tree_id) const;

    // Sizes for descriptor set binding
    VkDeviceSize totalSize() const { return total_size_; }
    VkDeviceSize frameSize() const { return frame_size_; }

    // Flush/invalidate for CPU<->GPU coherence
    void flushWrite();
    void invalidateRead();

    // Dimensions
    uint32_t leftTrees() const { return left_trees_; }
    uint32_t rightTrees() const { return right_trees_; }
    uint32_t signalDim() const { return signal_dim_; }

private:
    VmaAllocator allocator_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    uint8_t* mapped_ = nullptr;

    uint32_t left_trees_ = 0;
    uint32_t right_trees_ = 0;
    uint32_t signal_dim_ = 0;

    VkDeviceSize frame_size_ = 0;
    VkDeviceSize total_size_ = 0;
    std::atomic<uint32_t> read_frame_{0};

    // Section offsets within a frame (in bytes)
    VkDeviceSize left_signal_section_ = 0;
    VkDeviceSize right_signal_section_ = 0;
    VkDeviceSize left_field_section_ = 0;
    VkDeviceSize right_field_section_ = 0;

    // Mutex protecting write-frame access from concurrent threads
    // (main training thread writes signals, MFN sync thread enriches fields).
    mutable std::mutex write_mutex_;

    float* ptrAt(VkDeviceSize offset) const;
    const float* cptrAt(VkDeviceSize offset) const;
};

} // namespace Nova
