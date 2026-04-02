#include "nova_staging_pool.h"

#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Bucket sizing
// ---------------------------------------------------------------------------

// Boundaries: 4K, 16K, 64K, 256K, 1M, 4M, 16M, 64M, 256M
static constexpr size_t kBucketSizes[] = {
    4u << 10,    //   4 KB
    16u << 10,   //  16 KB
    64u << 10,   //  64 KB
    256u << 10,  // 256 KB
    1u << 20,    //   1 MB
    4u << 20,    //   4 MB
    16u << 20,   //  16 MB
    64u << 20,   //  64 MB
    256u << 20,  // 256 MB
};

size_t NovaStagingPool::bucketIndex(size_t nbytes) {
    for (size_t i = 0; i < kNumBuckets - 1; ++i) {
        if (nbytes <= kBucketSizes[i]) return i;
    }
    return kNumBuckets - 1;
}

size_t NovaStagingPool::bucketSize(size_t bucket) {
    if (bucket < kNumBuckets - 1) return kBucketSizes[bucket];
    return 0; // unbounded — allocate exact size
}

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

NovaStagingPool& NovaStagingPool::instance() {
    static NovaStagingPool pool;
    return pool;
}

void NovaStagingPool::init(VmaAllocator vma) {
    vma_ = vma;
}

// ---------------------------------------------------------------------------
// Create a new staging buffer via VMA
// ---------------------------------------------------------------------------

NovaStagingPool::StagingBuffer NovaStagingPool::createBuffer(size_t nbytes) {
    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size  = static_cast<VkDeviceSize>(nbytes);
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                   | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_ci{};
    alloc_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                   | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_ci.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                            | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    StagingBuffer sb{};
    VmaAllocationInfo info{};

    VkResult result = vmaCreateBuffer(
        vma_, &buf_info, &alloc_ci, &sb.buffer, &sb.alloc, &info);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "NovaStagingPool::createBuffer failed (VkResult "
            + std::to_string(static_cast<int>(result))
            + ", requested " + std::to_string(nbytes) + " bytes)");
    }

    sb.ptr = info.pMappedData;
    sb.capacity = nbytes;
    return sb;
}

// ---------------------------------------------------------------------------
// Acquire / Release
// ---------------------------------------------------------------------------

NovaStagingPool::StagingBuffer NovaStagingPool::acquire(size_t nbytes) {
    if (nbytes == 0) return {};

    size_t idx = bucketIndex(nbytes);
    size_t alloc_size = bucketSize(idx);
    if (alloc_size == 0) alloc_size = nbytes; // unbounded bucket

    {
        std::lock_guard<std::mutex> lock(mu_);
        auto& list = free_[idx];
        if (!list.empty()) {
            StagingBuffer sb = list.back();
            list.pop_back();
            return sb;
        }
    }

    // Nothing in pool — create fresh
    return createBuffer(alloc_size);
}

void NovaStagingPool::release(StagingBuffer buf) {
    if (buf.buffer == VK_NULL_HANDLE) return;

    size_t idx = bucketIndex(buf.capacity);
    std::lock_guard<std::mutex> lock(mu_);
    free_[idx].push_back(buf);
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

void NovaStagingPool::destroyAll() {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t i = 0; i < kNumBuckets; ++i) {
        for (auto& sb : free_[i]) {
            vmaDestroyBuffer(vma_, sb.buffer, sb.alloc);
        }
        free_[i].clear();
    }
}
