#include "nova_storage.h"
#include "nova_context.h"
#include "nova_batch_context.h"

#include <cstring>
#include <stdexcept>

namespace novatorch {

// -------------------------------------------------------------------------
// Basic accessors
// -------------------------------------------------------------------------

NovaAllocator::Allocation* getNovaAllocation(const at::Tensor& tensor) {
    void* ctx = tensor.storage().data_ptr().get_context();
    if (!ctx) {
        throw std::runtime_error(
            "getNovaAllocation: tensor has no Nova allocation context "
            "(is this a PrivateUse1 tensor?)");
    }
    return static_cast<NovaAllocator::Allocation*>(ctx);
}

VkBuffer getNovaBuffer(const at::Tensor& tensor) {
    return getNovaAllocation(tensor)->buffer;
}

// -------------------------------------------------------------------------
// Internal: synchronous staging transfers
// -------------------------------------------------------------------------

namespace {

/// staging → device
void transferUpload(VkBuffer staging, VkBuffer device, size_t nbytes) {
    NovaBatchContext::instance().flush();
    NovaContext::instance().executeSync([&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = static_cast<VkDeviceSize>(nbytes);
        vkCmdCopyBuffer(cmd, staging, device, 1, &region);
    });
}

/// device → staging
void transferDownload(VkBuffer device, VkBuffer staging, size_t nbytes) {
    NovaBatchContext::instance().flush();
    NovaContext::instance().executeSync([&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = static_cast<VkDeviceSize>(nbytes);
        vkCmdCopyBuffer(cmd, device, staging, 1, &region);
    });
}

} // anonymous namespace

// -------------------------------------------------------------------------
// Public transfer helpers
// -------------------------------------------------------------------------

void uploadToDevice(const at::Tensor& tensor, const void* src, size_t nbytes) {
    if (nbytes == 0) return;
    auto* alloc = getNovaAllocation(tensor);
    auto stg = NovaStagingPool::instance().acquire(nbytes);
    std::memcpy(stg.ptr, src, nbytes);
    transferUpload(stg.buffer, alloc->buffer, nbytes);
    NovaStagingPool::instance().release(stg);
}

void downloadFromDevice(const at::Tensor& tensor, void* dst, size_t nbytes) {
    if (nbytes == 0) return;
    auto* alloc = getNovaAllocation(tensor);
    auto stg = NovaStagingPool::instance().acquire(nbytes);
    transferDownload(alloc->buffer, stg.buffer, nbytes);
    std::memcpy(dst, stg.ptr, nbytes);
    NovaStagingPool::instance().release(stg);
}

void withStagingRead(const at::Tensor& tensor,
                     std::function<void(const void*, size_t)> fn) {
    auto* alloc = getNovaAllocation(tensor);
    size_t nbytes = alloc->size;
    if (nbytes == 0) { fn(nullptr, 0); return; }

    auto stg = NovaStagingPool::instance().acquire(nbytes);
    transferDownload(alloc->buffer, stg.buffer, nbytes);
    fn(stg.ptr, nbytes);
    NovaStagingPool::instance().release(stg);
}

void withStagingWrite(const at::Tensor& tensor,
                      std::function<void(void*, size_t)> fn) {
    auto* alloc = getNovaAllocation(tensor);
    size_t nbytes = alloc->size;
    if (nbytes == 0) { fn(nullptr, 0); return; }

    auto stg = NovaStagingPool::instance().acquire(nbytes);
    fn(stg.ptr, nbytes);
    transferUpload(stg.buffer, alloc->buffer, nbytes);
    NovaStagingPool::instance().release(stg);
}

void withStagingReadWrite(const at::Tensor& tensor,
                          std::function<void(void*, size_t)> fn) {
    auto* alloc = getNovaAllocation(tensor);
    size_t nbytes = alloc->size;
    if (nbytes == 0) { fn(nullptr, 0); return; }

    auto stg = NovaStagingPool::instance().acquire(nbytes);
    transferDownload(alloc->buffer, stg.buffer, nbytes);
    fn(stg.ptr, nbytes);
    transferUpload(stg.buffer, alloc->buffer, nbytes);
    NovaStagingPool::instance().release(stg);
}

void copyDeviceToDevice(const at::Tensor& src, const at::Tensor& dst,
                        size_t nbytes) {
    if (nbytes == 0) return;
    NovaBatchContext::instance().flush();
    VkBuffer src_buf = getNovaBuffer(src);
    VkBuffer dst_buf = getNovaBuffer(dst);
    NovaContext::instance().executeSync([&](VkCommandBuffer cmd) {
        VkBufferCopy region{};
        region.size = static_cast<VkDeviceSize>(nbytes);
        vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
    });
}

} // namespace novatorch
