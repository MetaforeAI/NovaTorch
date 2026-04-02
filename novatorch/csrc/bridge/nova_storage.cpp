#include "nova_storage.h"
#include "nova_context.h"

#include <stdexcept>

namespace novatorch {

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

void flushNovaBuffer(const at::Tensor& tensor) {
    auto* alloc = getNovaAllocation(tensor);
    VkResult result = vmaFlushAllocation(
        alloc->allocator, alloc->vma_alloc, 0, VK_WHOLE_SIZE);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "flushNovaBuffer: vmaFlushAllocation failed (VkResult "
            + std::to_string(static_cast<int>(result)) + ")");
    }
}

void invalidateNovaBuffer(const at::Tensor& tensor) {
    auto* alloc = getNovaAllocation(tensor);
    VkResult result = vmaInvalidateAllocation(
        alloc->allocator, alloc->vma_alloc, 0, VK_WHOLE_SIZE);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "invalidateNovaBuffer: vmaInvalidateAllocation failed (VkResult "
            + std::to_string(static_cast<int>(result)) + ")");
    }
}

} // namespace novatorch
