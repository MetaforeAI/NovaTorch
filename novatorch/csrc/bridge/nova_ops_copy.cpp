#include "nova_ops.h"

#include <c10/core/ScalarType.h>
#include <cstring>
#include <vector>

namespace {

bool isNova(const at::Tensor& t) {
    return t.device().type() == c10::DeviceType::PrivateUse1;
}

size_t tensorBytes(const at::Tensor& t) {
    return static_cast<size_t>(t.numel()) * t.element_size();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// _copy_from  (src = self, dst = dst)
// ---------------------------------------------------------------------------

at::Tensor nova_copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool /*non_blocking*/)
{
    if (self.numel() == 0) return dst;

    const bool src_nova = isNova(self);
    const bool dst_nova = isNova(dst);

    if (src_nova && dst_nova) {
        if (self.is_contiguous() && self.storage().nbytes() >= tensorBytes(dst)) {
            VkBuffer src_buf = novatorch::getNovaBuffer(self);
            VkBuffer dst_buf = novatorch::getNovaBuffer(dst);
            VkDeviceSize copy_size = static_cast<VkDeviceSize>(tensorBytes(dst));

            NovaContext::instance().executeSync([&](VkCommandBuffer cmd) {
                VkBufferCopy region{};
                region.srcOffset = 0;
                region.dstOffset = 0;
                region.size = copy_size;
                vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            });
        } else {
            // Non-contiguous source: element-by-element through mapped memory
            novatorch::invalidateNovaBuffer(self);
            auto* alloc_src = novatorch::getNovaAllocation(self);
            auto* alloc_dst = novatorch::getNovaAllocation(dst);
            const float* src_base = static_cast<const float*>(alloc_src->mapped_ptr);
            float* dst_ptr = static_cast<float*>(alloc_dst->mapped_ptr);
            int64_t n = dst.numel();
            auto src_strides = self.strides();
            auto src_sizes = self.sizes();
            int ndim = self.dim();
            std::vector<int64_t> idx(ndim, 0);
            for (int64_t i = 0; i < n; ++i) {
                int64_t src_offset = 0;
                for (int d = 0; d < ndim; ++d)
                    src_offset += idx[d] * src_strides[d];
                dst_ptr[i] = src_base[src_offset];
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++idx[d] < src_sizes[d]) break;
                    idx[d] = 0;
                }
            }
            novatorch::flushNovaBuffer(dst);
        }
        return dst;
    }

    if (!src_nova && dst_nova) {
        // CPU -> Nova: handle non-contiguous source
        auto* alloc = novatorch::getNovaAllocation(dst);
        if (self.is_contiguous()) {
            std::memcpy(alloc->mapped_ptr, self.data_ptr(), tensorBytes(self));
        } else {
            // Make contiguous on CPU first, then memcpy
            auto src_contig = self.contiguous();
            std::memcpy(alloc->mapped_ptr, src_contig.data_ptr(), tensorBytes(src_contig));
        }
        novatorch::flushNovaBuffer(dst);
        return dst;
    }

    if (src_nova && !dst_nova) {
        // Nova -> CPU: invalidate + memcpy
        novatorch::invalidateNovaBuffer(self);
        auto* alloc = novatorch::getNovaAllocation(self);
        if (self.is_contiguous() && self.storage().nbytes() >= tensorBytes(dst)) {
            std::memcpy(dst.data_ptr(), alloc->mapped_ptr, tensorBytes(dst));
        } else {
            // Non-contiguous source: element-by-element
            const float* src_base = static_cast<const float*>(alloc->mapped_ptr);
            float* dst_ptr = static_cast<float*>(dst.data_ptr());
            int64_t n = dst.numel();
            auto src_strides = self.strides();
            auto src_sizes = self.sizes();
            int ndim = self.dim();
            std::vector<int64_t> idx(ndim, 0);
            for (int64_t i = 0; i < n; ++i) {
                int64_t src_offset = 0;
                for (int d = 0; d < ndim; ++d)
                    src_offset += idx[d] * src_strides[d];
                dst_ptr[i] = src_base[src_offset];
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++idx[d] < src_sizes[d]) break;
                    idx[d] = 0;
                }
            }
        }
        return dst;
    }

    // CPU -> CPU fallback
    std::memcpy(dst.data_ptr(), self.data_ptr(), tensorBytes(self));
    return dst;
}

// ---------------------------------------------------------------------------
// _copy_from_and_resize
// ---------------------------------------------------------------------------

at::Tensor nova_copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst)
{
    if (dst.sizes() != self.sizes()) {
        const_cast<at::Tensor&>(dst).resize_(self.sizes());
    }
    return nova_copy_from(self, dst, /*non_blocking=*/false);
}

// ---------------------------------------------------------------------------
// _local_scalar_dense  –  read a single scalar from a Nova tensor
// ---------------------------------------------------------------------------

at::Scalar nova_local_scalar_dense(const at::Tensor& self) {
    TORCH_CHECK(
        self.numel() == 1,
        "nova_local_scalar_dense: expected 1-element tensor, got ",
        self.numel());

    novatorch::invalidateNovaBuffer(self);
    auto* alloc = novatorch::getNovaAllocation(self);

    switch (self.scalar_type()) {
        case c10::ScalarType::Float: {
            float v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Double: {
            double v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Half: {
            c10::Half v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(static_cast<float>(v));
        }
        case c10::ScalarType::BFloat16: {
            c10::BFloat16 v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(static_cast<float>(v));
        }
        case c10::ScalarType::Int: {
            int32_t v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Long: {
            int64_t v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Short: {
            int16_t v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Byte: {
            uint8_t v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Char: {
            int8_t v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Bool: {
            bool v;
            std::memcpy(&v, alloc->mapped_ptr, sizeof(v));
            return at::Scalar(v);
        }
        default:
            TORCH_CHECK(false,
                "nova_local_scalar_dense: unsupported dtype ",
                self.scalar_type());
    }
}
