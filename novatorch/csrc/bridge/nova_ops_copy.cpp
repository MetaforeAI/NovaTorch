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

/// Strided read from @p src_base into contiguous @p dst_ptr.
/// Handles arbitrary dimensionality via index vector iteration.
void stridedCopyToContiguous(
    const char* src_base,
    char* dst_ptr,
    size_t elem_size,
    int64_t numel,
    at::IntArrayRef sizes,
    at::IntArrayRef strides)
{
    const int ndim = static_cast<int>(sizes.size());
    std::vector<int64_t> idx(ndim, 0);
    for (int64_t i = 0; i < numel; ++i) {
        int64_t src_byte_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            src_byte_offset += idx[d] * strides[d] * static_cast<int64_t>(elem_size);
        }
        std::memcpy(dst_ptr + i * elem_size, src_base + src_byte_offset, elem_size);
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < sizes[d]) break;
            idx[d] = 0;
        }
    }
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
            // Contiguous Nova→Nova: device-to-device copy
            novatorch::copyDeviceToDevice(self, dst, tensorBytes(dst));
        } else {
            // Non-contiguous Nova→Nova: download src to staging, strided read,
            // then upload contiguous result to dst.
            const size_t elem_size = self.element_size();
            const int64_t numel = dst.numel();
            const size_t dst_bytes = tensorBytes(dst);

            novatorch::withStagingRead(self, [&](const void* src_staging, size_t) {
                // Allocate a temporary contiguous CPU buffer for the result
                std::vector<char> contiguous_buf(dst_bytes);
                stridedCopyToContiguous(
                    static_cast<const char*>(src_staging),
                    contiguous_buf.data(),
                    elem_size,
                    numel,
                    self.sizes(),
                    self.strides());
                // Upload the contiguous result to dst device buffer
                novatorch::uploadToDevice(dst, contiguous_buf.data(), dst_bytes);
            });
        }
        return dst;
    }

    if (!src_nova && dst_nova) {
        // CPU→Nova: upload CPU data to device
        if (self.is_contiguous()) {
            novatorch::uploadToDevice(dst, self.data_ptr(), tensorBytes(self));
        } else {
            auto src_contig = self.contiguous();
            novatorch::uploadToDevice(dst, src_contig.data_ptr(), tensorBytes(src_contig));
        }
        return dst;
    }

    if (src_nova && !dst_nova) {
        // Nova→CPU
        if (self.is_contiguous() && self.storage().nbytes() >= tensorBytes(dst)) {
            // Contiguous: direct download
            novatorch::downloadFromDevice(self, dst.data_ptr(), tensorBytes(dst));
        } else {
            // Non-contiguous: download to staging, strided read into CPU dst
            const size_t elem_size = self.element_size();
            const int64_t numel = dst.numel();

            novatorch::withStagingRead(self, [&](const void* staging, size_t) {
                stridedCopyToContiguous(
                    static_cast<const char*>(staging),
                    static_cast<char*>(dst.data_ptr()),
                    elem_size,
                    numel,
                    self.sizes(),
                    self.strides());
            });
        }
        return dst;
    }

    // CPU→CPU fallback
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

    switch (self.scalar_type()) {
        case c10::ScalarType::Float: {
            float v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Double: {
            double v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Half: {
            c10::Half v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(static_cast<float>(v));
        }
        case c10::ScalarType::BFloat16: {
            c10::BFloat16 v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(static_cast<float>(v));
        }
        case c10::ScalarType::Int: {
            int32_t v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Long: {
            int64_t v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(v);
        }
        case c10::ScalarType::Short: {
            int16_t v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Byte: {
            uint8_t v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Char: {
            int8_t v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(static_cast<int64_t>(v));
        }
        case c10::ScalarType::Bool: {
            bool v;
            novatorch::downloadFromDevice(self, &v, sizeof(v));
            return at::Scalar(v);
        }
        default:
            TORCH_CHECK(false,
                "nova_local_scalar_dense: unsupported dtype ",
                self.scalar_type());
    }
}
