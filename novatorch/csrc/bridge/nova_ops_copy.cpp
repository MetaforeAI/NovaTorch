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

// ---------------------------------------------------------------------------
// _to_copy — dtype/device conversion (used by AOTAutograd)
// ---------------------------------------------------------------------------

at::Tensor nova_to_copy(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> /*layout*/,
    std::optional<c10::Device> device,
    std::optional<bool> /*pin_memory*/,
    bool /*non_blocking*/,
    std::optional<c10::MemoryFormat> memory_format) {

    auto target_dtype = dtype.value_or(self.scalar_type());
    auto target_device = device.value_or(self.device());
    bool same_dtype = (target_dtype == self.scalar_type());
    bool self_nova = isNova(self);
    bool target_nova = (target_device.type() == c10::DeviceType::PrivateUse1);

    // Case 1: Same device, same dtype → clone
    if (same_dtype && self_nova && target_nova) {
        auto self_c = self.is_contiguous() ? self : self.contiguous();
        auto output = at::empty(self_c.sizes(), self_c.options());
        novatorch::copyDeviceToDevice(self_c, output,
            self_c.numel() * self_c.element_size());
        return output;
    }

    // Case 2: Same device, different dtype → CPU round-trip
    if (self_nova && target_nova && !same_dtype) {
        auto self_c = self.is_contiguous() ? self : self.contiguous();
        size_t src_bytes = self_c.numel() * self_c.element_size();
        auto cpu_src = at::empty(self_c.sizes(),
            self_c.options().device(c10::kCPU));
        novatorch::downloadFromDevice(self_c, cpu_src.data_ptr(), src_bytes);
        auto cpu_dst = cpu_src.to(target_dtype);
        auto output = at::empty(cpu_dst.sizes(),
            cpu_dst.options().device(target_device));
        novatorch::uploadToDevice(output, cpu_dst.data_ptr(),
            cpu_dst.numel() * cpu_dst.element_size());
        return output;
    }

    // Case 3: Nova → CPU
    if (self_nova && !target_nova) {
        auto self_c = self.is_contiguous() ? self : self.contiguous();
        size_t bytes = self_c.numel() * self_c.element_size();
        auto cpu_out = at::empty(self_c.sizes(),
            self_c.options().device(c10::kCPU));
        novatorch::downloadFromDevice(self_c, cpu_out.data_ptr(), bytes);
        if (!same_dtype) cpu_out = cpu_out.to(target_dtype);
        return cpu_out;
    }

    // Case 4: CPU → Nova
    if (!self_nova && target_nova) {
        auto src = same_dtype ? self : self.to(target_dtype);
        src = src.is_contiguous() ? src : src.contiguous();
        auto output = at::empty(src.sizes(),
            src.options().device(target_device));
        novatorch::uploadToDevice(output, src.data_ptr(),
            src.numel() * src.element_size());
        return output;
    }

    // Case 5: CPU → CPU fallback
    return self.to(target_dtype);
}
