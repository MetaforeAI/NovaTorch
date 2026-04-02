#include "nova_ops.h"

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Push-constant structs -- must match the GLSL layout(push_constant) exactly.
// ---------------------------------------------------------------------------

namespace {

/// elementwise_add.comp / elementwise_sub.comp: { uint numel; float alpha; }
struct PCAddSub {
    uint32_t numel;
    float alpha;
};

/// elementwise_mul.comp / elementwise_div.comp: { uint numel; }
struct PCNumel {
    uint32_t numel;
};

/// addcmul.comp / addcdiv.comp / mul_scalar.comp: { uint numel; float value; }
struct PCNumelValue {
    uint32_t numel;
    float value;
};

constexpr uint32_t kWorkgroupSize = 256;

uint32_t divRoundUp(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

bool isNova(const at::Tensor& t) {
    return t.device().type() == c10::DeviceType::PrivateUse1;
}

size_t tensorBytes(const at::Tensor& t) {
    return static_cast<size_t>(t.numel()) * t.element_size();
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// copy_  --  copy src into self
// ---------------------------------------------------------------------------

at::Tensor& nova_copy_inplace(
    at::Tensor& self,
    const at::Tensor& src,
    bool /*non_blocking*/) {

    if (self.numel() == 0) return self;

    const bool self_nova = isNova(self);
    const bool src_nova = isNova(src);

    if (self_nova && src_nova) {
        if (src.is_contiguous() && src.storage().nbytes() >= tensorBytes(self)) {
            // Nova -> Nova: GPU-side vkCmdCopyBuffer (contiguous, sufficient storage)
            VkBuffer src_buf = novatorch::getNovaBuffer(src);
            VkBuffer dst_buf = novatorch::getNovaBuffer(self);
            VkDeviceSize copy_size =
                static_cast<VkDeviceSize>(tensorBytes(self));

            NovaContext::instance().executeSync([&](VkCommandBuffer cmd) {
                VkBufferCopy region{};
                region.srcOffset = 0;
                region.dstOffset = 0;
                region.size = copy_size;
                vkCmdCopyBuffer(cmd, src_buf, dst_buf, 1, &region);
            });
        } else {
            // Non-contiguous source (e.g. expanded view): element-wise copy
            // through mapped memory. Both buffers are host-mapped.
            // We read element-by-element from src using its strides,
            // and write contiguously into dst.
            novatorch::invalidateNovaBuffer(src);
            auto* alloc_src = novatorch::getNovaAllocation(src);
            auto* alloc_dst = novatorch::getNovaAllocation(self);
            const float* src_base = static_cast<const float*>(alloc_src->mapped_ptr);
            float* dst_ptr = static_cast<float*>(alloc_dst->mapped_ptr);

            int64_t n = self.numel();
            auto src_strides = src.strides();
            auto src_sizes = src.sizes();
            int ndim = src.dim();

            // Flatten the strided iteration
            std::vector<int64_t> idx(ndim, 0);
            for (int64_t i = 0; i < n; ++i) {
                // Compute flat offset into source storage
                int64_t src_offset = 0;
                for (int d = 0; d < ndim; ++d)
                    src_offset += idx[d] * src_strides[d];
                dst_ptr[i] = src_base[src_offset];

                // Increment multi-dim index
                for (int d = ndim - 1; d >= 0; --d) {
                    if (++idx[d] < src_sizes[d]) break;
                    idx[d] = 0;
                }
            }
            novatorch::flushNovaBuffer(self);
        }
        return self;
    }

    if (!src_nova && self_nova) {
        // CPU -> Nova: memcpy + flush
        auto* alloc = novatorch::getNovaAllocation(self);
        std::memcpy(alloc->mapped_ptr, src.data_ptr(), tensorBytes(src));
        novatorch::flushNovaBuffer(self);
        return self;
    }

    if (src_nova && !self_nova) {
        // Nova -> CPU: invalidate + memcpy
        novatorch::invalidateNovaBuffer(src);
        auto* alloc = novatorch::getNovaAllocation(src);
        if (src.is_contiguous() && src.storage().nbytes() >= tensorBytes(self)) {
            std::memcpy(self.data_ptr(), alloc->mapped_ptr, tensorBytes(self));
        } else {
            // Non-contiguous source: element-by-element
            const float* src_base = static_cast<const float*>(alloc->mapped_ptr);
            float* dst_ptr = static_cast<float*>(self.data_ptr());
            int64_t n = self.numel();
            auto src_strides = src.strides();
            auto src_sizes = src.sizes();
            int ndim = src.dim();
            std::vector<int64_t> idx(ndim, 0);
            for (int64_t i = 0; i < n; ++i) {
                int64_t src_offset = 0;
                for (int d2 = 0; d2 < ndim; ++d2)
                    src_offset += idx[d2] * src_strides[d2];
                dst_ptr[i] = src_base[src_offset];
                for (int d2 = ndim - 1; d2 >= 0; --d2) {
                    if (++idx[d2] < src_sizes[d2]) break;
                    idx[d2] = 0;
                }
            }
        }
        return self;
    }

    // CPU -> CPU fallback
    std::memcpy(self.data_ptr(), src.data_ptr(), tensorBytes(src));
    return self;
}

// ---------------------------------------------------------------------------
// add_.Tensor  --  self = self + alpha * other
// ---------------------------------------------------------------------------

at::Tensor& nova_add_inplace(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {

    // Scalar broadcast: handle via mapped memory
    if (other.numel() == 1) {
        float val = alpha.toFloat() * other.item<float>();
        novatorch::invalidateNovaBuffer(self);
        float* ptr = static_cast<float*>(
            novatorch::getNovaAllocation(self)->mapped_ptr);
        int64_t n = self.numel();
        for (int64_t i = 0; i < n; ++i)
            ptr[i] += val;
        novatorch::flushNovaBuffer(self);
        return self;
    }

    auto other_c = other.is_contiguous() ? other : other.contiguous();
    if (other_c.sizes() != self.sizes())
        other_c = other_c.expand(self.sizes()).contiguous();

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    auto* alloc_self = novatorch::getNovaAllocation(self);
    auto* alloc_other = novatorch::getNovaAllocation(other_c);

    // Use elementwise_add shader: out = A + alpha * B
    // Here: A = self, B = other, Out = self (in-place)
    PCAddSub pc{numel, alpha.toFloat()};

    VkBuffer bufs[3] = {
        alloc_self->buffer, alloc_other->buffer, alloc_self->buffer};
    VkDeviceSize sizes[3] = {
        alloc_self->size, alloc_other->size, alloc_self->size};

    dispatchCompute("elementwise_add", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize));
    return self;
}

// ---------------------------------------------------------------------------
// sub_.Tensor  --  self = self - alpha * other
// ---------------------------------------------------------------------------

at::Tensor& nova_sub_inplace(
    at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha) {

    if (other.numel() == 1) {
        float val = alpha.toFloat() * other.item<float>();
        novatorch::invalidateNovaBuffer(self);
        float* ptr = static_cast<float*>(
            novatorch::getNovaAllocation(self)->mapped_ptr);
        int64_t n = self.numel();
        for (int64_t i = 0; i < n; ++i)
            ptr[i] -= val;
        novatorch::flushNovaBuffer(self);
        return self;
    }

    auto other_c = other.is_contiguous() ? other : other.contiguous();
    if (other_c.sizes() != self.sizes())
        other_c = other_c.expand(self.sizes()).contiguous();

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    auto* alloc_self = novatorch::getNovaAllocation(self);
    auto* alloc_other = novatorch::getNovaAllocation(other_c);

    PCAddSub pc{numel, alpha.toFloat()};

    VkBuffer bufs[3] = {
        alloc_self->buffer, alloc_other->buffer, alloc_self->buffer};
    VkDeviceSize sizes[3] = {
        alloc_self->size, alloc_other->size, alloc_self->size};

    dispatchCompute("elementwise_sub", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize));
    return self;
}

// ---------------------------------------------------------------------------
// mul_.Tensor  --  self = self * other
// ---------------------------------------------------------------------------

at::Tensor& nova_mul_inplace_tensor(
    at::Tensor& self,
    const at::Tensor& other) {

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    // Scalar tensor (0-dim or numel==1): use mul_scalar shader (1 buffer)
    if (other.numel() == 1) {
        float scalar_val = other.item<float>();
        auto* alloc_self = novatorch::getNovaAllocation(self);
        PCNumelValue pc{numel, scalar_val};
        VkBuffer bufs[1] = {alloc_self->buffer};
        VkDeviceSize sizes_arr[1] = {alloc_self->size};
        dispatchCompute("mul_scalar", 1, sizeof(pc), &pc, bufs, sizes_arr,
                        divRoundUp(numel, kWorkgroupSize));
        return self;
    }

    auto other_c = other.is_contiguous() ? other : other.contiguous();
    if (other_c.sizes() != self.sizes())
        other_c = other_c.expand(self.sizes()).contiguous();

    auto* alloc_self = novatorch::getNovaAllocation(self);
    auto* alloc_other = novatorch::getNovaAllocation(other_c);

    PCNumel pc{numel};

    VkBuffer bufs[3] = {
        alloc_self->buffer, alloc_other->buffer, alloc_self->buffer};
    VkDeviceSize sizes[3] = {
        alloc_self->size, alloc_other->size, alloc_self->size};

    dispatchCompute("elementwise_mul", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize));
    return self;
}

// ---------------------------------------------------------------------------
// mul_.Scalar  --  self = self * scalar (GPU shader)
// ---------------------------------------------------------------------------

at::Tensor& nova_mul_inplace_scalar(
    at::Tensor& self,
    const at::Scalar& other) {

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    auto* alloc_self = novatorch::getNovaAllocation(self);

    PCNumelValue pc{numel, other.toFloat()};

    VkBuffer bufs[1] = {alloc_self->buffer};
    VkDeviceSize sizes_arr[1] = {alloc_self->size};

    dispatchCompute("mul_scalar", 1, sizeof(pc), &pc, bufs, sizes_arr,
                    divRoundUp(numel, kWorkgroupSize));
    return self;
}

// ---------------------------------------------------------------------------
// div_.Tensor  --  self = self / other
// ---------------------------------------------------------------------------

at::Tensor& nova_div_inplace_tensor(
    at::Tensor& self,
    const at::Tensor& other) {

    if (other.numel() == 1) {
        float val = other.item<float>();
        float inv = 1.0f / val;
        novatorch::invalidateNovaBuffer(self);
        float* ptr = static_cast<float*>(
            novatorch::getNovaAllocation(self)->mapped_ptr);
        int64_t n = self.numel();
        for (int64_t i = 0; i < n; ++i)
            ptr[i] *= inv;
        novatorch::flushNovaBuffer(self);
        return self;
    }

    auto other_c = other.is_contiguous() ? other : other.contiguous();
    if (other_c.sizes() != self.sizes())
        other_c = other_c.expand(self.sizes()).contiguous();

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    auto* alloc_self = novatorch::getNovaAllocation(self);
    auto* alloc_other = novatorch::getNovaAllocation(other_c);

    PCNumel pc{numel};

    VkBuffer bufs[3] = {
        alloc_self->buffer, alloc_other->buffer, alloc_self->buffer};
    VkDeviceSize sizes[3] = {
        alloc_self->size, alloc_other->size, alloc_self->size};

    dispatchCompute("elementwise_div", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize));
    return self;
}

// ---------------------------------------------------------------------------
// addcmul_  --  self = self + value * tensor1 * tensor2
// ---------------------------------------------------------------------------

at::Tensor& nova_addcmul_inplace(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {

    TORCH_CHECK(
        self.sizes() == tensor1.sizes() &&
        self.sizes() == tensor2.sizes(),
        "nova_addcmul_inplace: mismatched sizes");

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    auto* alloc_self = novatorch::getNovaAllocation(self);
    auto* alloc_t1 = novatorch::getNovaAllocation(tensor1);
    auto* alloc_t2 = novatorch::getNovaAllocation(tensor2);

    PCNumelValue pc{numel, value.toFloat()};

    VkBuffer bufs[3] = {
        alloc_self->buffer, alloc_t1->buffer, alloc_t2->buffer};
    VkDeviceSize sizes_arr[3] = {
        alloc_self->size, alloc_t1->size, alloc_t2->size};

    dispatchCompute("addcmul", 3, sizeof(pc), &pc, bufs, sizes_arr,
                    divRoundUp(numel, kWorkgroupSize));
    return self;
}

// ---------------------------------------------------------------------------
// addcdiv_  --  self = self + value * tensor1 / tensor2
// ---------------------------------------------------------------------------

at::Tensor& nova_addcdiv_inplace(
    at::Tensor& self,
    const at::Tensor& tensor1,
    const at::Tensor& tensor2,
    const at::Scalar& value) {

    TORCH_CHECK(
        self.sizes() == tensor1.sizes() &&
        self.sizes() == tensor2.sizes(),
        "nova_addcdiv_inplace: mismatched sizes");

    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    auto* alloc_self = novatorch::getNovaAllocation(self);
    auto* alloc_t1 = novatorch::getNovaAllocation(tensor1);
    auto* alloc_t2 = novatorch::getNovaAllocation(tensor2);

    PCNumelValue pc{numel, value.toFloat()};

    VkBuffer bufs[3] = {
        alloc_self->buffer, alloc_t1->buffer, alloc_t2->buffer};
    VkDeviceSize sizes_arr[3] = {
        alloc_self->size, alloc_t1->size, alloc_t2->size};

    dispatchCompute("addcdiv", 3, sizeof(pc), &pc, bufs, sizes_arr,
                    divRoundUp(numel, kWorkgroupSize));
    return self;
}
