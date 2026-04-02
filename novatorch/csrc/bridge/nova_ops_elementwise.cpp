#include "nova_ops.h"

// ---------------------------------------------------------------------------
// Push-constant structs — must match the GLSL layout(push_constant) exactly.
// ---------------------------------------------------------------------------

namespace {

/// elementwise_add.comp / elementwise_sub.comp: { uint numel; float alpha; }
struct PCAddSub {
    uint32_t numel;
    float alpha;
};

/// elementwise_mul.comp / elementwise_div.comp / elementwise_neg.comp
struct PCNumel {
    uint32_t numel;
};

constexpr uint32_t kWorkgroupSize = 256;

uint32_t divRoundUp(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

/// Ensure a tensor is on the Nova device. If it's CPU, move it.
/// This handles the rsub/rtruediv case where PyTorch wraps a scalar
/// as a CPU tensor and dispatches to our PrivateUse1 kernel.
at::Tensor ensureNova(const at::Tensor& t) {
    if (t.device().type() == c10::DeviceType::PrivateUse1) return t;
    return t.to(c10::Device(c10::DeviceType::PrivateUse1, 0));
}

} // anonymous namespace

// Forward declarations for scalar variants (defined below)
at::Tensor nova_mul_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_add_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);

// ---------------------------------------------------------------------------
// add.Tensor  —  out = self + alpha * other
// ---------------------------------------------------------------------------

at::Tensor nova_add_tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha)
{
    auto self_n = ensureNova(self);
    auto other_n = ensureNova(other);

    // Scalar broadcast: only for true 0-dim scalars (not [1] or [1,1])
    if (other_n.dim() == 0) {
        return nova_add_scalar(self_n, other_n.item(), alpha);
    }
    if (self_n.dim() == 0) {
        return nova_add_scalar(other_n, self_n.item(), alpha);
    }

    auto self_c = self_n.is_contiguous() ? self_n : self_n.contiguous();
    auto other_c = other_n.is_contiguous() ? other_n : other_n.contiguous();

    auto out_sizes = at::infer_size(self_c.sizes(), other_c.sizes());
    if (self_c.sizes() != out_sizes)
        self_c = self_c.expand(out_sizes).contiguous();
    if (other_c.sizes() != out_sizes)
        other_c = other_c.expand(out_sizes).contiguous();

    at::Tensor output = at::empty(out_sizes, self_c.options());
    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return output;

    auto* alloc_a = novatorch::getNovaAllocation(self_c);
    auto* alloc_b = novatorch::getNovaAllocation(other_c);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    PCAddSub pc{numel, alpha.toFloat()};

    VkBuffer bufs[3] = {alloc_a->buffer, alloc_b->buffer, alloc_out->buffer};
    VkDeviceSize sizes[3] = {alloc_a->size, alloc_b->size, alloc_out->size};

    dispatchCompute("elementwise_add", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize), 1, 1,
                    {self_c, other_c, output});
    return output;
}

// ---------------------------------------------------------------------------
// sub.Tensor  —  out = self - alpha * other
// ---------------------------------------------------------------------------

at::Tensor nova_sub_tensor(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha)
{
    auto self_n = ensureNova(self);
    auto other_n = ensureNova(other);

    // Scalar broadcast: only for true 0-dim scalars
    if (other_n.dim() == 0) {
        float val = alpha.toFloat() * other_n.item<float>();
        return nova_add_scalar(self_n, at::Scalar(-val), at::Scalar(1.0));
    }
    if (self_n.dim() == 0) {
        float s = self_n.item<float>();
        auto neg_scaled = nova_mul_scalar(other_n, at::Scalar(-alpha.toFloat()));
        return nova_add_scalar(neg_scaled, at::Scalar(s), at::Scalar(1.0));
    }

    auto self_c = self_n.is_contiguous() ? self_n : self_n.contiguous();
    auto other_c = other_n.is_contiguous() ? other_n : other_n.contiguous();

    auto out_sizes = at::infer_size(self_c.sizes(), other_c.sizes());
    if (self_c.sizes() != out_sizes)
        self_c = self_c.expand(out_sizes).contiguous();
    if (other_c.sizes() != out_sizes)
        other_c = other_c.expand(out_sizes).contiguous();

    at::Tensor output = at::empty(out_sizes, self_c.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());
    if (numel == 0) return output;

    auto* alloc_a = novatorch::getNovaAllocation(self_c);
    auto* alloc_b = novatorch::getNovaAllocation(other_c);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    PCAddSub pc{numel, alpha.toFloat()};

    VkBuffer bufs[3] = {alloc_a->buffer, alloc_b->buffer, alloc_out->buffer};
    VkDeviceSize sizes[3] = {alloc_a->size, alloc_b->size, alloc_out->size};

    dispatchCompute("elementwise_sub", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize), 1, 1,
                    {self_c, other_c, output});
    return output;
}

// ---------------------------------------------------------------------------
// mul.Tensor  —  out = self * other
// ---------------------------------------------------------------------------

at::Tensor nova_mul_tensor(
    const at::Tensor& self,
    const at::Tensor& other)
{
    auto self_n = ensureNova(self);
    auto other_n = ensureNova(other);

    // Only use scalar shortcut for true 0-dim scalars, not [1] or [1,1] tensors.
    // Using numel()==1 here breaks broadcasting: [1]*[1,1] should yield [1,1].
    if (other_n.dim() == 0) {
        return nova_mul_scalar(self_n, other_n.item());
    }
    if (self_n.dim() == 0) {
        return nova_mul_scalar(other_n, self_n.item());
    }

    auto self_c = self_n.is_contiguous() ? self_n : self_n.contiguous();
    auto other_c = other_n.is_contiguous() ? other_n : other_n.contiguous();

    auto out_sizes = at::infer_size(self_c.sizes(), other_c.sizes());

    if (self_c.sizes() != out_sizes)
        self_c = self_c.expand(out_sizes).contiguous();
    if (other_c.sizes() != out_sizes)
        other_c = other_c.expand(out_sizes).contiguous();

    at::Tensor output = at::empty(out_sizes, self_c.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());
    if (numel == 0) return output;

    auto* alloc_a = novatorch::getNovaAllocation(self_c);
    auto* alloc_b = novatorch::getNovaAllocation(other_c);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    PCNumel pc{numel};

    VkBuffer bufs[3] = {alloc_a->buffer, alloc_b->buffer, alloc_out->buffer};
    VkDeviceSize sizes[3] = {alloc_a->size, alloc_b->size, alloc_out->size};

    dispatchCompute("elementwise_mul", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize), 1, 1,
                    {self_c, other_c, output});
    return output;
}

// ---------------------------------------------------------------------------
// div.Tensor  —  out = self / other
// ---------------------------------------------------------------------------

at::Tensor nova_div_tensor(
    const at::Tensor& self,
    const at::Tensor& other)
{
    auto self_n = ensureNova(self);
    auto other_n = ensureNova(other);

    if (other_n.dim() == 0) {
        float val = other_n.item<float>();
        return nova_mul_scalar(self_n, at::Scalar(1.0f / val));
    }
    if (self_n.dim() == 0) {
        // scalar / tensor — fall through to expand path
    }

    auto self_c = self_n.is_contiguous() ? self_n : self_n.contiguous();
    auto other_c = other_n.is_contiguous() ? other_n : other_n.contiguous();

    auto out_sizes = at::infer_size(self_c.sizes(), other_c.sizes());
    if (self_c.sizes() != out_sizes)
        self_c = self_c.expand(out_sizes).contiguous();
    if (other_c.sizes() != out_sizes)
        other_c = other_c.expand(out_sizes).contiguous();

    at::Tensor output = at::empty(out_sizes, self_c.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());
    if (numel == 0) return output;

    auto* alloc_a = novatorch::getNovaAllocation(self_c);
    auto* alloc_b = novatorch::getNovaAllocation(other_c);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    PCNumel pc{numel};

    VkBuffer bufs[3] = {alloc_a->buffer, alloc_b->buffer, alloc_out->buffer};
    VkDeviceSize sizes[3] = {alloc_a->size, alloc_b->size, alloc_out->size};

    dispatchCompute("elementwise_div", 3, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize), 1, 1,
                    {self_c, other_c, output});
    return output;
}

// ---------------------------------------------------------------------------
// mul.Scalar  —  out = self * scalar
// ---------------------------------------------------------------------------

struct PCMulScalar {
    uint32_t numel;
    float value;
};

at::Tensor nova_mul_scalar(
    const at::Tensor& self,
    const at::Scalar& other)
{
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    at::Tensor output = at::empty(self_c.sizes(), self_c.options());
    uint32_t numel = static_cast<uint32_t>(output.numel());
    if (numel == 0) return output;

    auto* alloc_a = novatorch::getNovaAllocation(self_c);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    // Copy self to output first, then apply scalar mul in-place
    // Actually, we need a shader that does out = a * scalar
    // We can use mul_scalar shader but it works in-place on 1 buffer.
    // Approach: copy self to output, then mul_scalar on output.
    output.copy_(self_c);
    PCMulScalar pc{numel, other.toFloat()};
    VkBuffer bufs[1] = {alloc_out->buffer};
    VkDeviceSize sizes[1] = {alloc_out->size};
    dispatchCompute("mul_scalar", 1, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize), 1, 1,
                    {output});
    return output;
}

// ---------------------------------------------------------------------------
// add.Scalar  —  out = self + other
// ---------------------------------------------------------------------------

at::Tensor nova_add_scalar(
    const at::Tensor& self,
    const at::Scalar& other,
    const at::Scalar& alpha)
{
    // self + alpha * other — implemented via staging transfers
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto output = at::empty(self_c.sizes(), self_c.options());
    int64_t n = output.numel();
    if (n == 0) return output;

    float val = alpha.toFloat() * other.toFloat();
    novatorch::withStagingRead(self_c, [&](const void* src_raw, size_t) {
        const float* src = static_cast<const float*>(src_raw);
        novatorch::withStagingWrite(output, [&](void* dst_raw, size_t) {
            float* dst = static_cast<float*>(dst_raw);
            for (int64_t i = 0; i < n; ++i)
                dst[i] = src[i] + val;
        });
    });
    return output;
}

// ---------------------------------------------------------------------------
// neg  —  out = -self
// ---------------------------------------------------------------------------

at::Tensor nova_neg(const at::Tensor& self) {
    at::Tensor output = at::empty(self.sizes(), self.options());
    uint32_t numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return output;

    auto* alloc_a = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    PCNumel pc{numel};

    VkBuffer bufs[2] = {alloc_a->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {alloc_a->size, alloc_out->size};

    dispatchCompute("elementwise_neg", 2, sizeof(pc), &pc, bufs, sizes,
                    divRoundUp(numel, kWorkgroupSize), 1, 1,
                    {self, output});
    return output;
}

// ---------------------------------------------------------------------------
// .out variants
// ---------------------------------------------------------------------------

at::Tensor& nova_add_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
    auto result = nova_add_tensor(self, other, alpha);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_sub_out(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Scalar& alpha,
    at::Tensor& out) {
    auto result = nova_sub_tensor(self, other, alpha);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_mul_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
    auto result = nova_mul_tensor(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_div_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
    auto result = nova_div_tensor(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_neg_out(
    const at::Tensor& self,
    at::Tensor& out) {
    auto result = nova_neg(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}
