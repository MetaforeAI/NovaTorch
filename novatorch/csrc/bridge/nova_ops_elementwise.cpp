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
    // Scalar broadcast: use add.Scalar path
    if (other.numel() == 1) {
        return nova_add_scalar(self, other.item(), alpha);
    }
    if (self.numel() == 1) {
        return nova_add_scalar(other, self.item(), alpha);
    }

    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto other_c = other.is_contiguous() ? other : other.contiguous();

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
                    divRoundUp(numel, kWorkgroupSize));
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
    // Scalar broadcast
    if (other.numel() == 1) {
        float val = alpha.toFloat() * other.item<float>();
        return nova_add_scalar(self, at::Scalar(-val), at::Scalar(1.0));
    }

    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto other_c = other.is_contiguous() ? other : other.contiguous();

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
                    divRoundUp(numel, kWorkgroupSize));
    return output;
}

// ---------------------------------------------------------------------------
// mul.Tensor  —  out = self * other
// ---------------------------------------------------------------------------

at::Tensor nova_mul_tensor(
    const at::Tensor& self,
    const at::Tensor& other)
{
    // Short-circuit: if either operand is a scalar (numel==1), use mul_scalar
    if (other.numel() == 1) {
        return nova_mul_scalar(self, other.item());
    }
    if (self.numel() == 1) {
        return nova_mul_scalar(other, self.item());
    }

    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto other_c = other.is_contiguous() ? other : other.contiguous();

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
                    divRoundUp(numel, kWorkgroupSize));
    return output;
}

// ---------------------------------------------------------------------------
// div.Tensor  —  out = self / other
// ---------------------------------------------------------------------------

at::Tensor nova_div_tensor(
    const at::Tensor& self,
    const at::Tensor& other)
{
    // Scalar broadcast: div by scalar = mul by 1/scalar
    if (other.numel() == 1) {
        float val = other.item<float>();
        return nova_mul_scalar(self, at::Scalar(1.0f / val));
    }
    if (self.numel() == 1) {
        // scalar / tensor — not easily optimized, fall through to expand
    }

    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto other_c = other.is_contiguous() ? other : other.contiguous();

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
                    divRoundUp(numel, kWorkgroupSize));
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
                    divRoundUp(numel, kWorkgroupSize));
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
                    divRoundUp(numel, kWorkgroupSize));
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
