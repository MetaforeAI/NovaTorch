#include "nova_ops.h"

#include <cmath>

// ---------------------------------------------------------------------------
// Push-constant structs -- must match the corresponding .comp shaders
// ---------------------------------------------------------------------------

namespace {

struct PCNumel {
    uint32_t numel;
};

struct PCPow {
    uint32_t numel;
    float exponent;
};

struct PCClamp {
    uint32_t numel;
    float min_val;
    float max_val;
    int32_t has_min;
    int32_t has_max;
};

static_assert(sizeof(PCNumel) == 4, "PCNumel must be 4 bytes");
static_assert(sizeof(PCPow) == 8, "PCPow must be 8 bytes");
static_assert(sizeof(PCClamp) == 20, "PCClamp must be 20 bytes");

constexpr uint32_t WG_SIZE = 256;

uint32_t divUp(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

// Shared helper for unary math ops (2 buffers: in, out)
at::Tensor dispatch_unary_math(
    const at::Tensor& self,
    const std::string& kernel_name,
    const char* op_name) {

    TORCH_CHECK(self.is_contiguous(),
        op_name, ": input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        op_name, ": only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return at::empty_like(self);

    auto output = at::empty_like(self);

    auto* alloc_in  = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[2]     = {alloc_in->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    PCNumel pc{numel};

    novatorch::flushNovaBuffer(self);
    dispatchCompute(kernel_name, 2, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE));
    return output;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Unary math ops
// ---------------------------------------------------------------------------

at::Tensor nova_exp(const at::Tensor& self) {
    return dispatch_unary_math(self, "math_exp", "nova_exp");
}

at::Tensor nova_log(const at::Tensor& self) {
    return dispatch_unary_math(self, "math_log", "nova_log");
}

at::Tensor nova_sqrt(const at::Tensor& self) {
    return dispatch_unary_math(self, "math_sqrt", "nova_sqrt");
}

at::Tensor nova_rsqrt(const at::Tensor& self) {
    return dispatch_unary_math(self, "math_rsqrt", "nova_rsqrt");
}

at::Tensor nova_abs(const at::Tensor& self) {
    return dispatch_unary_math(self, "math_abs", "nova_abs");
}

// ---------------------------------------------------------------------------
// pow.Tensor_Scalar -- out = self ^ exponent
// ---------------------------------------------------------------------------

at::Tensor nova_pow_tensor_scalar(
    const at::Tensor& self,
    const at::Scalar& exponent) {

    TORCH_CHECK(self.is_contiguous(),
        "nova_pow_tensor_scalar: input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_pow_tensor_scalar: only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return at::empty_like(self);

    auto output = at::empty_like(self);

    auto* alloc_in  = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[2]     = {alloc_in->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    PCPow pc{numel, exponent.toFloat()};

    novatorch::flushNovaBuffer(self);
    dispatchCompute("math_pow", 2, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE));
    return output;
}

// ---------------------------------------------------------------------------
// max.other / min.other -- elementwise binary
// ---------------------------------------------------------------------------

at::Tensor nova_max_other(
    const at::Tensor& self,
    const at::Tensor& other) {

    TORCH_CHECK(self.sizes() == other.sizes(),
        "nova_max_other: mismatched sizes");
    TORCH_CHECK(self.is_contiguous() && other.is_contiguous(),
        "nova_max_other: inputs must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_max_other: only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return at::empty_like(self);

    auto output = at::empty_like(self);

    auto* alloc_a   = novatorch::getNovaAllocation(self);
    auto* alloc_b   = novatorch::getNovaAllocation(other);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[3]     = {alloc_a->buffer, alloc_b->buffer, alloc_out->buffer};
    VkDeviceSize sizes[3] = {alloc_a->size, alloc_b->size, alloc_out->size};

    PCNumel pc{numel};

    novatorch::flushNovaBuffer(self);
    novatorch::flushNovaBuffer(other);
    dispatchCompute("math_max", 3, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE));
    return output;
}

at::Tensor nova_min_other(
    const at::Tensor& self,
    const at::Tensor& other) {

    TORCH_CHECK(self.sizes() == other.sizes(),
        "nova_min_other: mismatched sizes");
    TORCH_CHECK(self.is_contiguous() && other.is_contiguous(),
        "nova_min_other: inputs must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_min_other: only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return at::empty_like(self);

    auto output = at::empty_like(self);

    auto* alloc_a   = novatorch::getNovaAllocation(self);
    auto* alloc_b   = novatorch::getNovaAllocation(other);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[3]     = {alloc_a->buffer, alloc_b->buffer, alloc_out->buffer};
    VkDeviceSize sizes[3] = {alloc_a->size, alloc_b->size, alloc_out->size};

    PCNumel pc{numel};

    novatorch::flushNovaBuffer(self);
    novatorch::flushNovaBuffer(other);
    dispatchCompute("math_min", 3, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE));
    return output;
}

// ---------------------------------------------------------------------------
// clamp -- optional min/max bounds
// ---------------------------------------------------------------------------

at::Tensor nova_clamp(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min_val,
    const std::optional<at::Scalar>& max_val) {

    TORCH_CHECK(min_val.has_value() || max_val.has_value(),
        "nova_clamp: at least one of min or max must be provided");
    TORCH_CHECK(self.is_contiguous(),
        "nova_clamp: input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_clamp: only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return at::empty_like(self);

    auto output = at::empty_like(self);

    auto* alloc_in  = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[2]     = {alloc_in->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    PCClamp pc{};
    pc.numel   = numel;
    pc.min_val = min_val.has_value() ? min_val->toFloat() : 0.0f;
    pc.max_val = max_val.has_value() ? max_val->toFloat() : 0.0f;
    pc.has_min = min_val.has_value() ? 1 : 0;
    pc.has_max = max_val.has_value() ? 1 : 0;

    novatorch::flushNovaBuffer(self);
    dispatchCompute("clamp", 2, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE));
    return output;
}

// ---------------------------------------------------------------------------
// .out variants
// ---------------------------------------------------------------------------

at::Tensor& nova_exp_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_exp(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_log_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_log(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_sqrt_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_sqrt(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_rsqrt_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_rsqrt(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_abs_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_abs(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_pow_tensor_scalar_out(
    const at::Tensor& self,
    const at::Scalar& exponent,
    at::Tensor& out) {
    auto result = nova_pow_tensor_scalar(self, exponent);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_max_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
    auto result = nova_max_other(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_min_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
    auto result = nova_min_other(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_clamp_out(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min_val,
    const std::optional<at::Scalar>& max_val,
    at::Tensor& out) {
    auto result = nova_clamp(self, min_val, max_val);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

// ---------------------------------------------------------------------------
// lerp.Scalar  —  out = self + weight * (end - self)
// ---------------------------------------------------------------------------

at::Tensor nova_lerp_scalar(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Scalar& weight) {
    // lerp(a,b,w) = a + w*(b-a) = a*(1-w) + b*w
    float w = weight.toFloat();
    // Use existing ops: result = self * (1-w) + end * w
    // But we need to avoid broadcast issues, so do it via mapped memory
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto end_c = end.is_contiguous() ? end : end.contiguous();
    auto output = at::empty(self_c.sizes(), self_c.options());

    novatorch::invalidateNovaBuffer(self_c);
    novatorch::invalidateNovaBuffer(end_c);
    const float* a = static_cast<const float*>(
        novatorch::getNovaAllocation(self_c)->mapped_ptr);
    const float* b = static_cast<const float*>(
        novatorch::getNovaAllocation(end_c)->mapped_ptr);
    float* out_ptr = static_cast<float*>(
        novatorch::getNovaAllocation(output)->mapped_ptr);

    int64_t n = output.numel();
    for (int64_t i = 0; i < n; ++i)
        out_ptr[i] = a[i] + w * (b[i] - a[i]);

    novatorch::flushNovaBuffer(output);
    return output;
}

at::Tensor& nova_lerp_scalar_out(
    const at::Tensor& self,
    const at::Tensor& end,
    const at::Scalar& weight,
    at::Tensor& out) {
    auto result = nova_lerp_scalar(self, end, weight);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_lerp_scalar_inplace(
    at::Tensor& self,
    const at::Tensor& end,
    const at::Scalar& weight) {
    float w = weight.toFloat();
    auto end_c = end.is_contiguous() ? end : end.contiguous();

    novatorch::invalidateNovaBuffer(self);
    novatorch::invalidateNovaBuffer(end_c);
    float* a = static_cast<float*>(
        novatorch::getNovaAllocation(self)->mapped_ptr);
    const float* b = static_cast<const float*>(
        novatorch::getNovaAllocation(end_c)->mapped_ptr);

    int64_t n = self.numel();
    for (int64_t i = 0; i < n; ++i)
        a[i] = a[i] + w * (b[i] - a[i]);

    novatorch::flushNovaBuffer(self);
    return self;
}

// ---------------------------------------------------------------------------
// sqrt_  —  in-place sqrt (needed by Adam optimizer)
// ---------------------------------------------------------------------------

at::Tensor& nova_sqrt_inplace(at::Tensor& self) {
    novatorch::invalidateNovaBuffer(self);
    float* ptr = static_cast<float*>(
        novatorch::getNovaAllocation(self)->mapped_ptr);
    int64_t n = self.numel();
    for (int64_t i = 0; i < n; ++i)
        ptr[i] = std::sqrt(ptr[i]);
    novatorch::flushNovaBuffer(self);
    return self;
}

// ---------------------------------------------------------------------------
// div_.Scalar  —  in-place divide by scalar
// ---------------------------------------------------------------------------

at::Tensor& nova_div_inplace_scalar(
    at::Tensor& self,
    const at::Scalar& other) {
    float inv = 1.0f / other.toFloat();
    novatorch::invalidateNovaBuffer(self);
    float* ptr = static_cast<float*>(
        novatorch::getNovaAllocation(self)->mapped_ptr);
    int64_t n = self.numel();
    for (int64_t i = 0; i < n; ++i)
        ptr[i] *= inv;
    novatorch::flushNovaBuffer(self);
    return self;
}
