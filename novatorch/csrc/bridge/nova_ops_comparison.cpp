#include "nova_ops.h"

// ---------------------------------------------------------------------------
// Push-constant structs -- must match the corresponding .comp shaders
// ---------------------------------------------------------------------------

namespace {

struct PCCompare {
    uint32_t numel;
    float value;
};

struct PCNumel {
    uint32_t numel;
};

static_assert(sizeof(PCCompare) == 8, "PCCompare must be 8 bytes");
static_assert(sizeof(PCNumel) == 4, "PCNumel must be 4 bytes");

constexpr uint32_t WG_SIZE = 256;

uint32_t divUp(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

// Shared helper for comparison ops (2 buffers: in, out)
at::Tensor dispatch_comparison(
    const at::Tensor& self,
    const at::Scalar& other,
    const std::string& kernel_name,
    const char* op_name) {

    TORCH_CHECK(self.is_contiguous(),
        op_name, ": input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        op_name, ": only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return at::empty_like(self);

    // Output is float (1.0/0.0); PyTorch converts to bool at higher level
    auto output = at::empty_like(self);

    auto* alloc_in  = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[2]     = {alloc_in->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    PCCompare pc{numel, other.toFloat()};

    dispatchCompute(kernel_name, 2, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE), 1, 1,
                    {self, output});
    return output;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Scalar comparison ops
// ---------------------------------------------------------------------------

at::Tensor nova_eq_scalar(const at::Tensor& self, const at::Scalar& other) {
    return dispatch_comparison(self, other, "cmp_eq", "nova_eq_scalar");
}

at::Tensor nova_ne_scalar(const at::Tensor& self, const at::Scalar& other) {
    return dispatch_comparison(self, other, "cmp_ne", "nova_ne_scalar");
}

at::Tensor nova_gt_scalar(const at::Tensor& self, const at::Scalar& other) {
    return dispatch_comparison(self, other, "cmp_gt", "nova_gt_scalar");
}

at::Tensor nova_lt_scalar(const at::Tensor& self, const at::Scalar& other) {
    return dispatch_comparison(self, other, "cmp_lt", "nova_lt_scalar");
}

at::Tensor nova_ge_scalar(const at::Tensor& self, const at::Scalar& other) {
    return dispatch_comparison(self, other, "cmp_ge", "nova_ge_scalar");
}

at::Tensor nova_le_scalar(const at::Tensor& self, const at::Scalar& other) {
    return dispatch_comparison(self, other, "cmp_le", "nova_le_scalar");
}

// ---------------------------------------------------------------------------
// where.self -- out = cond ? self : other
// ---------------------------------------------------------------------------

at::Tensor nova_where_self(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other) {

    TORCH_CHECK(condition.sizes() == self.sizes(),
        "nova_where_self: condition and self must have same sizes");
    TORCH_CHECK(self.sizes() == other.sizes(),
        "nova_where_self: self and other must have same sizes");
    TORCH_CHECK(
        condition.is_contiguous() && self.is_contiguous() &&
        other.is_contiguous(),
        "nova_where_self: all inputs must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_where_self: only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return at::empty_like(self);

    auto output = at::empty_like(self);

    auto* alloc_cond  = novatorch::getNovaAllocation(condition);
    auto* alloc_self  = novatorch::getNovaAllocation(self);
    auto* alloc_other = novatorch::getNovaAllocation(other);
    auto* alloc_out   = novatorch::getNovaAllocation(output);

    VkBuffer bufs[4] = {
        alloc_cond->buffer, alloc_self->buffer,
        alloc_other->buffer, alloc_out->buffer
    };
    VkDeviceSize sizes[4] = {
        alloc_cond->size, alloc_self->size,
        alloc_other->size, alloc_out->size
    };

    PCNumel pc{numel};

    dispatchCompute("where", 4, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE), 1, 1,
                    {condition, self, other, output});
    return output;
}

// ---------------------------------------------------------------------------
// .out variants for comparison ops
// ---------------------------------------------------------------------------

at::Tensor& nova_eq_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
    auto result = nova_eq_scalar(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_ne_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
    auto result = nova_ne_scalar(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_gt_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
    auto result = nova_gt_scalar(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_lt_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
    auto result = nova_lt_scalar(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_ge_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
    auto result = nova_ge_scalar(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_le_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
    auto result = nova_le_scalar(self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_where_self_out(
    const at::Tensor& condition,
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& out) {
    auto result = nova_where_self(condition, self, other);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}
