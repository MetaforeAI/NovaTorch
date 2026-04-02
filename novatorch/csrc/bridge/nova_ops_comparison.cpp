#include "nova_ops.h"
#include "nova_batch_context.h"

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

    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_where_self: only float32 supported for self/other");

    // Broadcast all three to common shape
    auto out_sizes = at::infer_size(condition.sizes(), self.sizes());
    out_sizes = at::infer_size(out_sizes, other.sizes());

    auto cond_c = condition.expand(out_sizes).contiguous();
    auto self_c = self.expand(out_sizes).contiguous();
    auto other_c = other.expand(out_sizes).contiguous();

    const auto numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) return at::empty_like(self_c);

    auto output = at::empty_like(self_c);

    auto* alloc_cond  = novatorch::getNovaAllocation(cond_c);
    auto* alloc_self  = novatorch::getNovaAllocation(self_c);
    auto* alloc_other = novatorch::getNovaAllocation(other_c);
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
                    {cond_c, self_c, other_c, output});
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

// ---------------------------------------------------------------------------
// isnan — returns bool tensor
// ---------------------------------------------------------------------------

at::Tensor nova_isnan(const at::Tensor& self) {
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    TORCH_CHECK(self_c.scalar_type() == at::ScalarType::Float,
        "nova_isnan: only float32 supported");

    const auto numel = static_cast<uint32_t>(self_c.numel());
    if (numel == 0) {
        return at::empty_like(self_c, self_c.options().dtype(at::ScalarType::Bool));
    }

    // Compute as float (shader writes 0.0/1.0), then convert to bool
    auto float_out = at::empty_like(self_c);

    auto* alloc_in  = novatorch::getNovaAllocation(self_c);
    auto* alloc_out = novatorch::getNovaAllocation(float_out);

    VkBuffer bufs[2] = {alloc_in->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    PCNumel pc{numel};

    dispatchCompute("isnan", 2, sizeof(pc), &pc, bufs, sizes,
                    divUp(numel, WG_SIZE), 1, 1,
                    {self_c, float_out});

    // Convert float 0/1 → bool via staging
    auto bool_out = at::empty(self_c.sizes(),
        self_c.options().dtype(at::ScalarType::Bool));
    size_t f_bytes = numel * sizeof(float);
    size_t b_bytes = numel * sizeof(bool);
    std::vector<float> fbuf(numel);
    novatorch::downloadFromDevice(float_out, fbuf.data(), f_bytes);
    novatorch::withStagingWrite(bool_out, [&](void* dst, size_t) {
        bool* bptr = static_cast<bool*>(dst);
        for (uint32_t i = 0; i < numel; ++i)
            bptr[i] = (fbuf[i] != 0.0f);
    });
    return bool_out;
}

// ---------------------------------------------------------------------------
// any — reduce bool tensor to single bool
// ---------------------------------------------------------------------------

at::Tensor nova_any(const at::Tensor& self) {
    // Download to CPU, run any, upload result
    NovaBatchContext::instance().flush();
    auto self_cpu = at::empty(self.sizes(),
        self.options().device(c10::Device(c10::DeviceType::CPU)));
    novatorch::downloadFromDevice(self, self_cpu.data_ptr(),
        self.numel() * self.element_size());
    auto result_cpu = self_cpu.any();
    auto result = at::empty(result_cpu.sizes(),
        result_cpu.options().device(self.device()));
    novatorch::uploadToDevice(result, result_cpu.data_ptr(),
        result_cpu.numel() * result_cpu.element_size());
    return result;
}

at::Tensor& nova_any_all_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_any(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}
