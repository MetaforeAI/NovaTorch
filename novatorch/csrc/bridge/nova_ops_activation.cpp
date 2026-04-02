#include "nova_ops.h"

// ---------------------------------------------------------------------------
// Push constant layouts -- must match the corresponding .comp shaders
// ---------------------------------------------------------------------------

struct ActivationPC {
    uint32_t numel;
};

struct ThresholdBackwardPC {
    uint32_t numel;
    float threshold;
};

static_assert(sizeof(ActivationPC) == 4, "ActivationPC must be 4 bytes");
static_assert(
    sizeof(ThresholdBackwardPC) == 8,
    "ThresholdBackwardPC must be 8 bytes");

// ---------------------------------------------------------------------------
// Shared helper -- elementwise 2-buffer activation dispatch
// ---------------------------------------------------------------------------

static at::Tensor dispatch_elementwise_activation(
    const at::Tensor& self,
    const std::string& kernel_name,
    const char* op_name) {

    TORCH_CHECK(self.is_contiguous(),
        op_name, ": input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        op_name, ": only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) {
        return at::empty_like(self);
    }

    auto output = at::empty_like(self);

    VkBuffer buf_in = novatorch::getNovaBuffer(self);
    VkBuffer buf_out = novatorch::getNovaBuffer(output);

    auto* alloc_in = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[2] = {buf_in, buf_out};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    ActivationPC pc{numel};

    constexpr uint32_t WG_SIZE = 256;
    uint32_t groups = (numel + WG_SIZE - 1) / WG_SIZE;

    novatorch::flushNovaBuffer(self);
    dispatchCompute(kernel_name, 2, sizeof(pc), &pc, bufs, sizes, groups);

    return output;
}

// ---------------------------------------------------------------------------
// nova_relu
// ---------------------------------------------------------------------------

at::Tensor nova_relu(const at::Tensor& self) {
    return dispatch_elementwise_activation(self, "activation_relu", "nova_relu");
}

// ---------------------------------------------------------------------------
// nova_sigmoid
// ---------------------------------------------------------------------------

at::Tensor nova_sigmoid(const at::Tensor& self) {
    return dispatch_elementwise_activation(
        self, "activation_sigmoid", "nova_sigmoid");
}

// ---------------------------------------------------------------------------
// nova_tanh
// ---------------------------------------------------------------------------

at::Tensor nova_tanh(const at::Tensor& self) {
    return dispatch_elementwise_activation(
        self, "activation_tanh", "nova_tanh");
}

// ---------------------------------------------------------------------------
// nova_threshold_backward
// ---------------------------------------------------------------------------

at::Tensor nova_threshold_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold) {

    TORCH_CHECK(grad_output.is_contiguous(),
        "nova_threshold_backward: grad_output must be contiguous");
    TORCH_CHECK(self.is_contiguous(),
        "nova_threshold_backward: self must be contiguous");
    TORCH_CHECK(
        grad_output.scalar_type() == at::ScalarType::Float,
        "nova_threshold_backward: only float32 supported");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_threshold_backward: only float32 supported");
    TORCH_CHECK(
        grad_output.numel() == self.numel(),
        "nova_threshold_backward: grad_output and self must have "
        "the same number of elements");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) {
        return at::empty_like(grad_output);
    }

    auto grad_input = at::empty_like(grad_output);

    VkBuffer buf_grad_out = novatorch::getNovaBuffer(grad_output);
    VkBuffer buf_self = novatorch::getNovaBuffer(self);
    VkBuffer buf_grad_in = novatorch::getNovaBuffer(grad_input);

    auto* alloc_grad_out = novatorch::getNovaAllocation(grad_output);
    auto* alloc_self = novatorch::getNovaAllocation(self);
    auto* alloc_grad_in = novatorch::getNovaAllocation(grad_input);

    VkBuffer bufs[3] = {buf_grad_out, buf_self, buf_grad_in};
    VkDeviceSize sizes[3] = {
        static_cast<VkDeviceSize>(alloc_grad_out->size),
        static_cast<VkDeviceSize>(alloc_self->size),
        static_cast<VkDeviceSize>(alloc_grad_in->size)
    };

    ThresholdBackwardPC pc{numel, threshold.toFloat()};

    constexpr uint32_t WG_SIZE = 256;
    uint32_t groups = (numel + WG_SIZE - 1) / WG_SIZE;

    novatorch::flushNovaBuffer(grad_output);
    novatorch::flushNovaBuffer(self);

    dispatchCompute(
        "threshold_backward", 3, sizeof(pc), &pc, bufs, sizes, groups);

    return grad_input;
}

// ---------------------------------------------------------------------------
// .out variants
// ---------------------------------------------------------------------------

at::Tensor& nova_relu_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_relu(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_sigmoid_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_sigmoid(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_tanh_out(const at::Tensor& self, at::Tensor& out) {
    auto result = nova_tanh(self);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_threshold_backward_out(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& threshold,
    at::Tensor& out) {
    auto result = nova_threshold_backward(grad_output, self, threshold);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}
