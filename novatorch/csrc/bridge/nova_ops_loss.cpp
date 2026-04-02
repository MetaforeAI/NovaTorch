#include "nova_ops.h"
#include "nova_storage.h"
#include <cmath>

// ---------------------------------------------------------------------------
// Push-constant structs -- must match the corresponding .comp shaders
// ---------------------------------------------------------------------------

namespace {

struct PCSoftmax {
    uint32_t outer_size;
    uint32_t dim_size;
    uint32_t inner_size;
};

struct PCNllLoss {
    uint32_t batch_size;
    uint32_t num_classes;
    int32_t  reduction;     // 0=none, 1=mean, 2=sum
    int32_t  ignore_index;
};

static_assert(sizeof(PCSoftmax) == 12, "PCSoftmax must be 12 bytes");
static_assert(sizeof(PCNllLoss) == 16, "PCNllLoss must be 16 bytes");

constexpr uint32_t WG_SIZE = 256;

uint32_t divUp(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

// Compute outer_size, dim_size, inner_size for softmax along `dim`.
void compute_softmax_dims(
    const at::Tensor& self, int64_t dim,
    uint32_t& outer_size, uint32_t& dim_size, uint32_t& inner_size) {

    const auto ndim = self.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim,
        "softmax dim out of range");

    outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= static_cast<uint32_t>(self.size(i));
    }
    dim_size = static_cast<uint32_t>(self.size(dim));
    inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= static_cast<uint32_t>(self.size(i));
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// _softmax
// ---------------------------------------------------------------------------

at::Tensor nova_softmax(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float) {

    TORCH_CHECK(!half_to_float,
        "nova_softmax: half_to_float not supported");
    TORCH_CHECK(self.is_contiguous(),
        "nova_softmax: input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_softmax: only float32 supported");

    if (self.numel() == 0) return at::empty_like(self);

    uint32_t outer_size, dim_size, inner_size;
    compute_softmax_dims(self, dim, outer_size, dim_size, inner_size);

    TORCH_CHECK(dim_size <= WG_SIZE,
        "nova_softmax: dim_size (", dim_size,
        ") must be <= ", WG_SIZE, " for initial implementation");

    auto output = at::empty_like(self);

    auto* alloc_in  = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[2]     = {alloc_in->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    PCSoftmax pc{outer_size, dim_size, inner_size};

    // One workgroup per row
    uint32_t num_rows = outer_size * inner_size;

    dispatchCompute("softmax", 2, sizeof(pc), &pc, bufs, sizes, num_rows, 1, 1,
                    {self, output});
    return output;
}

// ---------------------------------------------------------------------------
// _log_softmax
// ---------------------------------------------------------------------------

at::Tensor nova_log_softmax(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float) {

    TORCH_CHECK(!half_to_float,
        "nova_log_softmax: half_to_float not supported");
    TORCH_CHECK(self.is_contiguous(),
        "nova_log_softmax: input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_log_softmax: only float32 supported");

    if (self.numel() == 0) return at::empty_like(self);

    uint32_t outer_size, dim_size, inner_size;
    compute_softmax_dims(self, dim, outer_size, dim_size, inner_size);

    TORCH_CHECK(dim_size <= WG_SIZE,
        "nova_log_softmax: dim_size (", dim_size,
        ") must be <= ", WG_SIZE, " for initial implementation");

    auto output = at::empty_like(self);

    auto* alloc_in  = novatorch::getNovaAllocation(self);
    auto* alloc_out = novatorch::getNovaAllocation(output);

    VkBuffer bufs[2]     = {alloc_in->buffer, alloc_out->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(alloc_in->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    PCSoftmax pc{outer_size, dim_size, inner_size};
    uint32_t num_rows = outer_size * inner_size;

    dispatchCompute("log_softmax", 2, sizeof(pc), &pc, bufs, sizes, num_rows,
                    1, 1, {self, output});
    return output;
}

// ---------------------------------------------------------------------------
// nll_loss_forward
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor> nova_nll_loss_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index) {

    TORCH_CHECK(!weight.has_value(),
        "nova_nll_loss_forward: class weights not yet supported");
    TORCH_CHECK(self.is_contiguous(),
        "nova_nll_loss_forward: input must be contiguous");
    TORCH_CHECK(target.is_contiguous(),
        "nova_nll_loss_forward: target must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_nll_loss_forward: only float32 input supported");
    TORCH_CHECK(self.dim() == 2,
        "nova_nll_loss_forward: input must be 2D (batch_size x num_classes)");
    TORCH_CHECK(target.dim() == 1,
        "nova_nll_loss_forward: target must be 1D (batch_size)");

    const auto batch_size  = static_cast<uint32_t>(self.size(0));
    const auto num_classes = static_cast<uint32_t>(self.size(1));

    // Output shape depends on reduction mode
    at::Tensor output;
    if (reduction == 0) {
        // none: per-element loss
        output = at::empty({static_cast<int64_t>(batch_size)}, self.options());
    } else {
        // mean or sum: scalar
        output = at::empty({}, self.options());
    }

    // total_weight is a scalar tensor (required by PyTorch API)
    at::Tensor total_weight = at::empty({}, self.options());

    if (batch_size == 0) {
        return std::make_tuple(output, total_weight);
    }

    auto* alloc_in     = novatorch::getNovaAllocation(self);
    auto* alloc_target = novatorch::getNovaAllocation(target);
    auto* alloc_out    = novatorch::getNovaAllocation(output);

    VkBuffer bufs[3] = {
        alloc_in->buffer, alloc_target->buffer, alloc_out->buffer
    };
    VkDeviceSize buf_sizes[3] = {
        alloc_in->size, alloc_target->size, alloc_out->size
    };

    PCNllLoss pc{};
    pc.batch_size   = batch_size;
    pc.num_classes  = num_classes;
    pc.reduction    = static_cast<int32_t>(reduction);
    pc.ignore_index = static_cast<int32_t>(ignore_index);

    if (reduction == 0) {
        // No reduction: dispatch one thread per batch element
        dispatchCompute("nll_loss", 3, sizeof(pc), &pc, bufs, buf_sizes,
                        divUp(batch_size, WG_SIZE), 1, 1,
                        {self, target, output});
    } else {
        // Reduction: single workgroup
        dispatchCompute("nll_loss", 3, sizeof(pc), &pc, bufs, buf_sizes, 1,
                        1, 1, {self, target, output});
    }

    // Compute total_weight on CPU (count of non-ignored samples)
    novatorch::withStagingRead(target, [&](const void* tgt_raw, size_t) {
        const int64_t* tgt_ptr = static_cast<const int64_t*>(tgt_raw);
        float tw = 0.0f;
        for (uint32_t i = 0; i < batch_size; ++i) {
            if (tgt_ptr[i] != static_cast<int64_t>(ignore_index)) {
                tw += 1.0f;
            }
        }
        novatorch::withStagingWrite(total_weight, [&](void* tw_raw, size_t) {
            *static_cast<float*>(tw_raw) = tw;
        });
    });

    return std::make_tuple(output, total_weight);
}

// ---------------------------------------------------------------------------
// nll_loss_backward.grad_input  (CPU via mapped memory)
// ---------------------------------------------------------------------------

at::Tensor& nova_nll_loss_backward_grad_input(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const std::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight,
    at::Tensor& grad_input) {

    TORCH_CHECK(self.dim() == 2,
        "nova_nll_loss_backward: input must be 2D");
    TORCH_CHECK(target.dim() == 1,
        "nova_nll_loss_backward: target must be 1D");
    TORCH_CHECK(self.scalar_type() == at::ScalarType::Float,
        "nova_nll_loss_backward: only float32 supported");

    const int64_t batch_size  = self.size(0);
    const int64_t num_classes = self.size(1);

    // Resize and zero grad_input
    grad_input.resize_as_(self);
    grad_input.zero_();

    // Download inputs to CPU-side buffers
    auto go_nbytes = static_cast<size_t>(
        grad_output.numel() * grad_output.element_size());
    auto tgt_nbytes = static_cast<size_t>(
        target.numel() * target.element_size());

    std::vector<uint8_t> go_buf(go_nbytes);
    std::vector<uint8_t> tgt_buf(tgt_nbytes);
    float tw = 0.0f;

    novatorch::downloadFromDevice(grad_output, go_buf.data(), go_nbytes);
    novatorch::downloadFromDevice(target, tgt_buf.data(), tgt_nbytes);
    novatorch::downloadFromDevice(total_weight, &tw, sizeof(float));

    const float* grad_out_ptr = reinterpret_cast<const float*>(go_buf.data());
    const int64_t* target_ptr = reinterpret_cast<const int64_t*>(tgt_buf.data());

    // Get weight data if present
    std::vector<uint8_t> w_buf;
    const float* weight_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        auto w_nbytes = static_cast<size_t>(
            weight->numel() * weight->element_size());
        w_buf.resize(w_nbytes);
        novatorch::downloadFromDevice(*weight, w_buf.data(), w_nbytes);
        weight_ptr = reinterpret_cast<const float*>(w_buf.data());
    }

    // Compute grad_input on CPU then upload
    novatorch::withStagingReadWrite(grad_input, [&](void* gi_raw, size_t) {
        float* gi_ptr = static_cast<float*>(gi_raw);

        for (int64_t i = 0; i < batch_size; ++i) {
            const int64_t t = target_ptr[i];
            if (t == ignore_index) continue;

            TORCH_CHECK(t >= 0 && t < num_classes,
                "Target ", t, " is out of bounds for num_classes=", num_classes);

            float w = weight_ptr ? weight_ptr[t] : 1.0f;

            if (reduction == 0) {
                // none: grad_output is per-sample
                gi_ptr[i * num_classes + t] = -grad_out_ptr[i] * w;
            } else if (reduction == 1) {
                // mean: grad_output is scalar, divide by total_weight
                if (tw > 0.0f) {
                    gi_ptr[i * num_classes + t] = -grad_out_ptr[0] * w / tw;
                }
            } else {
                // sum: grad_output is scalar
                gi_ptr[i * num_classes + t] = -grad_out_ptr[0] * w;
            }
        }
    });

    return grad_input;
}

// ---------------------------------------------------------------------------
// _log_softmax_backward_data  (CPU via mapped memory)
// ---------------------------------------------------------------------------

at::Tensor nova_log_softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {

    TORCH_CHECK(grad_output.is_contiguous(),
        "nova_log_softmax_backward: grad_output must be contiguous");
    TORCH_CHECK(output.is_contiguous(),
        "nova_log_softmax_backward: output must be contiguous");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Float,
        "nova_log_softmax_backward: only float32 supported");

    if (grad_output.numel() == 0) return at::empty_like(grad_output);

    uint32_t outer_size, dim_size, inner_size;
    compute_softmax_dims(output, dim, outer_size, dim_size, inner_size);

    auto grad_input = at::empty_like(grad_output);

    // Use staging to read grad_output and output, write grad_input
    novatorch::withStagingRead(grad_output, [&](const void* go_raw, size_t) {
    novatorch::withStagingRead(output, [&](const void* out_raw, size_t) {
    novatorch::withStagingWrite(grad_input, [&](void* gi_raw, size_t) {
        const float* go_ptr = static_cast<const float*>(go_raw);
        const float* out_ptr = static_cast<const float*>(out_raw);
        float* gi_ptr = static_cast<float*>(gi_raw);

        // For each row along the softmax dim:
        // grad_input[i] = grad_output[i] - exp(output[i]) * sum(grad_output)
        for (uint32_t o = 0; o < outer_size; ++o) {
            for (uint32_t inn = 0; inn < inner_size; ++inn) {
                // Compute sum of grad_output along dim
                float sum_go = 0.0f;
                for (uint32_t d = 0; d < dim_size; ++d) {
                    size_t idx = (o * dim_size + d) * inner_size + inn;
                    sum_go += go_ptr[idx];
                }
                // Compute grad_input
                for (uint32_t d = 0; d < dim_size; ++d) {
                    size_t idx = (o * dim_size + d) * inner_size + inn;
                    gi_ptr[idx] =
                        go_ptr[idx] - std::exp(out_ptr[idx]) * sum_go;
                }
            }
        }
    });
    });
    });

    return grad_input;
}

// ---------------------------------------------------------------------------
// .out variants
// ---------------------------------------------------------------------------

at::Tensor& nova_softmax_out(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    at::Tensor& out) {
    auto result = nova_softmax(self, dim, half_to_float);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_log_softmax_out(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    at::Tensor& out) {
    auto result = nova_log_softmax(self, dim, half_to_float);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

// ---------------------------------------------------------------------------
// _softmax_backward_data — GPU shader
// ---------------------------------------------------------------------------

at::Tensor nova_softmax_backward_data(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype) {

    auto grad_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    auto out_c = output.is_contiguous() ? output : output.contiguous();

    TORCH_CHECK(grad_c.scalar_type() == at::ScalarType::Float,
        "nova_softmax_backward: only float32 supported");

    // Normalize dim
    int64_t ndim = grad_c.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "nova_softmax_backward: invalid dim");

    // For the GPU shader, we need dim to be the last dimension.
    // If it's not, permute so that dim is last, make contiguous, dispatch, permute back.
    bool need_permute = (dim != ndim - 1);
    at::Tensor g = grad_c;
    at::Tensor o = out_c;

    if (need_permute) {
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < ndim; ++i)
            if (i != dim) perm.push_back(i);
        perm.push_back(dim);
        g = g.permute(perm).contiguous();
        o = o.permute(perm).contiguous();
    }

    // Now last dim is the softmax dim
    int64_t row_size = g.size(ndim - 1);
    int64_t num_rows = g.numel() / row_size;

    auto grad_input = at::empty_like(g);

    auto* alloc_go = novatorch::getNovaAllocation(g);
    auto* alloc_out = novatorch::getNovaAllocation(o);
    auto* alloc_gi = novatorch::getNovaAllocation(grad_input);

    VkBuffer bufs[3] = {alloc_go->buffer, alloc_out->buffer, alloc_gi->buffer};
    VkDeviceSize sizes[3] = {
        static_cast<VkDeviceSize>(alloc_go->size),
        static_cast<VkDeviceSize>(alloc_out->size),
        static_cast<VkDeviceSize>(alloc_gi->size)
    };

    struct { uint32_t num_rows; uint32_t row_size; } pc{
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(row_size)
    };

    // One workgroup per row
    dispatchCompute(
        "softmax_backward", 3, sizeof(pc), &pc, bufs, sizes,
        static_cast<uint32_t>(num_rows), 1, 1,
        {g, o, grad_input});

    if (need_permute) {
        // Inverse permute
        std::vector<int64_t> inv_perm(ndim);
        std::vector<int64_t> perm;
        for (int64_t i = 0; i < ndim; ++i)
            if (i != dim) perm.push_back(i);
        perm.push_back(dim);
        for (int64_t i = 0; i < ndim; ++i)
            inv_perm[perm[i]] = i;
        grad_input = grad_input.permute(inv_perm).contiguous();
    }

    return grad_input;
}

at::Tensor& nova_softmax_backward_data_out(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    int64_t dim,
    at::ScalarType input_dtype,
    at::Tensor& grad_input) {
    auto result = nova_softmax_backward_data(grad_output, output, dim, input_dtype);
    grad_input.resize_as_(result);
    grad_input.copy_(result);
    return grad_input;
}
