#include "nova_ops.h"
#include <cstring>

// ---------------------------------------------------------------------------
// nova_sum -- GPU reduction via reduce_sum compute shader
// ---------------------------------------------------------------------------

at::Tensor nova_sum(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype) {

    TORCH_CHECK(self.is_contiguous(), "nova_sum: input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_sum: only float32 supported");

    const auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) {
        return at::zeros({}, self.options());
    }

    constexpr uint32_t WG_SIZE = 256;
    uint32_t num_groups = (numel + WG_SIZE - 1) / WG_SIZE;

    auto partial = at::empty({static_cast<int64_t>(num_groups)}, self.options());

    auto* in_alloc = novatorch::getNovaAllocation(self);
    auto* out_alloc = novatorch::getNovaAllocation(partial);

    struct { uint32_t numel; } pc{numel};

    VkBuffer bufs[2] = {in_alloc->buffer, out_alloc->buffer};
    VkDeviceSize sizes[2] = {
        static_cast<VkDeviceSize>(in_alloc->size),
        static_cast<VkDeviceSize>(out_alloc->size)
    };

    novatorch::flushNovaBuffer(self);
    dispatchCompute("reduce_sum", 2, sizeof(pc), &pc, bufs, sizes, num_groups);

    // Iterative reduction until single workgroup remains
    uint32_t remaining = num_groups;
    while (remaining > 1) {
        uint32_t next_groups = (remaining + WG_SIZE - 1) / WG_SIZE;
        auto next = at::empty({static_cast<int64_t>(next_groups)}, self.options());

        auto* src_alloc = novatorch::getNovaAllocation(partial);
        auto* dst_alloc = novatorch::getNovaAllocation(next);

        struct { uint32_t numel; } iter_pc{remaining};

        VkBuffer iter_bufs[2] = {src_alloc->buffer, dst_alloc->buffer};
        VkDeviceSize iter_sizes[2] = {
            static_cast<VkDeviceSize>(src_alloc->size),
            static_cast<VkDeviceSize>(dst_alloc->size)
        };

        dispatchCompute("reduce_sum", 2, sizeof(iter_pc), &iter_pc,
                        iter_bufs, iter_sizes, next_groups);

        partial = next;
        remaining = next_groups;
    }

    // Read result from mapped memory
    novatorch::invalidateNovaBuffer(partial);
    float result = *static_cast<float*>(partial.data_ptr());

    auto output_dtype = dtype.value_or(self.scalar_type());
    auto output = at::empty({}, self.options().dtype(output_dtype));
    *static_cast<float*>(output.data_ptr()) = result;
    novatorch::flushNovaBuffer(output);

    return output;
}

// ---------------------------------------------------------------------------
// nova_mean -- sum / numel
// ---------------------------------------------------------------------------

at::Tensor nova_mean(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype) {

    TORCH_CHECK(self.is_contiguous(), "nova_mean: input must be contiguous");
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_mean: only float32 supported");
    TORCH_CHECK(self.numel() > 0, "nova_mean: input must be non-empty");

    at::Tensor sum_result = nova_sum(self, dtype);

    novatorch::invalidateNovaBuffer(sum_result);
    float sum_val = *static_cast<float*>(sum_result.data_ptr());
    float mean_val = sum_val / static_cast<float>(self.numel());

    auto output_dtype = dtype.value_or(self.scalar_type());
    auto output = at::empty({}, self.options().dtype(output_dtype));
    *static_cast<float*>(output.data_ptr()) = mean_val;
    novatorch::flushNovaBuffer(output);

    return output;
}

// ---------------------------------------------------------------------------
// sum.dim_IntList -- reduction along specified dimensions (CPU fallback)
// ---------------------------------------------------------------------------

at::Tensor nova_sum_dim_intlist(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {

    // If no dims specified, reduce all -- delegate to existing nova_sum
    if (!dim.has_value() || dim->empty()) {
        auto result = nova_sum(self, dtype);
        if (keepdim) {
            std::vector<int64_t> shape(self.dim(), 1);
            result = result.reshape(shape);
        }
        return result;
    }

    // CPU fallback: copy to CPU, compute, copy back
    auto cpu_self = self.to(at::kCPU);
    auto cpu_result = cpu_self.sum(*dim, keepdim, dtype);
    return cpu_result.to(self.device());
}

// ---------------------------------------------------------------------------
// mean.dim -- reduction along specified dimensions (CPU fallback)
// ---------------------------------------------------------------------------

at::Tensor nova_mean_dim(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {

    // If no dims specified, reduce all -- delegate to existing nova_mean
    if (!dim.has_value() || dim->empty()) {
        auto result = nova_mean(self, dtype);
        if (keepdim) {
            std::vector<int64_t> shape(self.dim(), 1);
            result = result.reshape(shape);
        }
        return result;
    }

    // CPU fallback: copy to CPU, compute, copy back
    auto cpu_self = self.to(at::kCPU);
    auto cpu_result = cpu_self.mean(*dim, keepdim, dtype);
    return cpu_result.to(self.device());
}

// ---------------------------------------------------------------------------
// .out variants for sum and mean
// ---------------------------------------------------------------------------

at::Tensor& nova_sum_intlist_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
    auto result = nova_sum_dim_intlist(self, dim, keepdim, dtype);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_mean_out(
    const at::Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
    auto result = nova_mean_dim(self, dim, keepdim, dtype);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

// ---------------------------------------------------------------------------
// cat -- concatenate tensors along a dimension
// ---------------------------------------------------------------------------

at::Tensor nova_cat(const at::ITensorListRef& tensors, int64_t dim) {
    auto materialized = tensors.materialize();
    TORCH_CHECK(!materialized.empty(), "nova_cat: expected non-empty tensor list");

    // Resolve negative dim
    const auto& first = materialized[0].get();
    int64_t ndim = first.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "nova_cat: dim out of range");

    // Compute output shape
    auto output_sizes = first.sizes().vec();
    int64_t total_dim_size = first.size(dim);

    for (size_t i = 1; i < materialized.size(); ++i) {
        const auto& t = materialized[i].get();
        TORCH_CHECK(t.dim() == ndim, "nova_cat: all tensors must have same ndim");
        for (int64_t d = 0; d < ndim; ++d) {
            if (d == dim) {
                total_dim_size += t.size(d);
            } else {
                TORCH_CHECK(t.size(d) == first.size(d),
                    "nova_cat: sizes mismatch at dim ", d);
            }
        }
    }
    output_sizes[dim] = total_dim_size;

    auto output = at::empty(output_sizes, first.options());

    // Copy each tensor's data into the output via CPU mapped memory
    // For contiguous tensors along dim=0, this is a simple sequential memcpy
    // For general case, use element-by-element copy via CPU fallback
    if (dim == 0) {
        // Fast path for dim=0 concat of contiguous tensors
        float* dst = static_cast<float*>(output.data_ptr());
        int64_t offset = 0;
        for (size_t i = 0; i < materialized.size(); ++i) {
            const auto& t = materialized[i].get();
            auto t_contig = t.contiguous();
            novatorch::invalidateNovaBuffer(t_contig);
            int64_t nbytes = t_contig.numel() * t_contig.element_size();
            std::memcpy(dst + offset, t_contig.data_ptr(),
                        static_cast<size_t>(nbytes));
            offset += t_contig.numel();
        }
        novatorch::flushNovaBuffer(output);
    } else {
        // General case: CPU fallback
        std::vector<at::Tensor> cpu_tensors;
        cpu_tensors.reserve(materialized.size());
        for (size_t i = 0; i < materialized.size(); ++i) {
            cpu_tensors.push_back(materialized[i].get().to(at::kCPU));
        }
        auto cpu_result = at::cat(cpu_tensors, dim);
        // Copy back to Nova
        std::memcpy(output.data_ptr(), cpu_result.data_ptr(),
                    static_cast<size_t>(cpu_result.numel() * cpu_result.element_size()));
        novatorch::flushNovaBuffer(output);
    }

    return output;
}

at::Tensor& nova_cat_out(
    const at::ITensorListRef& tensors,
    int64_t dim,
    at::Tensor& out) {
    auto result = nova_cat(tensors, dim);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}
