#include "nova_ops.h"

#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

// ===================================================================
// Helper: linear index to multi-dimensional index
// ===================================================================

namespace {

/// Convert a linear index to multi-dimensional indices for a given shape.
std::vector<int64_t> unravel_index(int64_t linear, c10::IntArrayRef sizes) {
    int64_t ndim = static_cast<int64_t>(sizes.size());
    std::vector<int64_t> indices(ndim);
    for (int64_t i = ndim - 1; i >= 0; --i) {
        indices[i] = linear % sizes[i];
        linear /= sizes[i];
    }
    return indices;
}

/// Compute the linear offset for a multi-dimensional index given strides.
int64_t compute_offset(
    const std::vector<int64_t>& indices,
    c10::IntArrayRef strides,
    int64_t storage_offset) {
    int64_t offset = storage_offset;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

/// Read a single float element from a tensor (handles non-contiguous).
float read_float_element(const at::Tensor& t, int64_t linear_idx) {
    auto indices = unravel_index(linear_idx, t.sizes());
    int64_t offset = compute_offset(indices, t.strides(), t.storage_offset());
    return static_cast<const float*>(t.data_ptr())[offset];
}

/// Write a single float element to a tensor (handles non-contiguous).
void write_float_element(at::Tensor& t, int64_t linear_idx, float val) {
    auto indices = unravel_index(linear_idx, t.sizes());
    int64_t offset = compute_offset(indices, t.strides(), t.storage_offset());
    static_cast<float*>(t.data_ptr())[offset] = val;
}

} // anonymous namespace

// ===================================================================
// 1. embedding
// ===================================================================

at::Tensor nova_embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool /*scale_grad_by_freq*/,
    bool /*sparse*/) {

    TORCH_CHECK(weight.dim() == 2, "nova_embedding: weight must be 2-D");
    TORCH_CHECK(
        weight.scalar_type() == at::ScalarType::Float,
        "nova_embedding: only float32 weight supported");

    // Make contiguous copies for simple indexed access
    auto weight_c = weight.contiguous();
    auto indices_c = indices.contiguous();

    novatorch::invalidateNovaBuffer(weight_c);
    novatorch::invalidateNovaBuffer(indices_c);

    int64_t embedding_dim = weight_c.size(1);
    int64_t num_indices = indices_c.numel();

    // Output shape = indices.shape + [embedding_dim]
    auto out_sizes = indices_c.sizes().vec();
    out_sizes.push_back(embedding_dim);
    auto output = at::empty(out_sizes, weight_c.options());

    const float* w_ptr = static_cast<const float*>(weight_c.data_ptr());
    float* out_ptr = static_cast<float*>(output.data_ptr());

    // Indices can be Long or Int
    if (indices_c.scalar_type() == at::ScalarType::Long) {
        const int64_t* idx_ptr = static_cast<const int64_t*>(indices_c.data_ptr());
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_ptr[i];
            TORCH_CHECK(
                idx >= 0 && idx < weight_c.size(0),
                "nova_embedding: index out of range");
            std::memcpy(
                out_ptr + i * embedding_dim,
                w_ptr + idx * embedding_dim,
                static_cast<size_t>(embedding_dim) * sizeof(float));
        }
    } else if (indices_c.scalar_type() == at::ScalarType::Int) {
        const int32_t* idx_ptr = static_cast<const int32_t*>(indices_c.data_ptr());
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = static_cast<int64_t>(idx_ptr[i]);
            TORCH_CHECK(
                idx >= 0 && idx < weight_c.size(0),
                "nova_embedding: index out of range");
            std::memcpy(
                out_ptr + i * embedding_dim,
                w_ptr + idx * embedding_dim,
                static_cast<size_t>(embedding_dim) * sizeof(float));
        }
    } else {
        TORCH_CHECK(false, "nova_embedding: indices must be Long or Int");
    }

    novatorch::flushNovaBuffer(output);
    return output;
}

// ===================================================================
// 2. embedding_dense_backward
// ===================================================================

at::Tensor nova_embedding_dense_backward(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool /*scale_grad_by_freq*/) {

    auto grad_c = grad_output.contiguous();
    auto indices_c = indices.contiguous();

    novatorch::invalidateNovaBuffer(grad_c);
    novatorch::invalidateNovaBuffer(indices_c);

    int64_t embedding_dim = grad_c.size(grad_c.dim() - 1);

    // grad_weight shape: [num_weights, embedding_dim]
    auto grad_weight = at::zeros(
        {num_weights, embedding_dim}, grad_c.options());

    float* gw_ptr = static_cast<float*>(grad_weight.data_ptr());
    const float* go_ptr = static_cast<const float*>(grad_c.data_ptr());
    int64_t num_indices = indices_c.numel();

    if (indices_c.scalar_type() == at::ScalarType::Long) {
        const int64_t* idx_ptr =
            static_cast<const int64_t*>(indices_c.data_ptr());
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = idx_ptr[i];
            if (idx == padding_idx) continue;
            for (int64_t j = 0; j < embedding_dim; ++j) {
                gw_ptr[idx * embedding_dim + j] +=
                    go_ptr[i * embedding_dim + j];
            }
        }
    } else {
        const int32_t* idx_ptr =
            static_cast<const int32_t*>(indices_c.data_ptr());
        for (int64_t i = 0; i < num_indices; ++i) {
            int64_t idx = static_cast<int64_t>(idx_ptr[i]);
            if (idx == padding_idx) continue;
            for (int64_t j = 0; j < embedding_dim; ++j) {
                gw_ptr[idx * embedding_dim + j] +=
                    go_ptr[i * embedding_dim + j];
            }
        }
    }

    novatorch::flushNovaBuffer(grad_weight);
    return grad_weight;
}

// ===================================================================
// 3. arange
// ===================================================================

at::Tensor nova_arange(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    std::optional<at::ScalarType> dtype_opt,
    std::optional<at::Layout> /*layout*/,
    std::optional<at::Device> device_opt,
    std::optional<bool> /*pin_memory*/) {

    double s = start.toDouble();
    double e = end.toDouble();
    double st = step.toDouble();

    TORCH_CHECK(st != 0, "nova_arange: step cannot be zero");
    TORCH_CHECK(
        (st > 0 && s < e) || (st < 0 && s > e) || s == e,
        "nova_arange: invalid range with given step");

    int64_t numel = 0;
    if (s != e) {
        numel = static_cast<int64_t>(std::ceil((e - s) / st));
    }
    if (numel < 0) numel = 0;

    auto dtype = dtype_opt.value_or(at::ScalarType::Float);
    auto device = device_opt.value_or(
        c10::Device(c10::DeviceType::PrivateUse1, 0));
    auto options = at::TensorOptions().dtype(dtype).device(device);

    auto output = at::empty({numel}, options);
    if (numel == 0) return output;

    if (dtype == at::ScalarType::Float) {
        float* ptr = static_cast<float*>(output.data_ptr());
        float fs = static_cast<float>(s);
        float fst = static_cast<float>(st);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = fs + static_cast<float>(i) * fst;
        }
    } else if (dtype == at::ScalarType::Long) {
        int64_t* ptr = static_cast<int64_t*>(output.data_ptr());
        int64_t is = static_cast<int64_t>(s);
        int64_t ist = static_cast<int64_t>(st);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = is + i * ist;
        }
    } else if (dtype == at::ScalarType::Int) {
        int32_t* ptr = static_cast<int32_t*>(output.data_ptr());
        int32_t is = static_cast<int32_t>(s);
        int32_t ist = static_cast<int32_t>(st);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = is + static_cast<int32_t>(i) * ist;
        }
    } else if (dtype == at::ScalarType::Double) {
        double* ptr = static_cast<double*>(output.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = s + static_cast<double>(i) * st;
        }
    } else {
        TORCH_CHECK(false,
            "nova_arange: unsupported dtype ", dtype);
    }

    novatorch::flushNovaBuffer(output);
    return output;
}

at::Tensor& nova_arange_out(
    const at::Scalar& start,
    const at::Scalar& end,
    const at::Scalar& step,
    at::Tensor& out) {

    auto result = nova_arange(
        start, end, step,
        out.scalar_type(),
        out.layout(),
        out.device(),
        false);

    // Resize out and copy
    out.resize_({result.numel()});
    auto nbytes = result.numel() * result.element_size();
    if (nbytes > 0) {
        novatorch::invalidateNovaBuffer(result);
        std::memcpy(out.data_ptr(), result.data_ptr(),
                     static_cast<size_t>(nbytes));
        novatorch::flushNovaBuffer(out);
    }
    return out;
}

// ===================================================================
// 4. native_dropout
// ===================================================================

std::tuple<at::Tensor, at::Tensor> nova_native_dropout(
    const at::Tensor& input,
    double p,
    std::optional<bool> train_opt) {

    bool train = train_opt.value_or(true);

    // If not training or p==0, return input and all-ones mask
    if (!train || p == 0.0) {
        auto mask = at::ones(input.sizes(),
            input.options().dtype(at::ScalarType::Bool));
        return std::make_tuple(input, mask);
    }

    // If p==1, return zeros and all-zeros mask
    if (p == 1.0) {
        auto output = at::zeros(input.sizes(), input.options());
        auto mask = at::zeros(input.sizes(),
            input.options().dtype(at::ScalarType::Bool));
        return std::make_tuple(output, mask);
    }

    auto input_c = input.contiguous();
    novatorch::invalidateNovaBuffer(input_c);

    int64_t numel = input_c.numel();
    auto output = at::empty(input_c.sizes(), input_c.options());
    auto mask = at::empty(input_c.sizes(),
        input_c.options().dtype(at::ScalarType::Bool));

    const float* in_ptr = static_cast<const float*>(input_c.data_ptr());
    float* out_ptr = static_cast<float*>(output.data_ptr());
    bool* mask_ptr = static_cast<bool*>(mask.data_ptr());

    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution dist(1.0 - p);
    float scale = 1.0f / (1.0f - static_cast<float>(p));

    for (int64_t i = 0; i < numel; ++i) {
        bool keep = dist(rng);
        mask_ptr[i] = keep;
        out_ptr[i] = keep ? in_ptr[i] * scale : 0.0f;
    }

    novatorch::flushNovaBuffer(output);
    novatorch::flushNovaBuffer(mask);
    return std::make_tuple(output, mask);
}

// ===================================================================
// 5. bernoulli_.float
// ===================================================================

at::Tensor& nova_bernoulli_float(
    at::Tensor& self,
    double p,
    std::optional<at::Generator> /*gen*/) {

    TORCH_CHECK(
        p >= 0.0 && p <= 1.0,
        "nova_bernoulli_: probability must be in [0, 1]");

    int64_t numel = self.numel();
    if (numel == 0) return self;

    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution dist(p);

    if (self.scalar_type() == at::ScalarType::Float) {
        float* ptr = static_cast<float*>(self.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = dist(rng) ? 1.0f : 0.0f;
        }
    } else if (self.scalar_type() == at::ScalarType::Bool) {
        bool* ptr = static_cast<bool*>(self.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = dist(rng);
        }
    } else if (self.scalar_type() == at::ScalarType::Double) {
        double* ptr = static_cast<double*>(self.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = dist(rng) ? 1.0 : 0.0;
        }
    } else {
        TORCH_CHECK(false,
            "nova_bernoulli_: unsupported dtype ", self.scalar_type());
    }

    novatorch::flushNovaBuffer(self);
    return self;
}

// ===================================================================
// 6. nonzero
// ===================================================================

at::Tensor nova_nonzero(const at::Tensor& self) {
    auto self_c = self.contiguous();
    novatorch::invalidateNovaBuffer(self_c);

    int64_t numel = self_c.numel();
    int64_t ndim = self_c.dim();
    if (ndim == 0) ndim = 1; // scalar treated as 1-D

    auto sizes = self_c.sizes();

    // First pass: count nonzero elements
    std::vector<int64_t> nz_linear;
    nz_linear.reserve(static_cast<size_t>(numel));

    if (self_c.scalar_type() == at::ScalarType::Float) {
        const float* ptr = static_cast<const float*>(self_c.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            if (ptr[i] != 0.0f) nz_linear.push_back(i);
        }
    } else if (self_c.scalar_type() == at::ScalarType::Long) {
        const int64_t* ptr =
            static_cast<const int64_t*>(self_c.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            if (ptr[i] != 0) nz_linear.push_back(i);
        }
    } else if (self_c.scalar_type() == at::ScalarType::Bool) {
        const bool* ptr = static_cast<const bool*>(self_c.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            if (ptr[i]) nz_linear.push_back(i);
        }
    } else if (self_c.scalar_type() == at::ScalarType::Double) {
        const double* ptr =
            static_cast<const double*>(self_c.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            if (ptr[i] != 0.0) nz_linear.push_back(i);
        }
    } else if (self_c.scalar_type() == at::ScalarType::Int) {
        const int32_t* ptr =
            static_cast<const int32_t*>(self_c.data_ptr());
        for (int64_t i = 0; i < numel; ++i) {
            if (ptr[i] != 0) nz_linear.push_back(i);
        }
    } else {
        TORCH_CHECK(false,
            "nova_nonzero: unsupported dtype ", self_c.scalar_type());
    }

    int64_t N = static_cast<int64_t>(nz_linear.size());

    // Output: [N, ndim] Long tensor
    auto output = at::empty(
        {N, ndim},
        self_c.options().dtype(at::ScalarType::Long));

    if (N == 0) return output;

    int64_t* out_ptr = static_cast<int64_t*>(output.data_ptr());

    // Convert linear indices to multi-dim indices
    for (int64_t i = 0; i < N; ++i) {
        int64_t linear = nz_linear[i];
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t dim_size = (self_c.dim() == 0) ? 1 : sizes[d];
            out_ptr[i * ndim + d] = linear % dim_size;
            linear /= dim_size;
        }
    }

    novatorch::flushNovaBuffer(output);
    return output;
}

// ===================================================================
// 7. scaled_dot_product_attention
// ===================================================================

at::Tensor nova_scaled_dot_product_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {

    // Q: [B, H, L, D], K: [B, H, S, D], V: [B, H, S, D]
    TORCH_CHECK(query.dim() == 4, "nova_sdpa: query must be 4-D");
    TORCH_CHECK(key.dim() == 4, "nova_sdpa: key must be 4-D");
    TORCH_CHECK(value.dim() == 4, "nova_sdpa: value must be 4-D");

    auto Q = query.contiguous();
    auto K = key.contiguous();
    auto V = value.contiguous();

    novatorch::invalidateNovaBuffer(Q);
    novatorch::invalidateNovaBuffer(K);
    novatorch::invalidateNovaBuffer(V);

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t L = Q.size(2);
    int64_t D = Q.size(3);
    int64_t S = K.size(2);

    TORCH_CHECK(K.size(3) == D, "nova_sdpa: key head dim must match query");
    TORCH_CHECK(V.size(2) == S, "nova_sdpa: value seq len must match key");

    double scale_val = scale.value_or(1.0 / std::sqrt(static_cast<double>(D)));

    // Output: [B, H, L, Dv]
    int64_t Dv = V.size(3);
    auto output = at::empty({B, H, L, Dv}, Q.options());

    const float* q_ptr = static_cast<const float*>(Q.data_ptr());
    const float* k_ptr = static_cast<const float*>(K.data_ptr());
    const float* v_ptr = static_cast<const float*>(V.data_ptr());
    float* out_ptr = static_cast<float*>(output.data_ptr());

    // Optional attention mask
    const float* mask_ptr = nullptr;
    at::Tensor mask_c;
    if (attn_mask.has_value() && attn_mask->defined()) {
        mask_c = attn_mask->contiguous();
        novatorch::invalidateNovaBuffer(mask_c);
        mask_ptr = static_cast<const float*>(mask_c.data_ptr());
    }

    // Temporary buffer for attention scores [L, S]
    std::vector<float> attn_scores(static_cast<size_t>(L * S));

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            int64_t qh_off = (b * H + h) * L * D;
            int64_t kh_off = (b * H + h) * S * D;
            int64_t vh_off = (b * H + h) * S * Dv;
            int64_t oh_off = (b * H + h) * L * Dv;

            // Compute Q @ K^T * scale -> attn_scores [L, S]
            for (int64_t i = 0; i < L; ++i) {
                for (int64_t j = 0; j < S; ++j) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += q_ptr[qh_off + i * D + d] *
                               k_ptr[kh_off + j * D + d];
                    }
                    attn_scores[static_cast<size_t>(i * S + j)] =
                        dot * static_cast<float>(scale_val);
                }
            }

            // Apply causal mask: mask future positions with -inf
            if (is_causal) {
                for (int64_t i = 0; i < L; ++i) {
                    for (int64_t j = i + 1; j < S; ++j) {
                        attn_scores[static_cast<size_t>(i * S + j)] =
                            -std::numeric_limits<float>::infinity();
                    }
                }
            }

            // Apply explicit attention mask (additive)
            if (mask_ptr) {
                // Mask can be [L, S], [1, 1, L, S], [B, H, L, S], etc.
                // For simplicity, assume broadcastable: use last 2 dims
                int64_t mask_numel = mask_c.numel();
                int64_t mask_ls = L * S;
                int64_t mask_offset = 0;
                if (mask_numel == mask_ls) {
                    mask_offset = 0;
                } else if (mask_numel == B * H * mask_ls) {
                    mask_offset = (b * H + h) * mask_ls;
                } else if (mask_numel == B * mask_ls) {
                    mask_offset = b * mask_ls;
                } else if (mask_numel == H * mask_ls) {
                    mask_offset = h * mask_ls;
                }
                // Apply mask as additive (for float masks)
                if (mask_c.scalar_type() == at::ScalarType::Float) {
                    for (int64_t idx = 0; idx < mask_ls; ++idx) {
                        attn_scores[static_cast<size_t>(idx)] +=
                            mask_ptr[mask_offset + idx];
                    }
                } else if (mask_c.scalar_type() == at::ScalarType::Bool) {
                    const bool* bmask =
                        static_cast<const bool*>(mask_c.data_ptr());
                    for (int64_t idx = 0; idx < mask_ls; ++idx) {
                        if (!bmask[mask_offset + idx]) {
                            attn_scores[static_cast<size_t>(idx)] =
                                -std::numeric_limits<float>::infinity();
                        }
                    }
                }
            }

            // Softmax over S dimension for each query position
            for (int64_t i = 0; i < L; ++i) {
                float max_val = -std::numeric_limits<float>::infinity();
                for (int64_t j = 0; j < S; ++j) {
                    float v = attn_scores[static_cast<size_t>(i * S + j)];
                    if (v > max_val) max_val = v;
                }
                float sum = 0.0f;
                for (int64_t j = 0; j < S; ++j) {
                    float e = std::exp(
                        attn_scores[static_cast<size_t>(i * S + j)] -
                        max_val);
                    attn_scores[static_cast<size_t>(i * S + j)] = e;
                    sum += e;
                }
                if (sum > 0.0f) {
                    float inv_sum = 1.0f / sum;
                    for (int64_t j = 0; j < S; ++j) {
                        attn_scores[static_cast<size_t>(i * S + j)] *=
                            inv_sum;
                    }
                }
            }

            // Compute attn_scores @ V -> output [L, Dv]
            for (int64_t i = 0; i < L; ++i) {
                for (int64_t d = 0; d < Dv; ++d) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < S; ++j) {
                        acc +=
                            attn_scores[static_cast<size_t>(i * S + j)] *
                            v_ptr[vh_off + j * Dv + d];
                    }
                    out_ptr[oh_off + i * Dv + d] = acc;
                }
            }
        }
    }

    novatorch::flushNovaBuffer(output);
    return output;
}
