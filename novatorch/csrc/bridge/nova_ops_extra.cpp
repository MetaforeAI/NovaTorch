#include "nova_ops.h"
#include "nova_storage.h"
#include "nova_batch_context.h"

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

    int64_t embedding_dim = weight_c.size(1);
    int64_t num_indices = indices_c.numel();

    // Output shape = indices.shape + [embedding_dim]
    auto out_sizes = indices_c.sizes().vec();
    out_sizes.push_back(embedding_dim);
    auto output = at::empty(out_sizes, weight_c.options());

    novatorch::withStagingRead(weight_c, [&](const void* w_raw, size_t) {
    novatorch::withStagingRead(indices_c, [&](const void* i_raw, size_t) {
    novatorch::withStagingWrite(output, [&](void* o_raw, size_t) {
        const float* w_ptr = static_cast<const float*>(w_raw);
        float* out_ptr = static_cast<float*>(o_raw);

        // Indices can be Long or Int
        if (indices_c.scalar_type() == at::ScalarType::Long) {
            const int64_t* idx_ptr = static_cast<const int64_t*>(i_raw);
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
            const int32_t* idx_ptr = static_cast<const int32_t*>(i_raw);
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
    });
    });
    });

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

    int64_t embedding_dim = grad_c.size(grad_c.dim() - 1);

    // grad_weight shape: [num_weights, embedding_dim]
    auto grad_weight = at::zeros(
        {num_weights, embedding_dim}, grad_c.options());

    int64_t num_indices = indices_c.numel();

    novatorch::withStagingRead(grad_c, [&](const void* go_raw, size_t) {
    novatorch::withStagingRead(indices_c, [&](const void* idx_raw, size_t) {
    novatorch::withStagingReadWrite(grad_weight, [&](void* gw_raw, size_t) {
        float* gw_ptr = static_cast<float*>(gw_raw);
        const float* go_ptr = static_cast<const float*>(go_raw);

        if (indices_c.scalar_type() == at::ScalarType::Long) {
            const int64_t* idx_ptr =
                static_cast<const int64_t*>(idx_raw);
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
                static_cast<const int32_t*>(idx_raw);
            for (int64_t i = 0; i < num_indices; ++i) {
                int64_t idx = static_cast<int64_t>(idx_ptr[i]);
                if (idx == padding_idx) continue;
                for (int64_t j = 0; j < embedding_dim; ++j) {
                    gw_ptr[idx * embedding_dim + j] +=
                        go_ptr[i * embedding_dim + j];
                }
            }
        }
    });
    });
    });

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

    novatorch::withStagingWrite(output, [&](void* raw, size_t) {
        if (dtype == at::ScalarType::Float) {
            float* ptr = static_cast<float*>(raw);
            float fs = static_cast<float>(s);
            float fst = static_cast<float>(st);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = fs + static_cast<float>(i) * fst;
            }
        } else if (dtype == at::ScalarType::Long) {
            int64_t* ptr = static_cast<int64_t*>(raw);
            int64_t is = static_cast<int64_t>(s);
            int64_t ist = static_cast<int64_t>(st);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = is + i * ist;
            }
        } else if (dtype == at::ScalarType::Int) {
            int32_t* ptr = static_cast<int32_t*>(raw);
            int32_t is = static_cast<int32_t>(s);
            int32_t ist = static_cast<int32_t>(st);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = is + static_cast<int32_t>(i) * ist;
            }
        } else if (dtype == at::ScalarType::Double) {
            double* ptr = static_cast<double*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = s + static_cast<double>(i) * st;
            }
        } else {
            TORCH_CHECK(false,
                "nova_arange: unsupported dtype ", dtype);
        }
    });

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

    // Resize out and copy device-to-device
    out.resize_({result.numel()});
    auto nbytes = static_cast<size_t>(result.numel() * result.element_size());
    if (nbytes > 0) {
        novatorch::copyDeviceToDevice(result, out, nbytes);
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

    int64_t numel = input_c.numel();
    auto output = at::empty(input_c.sizes(), input_c.options());
    auto mask = at::empty(input_c.sizes(),
        input_c.options().dtype(at::ScalarType::Bool));

    novatorch::withStagingRead(input_c, [&](const void* in_raw, size_t) {
    novatorch::withStagingWrite(output, [&](void* out_raw, size_t) {
    novatorch::withStagingWrite(mask, [&](void* mask_raw, size_t) {
        const float* in_ptr = static_cast<const float*>(in_raw);
        float* out_ptr = static_cast<float*>(out_raw);
        bool* mask_ptr = static_cast<bool*>(mask_raw);

        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(1.0 - p);
        float scale = 1.0f / (1.0f - static_cast<float>(p));

        for (int64_t i = 0; i < numel; ++i) {
            bool keep = dist(rng);
            mask_ptr[i] = keep;
            out_ptr[i] = keep ? in_ptr[i] * scale : 0.0f;
        }
    });
    });
    });

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

    novatorch::withStagingWrite(self, [&](void* raw, size_t) {
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(p);

        if (self.scalar_type() == at::ScalarType::Float) {
            float* ptr = static_cast<float*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = dist(rng) ? 1.0f : 0.0f;
            }
        } else if (self.scalar_type() == at::ScalarType::Bool) {
            bool* ptr = static_cast<bool*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = dist(rng);
            }
        } else if (self.scalar_type() == at::ScalarType::Double) {
            double* ptr = static_cast<double*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = dist(rng) ? 1.0 : 0.0;
            }
        } else {
            TORCH_CHECK(false,
                "nova_bernoulli_: unsupported dtype ", self.scalar_type());
        }
    });

    return self;
}

// ===================================================================
// 6. nonzero
// ===================================================================

at::Tensor nova_nonzero(const at::Tensor& self) {
    auto self_c = self.contiguous();

    int64_t numel = self_c.numel();
    int64_t ndim = self_c.dim();
    if (ndim == 0) ndim = 1; // scalar treated as 1-D

    auto sizes = self_c.sizes();

    // Download input to CPU, find nonzero indices
    std::vector<int64_t> nz_linear;
    nz_linear.reserve(static_cast<size_t>(numel));

    novatorch::withStagingRead(self_c, [&](const void* raw, size_t) {
        if (self_c.scalar_type() == at::ScalarType::Float) {
            const float* ptr = static_cast<const float*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                if (ptr[i] != 0.0f) nz_linear.push_back(i);
            }
        } else if (self_c.scalar_type() == at::ScalarType::Long) {
            const int64_t* ptr = static_cast<const int64_t*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                if (ptr[i] != 0) nz_linear.push_back(i);
            }
        } else if (self_c.scalar_type() == at::ScalarType::Bool) {
            const bool* ptr = static_cast<const bool*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                if (ptr[i]) nz_linear.push_back(i);
            }
        } else if (self_c.scalar_type() == at::ScalarType::Double) {
            const double* ptr = static_cast<const double*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                if (ptr[i] != 0.0) nz_linear.push_back(i);
            }
        } else if (self_c.scalar_type() == at::ScalarType::Int) {
            const int32_t* ptr = static_cast<const int32_t*>(raw);
            for (int64_t i = 0; i < numel; ++i) {
                if (ptr[i] != 0) nz_linear.push_back(i);
            }
        } else {
            TORCH_CHECK(false,
                "nova_nonzero: unsupported dtype ", self_c.scalar_type());
        }
    });

    int64_t N = static_cast<int64_t>(nz_linear.size());

    // Output: [N, ndim] Long tensor
    auto output = at::empty(
        {N, ndim},
        self_c.options().dtype(at::ScalarType::Long));

    if (N == 0) return output;

    // Convert linear indices to multi-dim indices and upload
    novatorch::withStagingWrite(output, [&](void* out_raw, size_t) {
        int64_t* out_ptr = static_cast<int64_t*>(out_raw);
        for (int64_t i = 0; i < N; ++i) {
            int64_t linear = nz_linear[i];
            for (int64_t d = ndim - 1; d >= 0; --d) {
                int64_t dim_size = (self_c.dim() == 0) ? 1 : sizes[d];
                out_ptr[i * ndim + d] = linear % dim_size;
                linear /= dim_size;
            }
        }
    });

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

    // Support both 3D (N, S, D) and 4D (B, H, S, D) inputs.
    // 3D is treated as (N, 1, S, D) — single head per batch element.
    TORCH_CHECK(query.dim() == 3 || query.dim() == 4,
        "nova_sdpa: query must be 3-D or 4-D, got ", query.dim(), "-D");
    TORCH_CHECK(key.dim() == query.dim(),
        "nova_sdpa: key must have same ndim as query");
    TORCH_CHECK(value.dim() == query.dim(),
        "nova_sdpa: value must have same ndim as query");

    bool was_3d = (query.dim() == 3);
    auto Q = was_3d ? query.unsqueeze(1).contiguous() : query.contiguous();
    auto K = was_3d ? key.unsqueeze(1).contiguous() : key.contiguous();
    auto V = was_3d ? value.unsqueeze(1).contiguous() : value.contiguous();

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

    // Download Q, K, V to CPU-side buffers
    auto q_nbytes = static_cast<size_t>(Q.numel() * Q.element_size());
    auto k_nbytes = static_cast<size_t>(K.numel() * K.element_size());
    auto v_nbytes = static_cast<size_t>(V.numel() * V.element_size());

    std::vector<uint8_t> q_buf(q_nbytes);
    std::vector<uint8_t> k_buf(k_nbytes);
    std::vector<uint8_t> v_buf(v_nbytes);

    novatorch::downloadFromDevice(Q, q_buf.data(), q_nbytes);
    novatorch::downloadFromDevice(K, k_buf.data(), k_nbytes);
    novatorch::downloadFromDevice(V, v_buf.data(), v_nbytes);

    const float* q_ptr = reinterpret_cast<const float*>(q_buf.data());
    const float* k_ptr = reinterpret_cast<const float*>(k_buf.data());
    const float* v_ptr = reinterpret_cast<const float*>(v_buf.data());

    // Optional attention mask
    std::vector<uint8_t> mask_buf;
    const void* mask_raw = nullptr;
    at::Tensor mask_c;
    if (attn_mask.has_value() && attn_mask->defined()) {
        mask_c = attn_mask->contiguous();
        auto m_nbytes = static_cast<size_t>(
            mask_c.numel() * mask_c.element_size());
        mask_buf.resize(m_nbytes);
        novatorch::downloadFromDevice(mask_c, mask_buf.data(), m_nbytes);
        mask_raw = mask_buf.data();
    }

    // Compute attention and upload result
    novatorch::withStagingWrite(output, [&](void* out_raw, size_t) {
        float* out_ptr = static_cast<float*>(out_raw);

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
                if (mask_raw) {
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
                    if (mask_c.scalar_type() == at::ScalarType::Float) {
                        const float* mask_ptr =
                            static_cast<const float*>(mask_raw);
                        for (int64_t idx = 0; idx < mask_ls; ++idx) {
                            attn_scores[static_cast<size_t>(idx)] +=
                                mask_ptr[mask_offset + idx];
                        }
                    } else if (mask_c.scalar_type() == at::ScalarType::Bool) {
                        const bool* bmask =
                            static_cast<const bool*>(mask_raw);
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
                    float max_val =
                        -std::numeric_limits<float>::infinity();
                    for (int64_t j = 0; j < S; ++j) {
                        float v =
                            attn_scores[static_cast<size_t>(i * S + j)];
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
                                attn_scores[static_cast<size_t>(
                                    i * S + j)] *
                                v_ptr[vh_off + j * Dv + d];
                        }
                        out_ptr[oh_off + i * Dv + d] = acc;
                    }
                }
            }
        }
    });

    // Squeeze back to 3D if input was 3D
    if (was_3d) output = output.squeeze(1);

    return output;
}

// ===================================================================
// 8. _thnn_fused_gru_cell — GRU cell (unfused implementation)
// ===================================================================

// Schema: _thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates,
//         Tensor hx, Tensor? input_bias, Tensor? hidden_bias)
//         -> (Tensor, Tensor)
//
// input_gates  = x @ W_ih.T    [batch, 3*hidden]
// hidden_gates = h @ W_hh.T    [batch, 3*hidden]
// hx           = previous hidden state [batch, hidden]
// Returns: (new_hidden, workspace) where workspace is used by backward

std::tuple<at::Tensor, at::Tensor> nova_thnn_fused_gru_cell(
    const at::Tensor& input_gates,
    const at::Tensor& hidden_gates,
    const at::Tensor& hx,
    const std::optional<at::Tensor>& input_bias,
    const std::optional<at::Tensor>& hidden_bias) {

    // Add biases if present
    auto ig = input_gates;
    auto hg = hidden_gates;
    if (input_bias.has_value() && input_bias->defined())
        ig = ig + *input_bias;
    if (hidden_bias.has_value() && hidden_bias->defined())
        hg = hg + *hidden_bias;

    int64_t hidden_size = hx.size(1);

    // Split gates: each is [batch, hidden_size]
    // ig = [ig_r, ig_z, ig_n], hg = [hg_r, hg_z, hg_n]
    auto ig_chunks = ig.chunk(3, 1);
    auto hg_chunks = hg.chunk(3, 1);

    auto r = at::sigmoid(ig_chunks[0] + hg_chunks[0]);  // reset gate
    auto z = at::sigmoid(ig_chunks[1] + hg_chunks[1]);  // update gate
    auto n = at::tanh(ig_chunks[2] + r * hg_chunks[2]); // new gate

    auto hy = (1.0 - z) * n + z * hx;  // new hidden state

    // Workspace tensor for backward pass — store intermediate gates + hx
    // Layout: [r, z, n, hg_n, hx] concatenated along dim 1
    auto workspace = at::cat({r, z, n, hg_chunks[2], hx}, 1);

    return std::make_tuple(hy, workspace);
}

// Backward: _thnn_fused_gru_cell_backward
// Returns: (grad_input_gates, grad_hidden_gates, grad_hx, grad_input_bias, grad_hidden_bias)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
nova_thnn_fused_gru_cell_backward(
    const at::Tensor& grad_hy,
    const at::Tensor& workspace,
    bool has_bias) {

    int64_t hidden_size = grad_hy.size(1);

    // Unpack workspace: [r, z, n, hg_n, hx]
    auto chunks = workspace.chunk(5, 1);
    auto r = chunks[0];
    auto z = chunks[1];
    auto n = chunks[2];
    auto hg_n = chunks[3];
    auto hx = chunks[4];

    // hy = (1-z)*n + z*hx
    // grad_n = grad_hy * (1-z)
    // grad_z = grad_hy * (hx - n)  -- but we don't have hx, use: grad_hy * (-n) for the (1-z) part
    // Actually: d(hy)/dz = hx - n, d(hy)/dn = 1-z, d(hy)/dhx = z
    // We need hx, which we can recover: hx = (hy - (1-z)*n) / z
    // But z could be 0. Instead, use the standard GRU backward formulas.

    // grad through hy = (1-z)*n + z*hx:
    auto grad_n = grad_hy * (1.0 - z);
    auto grad_hx = grad_hy * z;

    // For grad_z: d/dz[(1-z)*n + z*hx] = -n + hx = hx - n
    // But we don't have hx directly. Recover from: hx = (hy - (1-z)*n) / z
    // This is numerically unstable when z≈0. Use: hy = (1-z)*n + z*hx → hx-n = (hy-n)/z
    // Actually, use the chain rule on z directly:
    // grad_z_pre_sigmoid = grad_hy * (hx - n) * z * (1 - z)
    // Since we have workspace but not hx, we need a different approach.
    // The standard trick: store enough in workspace. Let's compute directly.

    // n = tanh(ig_n + r * hg_n)
    // grad through tanh: grad_tanh = grad_n * (1 - n*n)
    auto grad_tanh = grad_n * (1.0 - n * n);

    // grad_ig_n = grad_tanh
    // grad_r_hg_n = grad_tanh → grad_r = grad_tanh * hg_n, grad_hg_n = grad_tanh * r
    auto grad_ig_n = grad_tanh;
    auto grad_r_pre = grad_tanh * hg_n;
    auto grad_hg_n = grad_tanh * r;

    // r = sigmoid(ig_r + hg_r)
    // grad through sigmoid: grad_sigmoid = grad_r_pre * r * (1 - r)
    auto grad_r_sigmoid = grad_r_pre * r * (1.0 - r);
    auto grad_ig_r = grad_r_sigmoid;
    auto grad_hg_r = grad_r_sigmoid;

    // z = sigmoid(ig_z + hg_z)
    // We need grad_z, which requires hx - n. Approximate:
    // The workspace doesn't store hx. The PyTorch CUDA kernel stores it
    // in workspace. Our workspace stores [r, z, n, hg_n].
    // For correctness, set grad_z_pre = 0 for now (known limitation).
    // TODO: store hx in workspace for correct z gradient.
    // Actually, for many use cases the z gradient is small and training
    // still converges. But this IS a correctness issue.

    // Better approach: recompute hx from the relationship.
    // The caller passes grad_hy. We have z, n. If we had the original hx:
    // grad_z_pre = grad_hy * (hx - n) * z * (1-z)
    // Without hx we can't compute this exactly.
    // Workaround: compute via the output hy that autograd tracked.
    // Actually the autograd graph has hy as output. We have grad_hy.
    // Let's just not fuse — implement as decomposed ops and let autograd handle it.

    // z = sigmoid(ig_z + hg_z)
    // d(hy)/dz = hx - n
    auto grad_z_pre = grad_hy * (hx - n);
    auto grad_z_sigmoid = grad_z_pre * z * (1.0 - z);
    auto grad_ig_z = grad_z_sigmoid;
    auto grad_hg_z = grad_z_sigmoid;

    // Assemble grad_input_gates = [grad_ig_r, grad_ig_z, grad_ig_n]
    auto grad_input_gates = at::cat({grad_ig_r, grad_ig_z, grad_ig_n}, 1);
    auto grad_hidden_gates = at::cat({grad_hg_r, grad_hg_z, grad_hg_n}, 1);

    auto grad_input_bias = has_bias ? grad_input_gates.sum(0) : at::Tensor();
    auto grad_hidden_bias = has_bias ? grad_hidden_gates.sum(0) : at::Tensor();

    return std::make_tuple(grad_input_gates, grad_hidden_gates, grad_hx,
                           grad_input_bias, grad_hidden_bias);
}

// ===================================================================
// 9. index.Tensor — advanced integer/boolean indexing
// ===================================================================

at::Tensor nova_index_tensor(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {

    // Delegate to CPU: download self + index tensors, run CPU index,
    // upload result. Advanced indexing is complex (broadcasting,
    // multi-dim, bool masks) — correct CPU fallback first.
    NovaBatchContext::instance().flush();

    // Download self to CPU
    auto self_cpu = at::empty(self.sizes(),
        self.options().device(c10::Device(c10::DeviceType::CPU)));
    novatorch::downloadFromDevice(self, self_cpu.data_ptr(),
        self.numel() * self.element_size());

    // Download each index tensor to CPU
    c10::List<std::optional<at::Tensor>> cpu_indices;
    cpu_indices.reserve(indices.size());
    for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
        std::optional<at::Tensor> idx_opt =
            static_cast<std::optional<at::Tensor>>(indices.get(i));
        if (idx_opt.has_value() && idx_opt->defined()) {
            auto idx = *idx_opt;
            if (idx.device().type() == c10::DeviceType::PrivateUse1) {
                auto idx_cpu = at::empty(idx.sizes(),
                    idx.options().device(c10::Device(c10::DeviceType::CPU)));
                novatorch::downloadFromDevice(idx, idx_cpu.data_ptr(),
                    idx.numel() * idx.element_size());
                cpu_indices.push_back(idx_cpu);
            } else {
                cpu_indices.push_back(idx);
            }
        } else {
            cpu_indices.push_back(std::optional<at::Tensor>());
        }
    }

    // Run index on CPU
    auto result_cpu = at::index(self_cpu, cpu_indices);

    // Upload result to Nova
    auto result = at::empty(result_cpu.sizes(),
        result_cpu.options().device(self.device()));
    novatorch::uploadToDevice(result, result_cpu.contiguous().data_ptr(),
        result_cpu.numel() * result_cpu.element_size());

    return result;
}

at::Tensor& nova_index_tensor_out(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    at::Tensor& out) {
    auto result = nova_index_tensor(self, indices);
    out.resize_as_(result);
    novatorch::copyDeviceToDevice(result, out,
        result.numel() * result.element_size());
    return out;
}

// ===================================================================
// 10. linalg_vector_norm — used by clip_grad_norm_
// ===================================================================

at::Tensor nova_linalg_vector_norm(
    const at::Tensor& self,
    const at::Scalar& ord,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype) {

    float ord_val = ord.toFloat();

    // Common case: L2 norm (ord=2)
    if (ord_val == 2.0f) {
        auto squared = self * self;
        at::Tensor sum_sq;
        if (dim.has_value() && !dim->empty()) {
            sum_sq = squared.sum(*dim, keepdim);
        } else {
            sum_sq = squared.sum();
            if (keepdim) {
                std::vector<int64_t> shape(self.dim(), 1);
                sum_sq = sum_sq.reshape(shape);
            }
        }
        return at::sqrt(sum_sq);
    }

    // L1 norm (ord=1)
    if (ord_val == 1.0f) {
        auto abs_self = at::abs(self);
        if (dim.has_value() && !dim->empty()) {
            return abs_self.sum(*dim, keepdim);
        } else {
            auto result = abs_self.sum();
            if (keepdim) {
                std::vector<int64_t> shape(self.dim(), 1);
                result = result.reshape(shape);
            }
            return result;
        }
    }

    // Inf norm
    if (std::isinf(ord_val)) {
        auto abs_self = at::abs(self);
        // For inf: max of abs values. For -inf: min of abs values.
        // CPU fallback for simplicity
        NovaBatchContext::instance().flush();
        auto self_cpu = at::empty(self.sizes(),
            self.options().device(c10::Device(c10::DeviceType::CPU)));
        novatorch::downloadFromDevice(self, self_cpu.data_ptr(),
            self.numel() * self.element_size());
        auto result_cpu = at::linalg_vector_norm(self_cpu, ord, dim, keepdim, dtype);
        auto result = at::empty(result_cpu.sizes(),
            result_cpu.options().device(self.device()));
        novatorch::uploadToDevice(result, result_cpu.contiguous().data_ptr(),
            result_cpu.numel() * result_cpu.element_size());
        return result;
    }

    // General case: (sum(|x|^p))^(1/p)
    auto abs_self = at::abs(self);
    auto powered = at::pow(abs_self, ord);
    at::Tensor sum_p;
    if (dim.has_value() && !dim->empty()) {
        sum_p = powered.sum(*dim, keepdim);
    } else {
        sum_p = powered.sum();
        if (keepdim) {
            std::vector<int64_t> shape(self.dim(), 1);
            sum_p = sum_p.reshape(shape);
        }
    }
    return at::pow(sum_p, 1.0 / ord_val);
}

at::Tensor& nova_linalg_vector_norm_out(
    const at::Tensor& self,
    const at::Scalar& ord,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    std::optional<at::ScalarType> dtype,
    at::Tensor& out) {
    auto result = nova_linalg_vector_norm(self, ord, dim, keepdim, dtype);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}
