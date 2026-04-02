#include "nova_context.h"
#include "nova_allocator.h"
#include "nova_staging_pool.h"
#include "nova_guard.h"
#include "nova_hooks.h"
#include "nova_ops.h"

#include <ATen/ATen.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

// ---------------------------------------------------------------------------
// Pipeline cache & descriptor pool singletons
// ---------------------------------------------------------------------------

static NovaPipelineCache g_pipeline_cache;
static NovaDescriptorPool g_descriptor_pool;

NovaPipelineCache& getPipelineCache() { return g_pipeline_cache; }
NovaDescriptorPool& getDescriptorPool() { return g_descriptor_pool; }

// ---------------------------------------------------------------------------
// Forward declarations for ops implemented in other TUs
// ---------------------------------------------------------------------------

// Factory ops (nova_ops_factory.cpp)
at::Tensor nova_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);

at::Tensor nova_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory);

at::Tensor nova_as_strided(
    const at::Tensor& self, c10::IntArrayRef size,
    c10::IntArrayRef stride, std::optional<int64_t> storage_offset);
at::Tensor nova_reshape_alias(
    const at::Tensor& self, c10::IntArrayRef size, c10::IntArrayRef stride);
at::Tensor& nova_fill_scalar(at::Tensor& self, const at::Scalar& value);
at::Tensor& nova_zero_(at::Tensor& self);
const at::Tensor& nova_resize_(
    const at::Tensor& self, c10::IntArrayRef size,
    std::optional<c10::MemoryFormat> memory_format);

// View / reshape ops (nova_ops_factory.cpp)
at::Tensor nova_t(const at::Tensor& self);
at::Tensor nova_transpose_int(
    const at::Tensor& self, int64_t dim0, int64_t dim1);
at::Tensor nova_permute(
    const at::Tensor& self, c10::IntArrayRef dims);
at::Tensor nova_expand(
    const at::Tensor& self, c10::IntArrayRef size, bool implicit);
at::Tensor nova_view(
    const at::Tensor& self, c10::IntArrayRef size);
at::Tensor nova_reshape(
    const at::Tensor& self, c10::IntArrayRef shape);
at::Tensor nova_slice_tensor(
    const at::Tensor& self, int64_t dim,
    std::optional<int64_t> start, std::optional<int64_t> end, int64_t step);
at::Tensor nova_select_int(
    const at::Tensor& self, int64_t dim, int64_t index);
at::Tensor nova_unsqueeze(const at::Tensor& self, int64_t dim);
at::Tensor nova_squeeze_dim(const at::Tensor& self, int64_t dim);
at::Tensor nova_contiguous(
    const at::Tensor& self, c10::MemoryFormat memory_format);
at::Tensor nova_clone(
    const at::Tensor& self, std::optional<c10::MemoryFormat> memory_format);
at::Tensor nova_detach(const at::Tensor& self);

// Factory / utility ops (nova_ops_factory.cpp)
at::Tensor nova_zeros_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);
at::Tensor nova_ones_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);
at::Tensor nova_full_like(
    const at::Tensor& self,
    const at::Scalar& fill_value,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format);
at::Tensor nova_scalar_tensor(
    const at::Scalar& s,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory);
at::Tensor& nova_uniform_(
    at::Tensor& self, double from, double to,
    std::optional<at::Generator> gen);
at::Tensor& nova_normal_(
    at::Tensor& self, double mean, double std,
    std::optional<at::Generator> gen);

// Copy ops (nova_ops_copy.cpp)
at::Tensor nova_copy_from(
    const at::Tensor& self, const at::Tensor& dst, bool non_blocking);
at::Tensor nova_copy_from_and_resize(
    const at::Tensor& self, const at::Tensor& dst);
at::Scalar nova_local_scalar_dense(const at::Tensor& self);

// In-place ops (nova_ops_inplace.cpp)
at::Tensor& nova_copy_inplace(
    at::Tensor& self, const at::Tensor& src, bool non_blocking);
at::Tensor& nova_add_inplace(
    at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor& nova_sub_inplace(
    at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor& nova_mul_inplace_tensor(
    at::Tensor& self, const at::Tensor& other);
at::Tensor& nova_mul_inplace_scalar(
    at::Tensor& self, const at::Scalar& other);
at::Tensor& nova_div_inplace_tensor(
    at::Tensor& self, const at::Tensor& other);
at::Tensor& nova_addcmul_inplace(
    at::Tensor& self, const at::Tensor& tensor1,
    const at::Tensor& tensor2, const at::Scalar& value);
at::Tensor& nova_addcdiv_inplace(
    at::Tensor& self, const at::Tensor& tensor1,
    const at::Tensor& tensor2, const at::Scalar& value);

// Element-wise ops (nova_ops_elementwise.cpp)
at::Tensor nova_add_tensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor nova_sub_tensor(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha);
at::Tensor nova_mul_tensor(
    const at::Tensor& self, const at::Tensor& other);
at::Tensor nova_div_tensor(
    const at::Tensor& self, const at::Tensor& other);
at::Tensor nova_neg(const at::Tensor& self);
at::Tensor nova_mul_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_add_scalar(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha);

// Reduction ops (nova_ops_reduce.cpp)
at::Tensor nova_sum(const at::Tensor& self, std::optional<at::ScalarType> dtype);
at::Tensor nova_mean(const at::Tensor& self, std::optional<at::ScalarType> dtype);

// Matmul (nova_ops_matmul.cpp)
at::Tensor nova_mm(const at::Tensor& self, const at::Tensor& mat2);
at::Tensor nova_addmm(
    const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
    const at::Scalar& beta, const at::Scalar& alpha);
at::Tensor nova_bmm(const at::Tensor& self, const at::Tensor& mat2);

// Activation ops (nova_ops_activation.cpp)
at::Tensor nova_relu(const at::Tensor& self);
at::Tensor nova_sigmoid(const at::Tensor& self);
at::Tensor nova_tanh(const at::Tensor& self);
at::Tensor nova_threshold_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Scalar& threshold);

// Math ops (nova_ops_math.cpp)
at::Tensor nova_exp(const at::Tensor& self);
at::Tensor nova_log(const at::Tensor& self);
at::Tensor nova_sqrt(const at::Tensor& self);
at::Tensor nova_rsqrt(const at::Tensor& self);
at::Tensor nova_abs(const at::Tensor& self);
at::Tensor nova_pow_tensor_scalar(
    const at::Tensor& self, const at::Scalar& exponent);
at::Tensor nova_max_other(
    const at::Tensor& self, const at::Tensor& other);
at::Tensor nova_min_other(
    const at::Tensor& self, const at::Tensor& other);
at::Tensor nova_clamp(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min,
    const std::optional<at::Scalar>& max);

// Comparison ops (nova_ops_comparison.cpp)
at::Tensor nova_eq_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_ne_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_gt_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_lt_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_ge_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_le_scalar(const at::Tensor& self, const at::Scalar& other);
at::Tensor nova_where_self(
    const at::Tensor& condition, const at::Tensor& self,
    const at::Tensor& other);

// Loss / softmax ops (nova_ops_loss.cpp)
at::Tensor nova_softmax(
    const at::Tensor& self, int64_t dim, bool half_to_float);
at::Tensor nova_log_softmax(
    const at::Tensor& self, int64_t dim, bool half_to_float);
std::tuple<at::Tensor, at::Tensor> nova_nll_loss_forward(
    const at::Tensor& self, const at::Tensor& target,
    const std::optional<at::Tensor>& weight, int64_t reduction,
    int64_t ignore_index);

// ---------------------------------------------------------------------------
// .out variant forward declarations
// ---------------------------------------------------------------------------

// Matmul .out (nova_ops_matmul.cpp)
at::Tensor& nova_mm_out(
    const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out);
at::Tensor& nova_addmm_out(
    const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2,
    const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out);
at::Tensor& nova_bmm_out(
    const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out);

// Elementwise .out (nova_ops_elementwise.cpp)
at::Tensor& nova_add_out(
    const at::Tensor& self, const at::Tensor& other,
    const at::Scalar& alpha, at::Tensor& out);
at::Tensor& nova_sub_out(
    const at::Tensor& self, const at::Tensor& other,
    const at::Scalar& alpha, at::Tensor& out);
at::Tensor& nova_mul_out(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& nova_div_out(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& nova_neg_out(const at::Tensor& self, at::Tensor& out);

// Math .out (nova_ops_math.cpp)
at::Tensor& nova_exp_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_log_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_sqrt_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_rsqrt_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_abs_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_pow_tensor_scalar_out(
    const at::Tensor& self, const at::Scalar& exponent, at::Tensor& out);
at::Tensor& nova_max_out(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& nova_min_out(
    const at::Tensor& self, const at::Tensor& other, at::Tensor& out);
at::Tensor& nova_clamp_out(
    const at::Tensor& self,
    const std::optional<at::Scalar>& min_val,
    const std::optional<at::Scalar>& max_val,
    at::Tensor& out);
at::Tensor nova_lerp_scalar(
    const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight);
at::Tensor& nova_lerp_scalar_out(
    const at::Tensor& self, const at::Tensor& end,
    const at::Scalar& weight, at::Tensor& out);
at::Tensor& nova_lerp_scalar_inplace(
    at::Tensor& self, const at::Tensor& end, const at::Scalar& weight);
at::Tensor& nova_sqrt_inplace(at::Tensor& self);
at::Tensor& nova_div_inplace_scalar(at::Tensor& self, const at::Scalar& other);

// Activation .out (nova_ops_activation.cpp)
at::Tensor& nova_relu_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_sigmoid_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_tanh_out(const at::Tensor& self, at::Tensor& out);
at::Tensor& nova_threshold_backward_out(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Scalar& threshold, at::Tensor& out);

// Comparison .out (nova_ops_comparison.cpp)
at::Tensor& nova_eq_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& nova_ne_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& nova_gt_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& nova_lt_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& nova_ge_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& nova_le_scalar_out(
    const at::Tensor& self, const at::Scalar& other, at::Tensor& out);
at::Tensor& nova_where_self_out(
    const at::Tensor& condition, const at::Tensor& self,
    const at::Tensor& other, at::Tensor& out);

// Loss/softmax .out (nova_ops_loss.cpp)
at::Tensor& nova_softmax_out(
    const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& out);
at::Tensor& nova_log_softmax_out(
    const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& out);

// Extra ops (nova_ops_extra.cpp)
at::Tensor nova_embedding(
    const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
at::Tensor nova_embedding_dense_backward(
    const at::Tensor& grad_output, const at::Tensor& indices,
    int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
at::Tensor nova_arange(
    const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
    std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
    std::optional<at::Device> device, std::optional<bool> pin_memory);
at::Tensor& nova_arange_out(
    const at::Scalar& start, const at::Scalar& end, const at::Scalar& step,
    at::Tensor& out);
std::tuple<at::Tensor, at::Tensor> nova_native_dropout(
    const at::Tensor& input, double p, std::optional<bool> train);
at::Tensor& nova_bernoulli_float(
    at::Tensor& self, double p, std::optional<at::Generator> gen);
at::Tensor nova_nonzero(const at::Tensor& self);
at::Tensor nova_scaled_dot_product_attention(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale, bool enable_gqa);

// Index ops (nova_ops_extra.cpp)
at::Tensor nova_index_tensor(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices);
at::Tensor& nova_index_tensor_out(
    const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices,
    at::Tensor& out);

// Backward ops (nova_ops_loss.cpp)
at::Tensor& nova_nll_loss_backward_grad_input(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const std::optional<at::Tensor>& weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor& total_weight,
    at::Tensor& grad_input);
at::Tensor nova_log_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output,
    int64_t dim, at::ScalarType input_dtype);

// Reduction .out + dim variants (nova_ops_reduce.cpp)
at::Tensor nova_sum_dim_intlist(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype);
at::Tensor nova_mean_dim(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype);
at::Tensor& nova_sum_intlist_out(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype, at::Tensor& out);
at::Tensor& nova_mean_out(
    const at::Tensor& self, at::OptionalIntArrayRef dim,
    bool keepdim, std::optional<at::ScalarType> dtype, at::Tensor& out);

// Cat ops (nova_ops_reduce.cpp)
at::Tensor nova_cat(const at::ITensorListRef& tensors, int64_t dim);
at::Tensor& nova_cat_out(
    const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out);

// ---------------------------------------------------------------------------
// Static initialisation – runs when the shared library is loaded
// ---------------------------------------------------------------------------

namespace {

struct NovaBackendInit {
    NovaBackendInit() {
        // 0. Force-construct singletons in the correct destruction order.
        //    Meyer's singletons are destroyed in reverse construction order.
        //    Order: Allocator (last destroyed) → StagingPool → Context (first destroyed).
        //    ~NovaContext() calls StagingPool::destroyAll() then Allocator::releaseAll(),
        //    so both must outlive the context.
        (void)NovaAllocator::getInstance();
        (void)NovaStagingPool::instance();

        // 1. Bring up the Vulkan compute context
        auto& ctx = NovaContext::instance();

        // 2. Initialize pipeline cache, descriptor pool, and staging pool
        g_pipeline_cache.init(ctx.device());
        g_descriptor_pool.init(ctx.device());
        NovaStagingPool::instance().init(ctx.allocator());

        // 3. Register our VMA-backed allocator for PrivateUse1
        c10::SetAllocator(
            c10::DeviceType::PrivateUse1,
            NovaAllocator::getInstance());

        // 4. Register backend hooks (takes raw pointer, NOT unique_ptr)
        at::RegisterPrivateUse1HooksInterface(new NovaHooks());
    }
};

// Trigger construction at load time
static NovaBackendInit g_nova_init;

// Register the device guard implementation
C10_REGISTER_GUARD_IMPL(PrivateUse1, NovaDeviceGuardImpl);

} // anonymous namespace

// ---------------------------------------------------------------------------
// Op registrations – PrivateUse1 dispatch key
// ---------------------------------------------------------------------------

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Factory ops
    m.impl("empty.memory_format", nova_empty_memory_format);
    m.impl("empty_strided", nova_empty_strided);
    m.impl("as_strided", nova_as_strided);
    m.impl("_reshape_alias", nova_reshape_alias);
    m.impl("fill_.Scalar", nova_fill_scalar);
    m.impl("zero_", nova_zero_);
    m.impl("resize_", nova_resize_);

    // View / reshape ops
    m.impl("t", nova_t);
    m.impl("transpose.int", nova_transpose_int);
    m.impl("permute", nova_permute);
    m.impl("expand", nova_expand);
    m.impl("view", nova_view);
    m.impl("reshape", nova_reshape);
    m.impl("slice.Tensor", nova_slice_tensor);
    m.impl("select.int", nova_select_int);
    m.impl("unsqueeze", nova_unsqueeze);
    m.impl("squeeze.dim", nova_squeeze_dim);
    m.impl("contiguous", nova_contiguous);
    m.impl("clone", nova_clone);
    m.impl("detach", nova_detach);

    // Factory / utility ops
    m.impl("zeros_like", nova_zeros_like);
    m.impl("ones_like", nova_ones_like);
    m.impl("full_like", nova_full_like);
    m.impl("scalar_tensor", nova_scalar_tensor);
    m.impl("uniform_", nova_uniform_);
    m.impl("normal_", nova_normal_);

    // Copy ops
    m.impl("_copy_from", nova_copy_from);
    m.impl("_copy_from_and_resize", nova_copy_from_and_resize);
    m.impl("_local_scalar_dense", nova_local_scalar_dense);

    // In-place ops
    m.impl("copy_", nova_copy_inplace);
    m.impl("add_.Tensor", nova_add_inplace);
    m.impl("sub_.Tensor", nova_sub_inplace);
    m.impl("mul_.Tensor", nova_mul_inplace_tensor);
    m.impl("mul_.Scalar", nova_mul_inplace_scalar);
    m.impl("div_.Tensor", nova_div_inplace_tensor);
    m.impl("addcmul_", nova_addcmul_inplace);
    m.impl("addcdiv_", nova_addcdiv_inplace);

    // Element-wise ops
    m.impl("add.Tensor", nova_add_tensor);
    m.impl("sub.Tensor", nova_sub_tensor);
    m.impl("mul.Tensor", nova_mul_tensor);
    m.impl("div.Tensor", nova_div_tensor);
    m.impl("neg", nova_neg);
    m.impl("mul.Scalar", nova_mul_scalar);
    m.impl("add.Scalar", nova_add_scalar);

    // Reduction ops
    m.impl("sum", nova_sum);
    m.impl("mean", nova_mean);

    // Matmul
    m.impl("mm", nova_mm);
    m.impl("addmm", nova_addmm);
    m.impl("bmm", nova_bmm);

    // Activation ops
    m.impl("relu", nova_relu);
    m.impl("sigmoid", nova_sigmoid);
    m.impl("tanh", nova_tanh);
    m.impl("threshold_backward", nova_threshold_backward);

    // Math ops
    m.impl("exp", nova_exp);
    m.impl("log", nova_log);
    m.impl("sqrt", nova_sqrt);
    m.impl("rsqrt", nova_rsqrt);
    m.impl("abs", nova_abs);
    m.impl("pow.Tensor_Scalar", nova_pow_tensor_scalar);
    m.impl("max.other", nova_max_other);
    m.impl("min.other", nova_min_other);
    m.impl("clamp", nova_clamp);

    // Comparison ops
    m.impl("eq.Scalar", nova_eq_scalar);
    m.impl("ne.Scalar", nova_ne_scalar);
    m.impl("gt.Scalar", nova_gt_scalar);
    m.impl("lt.Scalar", nova_lt_scalar);
    m.impl("ge.Scalar", nova_ge_scalar);
    m.impl("le.Scalar", nova_le_scalar);
    m.impl("where.self", nova_where_self);

    // Loss / softmax ops
    m.impl("_softmax", nova_softmax);
    m.impl("_log_softmax", nova_log_softmax);
    m.impl("nll_loss_forward", nova_nll_loss_forward);

    // -----------------------------------------------------------------------
    // .out variant registrations
    // -----------------------------------------------------------------------

    // Matmul .out
    m.impl("mm.out", nova_mm_out);
    m.impl("addmm.out", nova_addmm_out);
    m.impl("bmm.out", nova_bmm_out);

    // Elementwise .out
    m.impl("add.out", nova_add_out);
    m.impl("sub.out", nova_sub_out);
    m.impl("mul.out", nova_mul_out);
    m.impl("div.out", nova_div_out);
    m.impl("neg.out", nova_neg_out);

    // Math .out
    m.impl("exp.out", nova_exp_out);
    m.impl("log.out", nova_log_out);
    m.impl("sqrt.out", nova_sqrt_out);
    m.impl("rsqrt.out", nova_rsqrt_out);
    m.impl("abs.out", nova_abs_out);
    m.impl("pow.Tensor_Scalar_out", nova_pow_tensor_scalar_out);
    m.impl("maximum.out", nova_max_out);
    m.impl("minimum.out", nova_min_out);
    m.impl("clamp.out", nova_clamp_out);

    // Lerp ops (needed by Adam optimizer)
    m.impl("lerp.Scalar", nova_lerp_scalar);
    m.impl("lerp.Scalar_out", nova_lerp_scalar_out);
    m.impl("lerp_.Scalar", nova_lerp_scalar_inplace);

    // Extra in-place ops for optimizers
    m.impl("sqrt_", nova_sqrt_inplace);
    m.impl("div_.Scalar", nova_div_inplace_scalar);

    // Activation .out
    m.impl("relu.out", nova_relu_out);
    m.impl("sigmoid.out", nova_sigmoid_out);
    m.impl("tanh.out", nova_tanh_out);
    m.impl("threshold_backward.grad_input", nova_threshold_backward_out);

    // Comparison .out
    m.impl("eq.Scalar_out", nova_eq_scalar_out);
    m.impl("ne.Scalar_out", nova_ne_scalar_out);
    m.impl("gt.Scalar_out", nova_gt_scalar_out);
    m.impl("lt.Scalar_out", nova_lt_scalar_out);
    m.impl("ge.Scalar_out", nova_ge_scalar_out);
    m.impl("le.Scalar_out", nova_le_scalar_out);
    m.impl("where.self_out", nova_where_self_out);

    // Loss/softmax .out
    m.impl("_softmax.out", nova_softmax_out);
    m.impl("_log_softmax.out", nova_log_softmax_out);

    // Backward ops
    m.impl("nll_loss_backward.grad_input", nova_nll_loss_backward_grad_input);
    m.impl("_log_softmax_backward_data", nova_log_softmax_backward_data);

    // Reduction dim variants + .out
    m.impl("sum.dim_IntList", nova_sum_dim_intlist);
    m.impl("sum.IntList_out", nova_sum_intlist_out);
    m.impl("mean.dim", nova_mean_dim);
    m.impl("mean.out", nova_mean_out);

    // Cat ops
    m.impl("cat", nova_cat);
    m.impl("cat.out", nova_cat_out);

    // Extra ops
    m.impl("embedding", nova_embedding);
    m.impl("embedding_dense_backward", nova_embedding_dense_backward);
    m.impl("arange.start_step", nova_arange);
    m.impl("arange.start_out", nova_arange_out);
    m.impl("native_dropout", nova_native_dropout);
    m.impl("bernoulli_.float", nova_bernoulli_float);
    m.impl("nonzero", nova_nonzero);
    m.impl("scaled_dot_product_attention", nova_scaled_dot_product_attention);

    // Index ops
    m.impl("index.Tensor", nova_index_tensor);
    m.impl("index.Tensor_out", nova_index_tensor_out);
}
