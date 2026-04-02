#include "nova_ops.h"

#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/ArrayRef.h>

#include <algorithm>
#include <numeric>
#include <random>

// ===================================================================
// Helper: create a view tensor sharing the same storage
// ===================================================================

namespace {

at::Tensor make_view(
    const at::Tensor& self,
    c10::IntArrayRef sizes,
    c10::IntArrayRef strides,
    int64_t storage_offset) {

    auto result = at::detail::make_tensor<c10::TensorImpl>(
        c10::Storage(self.storage()),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        self.dtype());
    auto* impl = result.unsafeGetTensorImpl();
    impl->set_sizes_and_strides(sizes, strides);
    impl->set_storage_offset(storage_offset);
    return result;
}

/// Compute contiguous strides for a given shape.
std::vector<int64_t> contiguous_strides(c10::IntArrayRef sizes) {
    int64_t ndim = static_cast<int64_t>(sizes.size());
    std::vector<int64_t> strides(ndim);
    if (ndim == 0) return strides;
    strides[ndim - 1] = 1;
    for (int64_t i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    return strides;
}

/// Check if tensor is contiguous (row-major).
bool is_contiguous(const at::Tensor& self) {
    auto sizes = self.sizes();
    auto strides = self.strides();
    int64_t ndim = self.dim();
    if (ndim == 0) return true;
    int64_t expected = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        if (sizes[i] != 1 && strides[i] != expected) return false;
        expected *= sizes[i];
    }
    return true;
}

} // anonymous namespace

// ===================================================================
// Factory ops
// ===================================================================

// ---------------------------------------------------------------------------
// empty.memory_format  --  creates an uninitialised tensor on the Nova device
// ---------------------------------------------------------------------------

at::Tensor nova_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> /*layout_opt*/,
    std::optional<c10::Device> device_opt,
    std::optional<bool> /*pin_memory_opt*/,
    std::optional<c10::MemoryFormat> memory_format_opt) {

    auto dtype = dtype_opt.value_or(c10::ScalarType::Float);
    (void)device_opt; // always PrivateUse1:0

    auto* allocator = NovaAllocator::getInstance();
    int64_t nelements = c10::multiply_integers(size);
    int64_t nbytes = nelements * static_cast<int64_t>(c10::elementSize(dtype));

    auto storage = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        allocator->allocate(static_cast<size_t>(nbytes)),
        allocator,
        /*resizable=*/true);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dtype));

    // Always set sizes -- even for 0-dim (scalar) tensors where size = {}
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

    auto memory_format =
        memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
    // Preserve is not a real format — treat as Contiguous
    if (memory_format == c10::MemoryFormat::Preserve)
        memory_format = c10::MemoryFormat::Contiguous;
    if (memory_format != c10::MemoryFormat::Contiguous) {
        tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
    }

    return tensor;
}

// ---------------------------------------------------------------------------
// empty_strided  --  creates an uninitialised tensor with explicit strides
// ---------------------------------------------------------------------------

at::Tensor nova_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> /*layout_opt*/,
    std::optional<c10::Device> /*device_opt*/,
    std::optional<bool> /*pin_memory_opt*/) {

    auto dtype = dtype_opt.value_or(c10::ScalarType::Float);
    auto* allocator = NovaAllocator::getInstance();

    // Compute the minimum storage size implied by (size, stride).
    int64_t storage_size = 1;
    for (size_t i = 0; i < size.size(); ++i) {
        if (size[i] != 0) {
            storage_size += (size[i] - 1) * stride[i];
        }
    }
    int64_t nbytes =
        storage_size * static_cast<int64_t>(c10::elementSize(dtype));

    auto storage = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        allocator->allocate(static_cast<size_t>(nbytes)),
        allocator,
        /*resizable=*/true);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
        c10::scalarTypeToTypeMeta(dtype));

    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);

    return tensor;
}

// ---------------------------------------------------------------------------
// as_strided  --  view with explicit size, stride, storage_offset
// ---------------------------------------------------------------------------

at::Tensor nova_as_strided(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<int64_t> storage_offset) {

    auto offset = storage_offset.value_or(self.storage_offset());
    return make_view(self, size, stride, offset);
}

// ---------------------------------------------------------------------------
// _reshape_alias  --  reshape as a view (no copy)
// ---------------------------------------------------------------------------

at::Tensor nova_reshape_alias(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {

    return make_view(self, size, stride, self.storage_offset());
}

// ---------------------------------------------------------------------------
// fill_.Scalar  --  fill tensor with a constant value
// ---------------------------------------------------------------------------

at::Tensor& nova_fill_scalar(at::Tensor& self, const at::Scalar& value) {
    auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    struct { uint32_t numel; float value; } pc = {
        numel, value.toFloat()
    };

    VkBuffer buf = novatorch::getNovaBuffer(self);
    VkDeviceSize buf_size = static_cast<VkDeviceSize>(
        self.numel() * self.element_size());
    uint32_t groups = (numel + 255) / 256;

    dispatchCompute("fill_scalar",
                    /*num_buffers=*/1,
                    /*push_constant_size=*/sizeof(pc),
                    /*push_data=*/&pc,
                    &buf, &buf_size,
                    groups, 1, 1,
                    {self});
    return self;
}

// ---------------------------------------------------------------------------
// zero_  --  fill tensor with zeros
// ---------------------------------------------------------------------------

at::Tensor& nova_zero_(at::Tensor& self) {
    auto numel = static_cast<uint32_t>(self.numel());
    if (numel == 0) return self;

    struct { uint32_t numel; float value; } pc = { numel, 0.0f };

    VkBuffer buf = novatorch::getNovaBuffer(self);
    VkDeviceSize buf_size = static_cast<VkDeviceSize>(
        self.numel() * self.element_size());
    uint32_t groups = (numel + 255) / 256;

    dispatchCompute("fill_scalar",
                    /*num_buffers=*/1,
                    /*push_constant_size=*/sizeof(pc),
                    /*push_data=*/&pc,
                    &buf, &buf_size,
                    groups, 1, 1,
                    {self});
    return self;
}

// ---------------------------------------------------------------------------
// resize_  --  resize tensor storage if needed
// ---------------------------------------------------------------------------

const at::Tensor& nova_resize_(
    const at::Tensor& self,
    c10::IntArrayRef size,
    std::optional<c10::MemoryFormat> memory_format) {

    auto* impl = self.unsafeGetTensorImpl();

    // Compute new number of elements
    int64_t new_numel = 1;
    for (auto s : size) new_numel *= s;

    int64_t old_numel = self.numel();
    int64_t new_nbytes = new_numel * self.element_size();
    int64_t old_nbytes = old_numel * self.element_size();

    if (new_nbytes > old_nbytes) {
        // Need to reallocate - create a new storage via our allocator
        auto new_storage = c10::Storage(
            c10::Storage::use_byte_size_t(),
            new_nbytes,
            NovaAllocator::getInstance(),
            /*resizable=*/true);
        impl->set_storage_keep_dtype(std::move(new_storage));
    }

    // Update sizes/strides (inline contiguous stride computation)
    int64_t ndim = static_cast<int64_t>(size.size());
    std::vector<int64_t> strides(ndim);
    if (ndim > 0) {
        strides[ndim - 1] = 1;
        for (int64_t i = ndim - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * size[i + 1];
    }
    impl->set_sizes_and_strides(size, strides);
    impl->set_storage_offset(0);

    return self;
}

// ===================================================================
// View / reshape ops  (zero-copy unless noted)
// ===================================================================

// ---------------------------------------------------------------------------
// t  --  transpose a 2D tensor (swap dims 0 and 1)
// ---------------------------------------------------------------------------

at::Tensor nova_t(const at::Tensor& self) {
    TORCH_CHECK(
        self.dim() <= 2,
        "nova_t: expected a tensor with <= 2 dimensions, got ",
        self.dim());

    if (self.dim() < 2) return self;

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[0], sizes[1]);
    std::swap(strides[0], strides[1]);

    return make_view(self, sizes, strides, self.storage_offset());
}

// ---------------------------------------------------------------------------
// transpose.int  --  swap two dimensions
// ---------------------------------------------------------------------------

at::Tensor nova_transpose_int(
    const at::Tensor& self, int64_t dim0, int64_t dim1) {

    int64_t ndim = self.dim();
    TORCH_CHECK(ndim > 0, "nova_transpose: expected non-scalar tensor");

    // Wrap negative dims
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    TORCH_CHECK(
        dim0 >= 0 && dim0 < ndim && dim1 >= 0 && dim1 < ndim,
        "nova_transpose: dim out of range");

    if (dim0 == dim1) return self;

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[dim0], sizes[dim1]);
    std::swap(strides[dim0], strides[dim1]);

    return make_view(self, sizes, strides, self.storage_offset());
}

// ---------------------------------------------------------------------------
// permute  --  reorder dimensions
// ---------------------------------------------------------------------------

at::Tensor nova_permute(
    const at::Tensor& self, c10::IntArrayRef dims) {

    int64_t ndim = self.dim();
    TORCH_CHECK(
        static_cast<int64_t>(dims.size()) == ndim,
        "nova_permute: number of dims doesn't match");

    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen(ndim, false);

    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        TORCH_CHECK(
            d >= 0 && d < ndim && !seen[d],
            "nova_permute: invalid or repeated dim");
        seen[d] = true;
        new_sizes[i] = old_sizes[d];
        new_strides[i] = old_strides[d];
    }

    return make_view(self, new_sizes, new_strides, self.storage_offset());
}

// ---------------------------------------------------------------------------
// expand  --  broadcast to larger size (stride=0 for broadcast dims)
// ---------------------------------------------------------------------------

at::Tensor nova_expand(
    const at::Tensor& self, c10::IntArrayRef size, bool /*implicit*/) {

    int64_t ndim = static_cast<int64_t>(size.size());
    TORCH_CHECK(
        ndim >= self.dim(),
        "nova_expand: target ndim (", ndim,
        ") must be >= self.dim() (", self.dim(),
        "). self.sizes()=", self.sizes(),
        ", target size=", size);

    auto old_sizes = self.sizes();
    auto old_strides = self.strides();
    int64_t old_ndim = self.dim();

    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);

    // Align from the right
    for (int64_t i = ndim - 1; i >= 0; --i) {
        int64_t old_i = i - (ndim - old_ndim);
        if (old_i < 0) {
            // New leading dimension: must broadcast
            TORCH_CHECK(
                size[i] >= 0,
                "nova_expand: expanded size must be non-negative");
            new_sizes[i] = size[i];
            new_strides[i] = 0;
        } else if (old_sizes[old_i] == 1) {
            // Broadcast from size 1
            new_sizes[i] = size[i];
            new_strides[i] = (size[i] == 1) ? old_strides[old_i] : 0;
        } else {
            // Must match exactly or be -1 (keep existing)
            TORCH_CHECK(
                size[i] == old_sizes[old_i] || size[i] == -1,
                "nova_expand: expanded size (",
                size[i],
                ") must match existing size (",
                old_sizes[old_i],
                ") at non-singleton dimension ",
                i);
            new_sizes[i] = old_sizes[old_i];
            new_strides[i] = old_strides[old_i];
        }
    }

    return make_view(self, new_sizes, new_strides, self.storage_offset());
}

// ---------------------------------------------------------------------------
// view  --  reshape (requires contiguous input)
// ---------------------------------------------------------------------------

at::Tensor nova_view(
    const at::Tensor& self, c10::IntArrayRef size) {

    // Make contiguous if needed (einsum and transpose produce non-contiguous views)
    at::Tensor input = is_contiguous(self) ? self : self.contiguous();

    // Resolve -1 dimension
    int64_t numel = input.numel();
    std::vector<int64_t> new_sizes(size.begin(), size.end());
    int64_t neg_idx = -1;
    int64_t product = 1;

    for (int64_t i = 0; i < static_cast<int64_t>(new_sizes.size()); ++i) {
        if (new_sizes[i] == -1) {
            TORCH_CHECK(neg_idx == -1, "nova_view: only one -1 allowed");
            neg_idx = i;
        } else {
            TORCH_CHECK(new_sizes[i] >= 0, "nova_view: invalid shape");
            product *= new_sizes[i];
        }
    }

    if (neg_idx >= 0) {
        TORCH_CHECK(
            product != 0 || numel == 0,
            "nova_view: cannot infer -1 with zero-sized dimensions");
        new_sizes[neg_idx] = (product != 0) ? numel / product : 0;
    }

    // Verify element count matches
    int64_t new_numel = c10::multiply_integers(new_sizes);
    TORCH_CHECK(
        new_numel == numel,
        "nova_view: shape [", c10::IntArrayRef(new_sizes),
        "] is invalid for input of size ", numel);

    auto new_strides = contiguous_strides(new_sizes);
    return make_view(input, new_sizes, new_strides, input.storage_offset());
}

// ---------------------------------------------------------------------------
// reshape  --  try view first, fall back to copy + view
// ---------------------------------------------------------------------------

at::Tensor nova_reshape(
    const at::Tensor& self, c10::IntArrayRef shape) {

    // If contiguous, just do a view
    if (is_contiguous(self)) {
        return nova_view(self, shape);
    }

    // Otherwise: make contiguous copy, then view
    at::Tensor contig = self.contiguous();
    return nova_view(contig, shape);
}

// ---------------------------------------------------------------------------
// slice.Tensor  --  slice along a dimension
// ---------------------------------------------------------------------------

at::Tensor nova_slice_tensor(
    const at::Tensor& self,
    int64_t dim,
    std::optional<int64_t> start_opt,
    std::optional<int64_t> end_opt,
    int64_t step) {

    int64_t ndim = self.dim();
    TORCH_CHECK(ndim > 0, "nova_slice: cannot slice a 0-dim tensor");

    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "nova_slice: dim out of range");

    int64_t dim_size = self.size(dim);
    int64_t start = start_opt.value_or(0);
    int64_t end = end_opt.value_or(dim_size);

    // Clamp negative indices
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;

    // Clamp to valid range
    start = std::max(int64_t(0), std::min(start, dim_size));
    end = std::max(start, std::min(end, dim_size));

    TORCH_CHECK(step > 0, "nova_slice: step must be positive");

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    int64_t offset = self.storage_offset() + start * strides[dim];

    sizes[dim] = (end - start + step - 1) / step;
    strides[dim] *= step;

    return make_view(self, sizes, strides, offset);
}

// ---------------------------------------------------------------------------
// select.int  --  select a single index along a dimension
// ---------------------------------------------------------------------------

at::Tensor nova_select_int(
    const at::Tensor& self, int64_t dim, int64_t index) {

    int64_t ndim = self.dim();
    TORCH_CHECK(ndim > 0, "nova_select: cannot select from a 0-dim tensor");

    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "nova_select: dim out of range");

    int64_t dim_size = self.size(dim);
    if (index < 0) index += dim_size;
    TORCH_CHECK(
        index >= 0 && index < dim_size,
        "nova_select: index ", index, " out of range for dim of size ",
        dim_size);

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    int64_t offset = self.storage_offset() + index * strides[dim];

    // Remove the selected dimension
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    // For 0-dim result (selecting from 1-dim tensor), return scalar tensor
    if (sizes.empty()) {
        return make_view(self, sizes, strides, offset);
    }

    return make_view(self, sizes, strides, offset);
}

// ---------------------------------------------------------------------------
// unsqueeze  --  insert a dimension of size 1
// ---------------------------------------------------------------------------

at::Tensor nova_unsqueeze(const at::Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    // dim can be in range [-ndim-1, ndim]
    if (dim < 0) dim += ndim + 1;
    TORCH_CHECK(
        dim >= 0 && dim <= ndim,
        "nova_unsqueeze: dim out of range");

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();

    // Stride for the new dim: if inserted at the end, 1; otherwise,
    // sizes[dim] * strides[dim] of the existing dim at that position.
    int64_t new_stride = (dim < ndim) ? sizes[dim] * strides[dim] : 1;
    // Special case: if ndim == 0 (scalar), stride is 1
    if (ndim == 0) new_stride = 1;

    sizes.insert(sizes.begin() + dim, 1);
    strides.insert(strides.begin() + dim, new_stride);

    return make_view(self, sizes, strides, self.storage_offset());
}

// ---------------------------------------------------------------------------
// squeeze.dim  --  remove a dimension of size 1
// ---------------------------------------------------------------------------

at::Tensor nova_squeeze_dim(const at::Tensor& self, int64_t dim) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    TORCH_CHECK(
        dim >= 0 && dim < ndim,
        "nova_squeeze_dim: dim out of range");

    // If the dimension is not size 1, return self unchanged
    if (self.size(dim) != 1) {
        return self;
    }

    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    return make_view(self, sizes, strides, self.storage_offset());
}

// ---------------------------------------------------------------------------
// contiguous  --  return contiguous tensor (no-op if already contiguous)
// ---------------------------------------------------------------------------

at::Tensor nova_contiguous(
    const at::Tensor& self, c10::MemoryFormat memory_format) {

    if (memory_format == c10::MemoryFormat::Contiguous && is_contiguous(self)) {
        return self;
    }

    // Allocate new contiguous tensor and copy
    at::Tensor output = at::empty(self.sizes(), self.options());
    auto numel = self.numel();
    if (numel == 0) return output;

    auto src_sizes   = self.sizes();
    auto src_strides = self.strides();
    int64_t ndim     = self.dim();
    int64_t src_off  = self.storage_offset();

    // Download source device buffer into staging, perform strided->contiguous
    // gather on CPU, then upload the contiguous result to the output device buffer.
    novatorch::withStagingRead(self, [&](const void* src_raw, size_t) {
        const float* src = static_cast<const float*>(src_raw);

        novatorch::withStagingWrite(output, [&](void* dst_raw, size_t) {
            float* dst = static_cast<float*>(dst_raw);

            if (ndim == 0) {
                dst[0] = src[src_off];
            } else {
                std::vector<int64_t> indices(ndim, 0);
                for (int64_t flat = 0; flat < numel; ++flat) {
                    int64_t src_offset = src_off;
                    for (int64_t d = 0; d < ndim; ++d) {
                        src_offset += indices[d] * src_strides[d];
                    }
                    dst[flat] = src[src_offset];

                    for (int64_t d = ndim - 1; d >= 0; --d) {
                        indices[d]++;
                        if (indices[d] < src_sizes[d]) break;
                        indices[d] = 0;
                    }
                }
            }
        });
    });

    return output;
}

// ---------------------------------------------------------------------------
// clone  --  always copy to new tensor
// ---------------------------------------------------------------------------

at::Tensor nova_clone(
    const at::Tensor& self,
    std::optional<c10::MemoryFormat> memory_format_opt) {

    auto memory_format =
        memory_format_opt.value_or(c10::MemoryFormat::Preserve);

    if (memory_format == c10::MemoryFormat::Preserve) {
        // Preserve the input layout
        if (is_contiguous(self)) {
            at::Tensor output = at::empty(self.sizes(), self.options());
            auto nbytes = self.numel() * self.element_size();
            if (nbytes > 0) {
                novatorch::copyDeviceToDevice(
                    self, output, static_cast<size_t>(nbytes));
            }
            return output;
        }
        // Non-contiguous: use contiguous copy
        return nova_contiguous(self, c10::MemoryFormat::Contiguous);
    }

    // Specific format requested: make contiguous in that format
    return nova_contiguous(self, memory_format);
}

// ---------------------------------------------------------------------------
// detach  --  return alias with autograd stripped
// ---------------------------------------------------------------------------

at::Tensor nova_detach(const at::Tensor& self) {
    return make_view(
        self, self.sizes(), self.strides(), self.storage_offset());
}

// ===================================================================
// Factory / utility ops
// ===================================================================

// ---------------------------------------------------------------------------
// zeros_like
// ---------------------------------------------------------------------------

at::Tensor nova_zeros_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {

    auto dtype = dtype_opt.value_or(self.scalar_type());
    auto result = nova_empty_memory_format(
        self.sizes(), dtype, layout_opt, device_opt,
        pin_memory_opt, memory_format_opt);
    nova_zero_(result);
    return result;
}

// ---------------------------------------------------------------------------
// ones_like
// ---------------------------------------------------------------------------

at::Tensor nova_ones_like(
    const at::Tensor& self,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {

    auto dtype = dtype_opt.value_or(self.scalar_type());
    auto result = nova_empty_memory_format(
        self.sizes(), dtype, layout_opt, device_opt,
        pin_memory_opt, memory_format_opt);
    nova_fill_scalar(result, at::Scalar(1.0));
    return result;
}

// ---------------------------------------------------------------------------
// full_like
// ---------------------------------------------------------------------------

at::Tensor nova_full_like(
    const at::Tensor& self,
    const at::Scalar& fill_value,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {

    auto dtype = dtype_opt.value_or(self.scalar_type());
    auto result = nova_empty_memory_format(
        self.sizes(), dtype, layout_opt, device_opt,
        pin_memory_opt, memory_format_opt);
    nova_fill_scalar(result, fill_value);
    return result;
}

// ---------------------------------------------------------------------------
// scalar_tensor  --  create a 0-dim tensor from a scalar
// ---------------------------------------------------------------------------

at::Tensor nova_scalar_tensor(
    const at::Scalar& s,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {

    auto dtype = dtype_opt.value_or(c10::ScalarType::Float);
    auto result = nova_empty_memory_format(
        /*size=*/{}, dtype, layout_opt, device_opt,
        pin_memory_opt, /*memory_format=*/std::nullopt);
    nova_fill_scalar(result, s);
    return result;
}

// ---------------------------------------------------------------------------
// uniform_  --  fill with uniform random values (CPU RNG, write to mapped mem)
// ---------------------------------------------------------------------------

at::Tensor& nova_uniform_(
    at::Tensor& self,
    double from,
    double to,
    std::optional<at::Generator> /*gen*/) {

    auto numel = self.numel();
    if (numel == 0) return self;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(
        static_cast<float>(from), static_cast<float>(to));

    novatorch::withStagingWrite(self, [&](void* ptr_raw, size_t) {
        float* ptr = static_cast<float*>(ptr_raw);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = dist(rng);
        }
    });
    return self;
}

// ---------------------------------------------------------------------------
// normal_  --  fill with normal random values (CPU RNG, write to mapped mem)
// ---------------------------------------------------------------------------

at::Tensor& nova_normal_(
    at::Tensor& self,
    double mean,
    double std_val,
    std::optional<at::Generator> /*gen*/) {

    auto numel = self.numel();
    if (numel == 0) return self;

    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(
        static_cast<float>(mean), static_cast<float>(std_val));

    novatorch::withStagingWrite(self, [&](void* ptr_raw, size_t) {
        float* ptr = static_cast<float*>(ptr_raw);
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = dist(rng);
        }
    });
    return self;
}
