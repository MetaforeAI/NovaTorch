#include "nova_ops.h"

// ---------------------------------------------------------------------------
// Push constant layout -- must match matmul.comp
// ---------------------------------------------------------------------------

struct MatmulPC {
    uint32_t M;  // rows of A / rows of C
    uint32_t N;  // cols of B / cols of C
    uint32_t K;  // cols of A / rows of B
};

static_assert(sizeof(MatmulPC) == 12, "MatmulPC must be 12 bytes");

// ---------------------------------------------------------------------------
// nova_mm -- matrix multiplication  C[M,N] = A[M,K] * B[K,N]
// ---------------------------------------------------------------------------

at::Tensor nova_mm(
    const at::Tensor& self,
    const at::Tensor& mat2) {

    TORCH_CHECK(self.dim() == 2,
        "nova_mm: self must be a 2D matrix, got ", self.dim(), "D");
    TORCH_CHECK(mat2.dim() == 2,
        "nova_mm: mat2 must be a 2D matrix, got ", mat2.dim(), "D");
    TORCH_CHECK(self.size(1) == mat2.size(0),
        "nova_mm: incompatible dimensions ",
        self.size(0), "x", self.size(1),
        " and ", mat2.size(0), "x", mat2.size(1));
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    auto mat2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    TORCH_CHECK(
        self.scalar_type() == at::ScalarType::Float,
        "nova_mm: only float32 supported");
    TORCH_CHECK(
        mat2.scalar_type() == at::ScalarType::Float,
        "nova_mm: only float32 supported");

    const auto M = static_cast<uint32_t>(self.size(0));
    const auto K = static_cast<uint32_t>(self.size(1));
    const auto N = static_cast<uint32_t>(mat2.size(1));

    auto output = at::empty({self_c.size(0), mat2_c.size(1)}, self_c.options());

    VkBuffer buf_a = novatorch::getNovaBuffer(self_c);
    VkBuffer buf_b = novatorch::getNovaBuffer(mat2_c);
    VkBuffer buf_c = novatorch::getNovaBuffer(output);

    auto* alloc_a = novatorch::getNovaAllocation(self_c);
    auto* alloc_b = novatorch::getNovaAllocation(mat2_c);
    auto* alloc_c = novatorch::getNovaAllocation(output);

    VkBuffer bufs[3] = {buf_a, buf_b, buf_c};
    VkDeviceSize sizes[3] = {
        static_cast<VkDeviceSize>(alloc_a->size),
        static_cast<VkDeviceSize>(alloc_b->size),
        static_cast<VkDeviceSize>(alloc_c->size)
    };

    MatmulPC pc{M, N, K};

    // Workgroup size is 16x16 in the shader
    constexpr uint32_t TILE = 16;
    uint32_t groups_x = (N + TILE - 1) / TILE;
    uint32_t groups_y = (M + TILE - 1) / TILE;

    novatorch::flushNovaBuffer(self_c);
    novatorch::flushNovaBuffer(mat2_c);

    dispatchCompute(
        "matmul", 3, sizeof(pc), &pc, bufs, sizes, groups_x, groups_y);

    return output;
}

// ---------------------------------------------------------------------------
// Push constant layout -- must match addmm.comp
// ---------------------------------------------------------------------------

struct AddmmPC {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    float beta;
    float alpha;
    uint32_t bias_is_1d;
};

static_assert(sizeof(AddmmPC) == 24, "AddmmPC must be 24 bytes");

// ---------------------------------------------------------------------------
// nova_addmm -- out = beta * self + alpha * (mat1 @ mat2)
// ---------------------------------------------------------------------------

at::Tensor nova_addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {

    TORCH_CHECK(mat1.dim() == 2,
        "nova_addmm: mat1 must be a 2D matrix, got ", mat1.dim(), "D");
    TORCH_CHECK(mat2.dim() == 2,
        "nova_addmm: mat2 must be a 2D matrix, got ", mat2.dim(), "D");
    TORCH_CHECK(mat1.size(1) == mat2.size(0),
        "nova_addmm: incompatible dimensions ",
        mat1.size(0), "x", mat1.size(1),
        " and ", mat2.size(0), "x", mat2.size(1));
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2,
        "nova_addmm: bias must be 1D or 2D, got ", self.dim(), "D");
    if (self.dim() == 1) {
        TORCH_CHECK(self.size(0) == mat2.size(1),
            "nova_addmm: 1D bias size ", self.size(0),
            " != N ", mat2.size(1));
    } else {
        TORCH_CHECK(self.size(0) == mat1.size(0) && self.size(1) == mat2.size(1),
            "nova_addmm: 2D bias shape [", self.size(0), ",", self.size(1),
            "] != output shape [", mat1.size(0), ",", mat2.size(1), "]");
    }
    auto mat1_c = mat1.is_contiguous() ? mat1 : mat1.contiguous();
    auto mat2_c = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    auto self_c = self.is_contiguous() ? self : self.contiguous();
    TORCH_CHECK(mat1_c.scalar_type() == at::ScalarType::Float,
        "nova_addmm: only float32 supported");
    TORCH_CHECK(mat2_c.scalar_type() == at::ScalarType::Float,
        "nova_addmm: only float32 supported");
    TORCH_CHECK(self_c.scalar_type() == at::ScalarType::Float,
        "nova_addmm: only float32 supported");

    const auto M = static_cast<uint32_t>(mat1_c.size(0));
    const auto K = static_cast<uint32_t>(mat1_c.size(1));
    const auto N = static_cast<uint32_t>(mat2_c.size(1));

    auto output = at::empty({mat1_c.size(0), mat2_c.size(1)}, mat1_c.options());

    VkBuffer buf_bias = novatorch::getNovaBuffer(self_c);
    VkBuffer buf_a    = novatorch::getNovaBuffer(mat1_c);
    VkBuffer buf_b    = novatorch::getNovaBuffer(mat2_c);
    VkBuffer buf_out  = novatorch::getNovaBuffer(output);

    auto* alloc_bias = novatorch::getNovaAllocation(self_c);
    auto* alloc_a    = novatorch::getNovaAllocation(mat1_c);
    auto* alloc_b    = novatorch::getNovaAllocation(mat2_c);
    auto* alloc_out  = novatorch::getNovaAllocation(output);

    VkBuffer bufs[4] = {buf_bias, buf_a, buf_b, buf_out};
    VkDeviceSize sizes[4] = {
        static_cast<VkDeviceSize>(alloc_bias->size),
        static_cast<VkDeviceSize>(alloc_a->size),
        static_cast<VkDeviceSize>(alloc_b->size),
        static_cast<VkDeviceSize>(alloc_out->size)
    };

    AddmmPC pc{M, N, K, beta.toFloat(), alpha.toFloat(),
               static_cast<uint32_t>(self_c.dim() == 1 ? 1 : 0)};

    constexpr uint32_t TILE = 16;
    uint32_t groups_x = (N + TILE - 1) / TILE;
    uint32_t groups_y = (M + TILE - 1) / TILE;

    novatorch::flushNovaBuffer(self_c);
    novatorch::flushNovaBuffer(mat1_c);
    novatorch::flushNovaBuffer(mat2_c);

    dispatchCompute(
        "addmm", 4, sizeof(pc), &pc, bufs, sizes, groups_x, groups_y);

    return output;
}

// ---------------------------------------------------------------------------
// Push constant layout -- must match bmm.comp
// ---------------------------------------------------------------------------

struct BmmPC {
    uint32_t batch;
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

static_assert(sizeof(BmmPC) == 16, "BmmPC must be 16 bytes");

// ---------------------------------------------------------------------------
// nova_bmm -- batched matrix multiplication  C[b,M,N] = A[b,M,K] * B[b,K,N]
// ---------------------------------------------------------------------------

at::Tensor nova_bmm(
    const at::Tensor& self,
    const at::Tensor& mat2) {

    TORCH_CHECK(self.dim() == 3,
        "nova_bmm: self must be a 3D tensor, got ", self.dim(), "D");
    TORCH_CHECK(mat2.dim() == 3,
        "nova_bmm: mat2 must be a 3D tensor, got ", mat2.dim(), "D");
    TORCH_CHECK(self.size(0) == mat2.size(0),
        "nova_bmm: batch sizes must match, got ",
        self.size(0), " and ", mat2.size(0));
    TORCH_CHECK(self.size(2) == mat2.size(1),
        "nova_bmm: incompatible dimensions ",
        self.size(1), "x", self.size(2),
        " and ", mat2.size(1), "x", mat2.size(2));
    auto self_c2 = self.is_contiguous() ? self : self.contiguous();
    auto mat2_c2 = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    TORCH_CHECK(self.scalar_type() == at::ScalarType::Float,
        "nova_bmm: only float32 supported");
    TORCH_CHECK(mat2.scalar_type() == at::ScalarType::Float,
        "nova_bmm: only float32 supported");

    const auto B = static_cast<uint32_t>(self_c2.size(0));
    const auto M = static_cast<uint32_t>(self_c2.size(1));
    const auto K = static_cast<uint32_t>(self_c2.size(2));
    const auto N = static_cast<uint32_t>(mat2_c2.size(2));

    auto output = at::empty({self_c2.size(0), self_c2.size(1), mat2_c2.size(2)},
                            self_c2.options());

    VkBuffer buf_a = novatorch::getNovaBuffer(self_c2);
    VkBuffer buf_b = novatorch::getNovaBuffer(mat2_c2);
    VkBuffer buf_c = novatorch::getNovaBuffer(output);

    auto* alloc_a = novatorch::getNovaAllocation(self_c2);
    auto* alloc_b = novatorch::getNovaAllocation(mat2_c2);
    auto* alloc_c = novatorch::getNovaAllocation(output);

    VkBuffer bufs[3] = {buf_a, buf_b, buf_c};
    VkDeviceSize sizes[3] = {
        static_cast<VkDeviceSize>(alloc_a->size),
        static_cast<VkDeviceSize>(alloc_b->size),
        static_cast<VkDeviceSize>(alloc_c->size)
    };

    BmmPC pc{B, M, N, K};

    constexpr uint32_t TILE = 16;
    uint32_t groups_x = (N + TILE - 1) / TILE;
    uint32_t groups_y = (M + TILE - 1) / TILE;
    uint32_t groups_z = B;

    novatorch::flushNovaBuffer(self_c2);
    novatorch::flushNovaBuffer(mat2_c2);

    dispatchCompute(
        "bmm", 3, sizeof(pc), &pc, bufs, sizes, groups_x, groups_y, groups_z);

    return output;
}

// ---------------------------------------------------------------------------
// .out variants
// ---------------------------------------------------------------------------

at::Tensor& nova_mm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
    auto result = nova_mm(self, mat2);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_addmm_out(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
    auto result = nova_addmm(self, mat1, mat2, beta, alpha);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}

at::Tensor& nova_bmm_out(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
    auto result = nova_bmm(self, mat2);
    out.resize_as_(result);
    out.copy_(result);
    return out;
}
