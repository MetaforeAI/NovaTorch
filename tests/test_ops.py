"""Comprehensive unit tests for all registered NovaTorch ops.

Tests every op registered in nova_register.cpp by:
1. Creating CPU reference tensors
2. Copying to nova device
3. Executing the op on nova
4. Copying result back to CPU
5. Comparing against CPU reference (atol=1e-4, rtol=1e-3)
"""
import pytest
import torch

DEVICE = torch.device("nova")
ATOL = 1e-4
RTOL = 1e-3
SIZES = [16, 256, 4096]


def assert_close(nova_result, cpu_expected, atol=ATOL, rtol=RTOL):
    """Move nova tensor to CPU and compare against expected."""
    actual = nova_result.cpu() if nova_result.device.type != "cpu" else nova_result
    expected = cpu_expected.cpu() if cpu_expected.device.type != "cpu" else cpu_expected
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def to_nova(t):
    """Copy a CPU tensor to nova device."""
    return t.to(DEVICE)


# =============================================================================
# Factory Ops
# =============================================================================

class TestFactoryOps:
    """Tests for empty.memory_format, empty_strided, fill_, zero_,
    zeros_like, ones_like, full_like, scalar_tensor, uniform_, normal_."""

    @pytest.mark.parametrize("size", SIZES)
    def test_empty_memory_format(self, size):
        t = torch.empty(size, device=DEVICE)
        assert t.shape == (size,)
        assert t.device.type == "privateuseone" or str(t.device).startswith("nova")

    @pytest.mark.parametrize("size", SIZES)
    def test_empty_strided(self, size):
        t = torch.empty_strided((size,), (1,), device=DEVICE)
        assert t.shape == (size,)
        assert t.stride() == (1,)

    @pytest.mark.parametrize("size", SIZES)
    def test_fill_scalar(self, size):
        t = torch.empty(size, device=DEVICE)
        t.fill_(3.14)
        expected = torch.full((size,), 3.14)
        assert_close(t, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_zero_(self, size):
        t = torch.empty(size, device=DEVICE)
        t.zero_()
        expected = torch.zeros(size)
        assert_close(t, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_zeros_like(self, size):
        ref = torch.randn(size, device=DEVICE)
        t = torch.zeros_like(ref)
        expected = torch.zeros(size)
        assert_close(t, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_ones_like(self, size):
        ref = torch.randn(size, device=DEVICE)
        t = torch.ones_like(ref)
        expected = torch.ones(size)
        assert_close(t, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_full_like(self, size):
        ref = torch.randn(size, device=DEVICE)
        t = torch.full_like(ref, 2.718)
        expected = torch.full((size,), 2.718)
        assert_close(t, expected)

    def test_scalar_tensor(self):
        t = torch.scalar_tensor(42.0, device=DEVICE)
        assert t.dim() == 0
        assert_close(t, torch.tensor(42.0))

    @pytest.mark.parametrize("size", SIZES)
    def test_uniform_(self, size):
        t = torch.empty(size, device=DEVICE)
        t.uniform_(0.0, 1.0)
        vals = t.cpu()
        assert vals.min() >= 0.0 - 1e-6
        assert vals.max() <= 1.0 + 1e-6

    @pytest.mark.parametrize("size", SIZES)
    def test_normal_(self, size):
        t = torch.empty(size, device=DEVICE)
        t.normal_(0.0, 1.0)
        vals = t.cpu()
        # Statistical check: mean should be roughly 0, std roughly 1
        # Use loose bounds since small samples can deviate
        if size >= 256:
            assert abs(vals.mean().item()) < 1.0
            assert abs(vals.std().item() - 1.0) < 1.0


# =============================================================================
# View / Reshape Ops
# =============================================================================

class TestViewReshapeOps:
    """Tests for t, transpose.int, permute, expand, view, reshape,
    slice.Tensor, select.int, unsqueeze, squeeze.dim, contiguous, clone, detach."""

    def test_t(self):
        cpu = torch.randn(4, 8)
        nova = to_nova(cpu)
        result = nova.t()
        expected = cpu.t()
        assert result.shape == expected.shape
        assert_close(result.contiguous(), expected.contiguous())

    def test_transpose_int(self):
        cpu = torch.randn(3, 5, 7)
        nova = to_nova(cpu)
        result = nova.transpose(0, 2)
        expected = cpu.transpose(0, 2)
        assert result.shape == expected.shape
        assert_close(result.contiguous(), expected.contiguous())

    def test_permute(self):
        cpu = torch.randn(2, 3, 4)
        nova = to_nova(cpu)
        result = nova.permute(2, 0, 1)
        expected = cpu.permute(2, 0, 1)
        assert result.shape == expected.shape
        assert_close(result.contiguous(), expected.contiguous())

    def test_expand(self):
        cpu = torch.randn(1, 4)
        nova = to_nova(cpu)
        result = nova.expand(3, 4)
        expected = cpu.expand(3, 4)
        assert result.shape == expected.shape
        assert_close(result.contiguous(), expected.contiguous())

    @pytest.mark.parametrize("size", SIZES)
    def test_view(self, size):
        cpu = torch.randn(size)
        nova = to_nova(cpu)
        result = nova.view(size // 4, 4)
        expected = cpu.view(size // 4, 4)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_reshape(self, size):
        cpu = torch.randn(size)
        nova = to_nova(cpu)
        result = nova.reshape(4, size // 4)
        expected = cpu.reshape(4, size // 4)
        assert_close(result, expected)

    def test_slice_tensor(self):
        cpu = torch.randn(10, 8)
        nova = to_nova(cpu)
        result = nova[2:7, :4]
        expected = cpu[2:7, :4]
        assert_close(result.contiguous(), expected.contiguous())

    def test_select_int(self):
        cpu = torch.randn(5, 8)
        nova = to_nova(cpu)
        result = nova.select(0, 3)
        expected = cpu.select(0, 3)
        assert_close(result.contiguous(), expected.contiguous())

    def test_unsqueeze(self):
        cpu = torch.randn(4, 8)
        nova = to_nova(cpu)
        result = nova.unsqueeze(1)
        expected = cpu.unsqueeze(1)
        assert result.shape == expected.shape
        assert_close(result.contiguous(), expected.contiguous())

    def test_squeeze_dim(self):
        cpu = torch.randn(4, 1, 8)
        nova = to_nova(cpu)
        result = nova.squeeze(1)
        expected = cpu.squeeze(1)
        assert result.shape == expected.shape
        assert_close(result.contiguous(), expected.contiguous())

    @pytest.mark.parametrize("size", SIZES)
    def test_contiguous(self, size):
        cpu = torch.randn(size)
        nova = to_nova(cpu)
        result = nova.contiguous()
        assert_close(result, cpu)

    @pytest.mark.parametrize("size", SIZES)
    def test_clone(self, size):
        cpu = torch.randn(size)
        nova = to_nova(cpu)
        result = nova.clone()
        assert_close(result, cpu)

    @pytest.mark.parametrize("size", SIZES)
    def test_detach(self, size):
        cpu = torch.randn(size)
        nova = to_nova(cpu)
        result = nova.detach()
        assert_close(result, cpu)


# =============================================================================
# Copy Ops
# =============================================================================

class TestCopyOps:
    """Tests for copy between CPU<->nova, _local_scalar_dense (tensor.item())."""

    @pytest.mark.parametrize("size", SIZES)
    def test_cpu_to_nova_and_back(self, size):
        cpu = torch.randn(size)
        nova = cpu.to(DEVICE)
        back = nova.cpu()
        assert_close(back, cpu)

    @pytest.mark.parametrize("size", SIZES)
    def test_copy_2d(self, size):
        cpu = torch.randn(size // 4, 4)
        nova = cpu.to(DEVICE)
        back = nova.cpu()
        assert_close(back, cpu)

    def test_local_scalar_dense(self):
        cpu = torch.tensor(3.14)
        nova = cpu.to(DEVICE)
        val = nova.item()
        assert abs(val - 3.14) < ATOL


# =============================================================================
# In-place Ops
# =============================================================================

class TestInplaceOps:
    """Tests for copy_, add_.Tensor, sub_.Tensor, mul_.Tensor, mul_.Scalar,
    div_.Tensor, addcmul_, addcdiv_."""

    @pytest.mark.parametrize("size", SIZES)
    def test_copy_(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        a_nova = to_nova(a_cpu.clone())
        b_nova = to_nova(b_cpu)
        a_nova.copy_(b_nova)
        assert_close(a_nova, b_cpu)

    @pytest.mark.parametrize("size", SIZES)
    def test_add_inplace(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = a_cpu.clone().add_(b_cpu)
        a_nova = to_nova(a_cpu)
        b_nova = to_nova(b_cpu)
        a_nova.add_(b_nova)
        assert_close(a_nova, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_add_inplace_alpha(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = a_cpu.clone().add_(b_cpu, alpha=0.5)
        a_nova = to_nova(a_cpu)
        b_nova = to_nova(b_cpu)
        a_nova.add_(b_nova, alpha=0.5)
        assert_close(a_nova, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_sub_inplace(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = a_cpu.clone().sub_(b_cpu)
        a_nova = to_nova(a_cpu)
        b_nova = to_nova(b_cpu)
        a_nova.sub_(b_nova)
        assert_close(a_nova, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_mul_inplace_tensor(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = a_cpu.clone().mul_(b_cpu)
        a_nova = to_nova(a_cpu)
        b_nova = to_nova(b_cpu)
        a_nova.mul_(b_nova)
        assert_close(a_nova, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_mul_inplace_scalar(self, size):
        a_cpu = torch.randn(size)
        expected = a_cpu.clone().mul_(2.5)
        a_nova = to_nova(a_cpu)
        a_nova.mul_(2.5)
        assert_close(a_nova, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_div_inplace_tensor(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size).clamp(min=0.1)  # avoid division by near-zero
        expected = a_cpu.clone().div_(b_cpu)
        a_nova = to_nova(a_cpu)
        b_nova = to_nova(b_cpu)
        a_nova.div_(b_nova)
        assert_close(a_nova, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_addcmul_(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        c_cpu = torch.randn(size)
        expected = a_cpu.clone().addcmul_(b_cpu, c_cpu, value=0.5)
        a_nova = to_nova(a_cpu)
        b_nova = to_nova(b_cpu)
        c_nova = to_nova(c_cpu)
        a_nova.addcmul_(b_nova, c_nova, value=0.5)
        assert_close(a_nova, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_addcdiv_(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        c_cpu = torch.randn(size).clamp(min=0.1)
        expected = a_cpu.clone().addcdiv_(b_cpu, c_cpu, value=0.5)
        a_nova = to_nova(a_cpu)
        b_nova = to_nova(b_cpu)
        c_nova = to_nova(c_cpu)
        a_nova.addcdiv_(b_nova, c_nova, value=0.5)
        assert_close(a_nova, expected)


# =============================================================================
# Elementwise Ops
# =============================================================================

class TestElementwiseOps:
    """Tests for add.Tensor, sub.Tensor, mul.Tensor, div.Tensor, neg."""

    @pytest.mark.parametrize("size", SIZES)
    def test_add_tensor(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = a_cpu + b_cpu
        result = to_nova(a_cpu) + to_nova(b_cpu)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_add_tensor_alpha(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = torch.add(a_cpu, b_cpu, alpha=2.0)
        result = torch.add(to_nova(a_cpu), to_nova(b_cpu), alpha=2.0)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_sub_tensor(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = a_cpu - b_cpu
        result = to_nova(a_cpu) - to_nova(b_cpu)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_mul_tensor(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = a_cpu * b_cpu
        result = to_nova(a_cpu) * to_nova(b_cpu)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_div_tensor(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size).clamp(min=0.1)
        expected = a_cpu / b_cpu
        result = to_nova(a_cpu) / to_nova(b_cpu)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_neg(self, size):
        cpu = torch.randn(size)
        expected = -cpu
        result = -to_nova(cpu)
        assert_close(result, expected)


# =============================================================================
# Reduction Ops
# =============================================================================

class TestReductionOps:
    """Tests for sum, mean."""

    @pytest.mark.parametrize("size", SIZES)
    def test_sum(self, size):
        cpu = torch.randn(size)
        expected = cpu.sum()
        result = to_nova(cpu).sum()
        # Reduction accumulates error, use looser tolerance for large sizes
        tol = ATOL * (size / 16)
        assert_close(result, expected, atol=tol, rtol=RTOL)

    @pytest.mark.parametrize("size", SIZES)
    def test_mean(self, size):
        cpu = torch.randn(size)
        expected = cpu.mean()
        result = to_nova(cpu).mean()
        assert_close(result, expected, atol=ATOL * 10, rtol=RTOL)

    def test_sum_2d(self):
        cpu = torch.randn(32, 16)
        expected = cpu.sum()
        result = to_nova(cpu).sum()
        assert_close(result, expected, atol=ATOL * 32, rtol=RTOL)

    def test_mean_2d(self):
        cpu = torch.randn(32, 16)
        expected = cpu.mean()
        result = to_nova(cpu).mean()
        assert_close(result, expected, atol=ATOL * 10, rtol=RTOL)


# =============================================================================
# Matmul Ops
# =============================================================================

class TestMatmulOps:
    """Tests for mm, addmm, bmm."""

    @pytest.mark.parametrize("m,k,n", [(4, 8, 6), (16, 32, 16), (64, 64, 64)])
    def test_mm(self, m, k, n):
        a_cpu = torch.randn(m, k)
        b_cpu = torch.randn(k, n)
        expected = torch.mm(a_cpu, b_cpu)
        result = torch.mm(to_nova(a_cpu), to_nova(b_cpu))
        assert_close(result, expected, atol=ATOL * k, rtol=RTOL)

    @pytest.mark.parametrize("m,k,n", [(4, 8, 6), (16, 32, 16), (64, 64, 64)])
    def test_addmm(self, m, k, n):
        bias_cpu = torch.randn(m, n)
        a_cpu = torch.randn(m, k)
        b_cpu = torch.randn(k, n)
        expected = torch.addmm(bias_cpu, a_cpu, b_cpu)
        result = torch.addmm(to_nova(bias_cpu), to_nova(a_cpu), to_nova(b_cpu))
        assert_close(result, expected, atol=ATOL * k, rtol=RTOL)

    @pytest.mark.parametrize("b,m,k,n", [(2, 4, 8, 6), (4, 16, 16, 16)])
    def test_bmm(self, b, m, k, n):
        a_cpu = torch.randn(b, m, k)
        b_cpu = torch.randn(b, k, n)
        expected = torch.bmm(a_cpu, b_cpu)
        result = torch.bmm(to_nova(a_cpu), to_nova(b_cpu))
        assert_close(result, expected, atol=ATOL * k, rtol=RTOL)


# =============================================================================
# Activation Ops
# =============================================================================

class TestActivationOps:
    """Tests for relu, sigmoid, tanh, threshold_backward."""

    @pytest.mark.parametrize("size", SIZES)
    def test_relu(self, size):
        cpu = torch.randn(size)
        expected = torch.relu(cpu)
        result = torch.relu(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_sigmoid(self, size):
        cpu = torch.randn(size)
        expected = torch.sigmoid(cpu)
        result = torch.sigmoid(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_tanh(self, size):
        cpu = torch.randn(size)
        expected = torch.tanh(cpu)
        result = torch.tanh(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_threshold_backward(self, size):
        grad_cpu = torch.randn(size)
        input_cpu = torch.randn(size)
        expected = torch.ops.aten.threshold_backward(grad_cpu, input_cpu, 0.0)
        result = torch.ops.aten.threshold_backward(
            to_nova(grad_cpu), to_nova(input_cpu), 0.0
        )
        assert_close(result, expected)


# =============================================================================
# Math Ops
# =============================================================================

class TestMathOps:
    """Tests for exp, log, sqrt, rsqrt, abs, pow, max, min, clamp."""

    @pytest.mark.parametrize("size", SIZES)
    def test_exp(self, size):
        cpu = torch.randn(size).clamp(-5, 5)  # avoid overflow
        expected = torch.exp(cpu)
        result = torch.exp(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_log(self, size):
        cpu = torch.rand(size).clamp(min=0.01)  # positive only
        expected = torch.log(cpu)
        result = torch.log(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_sqrt(self, size):
        cpu = torch.rand(size).clamp(min=0.01)
        expected = torch.sqrt(cpu)
        result = torch.sqrt(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_rsqrt(self, size):
        cpu = torch.rand(size).clamp(min=0.01)
        expected = torch.rsqrt(cpu)
        result = torch.rsqrt(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_abs(self, size):
        cpu = torch.randn(size)
        expected = torch.abs(cpu)
        result = torch.abs(to_nova(cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_pow_tensor_scalar(self, size):
        cpu = torch.rand(size).clamp(min=0.01)
        expected = torch.pow(cpu, 2.5)
        result = torch.pow(to_nova(cpu), 2.5)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_max_other(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = torch.max(a_cpu, b_cpu)
        result = torch.max(to_nova(a_cpu), to_nova(b_cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_min_other(self, size):
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = torch.min(a_cpu, b_cpu)
        result = torch.min(to_nova(a_cpu), to_nova(b_cpu))
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_clamp(self, size):
        cpu = torch.randn(size)
        expected = torch.clamp(cpu, min=-0.5, max=0.5)
        result = torch.clamp(to_nova(cpu), min=-0.5, max=0.5)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_clamp_min_only(self, size):
        cpu = torch.randn(size)
        expected = torch.clamp(cpu, min=-0.5)
        result = torch.clamp(to_nova(cpu), min=-0.5)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", SIZES)
    def test_clamp_max_only(self, size):
        cpu = torch.randn(size)
        expected = torch.clamp(cpu, max=0.5)
        result = torch.clamp(to_nova(cpu), max=0.5)
        assert_close(result, expected)


# =============================================================================
# Comparison Ops
# =============================================================================

class TestComparisonOps:
    """Tests for eq, ne, gt, lt, ge, le (all .Scalar), where.self."""

    @pytest.mark.parametrize("size", SIZES)
    def test_eq_scalar(self, size):
        cpu = torch.zeros(size)
        cpu[::2] = 1.0
        expected = cpu.eq(1.0)
        result = to_nova(cpu).eq(1.0)
        assert_close(result.float(), expected.float())

    @pytest.mark.parametrize("size", SIZES)
    def test_ne_scalar(self, size):
        cpu = torch.zeros(size)
        cpu[::2] = 1.0
        expected = cpu.ne(0.0)
        result = to_nova(cpu).ne(0.0)
        assert_close(result.float(), expected.float())

    @pytest.mark.parametrize("size", SIZES)
    def test_gt_scalar(self, size):
        cpu = torch.randn(size)
        expected = cpu.gt(0.0)
        result = to_nova(cpu).gt(0.0)
        assert_close(result.float(), expected.float())

    @pytest.mark.parametrize("size", SIZES)
    def test_lt_scalar(self, size):
        cpu = torch.randn(size)
        expected = cpu.lt(0.0)
        result = to_nova(cpu).lt(0.0)
        assert_close(result.float(), expected.float())

    @pytest.mark.parametrize("size", SIZES)
    def test_ge_scalar(self, size):
        cpu = torch.randn(size)
        expected = cpu.ge(0.0)
        result = to_nova(cpu).ge(0.0)
        assert_close(result.float(), expected.float())

    @pytest.mark.parametrize("size", SIZES)
    def test_le_scalar(self, size):
        cpu = torch.randn(size)
        expected = cpu.le(0.0)
        result = to_nova(cpu).le(0.0)
        assert_close(result.float(), expected.float())

    @pytest.mark.parametrize("size", SIZES)
    def test_where_self(self, size):
        cond_cpu = torch.randn(size) > 0
        a_cpu = torch.randn(size)
        b_cpu = torch.randn(size)
        expected = torch.where(cond_cpu, a_cpu, b_cpu)
        result = torch.where(
            to_nova(cond_cpu), to_nova(a_cpu), to_nova(b_cpu)
        )
        assert_close(result, expected)


# =============================================================================
# Loss / Softmax Ops
# =============================================================================

class TestLossSoftmaxOps:
    """Tests for _softmax, _log_softmax, nll_loss_forward."""

    @pytest.mark.parametrize("size", [(4, 16), (8, 64), (16, 256)])
    def test_softmax(self, size):
        cpu = torch.randn(*size)
        expected = torch.softmax(cpu, dim=1)
        result = torch.softmax(to_nova(cpu), dim=1)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", [(4, 16), (8, 64), (16, 256)])
    def test_softmax_dim0(self, size):
        cpu = torch.randn(*size)
        expected = torch.softmax(cpu, dim=0)
        result = torch.softmax(to_nova(cpu), dim=0)
        assert_close(result, expected)

    @pytest.mark.parametrize("size", [(4, 16), (8, 64), (16, 256)])
    def test_log_softmax(self, size):
        cpu = torch.randn(*size)
        expected = torch.log_softmax(cpu, dim=1)
        result = torch.log_softmax(to_nova(cpu), dim=1)
        assert_close(result, expected)

    @pytest.mark.parametrize("num_classes", [4, 16, 64])
    def test_nll_loss_forward(self, num_classes):
        batch = 8
        input_cpu = torch.randn(batch, num_classes)
        log_probs_cpu = torch.log_softmax(input_cpu, dim=1)
        target_cpu = torch.randint(0, num_classes, (batch,))

        expected_loss, expected_total_weight = torch.ops.aten.nll_loss_forward(
            log_probs_cpu, target_cpu, None, 1, -100  # reduction=mean
        )

        log_probs_nova = to_nova(log_probs_cpu)
        target_nova = to_nova(target_cpu)
        result_loss, result_total_weight = torch.ops.aten.nll_loss_forward(
            log_probs_nova, target_nova, None, 1, -100
        )

        assert_close(result_loss, expected_loss, atol=ATOL * 10, rtol=RTOL)
        assert_close(result_total_weight, expected_total_weight)

    @pytest.mark.parametrize("num_classes", [4, 16, 64])
    def test_nll_loss_forward_sum(self, num_classes):
        batch = 8
        input_cpu = torch.randn(batch, num_classes)
        log_probs_cpu = torch.log_softmax(input_cpu, dim=1)
        target_cpu = torch.randint(0, num_classes, (batch,))

        expected_loss, expected_total_weight = torch.ops.aten.nll_loss_forward(
            log_probs_cpu, target_cpu, None, 2, -100  # reduction=sum
        )

        log_probs_nova = to_nova(log_probs_cpu)
        target_nova = to_nova(target_cpu)
        result_loss, result_total_weight = torch.ops.aten.nll_loss_forward(
            log_probs_nova, target_nova, None, 2, -100
        )

        assert_close(result_loss, expected_loss, atol=ATOL * 10, rtol=RTOL)
        assert_close(result_total_weight, expected_total_weight)


# =============================================================================
# Multi-dimensional & Edge Case Tests
# =============================================================================

class TestMultiDim:
    """Test ops with multi-dimensional tensors to catch shape/stride bugs."""

    def test_elementwise_2d(self):
        a = torch.randn(32, 64)
        b = torch.randn(32, 64)
        for op in [torch.add, torch.sub, torch.mul]:
            expected = op(a, b)
            result = op(to_nova(a), to_nova(b))
            assert_close(result, expected)

    def test_activations_3d(self):
        cpu = torch.randn(4, 8, 16)
        for fn in [torch.relu, torch.sigmoid, torch.tanh]:
            expected = fn(cpu)
            result = fn(to_nova(cpu))
            assert_close(result, expected)

    def test_math_2d(self):
        cpu = torch.rand(16, 32).clamp(min=0.01)
        for fn in [torch.exp, torch.log, torch.sqrt, torch.rsqrt]:
            expected = fn(cpu)
            result = fn(to_nova(cpu))
            assert_close(result, expected)

    def test_neg_2d(self):
        cpu = torch.randn(16, 32)
        expected = -cpu
        result = -to_nova(cpu)
        assert_close(result, expected)

    def test_abs_2d(self):
        cpu = torch.randn(16, 32)
        expected = torch.abs(cpu)
        result = torch.abs(to_nova(cpu))
        assert_close(result, expected)
