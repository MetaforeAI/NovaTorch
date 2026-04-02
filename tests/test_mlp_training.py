"""End-to-end MLP training tests for the NovaTorch (nova) backend."""
import pytest
import torch
import torch.nn as nn

DEVICE = torch.device("nova")


def _make_mlp():
    """Build MLP: Linear(784,128) -> ReLU -> Linear(128,10) on CPU."""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


# ---------------------------------------------------------------------------
# 1. Tensor round-trip
# ---------------------------------------------------------------------------
def test_tensor_to_nova():
    torch.manual_seed(42)
    cpu_tensor = torch.randn(4, 8)
    nova_tensor = cpu_tensor.to(DEVICE)
    back_tensor = nova_tensor.to("cpu")
    assert torch.equal(cpu_tensor, back_tensor), (
        f"Round-trip mismatch: max diff = {(cpu_tensor - back_tensor).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 2. Single Linear forward
# ---------------------------------------------------------------------------
def test_linear_forward():
    torch.manual_seed(42)
    layer_cpu = nn.Linear(64, 32)
    layer_nova = nn.Linear(64, 32)
    # Copy weights so both layers are identical
    layer_nova.load_state_dict(layer_cpu.state_dict())
    layer_nova = layer_nova.to(DEVICE)

    x_cpu = torch.randn(8, 64)
    x_nova = x_cpu.to(DEVICE)

    out_cpu = layer_cpu(x_cpu)
    out_nova = layer_nova(x_nova)
    out_nova_cpu = out_nova.to("cpu")

    assert torch.allclose(out_cpu, out_nova_cpu, atol=1e-3), (
        f"Linear forward mismatch: max diff = {(out_cpu - out_nova_cpu).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 3. MLP forward
# ---------------------------------------------------------------------------
def test_mlp_forward():
    torch.manual_seed(42)
    mlp_cpu = _make_mlp()
    mlp_nova = _make_mlp()
    mlp_nova.load_state_dict(mlp_cpu.state_dict())
    mlp_nova = mlp_nova.to(DEVICE)

    x_cpu = torch.randn(4, 784)
    x_nova = x_cpu.to(DEVICE)

    out_cpu = mlp_cpu(x_cpu)
    out_nova = mlp_nova(x_nova)
    out_nova_cpu = out_nova.to("cpu")

    assert torch.allclose(out_cpu, out_nova_cpu, atol=1e-3), (
        f"MLP forward mismatch: max diff = {(out_cpu - out_nova_cpu).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# 4. MLP backward (gradients exist and non-zero)
# ---------------------------------------------------------------------------
def test_mlp_backward():
    torch.manual_seed(42)
    mlp = _make_mlp().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(4, 784, device=DEVICE)
    target = torch.randint(0, 10, (4,), device=DEVICE)

    out = mlp(x)
    loss = criterion(out, target)
    loss.backward()

    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"param '{name}' has no gradient"
        grad_cpu = param.grad.to("cpu")
        assert grad_cpu.abs().sum().item() > 0, (
            f"param '{name}' gradient is all zeros"
        )


# ---------------------------------------------------------------------------
# 5. MLP training with SGD
# ---------------------------------------------------------------------------
def test_mlp_training_sgd():
    torch.manual_seed(42)
    mlp = _make_mlp().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)

    x = torch.randn(16, 784, device=DEVICE)
    target = torch.randint(0, 10, (16,), device=DEVICE)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = mlp(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"SGD training: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}). "
        f"All losses: {losses}"
    )


# ---------------------------------------------------------------------------
# 6. MLP training with Adam
# ---------------------------------------------------------------------------
def test_mlp_training_adam():
    torch.manual_seed(42)
    mlp = _make_mlp().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

    x = torch.randn(16, 784, device=DEVICE)
    target = torch.randint(0, 10, (16,), device=DEVICE)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = mlp(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Adam training: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}). "
        f"All losses: {losses}"
    )


# ---------------------------------------------------------------------------
# 7. Gradient accuracy vs CPU
# ---------------------------------------------------------------------------
def test_gradient_accuracy():
    torch.manual_seed(42)
    layer_cpu = nn.Linear(64, 32)
    layer_nova = nn.Linear(64, 32)
    layer_nova.load_state_dict(layer_cpu.state_dict())
    layer_nova = layer_nova.to(DEVICE)

    x_cpu = torch.randn(8, 64)
    x_nova = x_cpu.to(DEVICE)

    # CPU forward + backward
    out_cpu = layer_cpu(x_cpu)
    out_cpu.sum().backward()

    # Nova forward + backward
    out_nova = layer_nova(x_nova)
    out_nova.sum().backward()

    for (name_c, p_cpu), (name_n, p_nova) in zip(
        layer_cpu.named_parameters(), layer_nova.named_parameters()
    ):
        grad_cpu = p_cpu.grad
        grad_nova = p_nova.grad.to("cpu")
        assert torch.allclose(grad_cpu, grad_nova, atol=1e-3), (
            f"Gradient mismatch for '{name_c}': "
            f"max diff = {(grad_cpu - grad_nova).abs().max().item()}"
        )
