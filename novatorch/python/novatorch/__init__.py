"""NovaTorch - PyTorch backend for Nova Vulkan compute engine."""

import atexit
import os
import sys
import types
import torch

# Load the C++ extension (triggers all backend registrations)
from novatorch import _C  # type: ignore[attr-defined]


def _cleanup():
    """Release all Nova GPU allocations before process exit.

    This runs before C++ static destructors, ensuring VMA buffers are freed
    while the Vulkan device and VMA allocator are still alive.
    """
    import gc
    gc.collect()  # Force Python GC to release tensor references first
    _C._shutdown()


atexit.register(_cleanup)


def set_log_level(level: str = "release") -> None:
    """Set Nova log level. Must be called BEFORE first device use.

    Levels: 'none', 'release', 'staging', 'development', 'debug'
    """
    _C.set_log_level(level)


# Allow env var override: NOVA_LOG_LEVEL=debug
_log_level = os.environ.get("NOVA_LOG_LEVEL", "release")
_C.set_log_level(_log_level)

# Rename PrivateUse1 to "nova"
torch.utils.rename_privateuse1_backend("nova")

# PyTorch 2.10+ and torch.compile need `torch.nova` as a proper module
# attribute (getattr(torch, "nova")) not just in sys.modules.
_nova_mod = types.ModuleType("torch.nova")
_nova_mod.__package__ = "torch"
_nova_mod.is_available = lambda: _C.device_count() > 0
_nova_mod.is_initialized = lambda: True
_nova_mod.device_count = lambda: _C.device_count()
_nova_mod.current_device = lambda: 0
_nova_mod.synchronize = lambda: _C.synchronize()
sys.modules["torch.nova"] = _nova_mod
torch.nova = _nova_mod  # make getattr(torch, "nova") work

# Generate .nova(), .is_nova, etc. methods on Tensor and Module
torch.utils.generate_methods_for_privateuse1_backend()


# Register torch.compile backend
import novatorch.compiler  # noqa: E402,F401


def is_available() -> bool:
    """Check if Nova backend is available."""
    return _C.device_count() > 0


def device_count() -> int:
    """Return number of Nova devices."""
    return _C.device_count()


def device_name(device_index: int = 0) -> str:
    """Return the name of the Nova device."""
    return _C.device_name()


def synchronize():
    """Wait for all Nova operations to complete."""
    _C.synchronize()


def reset_descriptor_pool():
    """Reset the Vulkan descriptor pool, reclaiming all allocated sets.

    Call at the start of each training step (before the forward pass) to
    reclaim descriptor sets used by the previous step.  Without periodic
    resets, the pool will exhaust after enough GPU dispatches.
    """
    _C.reset_descriptor_pool()


def flush():
    """Submit pending GPU dispatches and wait for completion.

    Called automatically at sync points (.item(), .cpu(), backward ops
    that read mapped memory). Call explicitly when you need to ensure
    all GPU work is complete before a timing measurement or checkpoint.
    """
    _C.flush()


def set_batching(enabled: bool = True):
    """Enable or disable automatic GPU command batching.

    When enabled (default), multiple GPU dispatches are recorded into
    a single Vulkan command buffer and submitted together at sync points.
    This eliminates per-dispatch submit+wait overhead (~20us each).

    Disable for debugging or when single-dispatch semantics are needed.
    """
    _C.set_batching(enabled)


def ssm_scan(A_bar, B_bar, u, C, D_val: float):
    """Fused SSM scan — replaces sequential Python loop with single GPU dispatch.

    For each timestep t:
        x[t] = A_bar * x[t-1] + B_bar[:, t, :] * u[:, t, :]
        y[t] = dot(C, x[t]) + D_val * sum(u[:, t, :])

    Args:
        A_bar: [state_dim] — constant decay factor
        B_bar: [batch, seq_len, state_dim] — input weight per step
        u:     [batch, seq_len, state_dim] — input sequence
        C:     [state_dim] — output projection
        D_val: scalar skip connection weight

    Returns:
        y: [batch, seq_len, 1] — output at every timestep
    """
    return _C.ssm_scan(A_bar, B_bar, u, C, D_val)


class profile:
    """Context manager for GPU profiling with per-kernel timing.

    Usage:
        with novatorch.profile() as prof:
            out = model(x)
            loss.backward()
        prof.summary()
    """

    def __enter__(self):
        _C.flush()
        _C.enable_profiling(True)
        self._start_mem = _C.get_memory_stats()
        self._results = []
        self._end_mem = {}
        self._gpu_util = -1
        return self

    def __exit__(self, *args):
        _C.flush()
        self._results = _C.get_profiling_results()
        self._end_mem = _C.get_memory_stats()
        self._gpu_util = _C.get_gpu_utilization()
        _C.enable_profiling(False)

    @property
    def results(self):
        return self._results

    def summary(self):
        """Print formatted profiling summary."""
        total_ns = sum(r['gpu_time_ns'] for r in self._results)
        total_ms = total_ns / 1e6

        print(f"Device: {device_name()}")
        print(f"GPU Utilization: {self._gpu_util}%")
        mem_alloc = self._end_mem.get('allocated_bytes', 0) / 1e9
        mem_used = self._end_mem.get('used_bytes', 0) / 1e9
        print(f"GPU Memory: {mem_used:.2f} GB allocated, {mem_alloc:.2f} GB reserved")
        print(f"Total GPU time: {total_ms:.2f} ms ({len(self._results)} dispatches)")
        print()

        # Aggregate by kernel name
        by_kernel = {}
        for r in self._results:
            name = r['kernel']
            by_kernel[name] = by_kernel.get(name, 0) + r['gpu_time_ns']

        print("Top kernels:")
        for name, ns in sorted(by_kernel.items(), key=lambda x: -x[1]):
            ms = ns / 1e6
            pct = 100 * ns / total_ns if total_ns > 0 else 0
            print(f"  {name:30s} {ms:8.2f} ms  ({pct:5.1f}%)")


def benchmark(fn, warmup: int = 5, iterations: int = 100):
    """Benchmark a callable with GPU timing.

    Args:
        fn: callable to benchmark
        warmup: number of warmup iterations (not timed)
        iterations: number of timed iterations

    Returns:
        dict with min_ms, mean_ms, std_ms, gpu_times_ms
    """
    import math

    # Warmup
    for _ in range(warmup):
        fn()
        _C.flush()

    # Timed runs
    _C.enable_profiling(True)
    gpu_times = []
    for _ in range(iterations):
        fn()
        _C.flush()
        results = _C.get_profiling_results()
        total_ns = sum(r['gpu_time_ns'] for r in results)
        gpu_times.append(total_ns / 1e6)  # ms
    _C.enable_profiling(False)

    mean = sum(gpu_times) / len(gpu_times)
    variance = sum((t - mean) ** 2 for t in gpu_times) / len(gpu_times)
    std = math.sqrt(variance)
    min_t = min(gpu_times)
    max_t = max(gpu_times)

    print(f"Benchmark ({iterations} iterations, {warmup} warmup):")
    print(f"  min: {min_t:.3f} ms  mean: {mean:.3f} ms  std: {std:.3f} ms  max: {max_t:.3f} ms")

    return {
        "min_ms": min_t,
        "mean_ms": mean,
        "std_ms": std,
        "max_ms": max_t,
        "gpu_times_ms": gpu_times,
    }


__version__ = "0.1.0"
__all__ = [
    "is_available", "device_count", "device_name",
    "synchronize", "set_log_level", "reset_descriptor_pool",
    "flush", "set_batching", "ssm_scan",
    "profile", "benchmark",
]
