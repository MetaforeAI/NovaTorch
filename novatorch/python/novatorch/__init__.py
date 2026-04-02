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

# PyTorch 2.10+ tries to `import torch.<backend>` when a custom device is used.
_nova_mod = types.ModuleType("torch.nova")
_nova_mod.__package__ = "torch"
sys.modules["torch.nova"] = _nova_mod

# Generate .nova(), .is_nova, etc. methods on Tensor and Module
torch.utils.generate_methods_for_privateuse1_backend()


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


__version__ = "0.1.0"
__all__ = [
    "is_available", "device_count", "device_name",
    "synchronize", "set_log_level",
]
