"""torch.compile backends for NovaTorch.

Usage:
    # Tier 1: Eager (no optimization, proves compatibility)
    model = torch.compile(model, backend="nova")

    # Tier 2: AOTAutograd (decomposes to Core ATen, auto-backward)
    model = torch.compile(model, backend="nova_aot")
"""

import torch
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.backends.common import aot_autograd


# ---------------------------------------------------------------------------
# Tier 1: Eager backend
# ---------------------------------------------------------------------------
# TorchDynamo captures FX graph, we execute it eagerly through existing
# 143+ PrivateUse1 ops. Eliminates Python frame overhead but no fusion.

@register_backend
def nova(gm: torch.fx.GraphModule, example_inputs):
    """Eager Nova backend for torch.compile."""
    return gm.forward


# ---------------------------------------------------------------------------
# Tier 2: AOTAutograd backend
# ---------------------------------------------------------------------------
# Decomposes all ops to ~180 Core ATen primitives, generates backward
# graphs automatically. Both forward and backward are compiled FX graphs
# dispatched through our registered ops.

nova_aot = aot_autograd(fw_compiler=nova)
register_backend(nova_aot, name="nova_aot")
