"""torch.compile backends for NovaTorch.

Usage:
    # Tier 1: Eager (no optimization)
    model = torch.compile(model, backend="nova")

    # Tier 2: AOTAutograd eager dispatch
    model = torch.compile(model, backend="nova_aot")

    # Tier 3: AOTAutograd + C++ execution loop (no Python per op)
    model = torch.compile(model, backend="nova_compiled")
"""

import torch
import torch.fx
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from novatorch import _C


# ===================================================================
# Tier 1: Eager backend
# ===================================================================

@register_backend
def nova(gm: torch.fx.GraphModule, example_inputs):
    """Eager Nova backend for torch.compile."""
    return gm.forward


# ===================================================================
# Tier 2: AOTAutograd + eager dispatch
# ===================================================================

def _nova_aot_compiler(gm, example_inputs):
    return make_boxed_func(gm.forward)

nova_aot = aot_autograd(fw_compiler=_nova_aot_compiler)
register_backend(nova_aot, name="nova_aot")


# ===================================================================
# Tier 3: AOTAutograd + C++ compiled plan
# ===================================================================

def _nova_compiled_compiler(gm: torch.fx.GraphModule, example_inputs):
    """Build a C++ execution plan from the FX graph.

    The plan calls the same C++ ops as eager (correctness preserved).
    But execution stays in C++ — no Python→C++ round trip per op.
    """
    from torch._subclasses.fake_tensor import FakeTensor

    plan = _C.CompiledPlan()
    node_to_slot = {}
    slot = 0
    all_in_registry = True

    # Map placeholders to input slots
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            node_to_slot[node.name] = slot
            slot += 1
    plan.num_inputs = slot

    # Walk graph, add steps
    output_names = []
    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            continue

        if node.op == 'output':
            args = node.args[0]
            if isinstance(args, (list, tuple)):
                output_names = [str(a) for a in args]
            else:
                output_names = [str(args)]
            continue

        if node.op != 'call_function':
            continue

        # Extract tensor input indices
        input_indices = []
        for a in node.args:
            if isinstance(a, torch.fx.Node) and a.name in node_to_slot:
                input_indices.append(node_to_slot[a.name])

        # Extract ALL non-tensor args as doubles.
        # Each op's C++ registry factory knows how to interpret these.
        scalar_args = []
        for a in node.args:
            if isinstance(a, torch.fx.Node):
                continue
            elif isinstance(a, bool):
                scalar_args.append(1.0 if a else 0.0)
            elif isinstance(a, (int, float)):
                scalar_args.append(float(a))
            elif isinstance(a, (list, tuple)):
                for item in a:
                    if isinstance(item, bool):
                        scalar_args.append(1.0 if item else 0.0)
                    elif isinstance(item, (int, float)):
                        scalar_args.append(float(item))
        for k, v in node.kwargs.items():
            if isinstance(v, bool):
                scalar_args.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                scalar_args.append(float(v))
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, (int, float)):
                        scalar_args.append(float(item))

        output_slot = slot
        node_to_slot[node.name] = output_slot
        slot += 1

        try:
            _C.add_op_step(plan, str(node.target), input_indices,
                           output_slot, scalar_args)
        except RuntimeError:
            # Op not in C++ registry — fall back to eager for this graph
            all_in_registry = False
            break

    if not all_in_registry or not output_names:
        # Fall back to eager dispatch (still correct, just slower)
        return make_boxed_func(gm.forward)

    # Handle None outputs (backward graphs return None for non-differentiable inputs)
    output_indices = []
    for name in output_names:
        if name == 'None' or name not in node_to_slot:
            output_indices.append(-1)  # sentinel for None
        else:
            output_indices.append(node_to_slot[name])

    plan.output_indices = output_indices
    plan.num_outputs = len(output_indices)
    plan.tensor_table_size = slot

    def execute(*args):
        # AOTAutograd calls with FakeTensors during compilation for validation
        if args and any(isinstance(a, FakeTensor) for a in args):
            return list(gm.forward(*args))

        outputs = _C.execute_plan(plan, list(args))

        # Replace -1 sentinel outputs with None
        result = []
        for i, idx in enumerate(output_indices):
            if idx == -1:
                result.append(None)
            else:
                result.append(outputs[i] if i < len(outputs) else None)
        return result

    return make_boxed_func(execute)


nova_compiled = aot_autograd(fw_compiler=_nova_compiled_compiler)
register_backend(nova_compiled, name="nova_compiled")
