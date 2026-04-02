"""torch.compile backend for NovaTorch.

Usage:
    model = torch.compile(model, backend="nova")

Internally uses AOTAutograd for decomposition + auto-backward, then
compiles the graph into a C++ execution plan with record-once/replay-many.
First call captures (~100ms), subsequent calls replay (~2ms).

Set NOVA_EAGER=1 to disable compilation and dispatch every op individually
through Python (for debugging).
"""

import os
import torch
import torch.fx
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from novatorch import _C


def _nova_compiler(gm: torch.fx.GraphModule, example_inputs):
    """Compile FX graph into a C++ execution plan.

    First call: execute C++ ops (correctness), record into a dedicated
    VkCommandBuffer with UPDATE_AFTER_BIND descriptors, save everything.
    Subsequent calls: rebind changed descriptors, resubmit same command
    buffer. O(1) CPU cost regardless of graph size.

    Falls back to eager dispatch if any op isn't in the C++ registry.
    """
    # Debug mode: skip compilation entirely
    if os.environ.get("NOVA_EAGER", "") == "1":
        return make_boxed_func(gm.forward)

    from torch._subclasses.fake_tensor import FakeTensor

    plan = _C.CompiledPlan()
    node_to_slot = {}
    slot = 0
    all_in_registry = True

    for node in gm.graph.nodes:
        if node.op == 'placeholder':
            node_to_slot[node.name] = slot
            slot += 1
    plan.num_inputs = slot

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

        input_indices = []
        for a in node.args:
            if isinstance(a, torch.fx.Node) and a.name in node_to_slot:
                input_indices.append(node_to_slot[a.name])

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
            # Log missing op — visible to user AND written to file
            num_ops = sum(1 for n in gm.graph.nodes if n.op == 'call_function')
            all_ops = [str(n.target) for n in gm.graph.nodes if n.op == 'call_function']
            msg = (f"[nova] FALLBACK to eager: '{node.target}' not in C++ registry "
                   f"(graph: {num_ops} ops)")
            print(msg)
            with open("/tmp/nova_fallback.log", "a") as f:
                f.write(msg + "\n")
                f.write(f"  All ops in graph: {all_ops}\n\n")
            all_in_registry = False
            break

    if not all_in_registry or not output_names:
        return make_boxed_func(gm.forward)

    output_indices = []
    for name in output_names:
        if name == 'None' or name not in node_to_slot:
            output_indices.append(-1)
        else:
            output_indices.append(node_to_slot[name])

    plan.output_indices = output_indices
    plan.num_outputs = len(output_indices)
    plan.tensor_table_size = slot

    def execute(*args):
        if args and any(isinstance(a, FakeTensor) for a in args):
            return list(gm.forward(*args))

        outputs = _C.execute_plan(plan, list(args))

        result = []
        for i, idx in enumerate(output_indices):
            if idx == -1:
                result.append(None)
            else:
                result.append(outputs[i] if i < len(outputs) else None)
        return result

    return make_boxed_func(execute)


_nova_backend = aot_autograd(fw_compiler=_nova_compiler)
register_backend(_nova_backend, name="nova")
