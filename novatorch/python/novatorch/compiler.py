"""torch.compile backends for NovaTorch.

Usage:
    # Tier 1: Eager (no optimization)
    model = torch.compile(model, backend="nova")

    # Tier 2: AOTAutograd (decomposed Core ATen, auto-backward)
    model = torch.compile(model, backend="nova_aot")
"""

import struct
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass

import torch
import torch.fx
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from novatorch import _C

WG_SIZE = 256
TILE_SIZE = 16


def _div_up(n: int, d: int) -> int:
    return (n + d - 1) // d


# ===================================================================
# Tier 1: Eager backend
# ===================================================================

@register_backend
def nova(gm: torch.fx.GraphModule, example_inputs):
    """Eager Nova backend for torch.compile."""
    return gm.forward


# ===================================================================
# Tier 2: AOTAutograd + Compiled Graph Execution
# ===================================================================

@dataclass
class KernelSpec:
    """How a Core ATen op maps to a Nova kernel dispatch."""
    kernel_name: str
    num_buffers: int
    push_data: bytes
    push_constant_size: int
    groups_x: int
    groups_y: int = 1
    groups_z: int = 1


# ---------------------------------------------------------------------------
# Op-to-kernel mapping table
# Each function takes (node, shape_env) and returns a KernelSpec
# ---------------------------------------------------------------------------

_OP_TABLE: Dict[Any, Callable] = {}


def _register_op(*targets):
    def decorator(fn):
        for t in targets:
            _OP_TABLE[t] = fn
        return fn
    return decorator


@_register_op(torch.ops.aten.add.Tensor)
def _compile_add(node, shape_env):
    numel = shape_env[node.name].numel()
    alpha = float(node.kwargs.get('alpha', 1.0))
    return KernelSpec(
        kernel_name='elementwise_add', num_buffers=3,
        push_data=struct.pack('If', numel, alpha),
        push_constant_size=8,
        groups_x=_div_up(numel, WG_SIZE))


@_register_op(torch.ops.aten.sub.Tensor)
def _compile_sub(node, shape_env):
    numel = shape_env[node.name].numel()
    alpha = float(node.kwargs.get('alpha', 1.0))
    return KernelSpec(
        kernel_name='elementwise_sub', num_buffers=3,
        push_data=struct.pack('If', numel, alpha),
        push_constant_size=8,
        groups_x=_div_up(numel, WG_SIZE))


@_register_op(torch.ops.aten.mul.Tensor)
def _compile_mul(node, shape_env):
    numel = shape_env[node.name].numel()
    return KernelSpec(
        kernel_name='elementwise_mul', num_buffers=3,
        push_data=struct.pack('I', numel),
        push_constant_size=4,
        groups_x=_div_up(numel, WG_SIZE))


@_register_op(torch.ops.aten.div.Tensor)
def _compile_div(node, shape_env):
    numel = shape_env[node.name].numel()
    return KernelSpec(
        kernel_name='elementwise_div', num_buffers=3,
        push_data=struct.pack('I', numel),
        push_constant_size=4,
        groups_x=_div_up(numel, WG_SIZE))


@_register_op(torch.ops.aten.neg.default)
def _compile_neg(node, shape_env):
    numel = shape_env[node.name].numel()
    return KernelSpec(
        kernel_name='elementwise_neg', num_buffers=2,
        push_data=struct.pack('I', numel),
        push_constant_size=4,
        groups_x=_div_up(numel, WG_SIZE))


@_register_op(torch.ops.aten.mm.default)
def _compile_mm(node, shape_env):
    out = shape_env[node.name]
    a_shape = shape_env[str(node.args[0])].shape
    b_shape = shape_env[str(node.args[1])].shape
    M, K = a_shape[0], a_shape[1]
    N = b_shape[1]
    return KernelSpec(
        kernel_name='matmul', num_buffers=3,
        push_data=struct.pack('III', M, N, K),
        push_constant_size=12,
        groups_x=_div_up(N, TILE_SIZE),
        groups_y=_div_up(M, TILE_SIZE))


@_register_op(torch.ops.aten.addmm.default)
def _compile_addmm(node, shape_env):
    bias_shape = shape_env[str(node.args[0])].shape
    mat1_shape = shape_env[str(node.args[1])].shape
    mat2_shape = shape_env[str(node.args[2])].shape
    M, K = mat1_shape[0], mat1_shape[1]
    N = mat2_shape[1]
    beta = float(node.kwargs.get('beta', 1.0))
    alpha = float(node.kwargs.get('alpha', 1.0))
    bias_is_1d = 1 if len(bias_shape) == 1 else 0
    return KernelSpec(
        kernel_name='addmm', num_buffers=4,
        push_data=struct.pack('IIIffI', M, N, K, beta, alpha, bias_is_1d),
        push_constant_size=24,
        groups_x=_div_up(N, TILE_SIZE),
        groups_y=_div_up(M, TILE_SIZE))


# Unary math ops (2 buffers, push_constant = numel)
def _compile_unary(kernel_name):
    def compiler(node, shape_env):
        numel = shape_env[node.name].numel()
        return KernelSpec(
            kernel_name=kernel_name, num_buffers=2,
            push_data=struct.pack('I', numel),
            push_constant_size=4,
            groups_x=_div_up(numel, WG_SIZE))
    return compiler


for _op, _kernel in [
    (torch.ops.aten.relu.default, 'activation_relu'),
    (torch.ops.aten.sigmoid.default, 'activation_sigmoid'),
    (torch.ops.aten.tanh.default, 'activation_tanh'),
    (torch.ops.aten.exp.default, 'math_exp'),
    (torch.ops.aten.log.default, 'math_log'),
    (torch.ops.aten.sqrt.default, 'math_sqrt'),
    (torch.ops.aten.rsqrt.default, 'math_rsqrt'),
    (torch.ops.aten.abs.default, 'math_abs'),
    (torch.ops.aten.reciprocal.default, 'math_reciprocal'),
]:
    _OP_TABLE[_op] = _compile_unary(_kernel)


# View ops — no dispatch, just metadata
_VIEW_OPS = {
    torch.ops.aten.view.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten._reshape_alias.default,
}


# ===================================================================
# Compiled Plan
# ===================================================================

class NovaCompiledPlan:
    """Compiled execution plan for an FX graph."""

    def __init__(self):
        self._plan = _C.NovaCompiledGraph()
        self._output_names: List[str] = []
        self._compiled = False

    @staticmethod
    def compile(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        plan = NovaCompiledPlan()
        device = torch.device('nova')

        # Shape propagation via eager execution on example inputs
        shape_env: Dict[str, torch.Tensor] = {}
        node_to_slot: Dict[str, int] = {}

        # Map placeholders to input slots
        input_idx = 0
        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                shape_env[node.name] = example_inputs[input_idx]
                node_to_slot[node.name] = input_idx
                input_idx += 1

        plan._plan.num_inputs = input_idx
        intermediates = []
        intermediate_offset = input_idx

        for node in gm.graph.nodes:
            if node.op == 'placeholder':
                continue

            if node.op == 'output':
                # node.args[0] is tuple of output nodes
                args = node.args[0]
                if isinstance(args, (list, tuple)):
                    plan._output_names = [str(a) for a in args]
                else:
                    plan._output_names = [str(args)]
                continue

            if node.op != 'call_function':
                continue

            target = node.target

            # Resolve arguments
            def get_arg(a):
                if isinstance(a, torch.fx.Node):
                    return shape_env[a.name]
                return a

            args = [get_arg(a) for a in node.args]
            kwargs = {k: get_arg(v) for k, v in node.kwargs.items()}

            # Execute eagerly for shape inference
            with torch.no_grad():
                result = target(*args, **kwargs)
            shape_env[node.name] = result

            # View ops: no dispatch, just track the tensor
            if target in _VIEW_OPS:
                # The result shares storage with an input.
                # Map this node to the same buffer slot as its source.
                src_node = node.args[0]
                if isinstance(src_node, torch.fx.Node) and src_node.name in node_to_slot:
                    node_to_slot[node.name] = node_to_slot[src_node.name]
                else:
                    # View created new storage (e.g. expand+contiguous) —
                    # treat as intermediate
                    inter = torch.empty(result.shape, dtype=result.dtype, device=device)
                    inter_idx = len(intermediates)
                    intermediates.append(inter)
                    node_to_slot[node.name] = intermediate_offset + inter_idx
                continue

            # Check if op is in our table
            if target not in _OP_TABLE:
                raise RuntimeError(
                    f"nova_aot compiled: unsupported op {target}. "
                    f"Fall back to backend='nova' for eager execution.")

            # Pre-allocate output intermediate
            inter = torch.empty(result.shape, dtype=result.dtype, device=device)
            inter_idx = len(intermediates)
            intermediates.append(inter)
            node_to_slot[node.name] = intermediate_offset + inter_idx

            # Build kernel spec
            spec = _OP_TABLE[target](node, shape_env)

            # Buffer indices: input buffers then output
            input_slots = []
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    input_slots.append(node_to_slot[a.name])
            output_slot = node_to_slot[node.name]
            all_slots = input_slots + [output_slot]

            # Buffer sizes
            buf_sizes = []
            for a in node.args:
                if isinstance(a, torch.fx.Node):
                    t = shape_env[a.name]
                    buf_sizes.append(t.nelement() * t.element_size())
            buf_sizes.append(result.nelement() * result.element_size())

            _C.add_dispatch_step(
                plan._plan,
                spec.kernel_name,
                spec.num_buffers,
                spec.push_constant_size,
                spec.push_data,
                all_slots,
                buf_sizes,
                spec.groups_x,
                spec.groups_y,
                spec.groups_z,
            )

        # Store intermediates and output mapping.
        # If an output references an input slot (not an intermediate),
        # create an intermediate for it so the index is valid.
        for name in plan._output_names:
            slot = node_to_slot.get(name)
            if slot is not None and slot < intermediate_offset:
                # Output references an input — need to allocate an intermediate
                t = shape_env[name]
                inter = torch.empty(t.shape, dtype=t.dtype, device=device)
                inter_idx = len(intermediates)
                intermediates.append(inter)
                node_to_slot[name] = intermediate_offset + inter_idx
                # Note: no dispatch step needed — the input IS the output.
                # We'll handle this in execute() by copying.

        plan._plan.intermediates = intermediates
        output_indices = []
        for name in plan._output_names:
            slot = node_to_slot[name]
            output_indices.append(slot - intermediate_offset)
        plan._plan.output_intermediate_indices = output_indices
        plan._plan.num_outputs = len(output_indices)

        # Init per-plan descriptor pool
        _C.init_compiled_graph_pool(plan._plan)
        plan._compiled = True

        return plan

    def execute(self, *args):
        inputs = list(args)
        # Ensure inputs on Nova + contiguous
        for i, a in enumerate(inputs):
            if a.device.type != 'nova':
                inputs[i] = a.to(torch.device('nova'))
            if not inputs[i].is_contiguous():
                inputs[i] = inputs[i].contiguous()

        outputs = _C.execute_compiled_graph(self._plan, inputs)

        # Clone outputs so intermediates can be reused next call
        return [o.clone() for o in outputs]


def _nova_compiled_compiler(gm: torch.fx.GraphModule, example_inputs):
    """Compiled graph backend — pre-records dispatch plan, replays in one submit."""
    try:
        plan = NovaCompiledPlan.compile(gm, example_inputs)
        return make_boxed_func(plan.execute)
    except RuntimeError:
        # Fall back to eager if unsupported op encountered
        return make_boxed_func(gm.forward)


# Register both AOT backends
def _nova_eager_compiler(gm, example_inputs):
    return make_boxed_func(gm.forward)

nova_aot = aot_autograd(fw_compiler=_nova_eager_compiler)
register_backend(nova_aot, name="nova_aot")

nova_compiled = aot_autograd(fw_compiler=_nova_compiled_compiler)
register_backend(nova_compiled, name="nova_compiled")
