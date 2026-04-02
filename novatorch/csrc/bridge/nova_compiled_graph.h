#pragma once

#include <ATen/ATen.h>
#include <functional>
#include <string>
#include <vector>
#include <cstdint>

/// A single step in a compiled execution plan.
/// Calls a real C++ op function (nova_addmm, nova_relu, etc.) with
/// tensor arguments resolved from the tensor table.
struct CompiledStep {
    /// The op to call. Takes tensor args + scalar args, returns result.
    /// Pre-bound with any scalar arguments (alpha, beta, etc.) at compile time.
    std::function<at::Tensor(const std::vector<at::Tensor>&)> op_fn;

    /// Indices into the tensor table for this op's inputs.
    std::vector<int> input_indices;

    /// Where to store the result in the tensor table.
    int output_index;
};

/// A compiled execution plan for an FX graph.
/// Replays a sequence of C++ op calls entirely in C++ without
/// returning to Python between ops.
struct CompiledPlan {
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<int> output_indices;  // which tensor table slots are outputs
    std::vector<CompiledStep> steps;
    int tensor_table_size = 0;
};

/// Execute a compiled plan with the given input tensors.
/// Calls C++ ops in sequence, batch context accumulates dispatches,
/// flushes once at the end. Returns output tensors.
std::vector<at::Tensor> executePlan(
    CompiledPlan& plan,
    const std::vector<at::Tensor>& inputs);

/// Add a step to the plan. The op_name is resolved to a C++ function.
/// scalar_args contains any non-tensor arguments (alpha, beta, dim, etc.)
/// serialized as doubles.
void addPlanStep(
    CompiledPlan& plan,
    const std::string& op_name,
    const std::vector<int>& input_indices,
    int output_index,
    const std::vector<double>& scalar_args);
