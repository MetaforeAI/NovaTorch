#pragma once

#include <ATen/ATen.h>
#include <vulkan/vulkan.h>
#include "nova_pipeline_cache.h"
#include "nova_descriptor_pool.h"

#include <cstdint>
#include <memory>
#include <vector>

/// A single dispatch step in a compiled execution plan.
/// All fields are resolved at compile time and immutable after.
struct NovaDispatchStep {
    const NovaPipelineInfo* pipeline;     // stable pointer into pipeline cache
    std::vector<uint8_t> push_data;       // serialized push constants
    uint32_t push_constant_size;
    uint32_t num_buffers;
    uint32_t groups_x, groups_y, groups_z;

    /// Indices into the flat buffer table maintained by executeCompiledGraph.
    /// Layout: [0..num_inputs-1] = input tensors, [num_inputs..] = intermediates.
    std::vector<uint32_t> buffer_indices;
    std::vector<VkDeviceSize> buffer_sizes;
};

/// A compiled execution plan for a full FX graph.
/// Pre-allocates intermediates, pre-resolves pipelines, replays in one submit.
struct NovaCompiledGraph {
    uint32_t num_inputs  = 0;
    uint32_t num_outputs = 0;

    /// Pre-allocated intermediate tensors (reused across calls).
    std::vector<at::Tensor> intermediates;

    /// Which intermediates are graph outputs (indices into intermediates).
    std::vector<uint32_t> output_intermediate_indices;

    /// The dispatch sequence.
    std::vector<NovaDispatchStep> steps;

    /// Per-plan descriptor pool — sized for steps.size(), reset each call.
    std::unique_ptr<NovaDescriptorPool> desc_pool;
};

/// Execute a compiled plan with the given input tensors.
/// Records all dispatches into one command buffer, submits once, waits once.
/// Returns output tensors (views of pre-allocated intermediates).
std::vector<at::Tensor> executeCompiledGraph(
    NovaCompiledGraph& plan,
    const std::vector<at::Tensor>& inputs);
