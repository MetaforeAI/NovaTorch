#pragma once

#include <ATen/ATen.h>
#include <vulkan/vulkan.h>
#include "nova_descriptor_pool.h"
#include "nova_pipeline_cache.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

/// A single step in a compiled execution plan.
struct CompiledStep {
    std::function<at::Tensor(const std::vector<at::Tensor>&)> op_fn;
    std::vector<int> input_indices;
    int output_index;
};

/// Tracks a descriptor set's buffer bindings for replay updates.
struct DescriptorMapping {
    VkDescriptorSet set;
    const NovaPipelineInfo* pipeline;   // for update_template
    uint32_t num_buffers;
    std::vector<VkBuffer> bound_buffers;  // current VkBuffer per binding
    std::vector<VkDeviceSize> bound_sizes;
};

/// Compiled execution plan with record-once/replay-many via
/// UPDATE_AFTER_BIND descriptor sets.
///
/// First call: execute C++ ops → record dedicated command buffer → save.
/// Subsequent calls: rebind changed descriptors → resubmit same command buffer.
struct CompiledPlan {
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<int> output_indices;
    std::vector<CompiledStep> steps;
    int tensor_table_size = 0;

    // --- Capture/replay state ---
    bool captured = false;

    // Dedicated Vulkan resources
    VkCommandPool dedicated_pool = VK_NULL_HANDLE;
    VkCommandBuffer dedicated_cmd = VK_NULL_HANDLE;
    VkFence dedicated_fence = VK_NULL_HANDLE;

    // UAB descriptor pool (persistent sets, not reset between replays)
    std::unique_ptr<NovaDescriptorPool> uab_desc_pool;

    // Descriptor tracking: all descriptor sets and their buffer bindings
    std::vector<DescriptorMapping> descriptor_map;

    // Input VkBuffer handles from capture (to detect changes on replay)
    std::vector<VkBuffer> captured_input_buffers;

    // Fixed tensor table from capture (keeps tensors alive)
    std::vector<at::Tensor> fixed_table;

    // ALL tensors retained during capture — keeps VkBuffers alive for
    // the command buffer's lifetime. Includes hidden intermediates from
    // .contiguous(), view materialization, etc. that aren't in fixed_table.
    std::vector<at::Tensor> captured_retained;

    ~CompiledPlan();
};

/// Execute a compiled plan. First call captures; subsequent calls replay.
std::vector<at::Tensor> executePlan(
    CompiledPlan& plan,
    const std::vector<at::Tensor>& inputs);

/// Add a step to the plan.
void addPlanStep(
    CompiledPlan& plan,
    const std::string& op_name,
    const std::vector<int>& input_indices,
    int output_index,
    const std::vector<double>& scalar_args);
