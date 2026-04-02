#include "nova_compiled_graph.h"
#include "nova_batch_context.h"

#include <unordered_map>
#include <stdexcept>

// =========================================================================
// Forward declarations for all registered C++ ops
// =========================================================================

// Elementwise
at::Tensor nova_add_tensor(const at::Tensor&, const at::Tensor&, const at::Scalar&);
at::Tensor nova_sub_tensor(const at::Tensor&, const at::Tensor&, const at::Scalar&);
at::Tensor nova_mul_tensor(const at::Tensor&, const at::Tensor&);
at::Tensor nova_div_tensor(const at::Tensor&, const at::Tensor&);
at::Tensor nova_neg(const at::Tensor&);

// Matmul
at::Tensor nova_mm(const at::Tensor&, const at::Tensor&);
at::Tensor nova_addmm(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                       const at::Scalar&, const at::Scalar&);
at::Tensor nova_bmm(const at::Tensor&, const at::Tensor&);

// Activation
at::Tensor nova_relu(const at::Tensor&);
at::Tensor nova_sigmoid(const at::Tensor&);
at::Tensor nova_tanh(const at::Tensor&);
at::Tensor nova_threshold_backward(const at::Tensor&, const at::Tensor&, const at::Scalar&);
at::Tensor nova_sigmoid_backward(const at::Tensor&, const at::Tensor&);
at::Tensor nova_tanh_backward(const at::Tensor&, const at::Tensor&);

// Math
at::Tensor nova_exp(const at::Tensor&);
at::Tensor nova_log(const at::Tensor&);
at::Tensor nova_sqrt(const at::Tensor&);
at::Tensor nova_rsqrt(const at::Tensor&);
at::Tensor nova_abs(const at::Tensor&);
at::Tensor nova_reciprocal(const at::Tensor&);

// Reduction
at::Tensor nova_sum(const at::Tensor&, std::optional<at::ScalarType>);
at::Tensor nova_sum_dim_intlist(const at::Tensor&, at::OptionalIntArrayRef, bool, std::optional<at::ScalarType>);
at::Tensor nova_mean(const at::Tensor&, std::optional<at::ScalarType>);
at::Tensor nova_mean_dim(const at::Tensor&, at::OptionalIntArrayRef, bool, std::optional<at::ScalarType>);

// View (these call through to PyTorch dispatch, handle contiguize etc.)
at::Tensor nova_view(const at::Tensor&, c10::IntArrayRef);
at::Tensor nova_t(const at::Tensor&);
at::Tensor nova_transpose_int(const at::Tensor&, int64_t, int64_t);
at::Tensor nova_permute(const at::Tensor&, c10::IntArrayRef);
at::Tensor nova_view(const at::Tensor&, c10::IntArrayRef);
at::Tensor nova_unsqueeze(const at::Tensor&, int64_t);
at::Tensor nova_squeeze_dim(const at::Tensor&, int64_t);
at::Tensor nova_clone(const at::Tensor&, std::optional<c10::MemoryFormat>);
at::Tensor nova_detach(const at::Tensor&);
at::Tensor nova_alias(const at::Tensor&);

// =========================================================================
// Op registry: maps op name strings to bound C++ functions
// =========================================================================

using OpFactory = std::function<
    std::function<at::Tensor(const std::vector<at::Tensor>&)>(
        const std::vector<double>&)>;

static std::unordered_map<std::string, OpFactory>& getOpRegistry() {
    static std::unordered_map<std::string, OpFactory> registry;
    static bool initialized = false;
    if (initialized) return registry;
    initialized = true;

    // --- Elementwise ---
    registry["aten.add.Tensor"] = [](const std::vector<double>& s) {
        double alpha = s.empty() ? 1.0 : s[0];
        return [alpha](const std::vector<at::Tensor>& t) {
            return nova_add_tensor(t[0], t[1], at::Scalar(alpha));
        };
    };
    registry["aten.sub.Tensor"] = [](const std::vector<double>& s) {
        double alpha = s.empty() ? 1.0 : s[0];
        return [alpha](const std::vector<at::Tensor>& t) {
            return nova_sub_tensor(t[0], t[1], at::Scalar(alpha));
        };
    };
    registry["aten.mul.Tensor"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_mul_tensor(t[0], t[1]);
        };
    };
    registry["aten.div.Tensor"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_div_tensor(t[0], t[1]);
        };
    };
    registry["aten.neg.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_neg(t[0]);
        };
    };

    // --- Matmul ---
    registry["aten.mm.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_mm(t[0], t[1]);
        };
    };
    registry["aten.addmm.default"] = [](const std::vector<double>& s) {
        double beta = s.size() > 0 ? s[0] : 1.0;
        double alpha = s.size() > 1 ? s[1] : 1.0;
        return [beta, alpha](const std::vector<at::Tensor>& t) {
            return nova_addmm(t[0], t[1], t[2], at::Scalar(beta), at::Scalar(alpha));
        };
    };
    registry["aten.bmm.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_bmm(t[0], t[1]);
        };
    };

    // --- Activation ---
    registry["aten.relu.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_relu(t[0]);
        };
    };
    registry["aten.sigmoid.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_sigmoid(t[0]);
        };
    };
    registry["aten.tanh.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_tanh(t[0]);
        };
    };
    registry["aten.threshold_backward.default"] = [](const std::vector<double>& s) {
        double threshold = s.empty() ? 0.0 : s[0];
        return [threshold](const std::vector<at::Tensor>& t) {
            return nova_threshold_backward(t[0], t[1], at::Scalar(threshold));
        };
    };
    registry["aten.sigmoid_backward.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_sigmoid_backward(t[0], t[1]);
        };
    };
    registry["aten.tanh_backward.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_tanh_backward(t[0], t[1]);
        };
    };

    // --- Math ---
    registry["aten.exp.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_exp(t[0]); };
    };
    registry["aten.log.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_log(t[0]); };
    };
    registry["aten.sqrt.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_sqrt(t[0]); };
    };
    registry["aten.rsqrt.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_rsqrt(t[0]); };
    };
    registry["aten.abs.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_abs(t[0]); };
    };
    registry["aten.reciprocal.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_reciprocal(t[0]); };
    };

    // --- Reduction ---
    // sum.dim_IntList: scalar_args = [dim0, dim1, ..., keepdim_flag]
    // Last value is keepdim (0 or 1). Everything before is dim indices.
    registry["aten.sum.dim_IntList"] = [](const std::vector<double>& s) {
        return [s](const std::vector<at::Tensor>& t) {
            bool keepdim = false;
            std::vector<int64_t> dims;
            if (!s.empty()) {
                keepdim = (s.back() != 0.0);
                for (size_t i = 0; i + 1 < s.size(); ++i)
                    dims.push_back(static_cast<int64_t>(s[i]));
            }
            return nova_sum_dim_intlist(t[0],
                dims.empty() ? at::OptionalIntArrayRef() : at::OptionalIntArrayRef(dims),
                keepdim, std::nullopt);
        };
    };
    registry["aten.sum.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_sum(t[0], std::nullopt);
        };
    };
    registry["aten.mean.dim"] = [](const std::vector<double>& s) {
        return [s](const std::vector<at::Tensor>& t) {
            bool keepdim = false;
            std::vector<int64_t> dims;
            if (!s.empty()) {
                keepdim = (s.back() != 0.0);
                for (size_t i = 0; i + 1 < s.size(); ++i)
                    dims.push_back(static_cast<int64_t>(s[i]));
            }
            return nova_mean_dim(t[0],
                dims.empty() ? at::OptionalIntArrayRef() : at::OptionalIntArrayRef(dims),
                keepdim, std::nullopt);
        };
    };

    // --- View ops (these handle contiguize internally) ---
    // view.default: scalar_args = shape dimensions
    registry["aten.view.default"] = [](const std::vector<double>& s) {
        return [s](const std::vector<at::Tensor>& t) {
            std::vector<int64_t> shape;
            for (double d : s) shape.push_back(static_cast<int64_t>(d));
            return nova_view(t[0], shape);
        };
    };
    registry["aten.unsqueeze.default"] = [](const std::vector<double>& s) {
        int64_t dim = s.empty() ? 0 : static_cast<int64_t>(s[0]);
        return [dim](const std::vector<at::Tensor>& t) {
            return nova_unsqueeze(t[0], dim);
        };
    };
    registry["aten.squeeze.dim"] = [](const std::vector<double>& s) {
        int64_t dim = s.empty() ? 0 : static_cast<int64_t>(s[0]);
        return [dim](const std::vector<at::Tensor>& t) {
            return nova_squeeze_dim(t[0], dim);
        };
    };
    registry["aten.transpose.int"] = [](const std::vector<double>& s) {
        int64_t d0 = s.size() > 0 ? static_cast<int64_t>(s[0]) : 0;
        int64_t d1 = s.size() > 1 ? static_cast<int64_t>(s[1]) : 1;
        return [d0, d1](const std::vector<at::Tensor>& t) {
            return nova_transpose_int(t[0], d0, d1);
        };
    };
    registry["aten.permute.default"] = [](const std::vector<double>& s) {
        return [s](const std::vector<at::Tensor>& t) {
            std::vector<int64_t> dims;
            for (double d : s) dims.push_back(static_cast<int64_t>(d));
            return nova_permute(t[0], dims);
        };
    };
    registry["aten.t.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_t(t[0]); };
    };
    registry["aten.clone.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) {
            return nova_clone(t[0], std::nullopt);
        };
    };
    registry["aten.detach.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_detach(t[0]); };
    };
    registry["aten.alias.default"] = [](const std::vector<double>&) {
        return [](const std::vector<at::Tensor>& t) { return nova_alias(t[0]); };
    };

    return registry;
}

// =========================================================================
// Plan construction
// =========================================================================

void addPlanStep(
    CompiledPlan& plan,
    const std::string& op_name,
    const std::vector<int>& input_indices,
    int output_index,
    const std::vector<double>& scalar_args)
{
    auto& reg = getOpRegistry();
    auto it = reg.find(op_name);
    if (it == reg.end()) {
        throw std::runtime_error(
            "nova: unsupported op in compiled graph '" + op_name + "'");
    }

    CompiledStep step;
    step.op_fn = it->second(scalar_args);
    step.input_indices = input_indices;
    step.output_index = output_index;
    plan.steps.push_back(std::move(step));

    if (output_index >= plan.tensor_table_size)
        plan.tensor_table_size = output_index + 1;
}

// =========================================================================
// Plan destructor — cleanup Vulkan resources
// =========================================================================

CompiledPlan::~CompiledPlan() {
    if (dedicated_cmd != VK_NULL_HANDLE || dedicated_pool != VK_NULL_HANDLE ||
        dedicated_fence != VK_NULL_HANDLE) {
        VkDevice dev = NovaContext::instance().device();
        if (dedicated_fence != VK_NULL_HANDLE) {
            vkWaitForFences(dev, 1, &dedicated_fence, VK_TRUE, UINT64_MAX);
            vkDestroyFence(dev, dedicated_fence, nullptr);
        }
        if (dedicated_cmd != VK_NULL_HANDLE && dedicated_pool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(dev, dedicated_pool, 1, &dedicated_cmd);
        }
        if (dedicated_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(dev, dedicated_pool, nullptr);
        }
    }
}

// =========================================================================
// Capture: first call — execute ops with UAB descriptors, save everything
// =========================================================================

static std::vector<at::Tensor> capturePlan(
    CompiledPlan& plan,
    const std::vector<at::Tensor>& inputs)
{
    auto& compute = NovaContext::instance().compute();
    VkDevice dev = NovaContext::instance().device();

    // 1. Flush any pending eager dispatches
    NovaBatchContext::instance().flush();

    // 2. Create dedicated Vulkan resources for this plan
    VkCommandPoolCreateInfo pool_ci{};
    pool_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_ci.queueFamilyIndex = compute.getComputeQueueFamily();
    vkCreateCommandPool(dev, &pool_ci, nullptr, &plan.dedicated_pool);

    VkCommandBufferAllocateInfo cmd_ci{};
    cmd_ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_ci.commandPool = plan.dedicated_pool;
    cmd_ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_ci.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev, &cmd_ci, &plan.dedicated_cmd);

    VkFenceCreateInfo fence_ci{};
    fence_ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(dev, &fence_ci, nullptr, &plan.dedicated_fence);

    // 3. Create UAB descriptor pool (persistent sets, not reset between replays)
    plan.uab_desc_pool = std::make_unique<NovaDescriptorPool>();
    plan.uab_desc_pool->initUAB(dev,
        std::max(1, static_cast<int>(plan.steps.size()) * 2));

    // 4. Set up tensor table with actual input tensors
    plan.fixed_table.resize(plan.tensor_table_size);
    plan.captured_input_buffers.resize(plan.num_inputs);
    for (int i = 0; i < plan.num_inputs && i < static_cast<int>(inputs.size()); ++i) {
        plan.fixed_table[i] = inputs[i];
        plan.captured_input_buffers[i] = novatorch::getNovaBuffer(inputs[i]);
    }

    // 5. Begin recording into the dedicated command buffer
    //    flags = 0 → reusable (NOT one-time-submit)
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = 0;
    vkBeginCommandBuffer(plan.dedicated_cmd, &begin);

    // 6. Redirect batch context to capture into our dedicated buffer + UAB pool
    auto& batch = NovaBatchContext::instance();
    batch.beginCapture(plan.dedicated_cmd);
    // Swap in the UAB descriptor pool for capture
    auto* old_pool = batch.swapDescPool(plan.uab_desc_pool.get());

    // 7. Execute the C++ op loop — records into dedicated_cmd with UAB descriptors
    for (auto& step : plan.steps) {
        std::vector<at::Tensor> args;
        args.reserve(step.input_indices.size());
        for (int idx : step.input_indices)
            args.push_back(plan.fixed_table[idx]);
        plan.fixed_table[step.output_index] = step.op_fn(args);
    }

    // 8. End capture — retrieve descriptor tracking data
    plan.descriptor_map = batch.endCaptureWithMap(&plan.captured_retained);
    batch.swapDescPool(old_pool);  // restore original pool

    // 9. End command buffer recording
    vkEndCommandBuffer(plan.dedicated_cmd);

    // 10. Submit and wait (first execution)
    compute.submitToQueue(plan.dedicated_cmd, plan.dedicated_fence);
    compute.waitForFence(plan.dedicated_fence);

    plan.captured = true;

    // 11. Extract outputs
    std::vector<at::Tensor> outputs;
    outputs.reserve(plan.output_indices.size());
    for (int idx : plan.output_indices) {
        if (idx >= 0 && idx < static_cast<int>(plan.fixed_table.size()))
            outputs.push_back(plan.fixed_table[idx]);
        else
            outputs.push_back(at::Tensor());
    }
    return outputs;
}

// =========================================================================
// Replay: rebind changed descriptors, resubmit saved command buffer
// =========================================================================

static std::vector<at::Tensor> replayPlan(
    CompiledPlan& plan,
    const std::vector<at::Tensor>& inputs)
{
    auto& compute = NovaContext::instance().compute();
    VkDevice dev = NovaContext::instance().device();

    // 1. Detect which input VkBuffers changed and rebind descriptors
    for (int i = 0; i < plan.num_inputs && i < static_cast<int>(inputs.size()); ++i) {
        VkBuffer new_buf = novatorch::getNovaBuffer(inputs[i]);
        VkBuffer old_buf = plan.captured_input_buffers[i];

        if (new_buf != old_buf) {
            // Scan descriptor map: update every binding that references old_buf
            for (auto& mapping : plan.descriptor_map) {
                bool set_dirty = false;
                for (uint32_t b = 0; b < mapping.num_buffers; ++b) {
                    if (mapping.bound_buffers[b] == old_buf) {
                        mapping.bound_buffers[b] = new_buf;
                        set_dirty = true;
                    }
                }
                if (set_dirty) {
                    // Rebuild buffer infos and update via template
                    std::vector<VkDescriptorBufferInfo> infos(mapping.num_buffers);
                    for (uint32_t b = 0; b < mapping.num_buffers; ++b) {
                        infos[b] = {mapping.bound_buffers[b], 0, mapping.bound_sizes[b]};
                    }
                    vkUpdateDescriptorSetWithTemplate(
                        dev, mapping.set, mapping.pipeline->update_template,
                        infos.data());
                }
            }

            plan.captured_input_buffers[i] = new_buf;
            plan.fixed_table[i] = inputs[i];
        }
    }

    // 2. Wait for previous replay, reset fence
    vkWaitForFences(dev, 1, &plan.dedicated_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(dev, 1, &plan.dedicated_fence);

    // 3. Submit the SAME command buffer — descriptors updated in-place
    compute.submitToQueue(plan.dedicated_cmd, plan.dedicated_fence);
    compute.waitForFence(plan.dedicated_fence);

    // 4. Extract outputs
    std::vector<at::Tensor> outputs;
    outputs.reserve(plan.output_indices.size());
    for (int idx : plan.output_indices) {
        if (idx >= 0 && idx < static_cast<int>(plan.fixed_table.size()))
            outputs.push_back(plan.fixed_table[idx]);
        else
            outputs.push_back(at::Tensor());
    }
    return outputs;
}

// =========================================================================
// Public entry point
// =========================================================================

std::vector<at::Tensor> executePlan(
    CompiledPlan& plan,
    const std::vector<at::Tensor>& inputs)
{
    if (plan.captured) {
        return replayPlan(plan, inputs);
    }
    return capturePlan(plan, inputs);
}
