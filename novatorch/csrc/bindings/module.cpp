#include <torch/extension.h>
#include <torch/python.h>
#include "nova_context.h"
#include "nova_compute.h"
#include "nova_allocator.h"
#include "nova_ops.h"
#include "nova_batch_context.h"
#include "nova_compiled_graph.h"

// Forward declaration for custom fused ops
at::Tensor nova_ssm_scan(
    const at::Tensor& A_bar, const at::Tensor& B_bar,
    const at::Tensor& u, const at::Tensor& C, double D_val);

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
    m.doc() = "NovaTorch: PyTorch backend for Nova Vulkan compute engine";

    // Must be called BEFORE any other novatorch call to take effect
    m.def("set_log_level", [](const std::string& level) {
        NovaContext::setLogLevel(level);
    }, py::arg("level"),
    "Set Nova log level: 'none', 'release', 'staging', 'development', 'debug'");

    m.def("device_count", []() -> int64_t {
        return 1;
    }, "Return number of Nova devices");

    m.def("device_name", []() -> std::string {
        return NovaContext::instance().deviceName();
    }, "Return Nova device name");

    m.def("synchronize", []() {
        NovaBatchContext::instance().flush();
        NovaContext::instance().compute().waitIdle();
    }, "Wait for all Nova operations to complete");

    m.def("_shutdown", []() {
        // Flush any pending batched dispatches before releasing allocations.
        NovaBatchContext::instance().flush();
        // Explicitly release all VMA allocations before process exit.
        // Called from Python atexit to ensure cleanup before static destruction.
        NovaAllocator::getInstance()->releaseAll();
    }, "Internal: release all GPU allocations (called at exit)");

    m.def("reset_descriptor_pool", []() {
        // Reset all descriptor sets back to the pool.
        // Call at the start of each forward pass or training step to reclaim
        // descriptor sets from the previous step.  Without this, the pool
        // exhausts after enough dispatches.
        NovaBatchContext::instance().flush();
        NovaContext::instance().compute().waitIdle();
        getDescriptorPool().reset();
    }, "Reset the Vulkan descriptor pool, reclaiming all allocated sets");

    m.def("flush", []() {
        NovaBatchContext::instance().flush();
    }, "Submit pending GPU dispatches and wait for completion");

    m.def("set_batching", [](bool enabled) {
        NovaBatchContext::instance().setEnabled(enabled);
    }, "Enable/disable automatic command batching");

    // Custom fused ops
    m.def("ssm_scan", &nova_ssm_scan,
        py::arg("A_bar"), py::arg("B_bar"), py::arg("u"),
        py::arg("C"), py::arg("D_val"),
        "Fused SSM scan: x[t] = A_bar * x[t-1] + B_bar[:,t,:] * u[:,t,:], "
        "y[t] = dot(C, x[t]) + D * sum(u[:,t,:]). Returns [batch, seq_len, 1].");

    // ---------------------------------------------------------------
    // Compiled graph execution (torch.compile nova_aot backend)
    // ---------------------------------------------------------------

    py::class_<NovaCompiledGraph, std::shared_ptr<NovaCompiledGraph>>(m, "NovaCompiledGraph")
        .def(py::init<>())
        .def_readwrite("num_inputs", &NovaCompiledGraph::num_inputs)
        .def_readwrite("num_outputs", &NovaCompiledGraph::num_outputs)
        .def_readwrite("intermediates", &NovaCompiledGraph::intermediates)
        .def_readwrite("output_intermediate_indices",
                       &NovaCompiledGraph::output_intermediate_indices);

    m.def("add_dispatch_step", [](
        NovaCompiledGraph& plan,
        const std::string& kernel_name,
        uint32_t num_buffers,
        uint32_t push_constant_size,
        py::bytes push_data_bytes,
        std::vector<uint32_t> buffer_indices,
        std::vector<uint64_t> buffer_sizes,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z
    ) {
        NovaDispatchStep step;
        step.pipeline = &getPipelineCache().get(
            kernel_name, num_buffers, push_constant_size);
        std::string pd = push_data_bytes;
        step.push_data.assign(pd.begin(), pd.end());
        step.push_constant_size = push_constant_size;
        step.num_buffers = num_buffers;
        step.groups_x = groups_x;
        step.groups_y = groups_y;
        step.groups_z = groups_z;
        step.buffer_indices = std::move(buffer_indices);
        step.buffer_sizes.reserve(buffer_sizes.size());
        for (auto s : buffer_sizes)
            step.buffer_sizes.push_back(static_cast<VkDeviceSize>(s));
        plan.steps.push_back(std::move(step));
    }, "Add a dispatch step to a compiled graph plan");

    m.def("init_compiled_graph_pool", [](NovaCompiledGraph& plan) {
        plan.desc_pool = std::make_unique<NovaDescriptorPool>();
        plan.desc_pool->init(NovaContext::instance().device(),
                             std::max(1u, static_cast<uint32_t>(plan.steps.size())));
    }, "Initialize the descriptor pool for a compiled graph");

    m.def("execute_compiled_graph", [](
        NovaCompiledGraph& plan,
        std::vector<at::Tensor> inputs
    ) -> std::vector<at::Tensor> {
        return executeCompiledGraph(plan, inputs);
    }, "Execute a compiled graph with given inputs");
}
