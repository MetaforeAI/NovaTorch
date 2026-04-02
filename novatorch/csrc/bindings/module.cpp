#include <torch/extension.h>
#include <torch/python.h>
#include <fstream>
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

    // ---------------------------------------------------------------
    // GPU Profiling
    // ---------------------------------------------------------------

    m.def("enable_profiling", [](bool enable) {
        NovaBatchContext::instance().setProfilingEnabled(enable);
    }, py::arg("enable"), "Enable/disable GPU timestamp profiling");

    m.def("get_profiling_results", []() -> py::list {
        auto& results = NovaBatchContext::instance().getProfilingResults();
        py::list out;
        for (auto& r : results) {
            py::dict d;
            d["kernel"] = r.kernel_name;
            d["gpu_time_ns"] = r.gpu_time_ns;
            out.append(d);
        }
        return out;
    }, "Get per-dispatch GPU timing from last batch");

    m.def("get_memory_stats", []() -> py::dict {
        VmaAllocator vma = NovaContext::instance().allocator();
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
        vmaGetHeapBudgets(vma, budgets);

        uint64_t total_allocated = 0, total_used = 0;
        for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i) {
            total_allocated += budgets[i].statistics.blockBytes;
            total_used += budgets[i].statistics.allocationBytes;
        }

        py::dict result;
        result["allocated_bytes"] = total_allocated;
        result["used_bytes"] = total_used;
        return result;
    }, "Get GPU memory statistics from VMA");

    m.def("get_gpu_utilization", []() -> int {
        std::ifstream f("/sys/class/drm/card1/device/gpu_busy_percent");
        if (!f.is_open()) return -1;
        int pct = 0;
        f >> pct;
        return pct;
    }, "Get GPU utilization percentage (AMD sysfs)");

    // Custom fused ops
    m.def("ssm_scan", &nova_ssm_scan,
        py::arg("A_bar"), py::arg("B_bar"), py::arg("u"),
        py::arg("C"), py::arg("D_val"),
        "Fused SSM scan: x[t] = A_bar * x[t-1] + B_bar[:,t,:] * u[:,t,:], "
        "y[t] = dot(C, x[t]) + D * sum(u[:,t,:]). Returns [batch, seq_len, 1].");

    // ---------------------------------------------------------------
    // Compiled graph execution (C++ op loop, no Python per op)
    // ---------------------------------------------------------------

    py::class_<CompiledPlan, std::shared_ptr<CompiledPlan>>(m, "CompiledPlan")
        .def(py::init<>())
        .def_readwrite("num_inputs", &CompiledPlan::num_inputs)
        .def_readwrite("num_outputs", &CompiledPlan::num_outputs)
        .def_readwrite("output_indices", &CompiledPlan::output_indices)
        .def_readwrite("tensor_table_size", &CompiledPlan::tensor_table_size);

    m.def("add_op_step", [](
        CompiledPlan& plan,
        const std::string& op_name,
        std::vector<int> input_indices,
        int output_index,
        std::vector<double> scalar_args
    ) {
        addPlanStep(plan, op_name, input_indices, output_index, scalar_args);
    }, "Add a C++ op step to a compiled plan");

    m.def("execute_plan", [](
        CompiledPlan& plan,
        std::vector<at::Tensor> inputs
    ) -> std::vector<at::Tensor> {
        return executePlan(plan, inputs);
    }, "Execute a compiled plan with given inputs");
}
