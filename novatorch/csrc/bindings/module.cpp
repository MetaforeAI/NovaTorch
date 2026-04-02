#include <torch/extension.h>
#include "nova_context.h"
#include "nova_compute.h"
#include "nova_allocator.h"
#include "nova_ops.h"
#include "nova_batch_context.h"

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
}
