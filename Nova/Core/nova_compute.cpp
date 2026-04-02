#include "./nova_compute.h"

// =========================================================================
// Construction / Destruction
// =========================================================================

NovaCompute::NovaCompute(const std::string& debug_level)
    : NovaCore(debug_level)
{
    report(LOGGER::INFO, "NovaCompute - Initializing compute-only mode ..");

    createVulkanInstance(false);   // No surface extensions
    createPhysicalDevice(false);   // No presentation support needed
    createLogicalDevice(false);    // No swapchain extension
    createSharedCommandPools();
    createImmediateContext();

    report(LOGGER::INFO, "NovaCompute - Initialized successfully");
}

NovaCompute::~NovaCompute()
{
    report(LOGGER::INFO, "NovaCompute - Destroying");
    vkDeviceWaitIdle(logical_device);
    releaseAllThreadResources();
}

// =========================================================================
// Per-thread resource management
// =========================================================================

NovaCompute::ThreadResources* NovaCompute::createThreadResources() {
    auto tr = std::make_unique<ThreadResources>();

    // Per-thread command pool — Vulkan spec requires separate pools
    // for concurrent recording from different threads.
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = getComputeQueueFamily()
    };
    VK_TRY(vkCreateCommandPool(logical_device, &pool_info, nullptr, &tr->pool));

    // Per-thread command buffer
    VkCommandBufferAllocateInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = tr->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VK_TRY(vkAllocateCommandBuffers(logical_device, &cmd_info, &tr->cmd));

    // Per-thread fence (created signaled so first wait is a no-op)
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    VK_TRY(vkCreateFence(logical_device, &fence_info, nullptr, &tr->fence));

    auto* raw = tr.get();

    std::lock_guard<std::mutex> lock(thread_map_mutex_);
    thread_resources_[std::this_thread::get_id()] = std::move(tr);

    return raw;
}

NovaCompute::ThreadResources& NovaCompute::getThreadResources() {
    auto tid = std::this_thread::get_id();

    {
        std::lock_guard<std::mutex> lock(thread_map_mutex_);
        auto it = thread_resources_.find(tid);
        if (it != thread_resources_.end()) return *it->second;
    }

    // First call from this thread — create resources
    return *createThreadResources();
}

void NovaCompute::releaseThreadResources() {
    auto tid = std::this_thread::get_id();
    std::unique_ptr<ThreadResources> tr;

    {
        std::lock_guard<std::mutex> lock(thread_map_mutex_);
        auto it = thread_resources_.find(tid);
        if (it == thread_resources_.end()) return;
        tr = std::move(it->second);
        thread_resources_.erase(it);
    }

    // Wait for any in-flight work before destroying
    if (tr->fence_submitted) {
        vkWaitForFences(logical_device, 1, &tr->fence, VK_TRUE, UINT64_MAX);
    }
    vkDestroyFence(logical_device, tr->fence, nullptr);
    vkFreeCommandBuffers(logical_device, tr->pool, 1, &tr->cmd);
    vkDestroyCommandPool(logical_device, tr->pool, nullptr);
}

void NovaCompute::releaseAllThreadResources() {
    std::lock_guard<std::mutex> lock(thread_map_mutex_);
    for (auto& [tid, tr] : thread_resources_) {
        if (tr->fence_submitted) {
            vkWaitForFences(logical_device, 1, &tr->fence, VK_TRUE, UINT64_MAX);
        }
        vkDestroyFence(logical_device, tr->fence, nullptr);
        vkFreeCommandBuffers(logical_device, tr->pool, 1, &tr->cmd);
        vkDestroyCommandPool(logical_device, tr->pool, nullptr);
    }
    thread_resources_.clear();
}

// =========================================================================
// Queue submission (serialized) and fence wait (lock-free)
// =========================================================================

void NovaCompute::submitToQueue(VkCommandBuffer cmd, VkFence fence) {
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };

    // Only hold the mutex for the vkQueueSubmit call itself —
    // microsecond-level, not during recording or fence wait.
    std::lock_guard<std::mutex> lock(submit_mutex_);
    VK_TRY(vkQueueSubmit(queues.compute, 1, &submit_info, fence));
}

void NovaCompute::waitForFence(VkFence fence) {
    // No mutex needed — vkWaitForFences is thread-safe per the spec
    // as long as no other thread is resetting the same fence.
    VK_TRY(vkWaitForFences(logical_device, 1, &fence, VK_TRUE, UINT64_MAX));
}

// =========================================================================
// Legacy synchronous API — uses calling thread's resources
// =========================================================================

void NovaCompute::executeCompute(
    std::function<void(VkCommandBuffer)>&& func)
{
    auto& tr = getThreadResources();

    // Wait for any previous submission on this thread
    if (tr.fence_submitted) {
        waitForFence(tr.fence);
        tr.fence_submitted = false;
    }

    // Reset and begin
    VK_TRY(vkResetFences(logical_device, 1, &tr.fence));
    VK_TRY(vkResetCommandBuffer(tr.cmd, 0));

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_TRY(vkBeginCommandBuffer(tr.cmd, &begin_info));

    // Record user commands
    func(tr.cmd);

    VK_TRY(vkEndCommandBuffer(tr.cmd));

    // Submit and wait
    submitToQueue(tr.cmd, tr.fence);
    waitForFence(tr.fence);
    // fence_submitted stays false since we already waited
}

void NovaCompute::waitIdle() {
    vkDeviceWaitIdle(logical_device);
}
