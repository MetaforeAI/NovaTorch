#include "nova_context.h"

#include "nova_allocator.h"
#include "nova_staging_pool.h"
#include "nova_compute.h"
#include <cstdlib>

std::string NovaContext::resolveLogLevel() {
    const char* env = std::getenv("NOVA_LOG_LEVEL");
    if (env && env[0] != '\0') return env;
    return log_level_.empty() ? "release" : log_level_;
}

std::string NovaContext::log_level_ = "release";

void NovaContext::setLogLevel(const std::string& level) {
    log_level_ = level;
}

NovaContext& NovaContext::instance() {
    static NovaContext ctx;
    return ctx;
}

NovaContext::NovaContext()
    : compute_(std::make_unique<NovaCompute>(resolveLogLevel())) {}

NovaContext::~NovaContext() {
    // Wait for all GPU work to complete before releasing allocations.
    if (compute_ && compute_->getDevice() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(compute_->getDevice());
    }

    // Destroy staging pool before VMA allocator — staging buffers are
    // VMA allocations that must be freed first.
    NovaStagingPool::instance().destroyAll();

    // Force-release all VMA allocations that PyTorch tensors may still hold.
    // This MUST happen before NovaCompute (and thus VMA) is destroyed,
    // otherwise vmaDestroyAllocator asserts that blocks are non-empty.
    NovaAllocator::getInstance()->releaseAll();
}

NovaCompute& NovaContext::compute() {
    return *compute_;
}

VmaAllocator NovaContext::allocator() {
    return compute_->getAllocator();
}

VkDevice NovaContext::device() {
    return compute_->getDevice();
}

void NovaContext::executeSync(std::function<void(VkCommandBuffer)>&& fn) {
    compute_->executeCompute(std::move(fn));
}

std::string NovaContext::deviceName() {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(compute_->getPhysicalDevice(), &props);
    return std::string(props.deviceName);
}
