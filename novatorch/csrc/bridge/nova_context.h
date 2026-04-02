#pragma once

#include <functional>
#include <memory>
#include <string>

#include <vulkan/vulkan.h>

// Forward declarations — full headers pulled in by .cpp files only
class NovaCompute;
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

/// Singleton owning the Nova Vulkan compute context.
/// Uses Meyer's singleton pattern (thread-safe static local).
class NovaContext {
public:
    /// Set the log level BEFORE calling instance() for the first time.
    /// Valid values: "none", "release", "staging", "development", "debug"
    /// Default: "release" (errors only). Env var NOVA_LOG_LEVEL overrides.
    static void setLogLevel(const std::string& level);

    static NovaContext& instance();

    NovaCompute& compute();
    VmaAllocator allocator();
    VkDevice device();

    void executeSync(std::function<void(VkCommandBuffer)>&& fn);
    std::string deviceName();

    NovaContext(const NovaContext&) = delete;
    NovaContext& operator=(const NovaContext&) = delete;

private:
    NovaContext();
    ~NovaContext();

    static std::string resolveLogLevel();
    static std::string log_level_;
    std::unique_ptr<NovaCompute> compute_;
};
