#pragma once
#include <functional>
#include <vector>
#include <string>

namespace NovaRAII {

/**
 * Resource Registry for Dynamic Cleanup Injection
 *
 * Principle: "Register cleanup at creation time"
 *
 * This solves the fundamental problem:
 * - Resources created conditionally (based on mode, config, etc.)
 * - Must be destroyed in reverse order of creation
 * - Only if they were actually created
 *
 * Usage:
 *   registry.register_resource("swapchain", [=]() {
 *       vkDestroySwapchainKHR(device, swapchain, nullptr);
 *   });
 *
 * On destruction, all registered cleanup functions are called in reverse order.
 */
class ResourceRegistry {
public:
    ResourceRegistry() = default;
    ~ResourceRegistry() { cleanup_all(); }

    // No copy, only move
    ResourceRegistry(const ResourceRegistry&) = delete;
    ResourceRegistry& operator=(const ResourceRegistry&) = delete;
    ResourceRegistry(ResourceRegistry&&) = default;
    ResourceRegistry& operator=(ResourceRegistry&&) = default;

    /**
     * Register a resource for cleanup
     * @param name Debug name for the resource
     * @param cleanup_fn Function to call to clean up this resource
     */
    void register_resource(const std::string& name, std::function<void()> cleanup_fn) {
        resources.push_back({name, cleanup_fn});
    }

    /**
     * Clean up all registered resources in reverse order (LIFO)
     * This ensures proper destruction order (last created, first destroyed)
     */
    void cleanup_all() {
        // Cleanup in reverse order (LIFO)
        for (auto it = resources.rbegin(); it != resources.rend(); ++it) {
            try {
                if (it->cleanup) {
                    it->cleanup();
                }
            } catch (...) {
                // Swallow exceptions during cleanup to allow other resources to be cleaned
                // In production, log this error
            }
        }
        resources.clear();
    }

    /**
     * Get count of registered resources (for debugging)
     */
    size_t count() const { return resources.size(); }

    /**
     * Check if any resources are registered
     */
    bool empty() const { return resources.empty(); }

private:
    struct Resource {
        std::string name;
        std::function<void()> cleanup;
    };

    std::vector<Resource> resources;
};

} // namespace NovaRAII
