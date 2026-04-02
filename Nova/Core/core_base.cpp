#include "./core_base.h"
#include "./components/vk_memory.h"
#include <SDL2/SDL_vulkan.h>
#include <set>
#include <string>
#include <stdexcept>

// Base class constructor
NovaCore::NovaCore(const std::string& debug_level)
{
    // Map debug_level string to LOGGER enum directly
    // (avoids static init order issues with DEBUG_MAP in debug_level.h)
    if (debug_level == "none")             LOG_LEVEL = LOGGER::OFF;
    else if (debug_level == "release")     LOG_LEVEL = LOGGER::ERROR;
    else if (debug_level == "staging")     LOG_LEVEL = LOGGER::INFO;
    else if (debug_level == "development") LOG_LEVEL = LOGGER::DEBUG;
    else if (debug_level == "debug")       LOG_LEVEL = LOGGER::VERBOSE;
    else                                   LOG_LEVEL = LOGGER::ERROR;  // safe default

    report(LOGGER::INFO, "NovaCore - Base initialization");
    _blankContext();
}

// Base class destructor
NovaCore::~NovaCore()
{
    report(LOGGER::INFO, "NovaCore - Base cleanup");

    // Flush legacy deletion queue
    queues.deletion.flush();

    // Resource registry will clean up automatically (RAII)
    // Cleans up in reverse order (LIFO)
}

void NovaCore::_blankContext()
{
    instance = VK_NULL_HANDLE;
    physical_device = VK_NULL_HANDLE;
    logical_device = VK_NULL_HANDLE;
    allocator = VK_NULL_HANDLE;

    queues.compute = VK_NULL_HANDLE;
    queues.transfer = VK_NULL_HANDLE;

    immediate.fence = VK_NULL_HANDLE;
    immediate.cmd = VK_NULL_HANDLE;
    immediate.pool = VK_NULL_HANDLE;

    compute_pool = VK_NULL_HANDLE;
    transfer_pool = VK_NULL_HANDLE;

    window_extent = {0, 0};
}

void NovaCore::setWindowExtent(VkExtent2D extent)
{
    window_extent = extent;
}

// Vulkan instance creation
void NovaCore::createVulkanInstance(bool need_surface_extensions)
{
    report(LOGGER::VLINE, "\t .. Instantiating Vulkan Instance ..");

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Nova Engine",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Nova Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .pApplicationInfo = &app_info,
    };

    // Handle Extensions
    report(LOGGER::VERBOSE, "Vulkan: Checking for extensions ..");

    uint32_t extension_count = 0;
    std::vector<const char*> extensions;

    if (need_surface_extensions) {
        // Graphics mode: need SDL surface extensions
        SDL_Vulkan_GetInstanceExtensions(nullptr, &extension_count, nullptr);
        extensions.resize(extension_count);
        SDL_Vulkan_GetInstanceExtensions(nullptr, &extension_count, extensions.data());
    }

    report(LOGGER::VLINE, "\t .. %d extensions found", extension_count);
    for (const auto& ext : extensions) {
        report(LOGGER::VLINE, "\t\t%s", ext);
    }

    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();

    // Validation layers (optional)
    const std::vector<const char*> VALIDATION_LAYERS = {};
    create_info.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
    create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();

    VK_TRY(vkCreateInstance(&create_info, nullptr, &instance));

    // Register for cleanup
    resource_registry.register_resource("vulkan_instance", [this]() {
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
    });

    report(LOGGER::INFO, "Vulkan instance created");
}

// Validation layer support check
bool NovaCore::checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    report(LOGGER::VERBOSE, "Checking for validation layers ..");
    report(LOGGER::VLINE, "Vulkan: %d layers supported:", layerCount);

    for (const auto& layer : availableLayers) {
        report(LOGGER::VLINE, "\t%s", layer.layerName);
    }

    return false;
}

// Queue family property setting
void NovaCore::setQueueFamilyProperties(unsigned int i, VkSurfaceKHR surface, bool need_presentation)
{
    VkQueueFamilyProperties* queue_family = &queues.families[i];
    std::string queue_name = "";

    if (queue_family->queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        queue_name += "{ Graphics } ";
        queues.indices.graphics_family = i;
        queues.priorities.push_back(std::vector<float>(queue_family->queueCount, 1.0f));
        report(LOGGER::VLINE, "\t\tGraphics Family Set.");
    }

    if (queue_family->queueFlags & VK_QUEUE_COMPUTE_BIT) {
        queue_name += "{ Compute } ";
        if (queues.indices.graphics_family.value() != i) {
            queues.indices.compute_family = i;
            queues.priorities.push_back(std::vector<float>(queue_family->queueCount, 1.0f));
            report(LOGGER::VLINE, "\t\tCompute Family Set.");
        }
    }

    if (queue_family->queueFlags & VK_QUEUE_TRANSFER_BIT) {
        queue_name += "{ Transfer } ";
        if (queues.indices.graphics_family.value() != i) {
            queues.indices.transfer_family = i;
            queues.priorities.push_back(std::vector<float>(queue_family->queueCount, 1.0f));
            report(LOGGER::VLINE, "\t\tTransfer Family Set.");
        }
    }

    if (queue_family->queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) {
        queue_name += "{ Sparse Binding } ";
    }

    if (queue_name.empty()) {
        queue_name = "~ Unknown ~";
    }

    report(LOGGER::VLINE, "\t\t\tQueue Count: %d", queue_family->queueCount);
    report(LOGGER::VLINE, "\t\t\t %s", queue_name.c_str());
}

// Get queue families
void NovaCore::getQueueFamilies(VkPhysicalDevice scanned_device, VkSurfaceKHR surface, bool need_presentation)
{
    report(LOGGER::VLINE, "\t .. Acquiring Queue Families ..");

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(scanned_device, &queue_family_count, nullptr);
    queues.families.resize(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(scanned_device, &queue_family_count, queues.families.data());

    for (int i = 0; i < queues.families.size(); i++) {
        report(LOGGER::VLINE, "\t\tQueue Family %d", i);

        // Check for presentation support if needed
        if (need_presentation && surface != VK_NULL_HANDLE) {
            if (queues.indices.present_family.value() == -1) {
                VkBool32 present_support = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(scanned_device, i, surface, &present_support);

                if (present_support) {
                    queues.indices.present_family = i;
                    report(LOGGER::VLINE, "\t\tPresent Family Set.");
                }
            }
        } else if (!need_presentation) {
            // In compute-only mode, set present family to first graphics family (dummy value)
            if (queues.indices.present_family.value() == -1) {
                if (queues.families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                    queues.indices.present_family = i;
                    report(LOGGER::VLINE, "\t\tPresent Family Set (compute-only, dummy value)");
                }
            }
        }

        setQueueFamilyProperties(i, surface, need_presentation);
    }

    // Check if the queues are complete and set transfer/compute family to graphics family if not set
    if (!queues.indices.isComplete()) {
        report(LOGGER::VLINE, "\t\tQueue Families Incomplete. Setting Transfer/Compute Family to Graphics Family.");
        queues.indices.transfer_family = queues.indices.graphics_family.value();
        queues.indices.compute_family = queues.indices.graphics_family.value();
    }
}

// Device extension support check
static bool checkDeviceExtensionSupport(VkPhysicalDevice device, const std::vector<const char*>& required_extensions)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(required_extensions.begin(), required_extensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

// Device provision check
bool NovaCore::deviceProvisioned(VkPhysicalDevice scanned_device, VkSurfaceKHR surface, bool need_swapchain)
{
    getQueueFamilies(scanned_device, surface, surface != VK_NULL_HANDLE);

    // In compute-only mode, we don't need extensions or swapchain support
    if (!need_swapchain) {
        return queues.indices.isComplete();
    }

    // Graphics mode: check device extensions
    const std::vector<const char*> DEVICE_EXTENSIONS = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    bool extensionsSupported = checkDeviceExtensionSupport(scanned_device, DEVICE_EXTENSIONS);
    return queues.indices.isComplete() && extensionsSupported;
}

// Physical device creation
void NovaCore::createPhysicalDevice(bool need_presentation, VkSurfaceKHR surface)
{
    report(LOGGER::VLINE, "\t .. Selecting Physical Device ..");

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        report(LOGGER::ERROR, "Failed to find GPUs with Vulkan support!");
        throw std::runtime_error("No Vulkan-capable GPU found");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        if (deviceProvisioned(device, surface, need_presentation)) {
            physical_device = device;
            break;
        }
    }

    if (physical_device == VK_NULL_HANDLE) {
        report(LOGGER::ERROR, "Failed to find a suitable GPU!");
        throw std::runtime_error("No suitable GPU found");
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_device, &props);
    report(LOGGER::INFO, "Selected GPU: %s", props.deviceName);
}

// Logical device creation
void NovaCore::createLogicalDevice(bool need_swapchain_extension)
{
    report(LOGGER::VLINE, "\t .. Creating Logical Device ..");

    // Create unique queue create infos
    std::set<uint32_t> uniqueQueueFamilies = {
        queues.indices.graphics_family.value(),
        queues.indices.transfer_family.value(),
        queues.indices.compute_family.value()
    };

    // Build per-family priority arrays and queue create infos.
    // Request all available queues for the compute family so that
    // NovaMultiQueue can use them; other families get a single queue.
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::vector<std::vector<float>> localPriorities;
    localPriorities.reserve(uniqueQueueFamilies.size());

    for (uint32_t queueFamily : uniqueQueueFamilies) {
        uint32_t count = (queueFamily == queues.indices.compute_family.value())
            ? queues.families[queueFamily].queueCount
            : 1;

        localPriorities.emplace_back(count, 1.0f);

        VkDeviceQueueCreateInfo queueCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queueFamily,
            .queueCount = count,
            .pQueuePriorities = localPriorities.back().data()
        };
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    // Vulkan 1.2 features: enable UPDATE_AFTER_BIND for descriptor sets.
    // This allows command buffer reuse — descriptor set contents can be
    // updated after recording but before submission.
    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    vulkan12Features.descriptorBindingPartiallyBound = VK_TRUE;
    vulkan12Features.runtimeDescriptorArray = VK_TRUE;

    // Device extensions
    std::vector<const char*> deviceExtensions;
    if (need_swapchain_extension) {
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    VkDeviceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &vulkan12Features,
        .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
        .pQueueCreateInfos = queueCreateInfos.data(),
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
        .pEnabledFeatures = &deviceFeatures
    };

    VK_TRY(vkCreateDevice(physical_device, &createInfo, nullptr, &logical_device));

    // Get queue handles
    vkGetDeviceQueue(logical_device, queues.indices.compute_family.value(), 0, &queues.compute);
    vkGetDeviceQueue(logical_device, queues.indices.transfer_family.value(), 0, &queues.transfer);

    // Create VMA allocator
    VmaAllocatorCreateInfo allocatorInfo = {
        .physicalDevice = physical_device,
        .device = logical_device,
        .instance = instance
    };
    VK_TRY(vmaCreateAllocator(&allocatorInfo, &allocator));

    // Register for cleanup — LIFO order, so logical_device must be
    // registered FIRST (destroyed last) and VMA SECOND (destroyed first).
    // VMA calls vkFreeMemory internally, so the device must still be alive.
    resource_registry.register_resource("logical_device", [this]() {
        if (logical_device != VK_NULL_HANDLE) {
            vkDestroyDevice(logical_device, nullptr);
            logical_device = VK_NULL_HANDLE;
        }
    });

    resource_registry.register_resource("vma_allocator", [this]() {
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
            allocator = VK_NULL_HANDLE;
        }
    });

    report(LOGGER::INFO, "Logical device created");
}

// Immediate context creation (Phase 1 fix)
void NovaCore::createImmediateContext()
{
    report(LOGGER::VLINE, "\t .. Creating Immediate Context ..");

    // Create command pool for immediate submissions
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queues.indices.transfer_family.value()
    };

    VK_TRY(vkCreateCommandPool(logical_device, &pool_info, nullptr, &immediate.pool));

    // Allocate command buffer
    VkCommandBufferAllocateInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = immediate.pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VK_TRY(vkAllocateCommandBuffers(logical_device, &cmd_info, &immediate.cmd));

    // Create fence
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    VK_TRY(vkCreateFence(logical_device, &fence_info, nullptr, &immediate.fence));

    // Register for cleanup
    resource_registry.register_resource("immediate_context", [this]() {
        if (immediate.fence != VK_NULL_HANDLE) {
            vkDestroyFence(logical_device, immediate.fence, nullptr);
            immediate.fence = VK_NULL_HANDLE;
        }
        if (immediate.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(logical_device, immediate.pool, nullptr);
            immediate.pool = VK_NULL_HANDLE;
        }
    });

    report(LOGGER::INFO, "Immediate context created");
}

// Shared command pools creation
void NovaCore::createSharedCommandPools()
{
    report(LOGGER::VLINE, "\t .. Creating Shared Command Pools ..");

    // Compute pool
    VkCommandPoolCreateInfo compute_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queues.indices.compute_family.value()
    };
    VK_TRY(vkCreateCommandPool(logical_device, &compute_pool_info, nullptr, &compute_pool));

    // Transfer pool
    VkCommandPoolCreateInfo transfer_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queues.indices.transfer_family.value()
    };
    VK_TRY(vkCreateCommandPool(logical_device, &transfer_pool_info, nullptr, &transfer_pool));

    // Register for cleanup
    resource_registry.register_resource("shared_command_pools", [this]() {
        if (compute_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(logical_device, compute_pool, nullptr);
            compute_pool = VK_NULL_HANDLE;
        }
        if (transfer_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(logical_device, transfer_pool, nullptr);
            transfer_pool = VK_NULL_HANDLE;
        }
    });

    report(LOGGER::INFO, "Shared command pools created");
}

// Immediate submit implementation
void NovaCore::immediateSubmit(std::function<void(VkCommandBuffer)>&& func)
{
    VK_TRY(vkResetFences(logical_device, 1, &immediate.fence));
    VK_TRY(vkResetCommandBuffer(immediate.cmd, 0));

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    VK_TRY(vkBeginCommandBuffer(immediate.cmd, &begin_info));
    func(immediate.cmd);
    VK_TRY(vkEndCommandBuffer(immediate.cmd));

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &immediate.cmd
    };

    VK_TRY(vkQueueSubmit(queues.transfer, 1, &submit_info, immediate.fence));
    VK_TRY(vkWaitForFences(logical_device, 1, &immediate.fence, VK_TRUE, UINT64_MAX));
}

// Multi-queue access: retrieve compute queue by index
VkQueue NovaCore::getComputeQueue(uint32_t index) const
{
    VkQueue q = VK_NULL_HANDLE;
    vkGetDeviceQueue(logical_device, queues.indices.compute_family.value(), index, &q);
    return q;
}

// Buffer creation
Buffer_T NovaCore::createEphemeralBuffer(size_t size, VkBufferUsageFlags flags, VmaMemoryUsage usage)
{
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = flags,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VmaAllocationCreateInfo alloc_info = {
        .usage = usage
    };

    Buffer_T buffer;
    VK_TRY(vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer.buffer, &buffer.allocation, nullptr));

    return buffer;
}
