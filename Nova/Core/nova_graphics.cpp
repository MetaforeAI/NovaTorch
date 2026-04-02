#include "./nova_graphics.h"
#include <set>

NovaGraphics::NovaGraphics(VkExtent2D extent, const std::string& debug_level, VkSurfaceKHR surf)
    : NovaCore(debug_level), surface(surf)
{
    report(LOGGER::INFO, "NovaGraphics - Initializing graphics mode ..");

    setWindowExtent(extent);

    // Initialize base resources WITH surface support
    createVulkanInstance(true);   // Need surface extensions (SDL/GLFW)
    createPhysicalDevice(true, surface);   // Need presentation support
    createLogicalDevice(true);    // Need swapchain extension
    createSharedCommandPools();
    createImmediateContext();

    // Get graphics and present queues
    vkGetDeviceQueue(logical_device, queues.indices.graphics_family.value(), 0, &graphics_queue);
    vkGetDeviceQueue(logical_device, queues.indices.present_family.value(), 0, &present_queue);

    // Initialize graphics-specific resources
    createSwapchain();
    createImageViews();
    createRenderPass();
    createFramebuffers();
    createFrameSyncObjects();

    report(LOGGER::INFO, "NovaGraphics - Initialized successfully");
}

NovaGraphics::~NovaGraphics()
{
    report(LOGGER::INFO, "NovaGraphics - Destroying");
    vkDeviceWaitIdle(logical_device);

    // Cleanup graphics resources
    if (graphics_pipeline) {
        delete graphics_pipeline;
        graphics_pipeline = nullptr;
    }

    if (render_pass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(logical_device, render_pass, nullptr);
    }

    // Cleanup swapchain
    for (auto framebuffer : swapchain.framebuffers) {
        if (framebuffer != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(logical_device, framebuffer, nullptr);
        }
    }

    for (auto imageView : swapchain.image_views) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(logical_device, imageView, nullptr);
        }
    }

    if (swapchain.instance != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(logical_device, swapchain.instance, nullptr);
    }

    // Cleanup frame sync objects
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        for (auto semaphore : frames[i].image_available) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(logical_device, semaphore, nullptr);
            }
        }
        for (auto semaphore : frames[i].render_finished) {
            if (semaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(logical_device, semaphore, nullptr);
            }
        }
        if (frames[i].in_flight != VK_NULL_HANDLE) {
            vkDestroyFence(logical_device, frames[i].in_flight, nullptr);
        }
        if (frames[i].cmd.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(logical_device, frames[i].cmd.pool, nullptr);
        }
    }

    // Surface will be destroyed by SDL/GLFW externally
}

// Swapchain creation (placeholder - will migrate from existing code)
void NovaGraphics::createSwapchain()
{
    report(LOGGER::VLINE, "\t .. Creating Swapchain ..");

    swapchain.support = querySwapChainSupport(physical_device);
    querySwapChainDetails();

    swapchain.extent = window_extent;
    swapchain.format = swapchain.details.surface_format.format;

    uint32_t image_count = swapchain.support.capabilities.minImageCount + 1;
    if (swapchain.support.capabilities.maxImageCount > 0 && image_count > swapchain.support.capabilities.maxImageCount) {
        image_count = swapchain.support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info = {};
    createSwapchainInfoKHR(&create_info, image_count);

    VK_TRY(vkCreateSwapchainKHR(logical_device, &create_info, nullptr, &swapchain.instance));

    // Get swapchain images
    vkGetSwapchainImagesKHR(logical_device, swapchain.instance, &image_count, nullptr);
    swapchain.images.resize(image_count);
    vkGetSwapchainImagesKHR(logical_device, swapchain.instance, &image_count, swapchain.images.data());

    report(LOGGER::INFO, "Swapchain created with %d images", image_count);
}

SwapChainSupportDetails NovaGraphics::querySwapChainSupport(VkPhysicalDevice device)
{
    SwapChainSupportDetails details = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t format_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);
    if (format_count != 0) {
        details.formats.resize(format_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
    }

    uint32_t present_mode_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);
    if (present_mode_count != 0) {
        details.present_modes.resize(present_mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.data());
    }

    return details;
}

void NovaGraphics::querySwapChainDetails()
{
    report(LOGGER::VLINE, "\t .. Querying SwapChain Details ..");

    if (swapchain.support.formats.empty() || swapchain.support.present_modes.empty()) {
        report(LOGGER::ERROR, "Vulkan: SwapChain support is not available.");
        return;
    }

    // Select surface format (prefer B8G8R8A8_SRGB)
    swapchain.details.surface_format = swapchain.support.formats.front();
    for (const auto& format : swapchain.support.formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            swapchain.details.surface_format = format;
            break;
        }
    }

    // Select present mode (prefer MAILBOX, fallback to FIFO)
    swapchain.details.present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (const auto& mode : swapchain.support.present_modes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            swapchain.details.present_mode = mode;
            break;
        }
    }

    // Select extent
    swapchain.details.extent = window_extent;
    if (swapchain.support.capabilities.currentExtent.width != UINT32_MAX) {
        swapchain.details.extent = swapchain.support.capabilities.currentExtent;
    } else {
        swapchain.details.extent.width = std::max(
            swapchain.support.capabilities.minImageExtent.width,
            std::min(swapchain.support.capabilities.maxImageExtent.width, window_extent.width)
        );
        swapchain.details.extent.height = std::max(
            swapchain.support.capabilities.minImageExtent.height,
            std::min(swapchain.support.capabilities.maxImageExtent.height, window_extent.height)
        );
    }
}

void NovaGraphics::createSwapchainInfoKHR(VkSwapchainCreateInfoKHR* create_info, uint32_t image_count)
{
    std::set<uint32_t> unique_queue_families = {
        queues.indices.graphics_family.value(),
        queues.indices.present_family.value()
    };
    std::vector<uint32_t> queue_families(unique_queue_families.begin(), unique_queue_families.end());

    *create_info = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = swapchain.details.surface_format.format,
        .imageColorSpace = swapchain.details.surface_format.colorSpace,
        .imageExtent = swapchain.extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform = swapchain.support.capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = swapchain.details.present_mode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE
    };

    if (queue_families.size() > 1) {
        create_info->imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info->queueFamilyIndexCount = static_cast<uint32_t>(queue_families.size());
        create_info->pQueueFamilyIndices = queue_families.data();
    } else {
        create_info->imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        create_info->queueFamilyIndexCount = 0;
        create_info->pQueueFamilyIndices = nullptr;
    }
}

void NovaGraphics::createImageViews()
{
    report(LOGGER::VLINE, "\t .. Creating Image Views ..");

    swapchain.image_views.resize(swapchain.images.size());

    for (size_t i = 0; i < swapchain.images.size(); i++) {
        VkImageViewCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = swapchain.images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = swapchain.format,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY
            },
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        VK_TRY(vkCreateImageView(logical_device, &create_info, nullptr, &swapchain.image_views[i]));
    }

    report(LOGGER::INFO, "Created %zu image views", swapchain.image_views.size());
}

void NovaGraphics::createRenderPass()
{
    report(LOGGER::VLINE, "\t .. Creating Render Pass ..");

    VkAttachmentDescription color_attachment = {
        .format = swapchain.format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    VkAttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref
    };

    VkRenderPassCreateInfo render_pass_info = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass
    };

    VK_TRY(vkCreateRenderPass(logical_device, &render_pass_info, nullptr, &render_pass));

    report(LOGGER::INFO, "Render pass created");
}

void NovaGraphics::createFramebuffers()
{
    report(LOGGER::VLINE, "\t .. Creating Framebuffers ..");

    swapchain.framebuffers.resize(swapchain.image_views.size());

    for (size_t i = 0; i < swapchain.image_views.size(); i++) {
        VkImageView attachments[] = { swapchain.image_views[i] };

        VkFramebufferCreateInfo framebuffer_info = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = attachments,
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1
        };

        VK_TRY(vkCreateFramebuffer(logical_device, &framebuffer_info, nullptr, &swapchain.framebuffers[i]));
    }

    report(LOGGER::INFO, "Created %zu framebuffers", swapchain.framebuffers.size());
}

void NovaGraphics::createFrameSyncObjects()
{
    report(LOGGER::VLINE, "\t .. Creating Frame Sync Objects ..");

    VkSemaphoreCreateInfo semaphore_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // Create command pool for this frame
        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queues.indices.graphics_family.value()
        };
        VK_TRY(vkCreateCommandPool(logical_device, &pool_info, nullptr, &frames[i].cmd.pool));

        // Allocate command buffer
        VkCommandBufferAllocateInfo cmd_alloc_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = frames[i].cmd.pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        VK_TRY(vkAllocateCommandBuffers(logical_device, &cmd_alloc_info, &frames[i].cmd.buffer));

        // Create semaphores for each swapchain image
        frames[i].image_available.resize(swapchain.images.size());
        frames[i].render_finished.resize(swapchain.images.size());

        for (size_t j = 0; j < swapchain.images.size(); j++) {
            VK_TRY(vkCreateSemaphore(logical_device, &semaphore_info, nullptr, &frames[i].image_available[j]));
            VK_TRY(vkCreateSemaphore(logical_device, &semaphore_info, nullptr, &frames[i].render_finished[j]));
        }

        // Create fence
        VK_TRY(vkCreateFence(logical_device, &fence_info, nullptr, &frames[i].in_flight));
    }

    report(LOGGER::INFO, "Frame sync objects created");
}

void NovaGraphics::drawFrame()
{
    // Placeholder - will implement full rendering loop
    report(LOGGER::VERBOSE, "NovaGraphics::drawFrame() - not yet implemented");
}

void NovaGraphics::setWindowExtent(VkExtent2D extent)
{
    window_extent = extent;
}

void NovaGraphics::constructGraphicsPipeline(const std::string& vert, const std::string& frag)
{
    report(LOGGER::DEBUG, "NovaGraphics::constructGraphicsPipeline() - not yet implemented");
    // Placeholder - will migrate from existing code
}

void NovaGraphics::recreateSwapchain()
{
    report(LOGGER::DEBUG, "NovaGraphics::recreateSwapchain() - not yet implemented");
    // Placeholder - will migrate from existing code
}
