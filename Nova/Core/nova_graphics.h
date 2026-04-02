#pragma once
#include "./core_base.h"
#include "./modules/pipeline/pipeline.h"
#include "./modules/camera/camera.h"

/**
 * NovaGraphics - Graphics rendering mode
 *
 * Traditional Vulkan rendering pipeline:
 * - Surface and swapchain
 * - Graphics and present queues
 * - Render pass
 * - Frame synchronization
 * - Graphics pipeline
 *
 * Use for:
 * - Traditional 3D rendering
 * - Window-based applications
 * - Game engines
 */
class NovaGraphics : public NovaCore {
private:
    // Graphics-specific resources
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkQueue present_queue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    SwapChainContext swapchain;
    FrameData frames[MAX_FRAMES_IN_FLIGHT];
    VkRenderPass render_pass = VK_NULL_HANDLE;

    Pipeline* graphics_pipeline = nullptr;
    DescriptorContext descriptor;

    const VkClearValue CLEAR_COLOR = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    unsigned int frame_ct = 0;

    FrameData& current_frame() { return frames[frame_ct % MAX_FRAMES_IN_FLIGHT]; }

    // Private graphics initialization methods
    void createSwapchain();
    void createImageViews();
    void createFramebuffers();
    void createRenderPass();
    void createFrameSyncObjects();
    void recreateSwapchain();

    void querySwapChainDetails();
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    void createSwapchainInfoKHR(VkSwapchainCreateInfoKHR* create_info, uint32_t image_count);

public:
    Camera player_camera;
    bool framebuffer_resized = false;

    /**
     * Constructor - Initialize graphics mode
     * @param extent Window extent (width, height)
     * @param debug_level Logging level
     * @param surface Vulkan surface (created by SDL/GLFW)
     */
    NovaGraphics(VkExtent2D extent, const std::string& debug_level, VkSurfaceKHR surface);

    /**
     * Destructor - Cleanup graphics resources
     */
    ~NovaGraphics() override;

    /**
     * Render a frame
     */
    void drawFrame();

    /**
     * Update window extent (for resize)
     */
    void setWindowExtent(VkExtent2D extent);

    /**
     * Construct graphics pipeline
     * @param vert Vertex shader path
     * @param frag Fragment shader path
     */
    void constructGraphicsPipeline(const std::string& vert = "", const std::string& frag = "");

    /**
     * Get swapchain context
     */
    SwapChainContext& getSwapchain() { return swapchain; }

    /**
     * Get graphics queue
     */
    VkQueue getGraphicsQueue() const { return graphics_queue; }

    /**
     * Get present queue
     */
    VkQueue getPresentQueue() const { return present_queue; }

    /**
     * Get surface
     */
    VkSurfaceKHR getSurface() const { return surface; }
};
