#include "../core.h"
#define VMA_IMPLEMENTATION
#include "../components/vk_memory.h"

    ///////////////////////////
    // PIPELINE CONSTRUCTION //
    ///////////////////////////

void NovaCoreLegacy::destroyPipeline(Pipeline* pipeline)
    {
        report(LOGGER::DEBUG, "Management - Destroying Pipeline ..");
        if (pipeline) {
            if (pipeline->instance != VK_NULL_HANDLE) {
                vkDestroyPipeline(logical_device, pipeline->instance, nullptr);
            }
            if (pipeline->layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(logical_device, pipeline->layout, nullptr);
            }
            delete pipeline;
        }
        return;
    }

void NovaCoreLegacy::constructGraphicsPipeline(const std::string& vert_shader, const std::string& frag_shader)
    {
        report(LOGGER::DEBUG, "Management - Constructing Graphics Pipeline ..");

        graphics_pipeline = new Pipeline();

        graphics_pipeline->shaders(&logical_device, vert_shader, frag_shader)
                .vertexInput()
                .inputAssembly()
                .viewportState()
                .rasterizer()
                .multisampling()
                .colorBlending()
                .dynamicState()
                .createLayout(&logical_device, &descriptor.layout)
                .pipe(&render_pass)
                .create(&logical_device);

        return;
    }
    
void NovaCoreLegacy::constructComputePipeline()
    {
        report(LOGGER::DEBUG, "Management - Constructing Compute Pipeline ..");

        compute_pipeline = new Pipeline();

        // Compute pipelines don't use graphics pipeline stages
        // Just allocate the Pipeline object for now
        // Actual shader loading and pipeline creation will be done per-shader

        return;
    }

    //////////////////////////
    // RENDER PASS CREATION //
    //////////////////////////

VkAttachmentDescription NovaCoreLegacy::colorAttachment()
    {
        report(LOGGER::VLINE, "\t\t .. Creating Color Attachment ..");

        return {
            .flags = 0,
            .format = swapchain.format,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        };
    }

static VkAttachmentReference colorAttachmentRef()
    {
        report(LOGGER::VLINE, "\t\t .. Creating Color Attachment Reference ..");

        return {
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        };
    }

static VkSubpassDescription subpassDescription(VkAttachmentReference* color_attachment_ref)
    {
        report(LOGGER::VLINE, "\t\t .. Creating Subpass Description");

        return {
            .flags = 0,
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = color_attachment_ref
        };
    }

static VkRenderPassCreateInfo renderPassInfo(VkAttachmentDescription* color_attachment, VkSubpassDescription* subpass_description)
    {
        report(LOGGER::VLINE, "\t\t .. Creating Render Pass Info ..");


        return {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = color_attachment,
            .subpassCount = 1,
            .pSubpasses = subpass_description
        };
    }

void NovaCoreLegacy::createRenderPass()
    {
        report(LOGGER::VLINE, "\t .. Creating Render Pass ..");

        //log();
        VkAttachmentDescription _color_attachment = colorAttachment();
        VkAttachmentReference _color_attachment_ref = colorAttachmentRef();
        VkSubpassDescription _subpass_description = subpassDescription(&_color_attachment_ref);
        VkRenderPassCreateInfo render_pass_info = renderPassInfo(&_color_attachment, &_subpass_description);
        
        VK_TRY(vkCreateRenderPass(logical_device, &render_pass_info, nullptr, &render_pass));

        return;
    }


VkRenderPassBeginInfo NovaCoreLegacy::getRenderPassInfo(size_t i)
    {
        return {
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .renderPass = render_pass,
                .framebuffer = swapchain.framebuffers[i],
                .renderArea = {
                    .offset = {0, 0},
                    .extent = swapchain.extent
                },
                .clearValueCount = 1,
                .pClearValues = &CLEAR_COLOR
            };
    }

    ///////////////////////////
    // DESCRIPTOR SET LAYOUT //
    ///////////////////////////

static inline VkDescriptorSetLayoutBinding _getLayoutBinding()
    {
        return {
                binding: 0,
                descriptorType: VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_VERTEX_BIT,
                pImmutableSamplers: nullptr
            };
    }

static inline VkDescriptorSetLayoutCreateInfo _getLayoutInfo(VkDescriptorSetLayoutBinding* bindings)
    {
        return {
                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount: 1,
                pBindings: bindings
            };
    }

void NovaCoreLegacy::createDescriptorSetLayout() 
    {
        report(LOGGER::VLINE, "\t .. Creating Descriptor Set Layout ..");

        VkDescriptorSetLayoutBinding _layout_binding = _getLayoutBinding();
        VkDescriptorSetLayoutCreateInfo _layout_info = _getLayoutInfo(&_layout_binding);

        VK_TRY(vkCreateDescriptorSetLayout(logical_device, &_layout_info, nullptr, &descriptor.layout));

        return;
    }

static inline VkDescriptorPoolSize getPoolSize(uint32_t ct)
    {
        report(LOGGER::VLINE, "\t\t .. Creating Descriptor Pool Size of %d ..", ct);

        return {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = ct
        };
    }

static inline VkDescriptorPoolCreateInfo getPoolInfo(uint32_t ct, VkDescriptorPoolSize* size)
    {
        report(LOGGER::VLINE, "\t\t .. Creating Descriptor Pool Info with size %d ..", size->descriptorCount);

        return {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = ct,
            .poolSizeCount = 1,
            .pPoolSizes = size
        };
    }

void NovaCoreLegacy::constructDescriptorPool() 
    {
        report(LOGGER::VLINE, "\t .. Constructing Descriptor Pool ..");

        VkDescriptorPoolSize _pool_size = getPoolSize(MAX_FRAMES_IN_FLIGHT);
        VkDescriptorPoolCreateInfo _pool_info = getPoolInfo(MAX_FRAMES_IN_FLIGHT, &_pool_size);

        VK_TRY(vkCreateDescriptorPool(logical_device, &_pool_info, nullptr, &descriptor.pool));

        return;
    }

static inline VkDescriptorSetAllocateInfo getDescriptorSetAllocateInfo(uint32_t ct, VkDescriptorPool* pool, std::vector<VkDescriptorSetLayout>& layouts)
    {
        report(LOGGER::VLINE, "\t\t .. Creating Descriptor Set Allocate Info ..");

        return {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = *pool,
            .descriptorSetCount = ct,
            .pSetLayouts = layouts.data()
        };
    }

static inline VkDescriptorBufferInfo getDescriptorBufferInfo(VkBuffer* buffer, VkDeviceSize size)
    {
        report(LOGGER::VLINE, "\t\t .. Creating Descriptor Buffer Info ..");

        return {
            .buffer = *buffer,
            .offset = 0,
            .range = size
        };
    }

static inline VkWriteDescriptorSet getDescriptorWrite(VkDescriptorSet* set, VkDescriptorBufferInfo* buffer_info)
    {
        report(LOGGER::VLINE, "\t\t .. Creating Descriptor Write ..");

        return {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = *set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = buffer_info,
            .pTexelBufferView = nullptr
        };
    }

void NovaCoreLegacy::createDescriptorSets() 
    {
        report(LOGGER::VLINE, "\t .. Creating Descriptor Sets ..");

        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptor.layout);
        VkDescriptorSetAllocateInfo _alloc_info = getDescriptorSetAllocateInfo(static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT), &descriptor.pool, layouts);

        descriptor.sets.resize(MAX_FRAMES_IN_FLIGHT);

        VK_TRY(vkAllocateDescriptorSets(logical_device, &_alloc_info, descriptor.sets.data()));

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
            {
                VkDescriptorBufferInfo _buffer_info = getDescriptorBufferInfo(&uniform[i].buffer, sizeof(MVP));
                VkWriteDescriptorSet _write_descriptor = getDescriptorWrite(&descriptor.sets[i], &_buffer_info);

                vkUpdateDescriptorSets(logical_device, 1, &_write_descriptor, 0, nullptr);
            }
    }

    /////////////////////
    // COMMAND BUFFERS //
    /////////////////////

static inline VkCommandPoolCreateInfo createCommandPoolInfo(unsigned int queue_family_index, char* name)
    {
        report(LOGGER::VLINE, "\t\t .. Creating %s Command Pool Info ..", name);
        return {
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                .queueFamilyIndex = queue_family_index
            };
    }
    
void NovaCoreLegacy::createCommandPool()
    {
        report(LOGGER::VLINE, "\t .. Creating Command Pool ..");

        // Skip graphics frame command pools in compute-only mode
        if (!compute_only) {
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                char name[32];
                sprintf(name, "Graphics %d", i);
                VkCommandPoolCreateInfo _gfx_cmd_pool_create_info = createCommandPoolInfo(queues.indices.graphics_family.value(), name);
                VK_TRY(vkCreateCommandPool(logical_device, &_gfx_cmd_pool_create_info, nullptr, &frames[i].cmd.pool));

                // Register each frame's command pool for cleanup
                resource_registry.register_resource("frame_cmd_pool", [this, i]() {
                    if (frames[i].cmd.pool != VK_NULL_HANDLE) {
                        vkDestroyCommandPool(logical_device, frames[i].cmd.pool, nullptr);
                        frames[i].cmd.pool = VK_NULL_HANDLE;
                    }
                });
            }
        }

        {
            char name[] = "Transfer";
            VkCommandPoolCreateInfo _xfr_cmd_pool_create_info = createCommandPoolInfo(queues.indices.transfer_family.value(), name);
            VK_TRY(vkCreateCommandPool(logical_device, &_xfr_cmd_pool_create_info, nullptr, &queues.xfr.pool));

            // Register transfer command pool for cleanup
            resource_registry.register_resource("transfer_cmd_pool", [this]() {
                if (queues.xfr.pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(logical_device, queues.xfr.pool, nullptr);
                    queues.xfr.pool = VK_NULL_HANDLE;
                }
            });
        }

        {
            char name[] = "Compute";
            VkCommandPoolCreateInfo _cmp_cmd_pool_create_info = createCommandPoolInfo(queues.indices.compute_family.value(), name);
            VK_TRY(vkCreateCommandPool(logical_device, &_cmp_cmd_pool_create_info, nullptr, &queues.cmp.pool));

            // Register compute command pool for cleanup
            resource_registry.register_resource("compute_cmd_pool", [this]() {
                if (queues.cmp.pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(logical_device, queues.cmp.pool, nullptr);
                    queues.cmp.pool = VK_NULL_HANDLE;
                }
            });
        }

        return;
    }

inline VkCommandBufferAllocateInfo NovaCoreLegacy::createCommandBuffersInfo(VkCommandPool& cmd_pool, char* name)
    {
        report(LOGGER::VLINE, "\t\t .. Creating %s Command Buffer Info  ..", name);

        return {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = cmd_pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1
            };
    }

void NovaCoreLegacy::createCommandBuffers()
    {
        report(LOGGER::VLINE, "\t .. Creating Command Buffers ..");

        // Only create graphics frame command buffers in graphics mode
        if (!compute_only) {
            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                char name[32];
                sprintf(name, "Graphics %d", i);
                VkCommandBufferAllocateInfo _gfx_cmd_buf_alloc_info = createCommandBuffersInfo(frames[i].cmd.pool, name);
                VK_TRY(vkAllocateCommandBuffers(logical_device, &_gfx_cmd_buf_alloc_info, &frames[i].cmd.buffer));
            }
        }

        {
            char name[] = "Transfer";
            VkCommandBufferAllocateInfo _xfr_cmd_buf_alloc_info = createCommandBuffersInfo(queues.xfr.pool, name);
            VK_TRY(vkAllocateCommandBuffers(logical_device, &_xfr_cmd_buf_alloc_info, &queues.xfr.buffer));
        }

        {
            char name[] = "Compute";
            VkCommandBufferAllocateInfo _cmp_cmd_buf_alloc_info = createCommandBuffersInfo(queues.cmp.pool, name);
            VK_TRY(vkAllocateCommandBuffers(logical_device, &_cmp_cmd_buf_alloc_info, &queues.cmp.buffer));
        }

        return;
    }


VkCommandBuffer NovaCoreLegacy::createEphemeralCommand(VkCommandPool& pool) 
    {
        report(LOGGER::VLINE, "\t .. Creating Ephemeral Command Buffer ..");

        VkCommandBuffer _buffer;

        char name[] = "Ephemeral";
        VkCommandBufferAllocateInfo _tmp_alloc_info = createCommandBuffersInfo(pool, name);

        VK_TRY(vkAllocateCommandBuffers(logical_device, &_tmp_alloc_info, &_buffer));

        VkCommandBufferBeginInfo _begin_info = createBeginInfo();
        VK_TRY(vkBeginCommandBuffer(_buffer, &_begin_info));

        return _buffer;
    }

static inline VkSubmitInfo createSubmitInfo(VkCommandBuffer* cmd)
    {
        report(LOGGER::VLINE, "\t .. Creating Submit Info ..");

        return {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr
        };
    }

void NovaCoreLegacy::flushCommandBuffer(VkCommandBuffer& buf, char* name) 
    {
        report(LOGGER::VLINE, "\t .. Ending %s Command Buffer ..", name);

        VK_TRY(vkEndCommandBuffer(buf));

        // Submit the command buffer
        VkSubmitInfo _submit_info = createSubmitInfo(&buf);
        VK_TRY(vkQueueSubmit(queues.transfer, 1, &_submit_info, VK_NULL_HANDLE));
        VK_TRY(vkQueueWaitIdle(queues.transfer));

        // Free the command buffer
        vkFreeCommandBuffers(logical_device, queues.xfr.pool, 1, &buf);

        return;
    }

void NovaCoreLegacy::destroyCommandContext()
    {
        report(LOGGER::VERBOSE, "Management - Destroying Semaphores, Fences and Command Pools ..");
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                // Destroy all per-swapchain-image semaphores
                for (size_t j = 0; j < frames[i].image_available.size(); j++) {
                    if (frames[i].image_available[j] != VK_NULL_HANDLE) {
                        vkDestroySemaphore(logical_device, frames[i].image_available[j], nullptr);
                    }
                    if (frames[i].render_finished[j] != VK_NULL_HANDLE) {
                        vkDestroySemaphore(logical_device, frames[i].render_finished[j], nullptr);
                    }
                }
                //vkDestroySemaphore(logical_device, frames[i].transfer_finished, nullptr);
                //vkDestroySemaphore(logical_device, frames[i].compute_finished, nullptr);
                if (frames[i].in_flight != VK_NULL_HANDLE) {
                    vkDestroyFence(logical_device, frames[i].in_flight, nullptr);
                }
                if (frames[i].cmd.pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(logical_device, frames[i].cmd.pool, nullptr);
                }
            }

        if (queues.xfr.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(logical_device, queues.xfr.pool, nullptr);
        }
        if (queues.cmp.pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(logical_device, queues.cmp.pool, nullptr);
        }
    }

    /////////////////////
    // SYNC STRUCTURES //
    /////////////////////

static VkSemaphoreCreateInfo createSemaphoreInfo()
    {
        report(LOGGER::VLINE, "\t .. Creating Semaphore Info ..");

        return {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0
        };
    }

static VkFenceCreateInfo createFenceInfo()
    {
        report(LOGGER::VLINE, "\t .. Creating Fence Info ..");

        return {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT
        };
    }

void NovaCoreLegacy::createImmediateContext()
{
    report(LOGGER::VLINE, "\t .. Creating Immediate Context ..");

    // Create command pool for immediate submissions
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queues.indices.transfer_family.value()
    };

    VkCommandPool immediate_pool;
    VK_TRY(vkCreateCommandPool(logical_device, &pool_info, nullptr, &immediate_pool));

    // Allocate command buffer
    VkCommandBufferAllocateInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = immediate_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };

    VK_TRY(vkAllocateCommandBuffers(logical_device, &cmd_info, &queues.immediate.cmd));

    // Create fence
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };

    VK_TRY(vkCreateFence(logical_device, &fence_info, nullptr, &queues.immediate.fence));

    // Register for cleanup
    resource_registry.register_resource("immediate_context", [this, immediate_pool]() {
        if (queues.immediate.fence != VK_NULL_HANDLE) {
            vkDestroyFence(logical_device, queues.immediate.fence, nullptr);
            queues.immediate.fence = VK_NULL_HANDLE;
        }
        if (immediate_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(logical_device, immediate_pool, nullptr);
        }
    });
}

void NovaCoreLegacy::createSyncObjects()
    {
        report(LOGGER::VLINE, "\t .. Creating Sync Objects ..");

        if (compute_only) {
            // Compute mode: No frame-based sync needed
            // immediateSubmit() uses queues.immediate.fence (created in createImmediateContext)
            report(LOGGER::INFO, "Compute-only mode: skipping frame sync objects");
            return;
        }

        // Get the number of swapchain images
        uint32_t swapchain_image_count = swapchain.images.size();
        if (swapchain_image_count == 0) {
            report(LOGGER::ERROR, "Swapchain images not created before sync objects!");
            return;
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkSemaphoreCreateInfo semaphore_info = createSemaphoreInfo();
            VkFenceCreateInfo fence_info = createFenceInfo();

            // Create per-swapchain-image semaphores to avoid reuse conflicts
            frames[i].image_available.resize(swapchain_image_count);
            frames[i].render_finished.resize(swapchain_image_count);

            for (uint32_t j = 0; j < swapchain_image_count; j++) {
                VK_TRY(vkCreateSemaphore(logical_device, &semaphore_info, nullptr, &frames[i].image_available[j]));
                VK_TRY(vkCreateSemaphore(logical_device, &semaphore_info, nullptr, &frames[i].render_finished[j]));

                // Register each semaphore for cleanup
                resource_registry.register_resource("frame_semaphores", [this, i, j]() {
                    if (frames[i].image_available[j] != VK_NULL_HANDLE) {
                        vkDestroySemaphore(logical_device, frames[i].image_available[j], nullptr);
                    }
                    if (frames[i].render_finished[j] != VK_NULL_HANDLE) {
                        vkDestroySemaphore(logical_device, frames[i].render_finished[j], nullptr);
                    }
                });
            }

            VK_TRY(vkCreateFence(logical_device, &fence_info, nullptr, &frames[i].in_flight));

            // Register fence for cleanup
            resource_registry.register_resource("frame_fence", [this, i]() {
                if (frames[i].in_flight != VK_NULL_HANDLE) {
                    vkDestroyFence(logical_device, frames[i].in_flight, nullptr);
                    frames[i].in_flight = VK_NULL_HANDLE;
                }
            });
        }

        return;
    }

static inline VmaAllocationCreateInfo getVMAAllocationInfo(VmaMemoryUsage mem_usage)
    {
        report(LOGGER::VLINE, "\t .. Creating VMA Allocation Info ..");

        return {
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = mem_usage,
        };
    }

Buffer_T NovaCoreLegacy::createEphemeralBuffer(size_t size, VkBufferUsageFlags flags, VmaMemoryUsage mem_usage)
    {
        report(LOGGER::VLINE, "\t .. Creating Ephemeral Buffer ..");

        Buffer_T buffer;
    
        VkBufferCreateInfo _buffer_info = getBufferInfo(size, flags);
        VmaAllocationCreateInfo _alloc_info = getVMAAllocationInfo(mem_usage);

        VK_TRY(vmaCreateBuffer(allocator, &_buffer_info, &_alloc_info, &buffer.buffer, &buffer.allocation, nullptr));

        return buffer;
    }

static inline VkCommandBufferSubmitInfo getBufferSubmitInfo(VkCommandBuffer cmd)
    {
        report(LOGGER::VLINE, "\t .. Creating Buffer Submit Info ..");

        return {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .pNext = nullptr,
            .commandBuffer = cmd,
            .deviceMask = 0,
        };
    }

static inline VkSubmitInfo2 getSubmitInfo2(VkCommandBufferSubmitInfo* cmd)
    {
        report(LOGGER::VLINE, "\t .. Creating Submit Info 2 ..");

        return {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .pNext = nullptr,
            .waitSemaphoreInfoCount = 0,
            .pWaitSemaphoreInfos = nullptr,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = cmd,
            .signalSemaphoreInfoCount = 0,
            .pSignalSemaphoreInfos = nullptr
        };
    }

void NovaCoreLegacy::immediateSubmit(std::function<void(VkCommandBuffer)>&& func)
    {
        VK_TRY(vkResetFences(logical_device, 1, &queues.immediate.fence));
        VK_TRY(vkResetCommandBuffer(queues.immediate.cmd, 0));

        VkCommandBuffer cmd = queues.immediate.cmd;
        VkCommandBufferBeginInfo _begin_info = createBeginInfo();
        _begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_TRY(vkBeginCommandBuffer(cmd, &_begin_info));
        func(cmd);
        VK_TRY(vkEndCommandBuffer(cmd));

        // Use Vulkan 1.0-compatible vkQueueSubmit instead of vkQueueSubmit2 (which requires Vulkan 1.3)
        VkSubmitInfo _submit_info = {};
        _submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        _submit_info.commandBufferCount = 1;
        _submit_info.pCommandBuffers = &cmd;

        VK_TRY(vkQueueSubmit(queues.transfer, 1, &_submit_info, queues.immediate.fence));
        VK_TRY(vkWaitForFences(logical_device, 1, &queues.immediate.fence, VK_TRUE, UINT64_MAX));
    }

MeshBuffer NovaCoreLegacy::createMeshBuffer(std::span<uint32_t> idx, std::span<Vertex_T> vtx)
    {
        const size_t VERTEX_BUFFER_SIZE = vtx.size() * sizeof(Vertex_T);
        const size_t INDEX_BUFFER_SIZE = idx.size() * sizeof(uint32_t);

        MeshBuffer buffer;

        buffer.vtx_buffer = createEphemeralBuffer(VERTEX_BUFFER_SIZE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VMA_MEMORY_USAGE_GPU_ONLY);
        VkBufferDeviceAddressInfo _buffer_info = { 
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = nullptr,
            .buffer = buffer.vtx_buffer.buffer
        };
        buffer.buffer_address = vkGetBufferDeviceAddress(logical_device, &_buffer_info);

        buffer.idx_buffer = createEphemeralBuffer(INDEX_BUFFER_SIZE,  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY);

        Buffer_T staging_buffer = createEphemeralBuffer(VERTEX_BUFFER_SIZE + INDEX_BUFFER_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
        void *data = staging_buffer.allocation->GetMappedData();

        memcpy(data, vtx.data(), VERTEX_BUFFER_SIZE);
        memcpy((char*)data + VERTEX_BUFFER_SIZE, idx.data(), INDEX_BUFFER_SIZE);

        immediateSubmit([&](VkCommandBuffer cmd) {
            VkBufferCopy _vertex_copy = {
                .srcOffset = 0,
                .dstOffset = 0,
                .size = VERTEX_BUFFER_SIZE
            };

            vkCmdCopyBuffer(cmd, staging_buffer.buffer, buffer.vtx_buffer.buffer, 1, &_vertex_copy);

            VkBufferCopy _index_copy = {
                .srcOffset = VERTEX_BUFFER_SIZE,
                .dstOffset = 0,
                .size = INDEX_BUFFER_SIZE
            };
            vkCmdCopyBuffer(cmd, staging_buffer.buffer, buffer.idx_buffer.buffer, 1, &_index_copy);
        });

        vmaDestroyBuffer(allocator, staging_buffer.buffer, staging_buffer.allocation);
        return buffer;
    }