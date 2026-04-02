#include "nova_pipeline_cache.h"

#include <fstream>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void NovaPipelineCache::init(VkDevice device) {
    device_ = device;
}

void NovaPipelineCache::shutdown() {
    for (auto& [name, info] : cache_) {
        if (info.update_template != VK_NULL_HANDLE)
            vkDestroyDescriptorUpdateTemplate(device_, info.update_template, nullptr);
        if (info.pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device_, info.pipeline, nullptr);
        if (info.layout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device_, info.layout, nullptr);
        // desc_layout and desc_layout_uab are the same handle now
        if (info.desc_layout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device_, info.desc_layout, nullptr);
    }
    cache_.clear();
    device_ = VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const NovaPipelineInfo& NovaPipelineCache::get(
    const std::string& kernel_name,
    uint32_t num_buffers,
    uint32_t push_constant_size) {

    auto it = cache_.find(kernel_name);
    if (it != cache_.end()) {
        return it->second;
    }

    NovaPipelineInfo info = createPipeline(
        kernel_name, num_buffers, push_constant_size);
    auto [inserted, _] = cache_.emplace(kernel_name, info);
    return inserted->second;
}

// ---------------------------------------------------------------------------
// SPIR-V loader
// ---------------------------------------------------------------------------

std::vector<uint32_t> NovaPipelineCache::loadSPIRV(
    const std::string& kernel_name) {

#ifndef NOVATORCH_SHADER_DIR
#error "NOVATORCH_SHADER_DIR must be defined by CMake"
#endif

    std::string path =
        std::string(NOVATORCH_SHADER_DIR) + "/" + kernel_name + ".spv";

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error(
            "NovaPipelineCache: failed to open SPIR-V file: " + path);
    }

    auto file_size = static_cast<std::streamsize>(file.tellg());
    if (file_size <= 0 || file_size % sizeof(uint32_t) != 0) {
        throw std::runtime_error(
            "NovaPipelineCache: invalid SPIR-V file size: " + path);
    }

    std::vector<uint32_t> code(
        static_cast<size_t>(file_size) / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), file_size);

    if (!file.good()) {
        throw std::runtime_error(
            "NovaPipelineCache: failed to read SPIR-V file: " + path);
    }

    return code;
}

// ---------------------------------------------------------------------------
// Pipeline creation
// ---------------------------------------------------------------------------

NovaPipelineInfo NovaPipelineCache::createPipeline(
    const std::string& kernel_name,
    uint32_t num_buffers,
    uint32_t push_constant_size) {

    // 1. Load SPIR-V bytecode
    std::vector<uint32_t> spirv = loadSPIRV(kernel_name);

    // 2. Create shader module
    VkShaderModuleCreateInfo module_ci{};
    module_ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_ci.codeSize = spirv.size() * sizeof(uint32_t);
    module_ci.pCode = spirv.data();

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(
        device_, &module_ci, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        throw std::runtime_error(
            "NovaPipelineCache: vkCreateShaderModule failed for: "
            + kernel_name);
    }

    // 3. Create descriptor set layout (bindings 0..N-1, all storage buffers)
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    // Descriptor set layout with UPDATE_AFTER_BIND — used for ALL paths.
    // UAB is a superset: allows post-record descriptor updates (for compiled
    // path replay) while being fully compatible with normal eager use.
    // Requires UAB descriptor pools everywhere.
    std::vector<VkDescriptorBindingFlags> bind_flags(
        num_buffers, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{};
    flags_ci.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flags_ci.bindingCount = num_buffers;
    flags_ci.pBindingFlags = bind_flags.data();

    VkDescriptorSetLayoutCreateInfo ds_layout_ci{};
    ds_layout_ci.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds_layout_ci.pNext = &flags_ci;
    ds_layout_ci.flags =
        VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    ds_layout_ci.bindingCount = num_buffers;
    ds_layout_ci.pBindings = bindings.data();

    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    result = vkCreateDescriptorSetLayout(
        device_, &ds_layout_ci, nullptr, &desc_layout);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device_, shader_module, nullptr);
        throw std::runtime_error(
            "NovaPipelineCache: vkCreateDescriptorSetLayout failed for: "
            + kernel_name);
    }
    // Both paths use the same layout now
    VkDescriptorSetLayout desc_layout_uab = desc_layout;

    // Descriptor update template (compiled path)
    std::vector<VkDescriptorUpdateTemplateEntry> tmpl_entries(num_buffers);
    for (uint32_t i = 0; i < num_buffers; ++i) {
        tmpl_entries[i] = {};
        tmpl_entries[i].dstBinding = i;
        tmpl_entries[i].dstArrayElement = 0;
        tmpl_entries[i].descriptorCount = 1;
        tmpl_entries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        tmpl_entries[i].offset = i * sizeof(VkDescriptorBufferInfo);
        tmpl_entries[i].stride = sizeof(VkDescriptorBufferInfo);
    }
    VkDescriptorUpdateTemplateCreateInfo tmpl_ci{};
    tmpl_ci.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    tmpl_ci.templateType =
        VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
    tmpl_ci.descriptorUpdateEntryCount = num_buffers;
    tmpl_ci.pDescriptorUpdateEntries = tmpl_entries.data();
    tmpl_ci.descriptorSetLayout = desc_layout_uab;

    VkDescriptorUpdateTemplate update_template = VK_NULL_HANDLE;
    vkCreateDescriptorUpdateTemplate(
        device_, &tmpl_ci, nullptr, &update_template);

    // 4. Create pipeline layout (descriptor set + optional push constants)
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size;

    VkPipelineLayoutCreateInfo pl_layout_ci{};
    pl_layout_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_layout_ci.setLayoutCount = 1;
    pl_layout_ci.pSetLayouts = &desc_layout;
    if (push_constant_size > 0) {
        pl_layout_ci.pushConstantRangeCount = 1;
        pl_layout_ci.pPushConstantRanges = &push_range;
    }

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    result = vkCreatePipelineLayout(
        device_, &pl_layout_ci, nullptr, &pipeline_layout);
    if (result != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, desc_layout, nullptr);
        vkDestroyShaderModule(device_, shader_module, nullptr);
        throw std::runtime_error(
            "NovaPipelineCache: vkCreatePipelineLayout failed for: "
            + kernel_name);
    }

    // 5. Create compute pipeline
    VkComputePipelineCreateInfo pipeline_ci{};
    pipeline_ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_ci.stage.module = shader_module;
    pipeline_ci.stage.pName = "main";
    pipeline_ci.layout = pipeline_layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    result = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device_, desc_layout, nullptr);
        vkDestroyShaderModule(device_, shader_module, nullptr);
        throw std::runtime_error(
            "NovaPipelineCache: vkCreateComputePipelines failed for: "
            + kernel_name);
    }

    // 6. Shader module no longer needed after pipeline creation
    vkDestroyShaderModule(device_, shader_module, nullptr);

    // 7. Package result
    NovaPipelineInfo info{};
    info.pipeline = pipeline;
    info.layout = pipeline_layout;
    info.desc_layout = desc_layout;
    info.desc_layout_uab = desc_layout_uab;
    info.update_template = update_template;
    info.num_buffers = num_buffers;
    info.push_constant_size = push_constant_size;
    return info;
}
