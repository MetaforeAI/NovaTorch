#include "obj_loader.h"
#include "logger.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "./extern/fastgltf/include/fastgltf/glm_element_traits.hpp"
#include "./extern/fastgltf/include/fastgltf/core.hpp"
#include "./extern/fastgltf/include/fastgltf/tools.hpp"

std::optional<std::vector<std::shared_ptr<MeshType>>> loadGLTFMeshes(NovaCoreLegacy* engine, std::filesystem::path path) {
    report(LOGGER::ILINE, "GLTF Mesh: %s", path.c_str());

    // We load the data from the path
    fastgltf::GltfDataBuffer data;
    data.FromPath(path);

    // Then we set the options for the parser
    constexpr auto gltf_ops = fastgltf::Options::LoadExternalBuffers;
    fastgltf::Asset gltf;
    fastgltf::Parser parser;

    // The we load the data from the parser
    auto load = parser.loadGltfBinary(data, path.parent_path(), gltf_ops);

    if (!load) {
        report(LOGGER::ERROR, "Failed to load GLTF Mesh: %s", path.c_str());
        return std::nullopt;
    }

    gltf = std::move(load.get());
    std::vector<std::shared_ptr<MeshType>> meshes;
    std::vector<uint32_t> indices;
    std::vector<Vertex_T> vertices;

    // Works for loading entire scenes
    for (fastgltf::Mesh& mesh : gltf.meshes) {
        MeshType new_mesh;
        new_mesh.name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto&& prim : mesh.primitives) {
            VertexRange range;
            range.start = (uint32_t)indices.size();
            range.end = (uint32_t)gltf.accessors[prim.indicesAccessor.value()].count;

            size_t init_vtx = vertices.size();

            // Load indices
            {
                auto& accessor = gltf.accessors[prim.indicesAccessor.value()];
                indices.reserve(indices.size() + accessor.count);

                auto idx_push_back = [&](uint32_t idx) 
                    { indices.push_back(idx + init_vtx); };
             
                fastgltf::iterateAccessor<uint32_t>(gltf, accessor, idx_push_back);
            }

            // Load vertices
            {
                auto& accessor = gltf.accessors[prim.findAttribute("POSITION")->second];
                vertices.resize(init_vtx + accessor.count);

                auto vtx_push_back = [&](glm::vec3 v, size_t idx) 
                    {
                        Vertex_T new_vtx;
                        new_vtx.position = v;
                        new_vtx.normal = { 1, 0, 0 };
                        new_vtx.uv_coord = glm::vec2(0);
                        new_vtx.color = glm::vec4(1.0f);
                        vertices[init_vtx + idx] = new_vtx;
                    };
             
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, accessor, vtx_push_back);
            }

            // Load normals
            auto normals = prim.findAttribute("NORMAL");

            if (normals != prim.attributes.end()) 
                {
                    auto& accessor = gltf.accessors[normals->second];

                    auto vtx_push_back = [&](glm::vec3 v, size_t idx) 
                        { vertices[init_vtx + idx].normal = v; };
                
                    fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, accessor, vtx_push_back);
                }

            // Load UVs
            auto uvs = prim.findAttribute("TEXCOORD_0");

            if (uvs != prim.attributes.end())
                {
                    auto& accessor = gltf.accessors[uvs->second];

                    auto vtx_push_back = [&](glm::vec2 v, size_t idx) 
                        { vertices[init_vtx + idx].uv_coord = v; };
                
                    fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, accessor, vtx_push_back);
                }

            // Load colors
            auto colors = prim.findAttribute("COLOR_0");

            if (colors != prim.attributes.end())
                {
                    auto& accessor = gltf.accessors[colors->second];

                    auto vtx_push_back = [&](glm::vec4 v, size_t idx) 
                        { vertices[init_vtx + idx].color = v; };
                
                    fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, accessor, vtx_push_back);
                }

            new_mesh.meshes.push_back(range);
        }

        constexpr bool color_override = true;

        if (color_override) 
            {
                for (auto& vtx : vertices) 
                    { vtx.color = glm::vec4(vtx.normal, 1.0f); }
            }

        new_mesh.buffer = engine->createMeshBuffer(indices, vertices);
        meshes.emplace_back(std::make_shared<MeshType>(new_mesh));
    }

    return meshes;
}

std::optional<std::vector<std::shared_ptr<MeshType>>> loadOBJMeshes(NovaCoreLegacy* engine, std::filesystem::path path) {
    // TODO: Implement tinyobjloader
    return std::nullopt;
}

std::optional<std::vector<std::shared_ptr<MeshType>>> loadMeshes(NovaCoreLegacy* engine, std::filesystem::path path) {
    report(LOGGER::INFO, "Loading Mesh:");
    if (path.extension() == ".gltf") 
        { return loadGLTFMeshes(engine, path); } 
    else if (path.extension() == ".obj") 
        { return loadOBJMeshes(engine, path); } 
    else 
        { return std::nullopt; }
}