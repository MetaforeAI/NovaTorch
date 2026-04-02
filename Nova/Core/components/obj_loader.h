#pragma once
#include <unordered_map>
#include <filesystem>
#include <optional>
#include <vector>
#include "../core.h"

std::optional<std::vector<std::shared_ptr<MeshType>>> loadGLTFMeshes(NovaCoreLegacy* engine, std::filesystem::path path);
std::optional<std::vector<std::shared_ptr<MeshType>>> loadOBJMeshes(NovaCoreLegacy* engine, std::filesystem::path path);
std::optional<std::vector<std::shared_ptr<MeshType>>> loadMeshes(NovaCoreLegacy* engine, std::filesystem::path path);