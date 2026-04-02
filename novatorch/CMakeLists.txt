cmake_minimum_required(VERSION 3.20)
project(NovaTorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Vulkan REQUIRED COMPONENTS glslangValidator)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

# Find pybind11 (bundled with PyTorch)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'share' / 'cmake')"
    OUTPUT_VARIABLE TORCH_SHARE_CMAKE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Build Nova as static library with PIC (required for shared .so linkage)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Nova ${CMAKE_CURRENT_BINARY_DIR}/Nova)

# Compile shaders
include(${CMAKE_CURRENT_SOURCE_DIR}/shaders/compile_shaders.cmake)
compile_nova_shaders(nova_shaders
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# NovaTorch extension sources
set(NOVATORCH_SOURCES
    csrc/bridge/nova_context.cpp
    csrc/bridge/nova_allocator.cpp
    csrc/bridge/nova_storage.cpp
    csrc/bridge/nova_guard.cpp
    csrc/bridge/nova_hooks.cpp
    csrc/bridge/nova_register.cpp
    csrc/bridge/nova_ops_factory.cpp
    csrc/bridge/nova_batch_context.cpp
    csrc/bridge/nova_ops_dispatch.cpp
    csrc/bridge/nova_ops_copy.cpp
    csrc/bridge/nova_ops_elementwise.cpp
    csrc/bridge/nova_ops_reduce.cpp
    csrc/bridge/nova_ops_matmul.cpp
    csrc/bridge/nova_ops_activation.cpp
    csrc/bridge/nova_ops_math.cpp
    csrc/bridge/nova_ops_comparison.cpp
    csrc/bridge/nova_ops_loss.cpp
    csrc/bridge/nova_ops_inplace.cpp
    csrc/bridge/nova_ops_extra.cpp
    csrc/bridge/nova_pipeline_cache.cpp
    csrc/bridge/nova_descriptor_pool.cpp
    csrc/bindings/module.cpp
)

# Build as shared library (Python extension)
Python3_add_library(novatorch_C MODULE ${NOVATORCH_SOURCES})

# Set output name to match Python import
set_target_properties(novatorch_C PROPERTIES
    OUTPUT_NAME "_C"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/novatorch
)

target_include_directories(novatorch_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/bridge
    ${CMAKE_CURRENT_SOURCE_DIR}/../Nova
    ${CMAKE_CURRENT_SOURCE_DIR}/../Nova/Core
    ${CMAKE_CURRENT_SOURCE_DIR}/../Nova/Core/components
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(novatorch_C PRIVATE
    Nova
    ${TORCH_LIBRARIES}
    Vulkan::Vulkan
    Python3::Module
)

# Define shader directory for runtime loading
target_compile_definitions(novatorch_C PRIVATE
    NOVATORCH_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}/shaders"
)

add_dependencies(novatorch_C nova_shaders)

# Install
install(TARGETS novatorch_C LIBRARY DESTINATION novatorch)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders/
    DESTINATION novatorch/shaders
    FILES_MATCHING PATTERN "*.spv")
