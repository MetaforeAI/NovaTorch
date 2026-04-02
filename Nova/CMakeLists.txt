cmake_minimum_required(VERSION 3.20)
project(Nova LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan
find_package(Vulkan REQUIRED)

# Find SDL2
find_package(SDL2 REQUIRED)

# Collect Nova source files
set(NOVA_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/Nova.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/placeholder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/core_base.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/nova_compute.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/nova_buffer_arena.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/nova_command_batch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/nova_multi_queue.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/nova_cross_talk_buffer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/nova_graphics.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/modules/device.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/modules/management.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/modules/presentation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/modules/render.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/modules/pipeline/pipeline.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/modules/camera/camera.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/modules/camera/perspective.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/components/genesis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/components/logger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Core/components/obj_loader.cpp"
)

message(STATUS "NOVA_SOURCES: ${NOVA_SOURCES}")
foreach(src ${NOVA_SOURCES})
    message(STATUS "  - ${src}")
endforeach()

# Create Nova library
add_library(Nova STATIC ${NOVA_SOURCES})

# Disable warnings-as-errors for Nova (external library)
target_compile_options(Nova PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-w>  # Suppress all warnings for Nova
)

# Include directories
target_include_directories(Nova PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/Core
    ${Vulkan_INCLUDE_DIRS}
    ${SDL2_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(Nova PUBLIC
    ${Vulkan_LIBRARIES}
    ${SDL2_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

# Compile definitions
target_compile_definitions(Nova PUBLIC
    NOVA_COMPUTE_ENABLED
)

# Export for parent project
set(NOVA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(NOVA_LIBRARIES Nova PARENT_SCOPE)