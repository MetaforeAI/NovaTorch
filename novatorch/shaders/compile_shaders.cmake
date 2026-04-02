# compile_shaders.cmake
# CMake function to compile GLSL compute shaders to SPIR-V via glslangValidator.
#
# Usage:
#   include(compile_shaders.cmake)
#   compile_nova_shaders(nova_shaders
#       "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
#       "${CMAKE_CURRENT_BINARY_DIR}/shaders")

function(compile_nova_shaders target shader_dir output_dir)
    file(MAKE_DIRECTORY ${output_dir})
    file(GLOB SHADER_SOURCES "${shader_dir}/*.comp")

    set(SPV_FILES "")
    foreach(shader ${SHADER_SOURCES})
        get_filename_component(name ${shader} NAME_WE)
        set(spv "${output_dir}/${name}.spv")
        add_custom_command(
            OUTPUT ${spv}
            COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V ${shader} -o ${spv}
            DEPENDS ${shader}
            COMMENT "Compiling ${name}.comp -> ${name}.spv"
        )
        list(APPEND SPV_FILES ${spv})
    endforeach()

    add_custom_target(${target} ALL DEPENDS ${SPV_FILES})
endfunction()
