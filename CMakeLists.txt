cmake_minimum_required(VERSION 3.10)
project(Vulkan)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm CONFIG REQUIRED)

set(SOURCES
    main.cpp
    Engine.cpp
)
set(SHADER_FILES
    # ../shader.vert
    ../shader.frag
    ../shader.mesh
)
set(COMPILED_SHADERS "")

foreach(SHADER ${SHADER_FILES})
    get_filename_component(FILE_NAME ${SHADER} NAME)
    set(SPIRV ../${FILE_NAME}.spv)

    add_custom_command(
        OUTPUT ${SPIRV}
        COMMAND glslc ${SHADER} -o ${SPIRV} --target-env=vulkan1.4
        DEPENDS ${SHADER}
        COMMENT "Compiling ${FILE_NAME} to SPIR-V"
        VERBATIM
    )

    list(APPEND COMPILED_SHADERS ${SPIRV})
endforeach()

add_custom_target(Shaders DEPENDS ${COMPILED_SHADERS})
add_executable(${PROJECT_NAME} ${SOURCES})
add_dependencies(Vulkan Shaders)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
)
