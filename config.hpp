#pragma once
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <optional>
#include <limits>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

#include "tiny_obj_loader.h"
#include "stb_image.h"

#include <vulkan/vulkan.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define VK_CHECK(x) vk_check_result((x), #x, __FILE__, __LINE__)
inline const char* vk_result_to_string(VkResult result) {
    switch (result) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        default: return "ERROR NOT RECOGNIZED";
    }
}
inline void vk_check_result(VkResult result, const char* expr, const char* file, int line) {
    if (result != VK_SUCCESS) {
        std::ostringstream oss;
        oss << "Vulkan error: " << vk_result_to_string(result)
            << " (" << result << ")"
            << " at " << file << ":" << line
            << " in call: " << expr;
        throw std::runtime_error(oss.str());
    }
}

static std::vector<char> readFile(std::string filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary); // start reading at the end, read file as binary
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}