#define STB_IMAGE_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include "Engine.hpp"

Engine::Engine() {
    loadModel();
    createMeshlets();
    createWindow();
    createInstance();
    createSurface();
    createDevice();
    createSwapchain();
    // renderpass specifies attachments that will be used during rendering with this graphics pipeline
    // we need to specify how many color and depth buffers we will provide, how many samples to use for each of them
    // and how their content should be handled
    createRenderpass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createColorAttachment();
    createDepthAttachment();
    createFramebuffers();
    createTextureImage();
    createTextureImageView();
    createTextureImageSampler();
    createVertexBuffer();
    createMeshletBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSet();
}

Engine::~Engine() {
    vkDestroySampler(device, textureSampler, nullptr);
    vkDestroyImageView(device, textureImageView, nullptr);
    vkDestroyImage(device, textureImage, nullptr);
    vkFreeMemory(device, textureImageMemory, nullptr);
    cleanupSwapchain();
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, meshDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);
    vkDestroyBuffer(device, meshletBuffer, nullptr);
    vkFreeMemory(device, meshletBufferMemory, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyRenderPass(device, renderpass, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void Engine::run() {
    std::vector<VkSemaphore> imageAvailable(MAX_FRAMES_IN_FLIGHT);
    std::vector<VkSemaphore> renderDone(MAX_FRAMES_IN_FLIGHT);
    std::vector<VkFence> cmdBufferReady(MAX_FRAMES_IN_FLIGHT);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) createSemaphore(imageAvailable[i]);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) createSemaphore(renderDone[i]);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) createFence(cmdBufferReady[i]);

    VkCommandPool graphicsCmdPool = createCommandPool(queueFamilies.graphicsFamily.value(), 
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
    std::vector<VkCommandBuffer> graphicsCmdBuffer = createCommandBuffer(graphicsCmdPool, MAX_FRAMES_IN_FLIGHT);
    double lastTime = glfwGetTime();
    int frames = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // fence indicates the cmd buffer is ready to be rerecorded
        // this fence also indicates that the semaphores are available
        VK_CHECK(vkWaitForFences(device, 1, &cmdBufferReady[currFrame], VK_TRUE, ~0ull));
        
        uint32_t imageIndex;
        VkResult r = vkAcquireNextImageKHR(device, swapchain, ~0ull, imageAvailable[currFrame], VK_NULL_HANDLE, &imageIndex);
        if (r==VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            continue;
        } else if (r!=VK_SUCCESS && r!=VK_SUBOPTIMAL_KHR) {
            VK_CHECK(r);
        }

        VK_CHECK(vkResetFences(device, 1, &cmdBufferReady[currFrame]));
        VK_CHECK(vkResetCommandBuffer(graphicsCmdBuffer[currFrame], 0));
        
        VkCommandBufferBeginInfo beginCmdBufferInfo{};
        beginCmdBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT specifies that each recording of the command buffer will only be submitted once, 
        // and the command buffer will be reset and recorded again between each submission.
        // VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT means it is a seconday buffer that will be entirely within a single render pass
        // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT means this cmd buffer will be resubmitted while it is also already pending
        beginCmdBufferInfo.flags = 0;
        // for seconday buffers, this info specifies which state to inherit from the primary cmd buffer
        beginCmdBufferInfo.pInheritanceInfo = nullptr;
        VK_CHECK(vkBeginCommandBuffer(graphicsCmdBuffer[currFrame], &beginCmdBufferInfo));
        {
            VkRenderPassBeginInfo beginRenderPassInfo{};
            beginRenderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            beginRenderPassInfo.renderPass = renderpass;
            beginRenderPassInfo.framebuffer = framebuffers[imageIndex];
            beginRenderPassInfo.renderArea.offset = {0, 0};
            beginRenderPassInfo.renderArea.extent = swapchainExtent;
            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            clearValues[1].depthStencil = {1.0f, 0};
            beginRenderPassInfo.clearValueCount = clearValues.size();
            beginRenderPassInfo.pClearValues = clearValues.data(); // the value to set for clearing during load op

            // VK_SUBPASS_CONTENTS_INLINE means the renderpass commands will be embedded in the primary cmd buffer and
            // no seconday buffer will be executed
            // VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS means the renderpass commands will be executed from seconday cmd buffer
            vkCmdBeginRenderPass(graphicsCmdBuffer[currFrame], &beginRenderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            {
                vkCmdBindPipeline(graphicsCmdBuffer[currFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
                VkBuffer vertexBuffers[] = {vertexBuffer};
                VkDeviceSize offsets[] = {0};
                // vkCmdBindVertexBuffers(graphicsCmdBuffer[currFrame], 0, 1, vertexBuffers, offsets);
                // vkCmdBindIndexBuffer(graphicsCmdBuffer[currFrame], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                VkViewport viewport{};
                viewport.height = swapchainExtent.height;
                viewport.width = swapchainExtent.width;
                viewport.x = 0.0f;
                viewport.y = 0.0f;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(graphicsCmdBuffer[currFrame], 0, 1, &viewport);
                VkRect2D scissor{};
                scissor.extent = swapchainExtent;
                scissor.offset = {0, 0};
                vkCmdSetScissor(graphicsCmdBuffer[currFrame], 0, 1, &scissor);

                updateUniformBuffers();

                PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = 
                    (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
                if (!vkCmdPushDescriptorSetKHR) {
                    throw std::runtime_error("Failed to load vkCmdPushDescriptorSetKHR function");
                }
                VkDescriptorBufferInfo vertexBufferInfo{};
                vertexBufferInfo.buffer = vertexBuffer;
                vertexBufferInfo.offset = 0;
                vertexBufferInfo.range = vertexBufferSize;
                VkWriteDescriptorSet writeBuffer{};
                writeBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeBuffer.dstBinding = 0;
                writeBuffer.dstArrayElement = 0;
                writeBuffer.descriptorCount = 1;
                writeBuffer.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writeBuffer.pBufferInfo = &vertexBufferInfo;
                vkCmdPushDescriptorSetKHR(graphicsCmdBuffer[currFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &writeBuffer);

                VkDescriptorBufferInfo meshletBufferInfo{};
                meshletBufferInfo.buffer = meshletBuffer;
                meshletBufferInfo.offset = 0;
                meshletBufferInfo.range = meshletBufferSize;
                writeBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writeBuffer.dstBinding = 1;
                writeBuffer.dstArrayElement = 0;
                writeBuffer.descriptorCount = 1;
                writeBuffer.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writeBuffer.pBufferInfo = &meshletBufferInfo;
                vkCmdPushDescriptorSetKHR(graphicsCmdBuffer[currFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &writeBuffer);

                // descriptor sets are not unique to graphics pipeline, so we need to specify the pipeline
                vkCmdBindDescriptorSets(graphicsCmdBuffer[currFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, 
                    &descriptorSets[currFrame], 0, nullptr);

                PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT = 
                    (PFN_vkCmdDrawMeshTasksEXT) vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT");
                // vkCmdDrawIndexed(graphicsCmdBuffer[currFrame], indices.size(), 1, 0, 0, 0);
                vkCmdDrawMeshTasksEXT(graphicsCmdBuffer[currFrame], mesh.meshlets.size(), 1, 1);
            }
            vkCmdEndRenderPass(graphicsCmdBuffer[currFrame]);
        }
        VK_CHECK(vkEndCommandBuffer(graphicsCmdBuffer[currFrame]));
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &graphicsCmdBuffer[currFrame];
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &imageAvailable[currFrame];
        submitInfo.pSignalSemaphores = &renderDone[currFrame];
        submitInfo.signalSemaphoreCount = 1;

        // specifies in which stages of the pipeline we wait for pWaitSemaphores
        // in this case we need to wait until writing colors since we may not have the image yet
        // so it means we can start executing vertex shader and etc, but we will wait to write colors to the image
        std::vector<VkPipelineStageFlags> stages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.pWaitDstStageMask = stages.data();

        VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, cmdBufferReady[currFrame]));
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pWaitSemaphores = &renderDone[currFrame];
        presentInfo.waitSemaphoreCount = 1;
        r = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (r==VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            continue;
        } else if (r!=VK_SUCCESS && r!=VK_SUBOPTIMAL_KHR) {
            VK_CHECK(r);
        }
        currFrame = (currFrame+1)%MAX_FRAMES_IN_FLIGHT;

        frames++;
        double currentTime = glfwGetTime();
        double elapsed = currentTime - lastTime; 
        if (elapsed >= 1.0) {
            std::ostringstream title;
            title << "CPU: " << frames/elapsed << ", Triangles: "<<mesh.indices.size()/3<<", Meshlets: "<<mesh.meshlets.size()<<'\n';
            glfwSetWindowTitle(window, title.str().c_str());
            frames = 0;
            lastTime = currentTime;
        }
    }
    VK_CHECK(vkDeviceWaitIdle(device));
    for (auto fence: cmdBufferReady) vkDestroyFence(device, fence, nullptr);
    for (auto sem: imageAvailable) vkDestroySemaphore(device, sem, nullptr);
    for (auto sem: renderDone) vkDestroySemaphore(device, sem, nullptr);
    vkDestroyCommandPool(device, graphicsCmdPool, nullptr);
}

void Engine::loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;
    std::string warn;
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_OBJ.c_str(), nullptr)) {
        throw std::runtime_error("Error loading the model");
    }
    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};
            vertex.x = attrib.vertices[3 * index.vertex_index + 0];
            vertex.y = attrib.vertices[3 * index.vertex_index + 1];
            vertex.z = attrib.vertices[3 * index.vertex_index + 2];  
            vertex.tx = attrib.texcoords[2 * index.texcoord_index + 0];
            vertex.ty = 1.0f - attrib.texcoords[2 * index.texcoord_index + 1];
            glm::vec3 normal = {
                attrib.normals[3 * index.normal_index + 0],
                attrib.normals[3 * index.normal_index + 1],
                attrib.normals[3 * index.normal_index + 2],
            };
            normal = glm::normalize(normal);
            // input float is [-1, 1], we need to convert it to [0, 255] to fit into uint8_t
            vertex.nx = uint8_t((normal.x*0.5f+0.5f)*255.0f);
            vertex.ny = uint8_t((normal.y*0.5f+0.5f)*255.0f);
            vertex.nz = uint8_t((normal.z*0.5f+0.5f)*255.0f);
            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = mesh.vertices.size();
                mesh.vertices.push_back(vertex);
            }

            mesh.indices.push_back(uniqueVertices[vertex]);
        }
    }
}
void Engine::createMeshlets() {
    const uint32_t MAX_VERTICES = 64;
    const uint32_t MAX_TRIANGLES = 42;
    
    uint32_t triangleCount = mesh.indices.size() / 3;
    uint32_t currentTriangle = 0;
    
    while (currentTriangle < triangleCount) {
        Meshlet meshlet = {};
    
        std::unordered_set<uint32_t> meshletVertexSet;
        std::vector<uint32_t> meshletVertexList;
        std::vector<uint8_t> meshletIndicesList;
        uint32_t trianglesInMeshlet = 0;
    
        while (currentTriangle < triangleCount && trianglesInMeshlet < MAX_TRIANGLES) {
            uint32_t i0 = mesh.indices[currentTriangle * 3 + 0];
            uint32_t i1 = mesh.indices[currentTriangle * 3 + 1];
            uint32_t i2 = mesh.indices[currentTriangle * 3 + 2];
            
            std::unordered_set<uint32_t> potentialVertices = meshletVertexSet;
            potentialVertices.insert(i0);
            potentialVertices.insert(i1);
            potentialVertices.insert(i2);
            if (potentialVertices.size() > MAX_VERTICES) {
                break;
            }
            
            auto addVertex = [&](uint32_t globalIdx) -> uint8_t {
                if (meshletVertexSet.find(globalIdx) == meshletVertexSet.end()) {
                    meshletVertexSet.insert(globalIdx);
                    meshletVertexList.push_back(globalIdx);
                }
                for (uint8_t i = 0; i < meshletVertexList.size(); i++) {
                    if (meshletVertexList[i] == globalIdx) return i;
                }
                return 0;
            };
            
            uint8_t localI0 = addVertex(i0);
            uint8_t localI1 = addVertex(i1);  
            uint8_t localI2 = addVertex(i2);
            
            meshletIndicesList.push_back(localI0);
            meshletIndicesList.push_back(localI1);
            meshletIndicesList.push_back(localI2);
            
            trianglesInMeshlet++;
            currentTriangle++;
        }
    
        meshlet.vertexCount = meshletVertexList.size();
        meshlet.indexCount = meshletIndicesList.size();
        for (size_t i = 0; i < meshletVertexList.size(); i++) {
            meshlet.vertices[i] = meshletVertexList[i];
        }
        for (size_t i = 0; i < meshletIndicesList.size(); i++) {
            meshlet.indices[i] = meshletIndicesList[i];
        }
        mesh.meshlets.push_back(meshlet);
    }
}
void Engine::createWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);
    glfwMakeContextCurrent(window);
}
void Engine::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_4;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pApplicationName = "Vulkan application";
    appInfo.pEngineName = "Vulkan Engine";
    
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    
    std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    if (!checkInstanceLayersSupport(validationLayers)) throw std::runtime_error("Validation layers not supported");
    instanceInfo.enabledLayerCount = validationLayers.size();
    instanceInfo.ppEnabledLayerNames = validationLayers.data();

    std::vector<const char*> instanceExtensions = {
        "VK_KHR_surface",
        "VK_KHR_xcb_surface"
    };
    if (!checkInstanceExtensionsSupport(instanceExtensions)) throw std::runtime_error("Instance extensions not supported");
    instanceInfo.enabledExtensionCount = instanceExtensions.size();
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();

    instanceInfo.pApplicationInfo = &appInfo;
    
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));
}
void Engine::createSurface() {
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}
void Engine::createDevice() {
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (const auto& device: devices) {
        if (isDeviceSuitable(device)) {
            pDevice = device;
            msaaSamples = getMaxSampleCount();
            break;
        }
    }
    if (pDevice == VK_NULL_HANDLE) throw std::runtime_error("No physical devices");
    
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.features.geometryShader = VK_TRUE;
    features.features.samplerAnisotropy = VK_TRUE;
    features.features.sampleRateShading = VK_TRUE;
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.shaderInt8 = VK_TRUE;
    features12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    features12.storageBuffer8BitAccess = VK_TRUE;
    VkPhysicalDeviceMaintenance4Features maintenanceFeatures{};
    maintenanceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES;
    maintenanceFeatures.maintenance4 = VK_TRUE;
    VkPhysicalDeviceMeshShaderFeaturesEXT meshFeatures{};
    meshFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    meshFeatures.meshShader = VK_TRUE;
    meshFeatures.pNext = &maintenanceFeatures;
    features12.pNext = &meshFeatures;
    features.pNext = &features12;

    queueFamilies = getQueueFamilies(pDevice);
    std::set<uint32_t> uniqueQueueFamilies = {
        queueFamilies.transferFamily.value(), 
        queueFamilies.graphicsFamily.value(), 
        queueFamilies.presentFamily.value()
    };
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    float priority = 1.0f;
    for (const auto& family: uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.pQueuePriorities = &priority;
        queueInfo.queueCount = 1;
        queueInfo.queueFamilyIndex = family;
        queueInfos.push_back(queueInfo);
    }

    std::vector<const char*> extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
        VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        VK_EXT_MESH_SHADER_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
    };
    deviceInfo.enabledExtensionCount = extensions.size();
    deviceInfo.ppEnabledExtensionNames = extensions.data();
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.queueCreateInfoCount = queueInfos.size();
    deviceInfo.pNext = &features;

    VK_CHECK(vkCreateDevice(pDevice, &deviceInfo, nullptr, &device));

    vkGetDeviceQueue(device, queueFamilies.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFamilies.presentFamily.value(), 0, &presentQueue);
    vkGetDeviceQueue(device, queueFamilies.transferFamily.value(), 0, &transferQueue);
}
void Engine::createSwapchain() {
    SurfaceDetails surfaceDetails = getSurfaceDetails(pDevice);
    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(surfaceDetails.formats);
    VkPresentModeKHR presentMode = choosePresentMode(surfaceDetails.presentModes);
    VkExtent2D extent = chooseExtent(surfaceDetails.capabilities);

    uint32_t imageCount = surfaceDetails.capabilities.minImageCount+1;
    if (imageCount > surfaceDetails.capabilities.maxImageCount && surfaceDetails.capabilities.maxImageCount>0) {
        imageCount = surfaceDetails.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.surface = surface;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = surfaceFormat.format;
    swapchainInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapchainInfo.imageExtent = extent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = {queueFamilies.graphicsFamily.value(), queueFamilies.presentFamily.value()};
    if (queueFamilies.graphicsFamily.value()==queueFamilies.presentFamily.value()) {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; // images are not shared among queues
        swapchainInfo.queueFamilyIndexCount = 1;
        swapchainInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainInfo.queueFamilyIndexCount = 2;
        swapchainInfo.pQueueFamilyIndices = queueFamilyIndices;
    }

    swapchainInfo.preTransform = surfaceDetails.capabilities.currentTransform;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // to use alpha channel for blending with other windows
    swapchainInfo.clipped = VK_TRUE; // discard pixels that are obscured, i.e. by another window
    swapchainInfo.presentMode = presentMode;
    swapchainInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &swapchain));

    swapchainExtent = extent;
    swapchainFormat = surfaceFormat.format;

    uint32_t count;
    vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr);
    swapchainImages.resize(count);
    swapchainImageViews.resize(count);
    vkGetSwapchainImagesKHR(device, swapchain, &count, swapchainImages.data());
    for (int i=0; i<swapchainImageViews.size(); i++) {
        createImageView(swapchainImageViews[i], swapchainImages[i], swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }
}
void Engine::createDescriptorSetLayout() {
    // layout describes what type of resources the pipeline is supposed to expect, i.e how render pass expects certain attachments
    // descriptor set itself describes actual data, i.e how framebuffer has actual image views

    // we need to provide details about each binding
    VkDescriptorSetLayoutBinding uniformBufferLayoutBinding{};
    uniformBufferLayoutBinding.binding = 0;
    uniformBufferLayoutBinding.descriptorCount = 1; // might be an array of uniforms of the same type, i.e array of buffers
    uniformBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    uniformBufferLayoutBinding.pImmutableSamplers = nullptr; // image sampling descriptors

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding vertexBufferLayoutBinding{};
    vertexBufferLayoutBinding.binding = 0;
    vertexBufferLayoutBinding.descriptorCount = 1;
    vertexBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vertexBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    vertexBufferLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding meshletBufferLayoutBinding{};
    meshletBufferLayoutBinding.binding = 1;
    meshletBufferLayoutBinding.descriptorCount = 1;
    meshletBufferLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    meshletBufferLayoutBinding.stageFlags = VK_SHADER_STAGE_MESH_BIT_EXT;
    meshletBufferLayoutBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding bindings[] = {uniformBufferLayoutBinding, samplerLayoutBinding};

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = sizeof(bindings)/sizeof(bindings[0]);
    layoutInfo.pBindings = bindings;
    layoutInfo.flags = 0;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));

    VkDescriptorSetLayoutBinding meshBindings[] = {vertexBufferLayoutBinding, meshletBufferLayoutBinding};
    VkDescriptorSetLayoutCreateInfo meshLayoutInfo{};
    meshLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    meshLayoutInfo.bindingCount = sizeof(meshBindings)/sizeof(meshBindings[0]);
    meshLayoutInfo.pBindings = meshBindings;
    meshLayoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &meshLayoutInfo, nullptr, &meshDescriptorSetLayout));
}
void Engine::createGraphicsPipeline() {
    std::vector<char> meshCode = readFile("../shader.mesh.spv");
    std::vector<char> fragCode = readFile("../shader.frag.spv");
    VkShaderModule meshModule = createShaderModule(meshCode);
    VkShaderModule fragModule = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo meshStageInfo{};
    meshStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    meshStageInfo.module = meshModule;
    meshStageInfo.pName = "main";
    meshStageInfo.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
    meshStageInfo.pSpecializationInfo = nullptr; // allows us to specify shader constants

    VkPipelineShaderStageCreateInfo fragStageInfo{};
    fragStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStageInfo.module = fragModule;
    fragStageInfo.pName = "main";
    fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStageInfo.pSpecializationInfo = nullptr;

    VkPipelineShaderStageCreateInfo shaderStages[] = {meshStageInfo, fragStageInfo};

    // how to treat incoming data
    VkPipelineVertexInputStateCreateInfo vInputInfo{};
    // auto bindingDescription = Vertex::getBindingDescription();
    // auto attributeDescription = Vertex::getAttributeDescription();
    vInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    // vInputInfo.vertexAttributeDescriptionCount = attributeDescription.size();
    // vInputInfo.pVertexAttributeDescriptions = attributeDescription.data();
    // vInputInfo.vertexBindingDescriptionCount = 1;
    // vInputInfo.pVertexBindingDescriptions = &bindingDescription;

    // what kind of geometry to draw and how to order vertices
    VkPipelineInputAssemblyStateCreateInfo assembleInfo{};
    assembleInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembleInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // triangle from every 3 vertices
    assembleInfo.primitiveRestartEnable = VK_FALSE; // only used to break up lines and triangles in STRIP topology

    // the region of the framebuffer attachment that the output will be rendered to
    VkPipelineViewportStateCreateInfo viewportInfo{};
    viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportInfo.scissorCount = 1;
    viewportInfo.viewportCount = 1;

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicInfo{};
    dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicInfo.dynamicStateCount = dynamicStates.size();
    dynamicInfo.pDynamicStates = dynamicStates.data();

    // rasterizer takes geometry and turns into fragments, performs depth testing, face culling and scissor test
    VkPipelineRasterizationStateCreateInfo rasterInfo{};
    rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterInfo.depthClampEnable = VK_FALSE; // clamp fragments beyond the near and far planes to them (as opposed to discarding)
    rasterInfo.rasterizerDiscardEnable = VK_FALSE; // transform feedback enabled, requires GPU feature enabled
    rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterInfo.lineWidth = 1.0f;
    rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterInfo.depthBiasEnable = VK_FALSE; // add a constant value to depth values or bias them based on a fragment's slope

    // sampling multiple fragments that rasterize to the same pixel
    VkPipelineMultisampleStateCreateInfo msaaInfo{};
    msaaInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaaInfo.sampleShadingEnable = VK_TRUE;
    msaaInfo.minSampleShading = 0.2f;
    msaaInfo.rasterizationSamples = msaaSamples;

    // blend the calculated color with the color already in the framebuffer
    VkPipelineColorBlendAttachmentState colorBlendAttachment{}; // configuration per attached framebuffer
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | 
        VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo blendInfo{}; // global color blending settings
    blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blendInfo.logicOpEnable = VK_FALSE;
    blendInfo.attachmentCount = 1;
    blendInfo.pAttachments = &colorBlendAttachment;

    // layout used to pass uniform values and push constants into shaders
    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    VkDescriptorSetLayout descriptorSetLayouts[] = {descriptorSetLayout, meshDescriptorSetLayout};
    layoutInfo.pSetLayouts = descriptorSetLayouts;
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pPushConstantRanges = nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout)); 

    VkPipelineDepthStencilStateCreateInfo depthStencilInfo{};
    depthStencilInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilInfo.depthTestEnable = VK_TRUE;
    depthStencilInfo.depthWriteEnable = VK_TRUE;
    depthStencilInfo.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencilInfo.depthBoundsTestEnable = VK_FALSE;
    depthStencilInfo.minDepthBounds = 0.0f;
    depthStencilInfo.maxDepthBounds = 1.0f;
    depthStencilInfo.stencilTestEnable = VK_FALSE;
    depthStencilInfo.front = {};
    depthStencilInfo.back = {};

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vInputInfo;
    pipelineInfo.pInputAssemblyState = &assembleInfo;
    pipelineInfo.pViewportState = &viewportInfo;
    pipelineInfo.pRasterizationState = &rasterInfo;
    pipelineInfo.pMultisampleState = &msaaInfo;
    pipelineInfo.pColorBlendState = &blendInfo;
    pipelineInfo.pDynamicState = &dynamicInfo;
    pipelineInfo.pDepthStencilState = &depthStencilInfo;
    pipelineInfo.layout = layout;
    pipelineInfo.renderPass = renderpass;
    pipelineInfo.subpass = 0; // index of the subpass to be used
    // in case we want to derive pipeline from another pipeline
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;
    
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline));

    vkDestroyShaderModule(device, meshModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
}
void Engine::createFramebuffers() {
    framebuffers.resize(swapchainImageViews.size());
    for (int i=0; i<framebuffers.size(); i++) {
        std::vector<VkImageView> attachments = {colorImageView, depthImageView, swapchainImageViews[i]};
        VkFramebufferCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        createInfo.attachmentCount = attachments.size();
        createInfo.pAttachments = attachments.data();
        // provide renderpass that framebuffer needs to be compatible with
        createInfo.renderPass = renderpass; 
        createInfo.width = swapchainExtent.width;
        createInfo.height = swapchainExtent.height;
        createInfo.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device, &createInfo, nullptr, &framebuffers[i])); 
    }
}
void Engine::createColorAttachment() {
    VkFormat colorFormat = swapchainFormat;
    createImage(swapchainExtent.width, swapchainExtent.height, 1, msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 
        colorImage, colorImageMemory);
    createImageView(colorImageView, colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
}
void Engine::createDepthAttachment() {
    VkFormat depthFormat = findDepthFormat();
    createImage(swapchainExtent.width, swapchainExtent.height, 1, msaaSamples, depthFormat, VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    createImageView(depthImageView, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
}
void Engine::createRenderpass() {
    // attachments must be in the same order they are provided by framebuffer
    VkAttachmentDescription colorAttachment{};
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; // we don't care about layout at the beginning
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.format = swapchainFormat;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // clear the values to a constant at the start 
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // store rendered content in memory to be read
    colorAttachment.samples = msaaSamples;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.format = findDepthFormat();
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.samples = msaaSamples;

    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    colorAttachmentResolve.format = swapchainFormat;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;

    VkAttachmentDescription attachments[] = {colorAttachment, depthAttachment, colorAttachmentResolve};
    
    // each subpass references 1+ attachments from attachment description array
    VkAttachmentReference colorAttachmentRef;
    // which attachment to reference by its index in the attachment description array
    // this index is directly referenced by fragment shader output, i.e layout (location=0) out vec4 outColor
    colorAttachmentRef.attachment = 0;
    // what layout we want attachment to have when subpass starts
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // used as a color buffer 

    VkAttachmentReference depthAttachmentRef;
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResovleRef;
    colorAttachmentResovleRef.attachment = 2;
    colorAttachmentResovleRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // each renderpass consists of multiple subpasses
    // each subpass is a sequence of rendering operations that operate on contents of framebuffers from the previous passes
    std::vector<VkSubpassDescription> subpasses(1);
    subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; // graphics subpass
    subpasses[0].colorAttachmentCount = 1;
    subpasses[0].pColorAttachments = &colorAttachmentRef;
    subpasses[0].pDepthStencilAttachment = &depthAttachmentRef;
    subpasses[0].pResolveAttachments = &colorAttachmentResovleRef;

    // subpass dependency controls how to do image layout transition
    // there are two implicit transitions: at the beginning and at the end of the render pass
    // the one at the beginning transitions image right away at the beginning of the pipeline, but the image hasn't been acquired yet
    // due to imageAvailable semaphore
    // so we wait until we hit VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT stage to actually start render pass and transition the image
    // thus, we make the subpass wait until image is acquired
    std::vector<VkSubpassDependency> dependencies(1);
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL; // subpass before current subpass
    dependencies[0].dstSubpass = 0; // current subpass, must always be bigger value than srcSubpass
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | 
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT; // wait for these stages to finish in srcSubpass
    dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; // operations to wait on
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | 
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT; // operations here are not executed, wait for srcStageMask
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | 
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; // operations that wait for transition to be done

    VkRenderPassCreateInfo renderpassInfo{};
    renderpassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderpassInfo.attachmentCount = sizeof(attachments)/sizeof(attachments[0]);
    renderpassInfo.pAttachments = attachments;
    renderpassInfo.subpassCount = subpasses.size();
    renderpassInfo.pSubpasses = subpasses.data();
    renderpassInfo.dependencyCount = dependencies.size();
    renderpassInfo.pDependencies = dependencies.data();
    VK_CHECK(vkCreateRenderPass(device, &renderpassInfo, nullptr, &renderpass));
}
void Engine::recreateSwapchain() {
    vkDeviceWaitIdle(device);

    cleanupSwapchain();

    createSwapchain();
    createColorAttachment();
    createDepthAttachment();
    createFramebuffers();
}
void Engine::cleanupSwapchain() {
    vkDestroyImageView(device, colorImageView, nullptr);
    vkDestroyImage(device, colorImage, nullptr);
    vkFreeMemory(device, colorImageMemory, nullptr);
    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);
    for (auto imageView: swapchainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    for (auto framebuffer: framebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}
void Engine::createTextureImage() {
    int texWidth;
    int texHeight;
    int texChannels;
    stbi_uc* pixels = stbi_load(MODEL_TEX.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    mipLevels = std::floor(std::log2(std::max(texWidth, texHeight)))+1;
    VkDeviceSize imageSize = texWidth*texHeight*4;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, imageSize);
    vkUnmapMemory(device, stagingBufferMemory);
    stbi_image_free(pixels);

    createImage(texWidth, texHeight, mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
    
    VkCommandPool transferCmdPool = createCommandPool(queueFamilies.transferFamily.value(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    std::vector<VkCommandBuffer> transferCmdBuffer = createCommandBuffer(transferCmdPool, 1);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(transferCmdBuffer[0], &beginInfo);
    transitionImageLayout(transferCmdBuffer[0], textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, 
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
    vkEndCommandBuffer(transferCmdBuffer[0]);
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = transferCmdBuffer.data();
    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);
    vkQueueSubmit(transferQueue, 1, &submit, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, ~0ull);
    vkDestroyCommandPool(device, transferCmdPool, nullptr);
    vkDestroyFence(device, fence, nullptr);

    copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
    
    generateMipMaps(textureImage, VK_FORMAT_R8G8B8A8_UNORM, texWidth, texHeight, mipLevels);
    
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void Engine::createTextureImageView() {
    createImageView(textureImageView, textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
}
void Engine::createTextureImageSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR; // how to interpolate texels that are magnified - helps with oversampling
    samplerInfo.minFilter = VK_FILTER_LINEAR; // how to interpolate texels that are minified - helps with undersampling
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    VkPhysicalDeviceProperties prop{};
    vkGetPhysicalDeviceProperties(pDevice, &prop);
    samplerInfo.maxAnisotropy = prop.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE; // whether to use (texWidth, texHeight) as coordinates
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler));
}
void Engine::createVertexBuffer() {
    vertexBufferSize = sizeof(mesh.vertices[0])*mesh.vertices.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means GPU keeps track of writes memory copies to this buffer
    // if the writes are done to cache or they are not done yet, GPU will take it into account
    // without this flag we have to manually flush writes
    createBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data;
    // this function allows us to access a region of the specified memory resource defined by an offset and size
    vkMapMemory(device, stagingBufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, mesh.vertices.data(), sizeof(mesh.vertices[0])*mesh.vertices.size()); // copy vertices into data, which is bound to stagingBufferMemory
    vkUnmapMemory(device, stagingBufferMemory);

    // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT is not accessible from CPU, so we can't use vkMapMemory
    createBuffer(vertexBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        vertexBuffer, vertexBufferMemory);

    copyBuffer(stagingBuffer, vertexBuffer, vertexBufferSize);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void Engine::createMeshletBuffer() {
    meshletBufferSize = sizeof(mesh.meshlets[0])*mesh.meshlets.size();
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(meshletBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, meshletBufferSize, 0, &data);
    memcpy(data, mesh.meshlets.data(), sizeof(mesh.meshlets[0])*mesh.meshlets.size());
    vkUnmapMemory(device, stagingBufferMemory);

    createBuffer(meshletBufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        meshletBuffer, meshletBufferMemory);

    copyBuffer(stagingBuffer, meshletBuffer, meshletBufferSize);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}
void Engine::createUniformBuffers() {
    VkDeviceSize size = sizeof(UniformBufferObject);
    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
        vkMapMemory(device, uniformBuffersMemory[i], 0, size, 0, &uniformBuffersMapped[i]);
    }
}
void Engine::updateUniformBuffers() {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currTime-startTime).count();

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time*0.5f*glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), swapchainExtent.width/(float)swapchainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1]*=-1;

    memcpy(uniformBuffersMapped[currFrame], &ubo, sizeof(ubo));
}

VkShaderModule Engine::createShaderModule(std::vector<char> code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size(); // size in bytes
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
    return shaderModule;
}
void Engine::createFence(VkFence& fence) {
    VkFenceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    createInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(device, &createInfo, nullptr, &fence));  
}
void Engine::createSemaphore(VkSemaphore& semaphore) {
    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VK_CHECK(vkCreateSemaphore(device, &createInfo, nullptr, &semaphore));
}
VkCommandPool Engine::createCommandPool(uint32_t queueFamily, VkCommandPoolCreateFlagBits flags) {
    VkCommandPoolCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    createInfo.queueFamilyIndex = queueFamily;

    // VK_COMMAND_POOL_CREATE_TRANSIENT_BIT specifies that command buffers allocated from the pool
    // will be short-lived, meaning that they will be reset or freed in a relatively short timeframe
    // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT allows any command buffer allocated from a pool
    // to be individually reset to the initial state; either by calling vkResetCommandBuffer, or 
    // via the implicit reset when calling vkBeginCommandBuffer. 
    // If this flag is not set on a pool, then vkResetCommandBuffer must not be called for any command buffer allocated from that pool.
    createInfo.flags = flags;

    VkCommandPool cmdPool;
    VK_CHECK(vkCreateCommandPool(device, &createInfo, nullptr, &cmdPool));
    return cmdPool;
}
std::vector<VkCommandBuffer> Engine::createCommandBuffer(VkCommandPool& pool, uint32_t count) {
    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = pool;

    // VK_COMMAND_BUFFER_LEVEL_SECONDARY level cannot be submitted to a queue directy, but can be called from primary cmd buffer
    // VK_COMMAND_BUFFER_LEVEL_PRIMARY can be submitted to a queue but cannot becalled from another cmd buffer
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = count;

    std::vector<VkCommandBuffer> cmdBuffer(count);
    VK_CHECK(vkAllocateCommandBuffers(device, &allocateInfo, cmdBuffer.data()));
    return cmdBuffer;
}
void Engine::createDescriptorPool() {
    // pool size to allocate descriptors of certain type
    VkDescriptorPoolSize poolSizeUBO{};
    poolSizeUBO.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizeUBO.descriptorCount = MAX_FRAMES_IN_FLIGHT; // how many descriptors of the type will be allocated from this pool
    
    VkDescriptorPoolSize poolSizeSampler{};
    poolSizeSampler.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizeSampler.descriptorCount = MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolSize poolSizes[] = {poolSizeUBO, poolSizeSampler};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.poolSizeCount = sizeof(poolSizes)/sizeof(poolSizes[0]);
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT; // maximum number of descriptor sets that may be allocated

    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
}
void Engine::createDescriptorSet() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT; // how many descriptor sets to allocate
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()));
    
    for (int i=0; i<MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = textureImageView;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.sampler = textureSampler;

        VkWriteDescriptorSet descriptorWriteBuffer{};
        descriptorWriteBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteBuffer.dstSet = descriptorSets[i];
        descriptorWriteBuffer.dstBinding = 0;
        descriptorWriteBuffer.dstArrayElement = 0; // uniform might be an array of buffers
        descriptorWriteBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWriteBuffer.descriptorCount = 1; // how many array elements to update
        descriptorWriteBuffer.pBufferInfo = &bufferInfo;

        VkWriteDescriptorSet descriptorWriteImage{};
        descriptorWriteImage.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWriteImage.dstSet = descriptorSets[i];
        descriptorWriteImage.dstBinding = 1;
        descriptorWriteImage.dstArrayElement = 0;
        descriptorWriteImage.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWriteImage.descriptorCount = 1;
        descriptorWriteImage.pImageInfo = &imageInfo;
        VkWriteDescriptorSet writeSets[] = {descriptorWriteBuffer, descriptorWriteImage};
        vkUpdateDescriptorSets(device, sizeof(writeSets)/sizeof(writeSets[0]), writeSets, 0, nullptr);
    }
}
bool Engine::checkInstanceLayersSupport(std::vector<const char*> layers) {
    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> availableLayers(count);
    vkEnumerateInstanceLayerProperties(&count, availableLayers.data());
    std::set<std::string> requested(layers.begin(), layers.end());
    for (const auto layer: availableLayers) {
        requested.erase(layer.layerName);
    }
    return requested.empty();
}
bool Engine::checkInstanceExtensionsSupport(std::vector<const char*> extensions) {
    uint32_t count;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, availableExtensions.data());
    std::set<std::string> requested(extensions.begin(), extensions.end());
    for (const auto ext: availableExtensions) {
        requested.erase(ext.extensionName);
    }
    return requested.empty();
}
bool Engine::checkDeviceExtensionsSupport(std::vector<const char*> extensions, VkPhysicalDevice device) {
    uint32_t count;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, availableExtensions.data());
    std::set<std::string> requested(extensions.begin(), extensions.end());

    for (const auto ext: availableExtensions) {
        requested.erase(ext.extensionName);
    }
    return requested.empty();
}
QueueFamilies Engine::getQueueFamilies(VkPhysicalDevice device) {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queueFamilies.data());

    QueueFamilies _queueFamilies;
    int i=0;
    for (const auto& queueFamily: queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            _queueFamilies.graphicsFamily = i;
        }
        if (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT && !(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            _queueFamilies.transferFamily = i;
        }
        VkBool32 presentSupported;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupported);
        if (presentSupported==VK_TRUE) {
            _queueFamilies.presentFamily = i;
        }
        if (_queueFamilies.isComplete()) {
            break;
        }
    }
    return _queueFamilies;
}
bool Engine::isDeviceSuitable(VkPhysicalDevice device) {
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);

    VkPhysicalDeviceProperties prop;
    vkGetPhysicalDeviceProperties(device, &prop);

    std::vector<const char*> extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    QueueFamilies _queueFamilies = getQueueFamilies(device);
    SurfaceDetails surfaceDetails = getSurfaceDetails(device);

    return !surfaceDetails.formats.empty() &&
        !surfaceDetails.presentModes.empty() && 
        _queueFamilies.isComplete() &&
        checkDeviceExtensionsSupport(extensions, device) &&
        features.geometryShader == VK_TRUE && 
        prop.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
        features.samplerAnisotropy == VK_TRUE &&
        features.sampleRateShading == VK_TRUE;
}
SurfaceDetails Engine::getSurfaceDetails(VkPhysicalDevice device) {
    SurfaceDetails surfaceDetails;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &surfaceDetails.capabilities);

    uint32_t count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
    surfaceDetails.formats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, surfaceDetails.formats.data());

    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
    surfaceDetails.presentModes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, surfaceDetails.presentModes.data());

    return surfaceDetails;
}
VkSurfaceFormatKHR Engine::chooseSurfaceFormat(std::vector<VkSurfaceFormatKHR> formats) {
    for (const auto format: formats) {
        if (format.format == VK_FORMAT_R8G8B8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats[0];
}
VkPresentModeKHR Engine::choosePresentMode(std::vector<VkPresentModeKHR> presentModes) {
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
    for (const auto mode: presentModes) {
        if (mode==VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D Engine::chooseExtent(VkSurfaceCapabilitiesKHR capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.minImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.minImageExtent.height);
        return actualExtent;
    }
}
uint32_t Engine::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    /*
    memory heaps are distinct memory resources like VRAM or swap space in RAM in case memory spills from VRAM, 
    different types of memory exist within those heaps

    typedef struct VkPhysicalDeviceMemoryProperties {
        uint32_t        memoryTypeCount;
        VkMemoryType    memoryTypes[VK_MAX_MEMORY_TYPES];
        uint32_t        memoryHeapCount;
        VkMemoryHeap    memoryHeaps[VK_MAX_MEMORY_HEAPS];
    } VkPhysicalDeviceMemoryProperties;
    */
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(pDevice, &memProperties); // get memory types and heaps
    for (int i=0; i<memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1<<i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i; // index of the suitable memory type
        }
    }
    throw std::runtime_error("No suitable memory type");
}
void Engine::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
        VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = size; // in bytes
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is used only by transfer queue
    VK_CHECK(vkCreateBuffer(device, &info, nullptr, &buffer));
    
    // buffer is created, now we need to assign memory to it
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    // memRequirements.memoryTypeBits specifies suitable memory types
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));

    // memory offset means offset within the region of memory
    // if the offset is non-zero, then it is required to be divisible by memRequirements.alignment
    VK_CHECK(vkBindBufferMemory(device, buffer, bufferMemory, 0)); // associate memory with the buffer
}
void Engine::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits samples, VkFormat format,
    VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    // how to lay out texels (pixels within the texture) - either row by row or up implementation
    imageInfo.tiling = tiling;
    // VK_IMAGE_LAYOUT_UNDEFINED means it's not usable by the GPU and the very first transition will discard the texels
    // VK_IMAGE_LAYOUT_PREINITIALIZED means it's not usable by the GPU, but the first transition will preserve the texels.
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // VK_IMAGE_USAGE_SAMPLED_BIT so that shaders can read it
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = samples;
    VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &image));

    VkMemoryRequirements memRequirements{};
    vkGetImageMemoryRequirements(device, image, &memRequirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory));
    VK_CHECK(vkBindImageMemory(device, image, imageMemory, 0));
}
void Engine::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandPool transferCmdPool = createCommandPool(queueFamilies.transferFamily.value(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    std::vector<VkCommandBuffer> transferCmdBuffer = createCommandBuffer(transferCmdPool, 1);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(transferCmdBuffer[0], &beginInfo);
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(transferCmdBuffer[0], srcBuffer, dstBuffer, 1, &copyRegion);
    vkEndCommandBuffer(transferCmdBuffer[0]);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = transferCmdBuffer.data();
    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);
    vkQueueSubmit(transferQueue, 1, &submit, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, ~0ull);
    vkDestroyCommandPool(device, transferCmdPool, nullptr);
    vkDestroyFence(device, fence, nullptr);
}
void Engine::transitionImageLayout(VkCommandBuffer cmdBuffer, VkImage image, VkFormat format, VkImageLayout oldLayout, 
    VkImageLayout newLayout, uint32_t mipLevels) {
    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;
    VkAccessFlags srcAccess;
    VkAccessFlags dstAccess;
    VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        // transfer writes don't really have to wait for anything
        srcAccess = 0;
        dstAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstAccess = VK_ACCESS_SHADER_READ_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
        srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        srcAccess = 0;
        dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
        srcStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dstStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        srcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dstAccess = 0;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) 
            aspectFlags |= VK_IMAGE_ASPECT_STENCIL_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        srcAccess = 0;
        dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    // if we need to transfer queue ownership
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspectFlags;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmdBuffer, 
        srcStage, dstStage, 
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier);
}
void Engine::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandPool transferCmdPool = createCommandPool(queueFamilies.transferFamily.value(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    std::vector<VkCommandBuffer> transferCmdBuffer = createCommandBuffer(transferCmdPool, 1);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(transferCmdBuffer[0], &beginInfo);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    // bufferRowLength and bufferImageHeight are used in case of padding, 
    // 0 value is used to indicate that the pixels are tightly packed
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(transferCmdBuffer[0], buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    vkEndCommandBuffer(transferCmdBuffer[0]);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = transferCmdBuffer.data();
    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);
    vkQueueSubmit(transferQueue, 1, &submit, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, ~0ull);
    vkDestroyCommandPool(device, transferCmdPool, nullptr);
    vkDestroyFence(device, fence, nullptr);
}
void Engine::createImageView(VkImageView& imageView, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
    VkImageViewCreateInfo imageViewInfo{};
    imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.image = image;
    imageViewInfo.format = format;
    imageViewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

    // subresourceRange describes what the image's purpose is and which part of the image should be accessed
    imageViewInfo.subresourceRange.aspectMask = aspectFlags; 
    imageViewInfo.subresourceRange.layerCount = 1;
    imageViewInfo.subresourceRange.levelCount = mipLevels;
    imageViewInfo.subresourceRange.layerCount = 1;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.baseMipLevel = 0;

    VK_CHECK(vkCreateImageView(device, &imageViewInfo, nullptr, &imageView));
}
VkFormat Engine::findDepthFormat() {
    return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}
VkFormat Engine::findSupportedFormat(const std::vector<VkFormat>& formats, VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (VkFormat format: formats) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(pDevice, format, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("Cannot find a suitable image format");
}
void Engine::generateMipMaps(VkImage image, VkFormat format, uint32_t texWidth, uint32_t texHeight, uint32_t mipLevels) {
    // check if blit operation is supported for this format
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(pDevice, format, &props);
    if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        throw std::runtime_error("Texture image format doesn't support linear blitting");
    }

    VkCommandPool graphicsCmdPool = createCommandPool(queueFamilies.graphicsFamily.value(), VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
    std::vector<VkCommandBuffer> graphicsCmdBuffer = createCommandBuffer(graphicsCmdPool, 1);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(graphicsCmdBuffer[0], &beginInfo);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (int i=1; i<mipLevels; i++) {
        barrier.subresourceRange.baseMipLevel = i-1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(graphicsCmdBuffer[0], VK_PIPELINE_STAGE_TRANSFER_BIT, 
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        
        VkImageBlit blit{};
        // 3D region that data is copied from
        blit.srcOffsets[0] = {0,0,0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i-1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0,0,0};
        blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth/2 : 1, mipHeight > 1 ? mipHeight/2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;        
        vkCmdBlitImage(graphicsCmdBuffer[0], image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(graphicsCmdBuffer[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, nullptr,
            0, nullptr,
            1, &barrier);
        
        if (mipWidth > 1) mipWidth/=2;
        if (mipHeight > 1) mipHeight/=2;
    }
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(graphicsCmdBuffer[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr,
        0, nullptr,
        1, &barrier);

    vkEndCommandBuffer(graphicsCmdBuffer[0]);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = graphicsCmdBuffer.data();
    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkCreateFence(device, &fenceInfo, nullptr, &fence);
    vkQueueSubmit(graphicsQueue, 1, &submit, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, ~0ull);
    vkDestroyCommandPool(device, graphicsCmdPool, nullptr);
    vkDestroyFence(device, fence, nullptr);
}
VkSampleCountFlagBits Engine::getMaxSampleCount() {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pDevice, &props);
    VkSampleCountFlags counts = props.limits.framebufferDepthSampleCounts & props.limits.framebufferColorSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }
    return VK_SAMPLE_COUNT_1_BIT;
}