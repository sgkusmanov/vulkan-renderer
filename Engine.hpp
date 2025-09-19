#pragma once
#include "config.hpp"

struct Vertex {
    float x, y, z;
    uint8_t nx, ny, nz, nw;
    float tx, ty;
    bool operator==(const Vertex& other) const {
        return x == other.x && y == other.y && z == other.z && 
            nx == other.nx && ny == other.ny && nz == other.nz &&
            tx == other.tx && ty == other.ty;
    }

    // basically describes the vertex buffer
    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0; // index of binding in the binding array, would be a different value if the vertices were
        // packed in different arrays
        binding.stride = sizeof(Vertex); // number of bytes between vertices
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // instance vs vertex
        return binding;
    }
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescription() {
        std::array<VkVertexInputAttributeDescription, 3> attribute{};
        attribute[0].binding = 0;
        attribute[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute[0].location = 0; // location references location directive of the input in VS
        attribute[0].offset = offsetof(Vertex, x);

        attribute[1].binding = 0;
        attribute[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute[1].location = 1;
        attribute[1].offset = offsetof(Vertex, nx);

        attribute[2].binding = 0;
        attribute[2].format = VK_FORMAT_R32G32_SFLOAT;
        attribute[2].location = 2;
        attribute[2].offset = offsetof(Vertex, tx);

        return attribute;
    }
};
struct Meshlet {
	uint32_t vertices[64];
	uint8_t indices[128];
	uint8_t indexCount;
	uint8_t vertexCount;
};
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Meshlet> meshlets;
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(glm::vec3(vertex.x, vertex.y, vertex.z)) ^
                   (hash<glm::vec3>()(glm::vec3(vertex.nx, vertex.ny, vertex.nz)) << 1)) >> 1) ^
                   (hash<glm::vec2>()(glm::vec2(vertex.tx, vertex.ty)) << 1);
        }
    };
}

struct QueueFamilies {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> transferFamily;
    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value() && transferFamily.has_value();
    }
};

struct SurfaceDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class Engine {
public:
    Engine();
    ~Engine();

    void run();

private:
    void loadModel();
    void createMeshlets();
    void createWindow();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapchain();
    void createFence(VkFence& fence);
    void createSemaphore(VkSemaphore& semaphore);
    VkCommandPool createCommandPool(uint32_t queueFamily, VkCommandPoolCreateFlagBits flags);
    std::vector<VkCommandBuffer> createCommandBuffer(VkCommandPool& pool, uint32_t count);
    void createColorAttachment();
    void createDepthAttachment();
    void createDescriptorPool();
    void createDescriptorSet();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createRenderpass();
    void createFramebuffers();
    void recreateSwapchain();
    void cleanupSwapchain();
    void createTextureImage();
    void createTextureImageView();
    void createTextureImageSampler();
    void createVertexBuffer();
    void createMeshletBuffer();
    void createUniformBuffers();
    void updateUniformBuffers();

    VkShaderModule createShaderModule(std::vector<char> code);
    bool checkInstanceLayersSupport(std::vector<const char*> layers);
    bool checkInstanceExtensionsSupport(std::vector<const char*> extensions);
    bool checkDeviceExtensionsSupport(std::vector<const char*> extensions, VkPhysicalDevice device);
    QueueFamilies getQueueFamilies(VkPhysicalDevice device);
    bool isDeviceSuitable(VkPhysicalDevice device);
    SurfaceDetails getSurfaceDetails(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSurfaceFormat(std::vector<VkSurfaceFormatKHR> formats);
    VkPresentModeKHR choosePresentMode(std::vector<VkPresentModeKHR> presentModes);
    VkExtent2D chooseExtent(VkSurfaceCapabilitiesKHR capabilities);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
        VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits samples, VkFormat format, 
        VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    void transitionImageLayout(VkCommandBuffer cmdBuffer, VkImage image, VkFormat format, 
        VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    void createImageView(VkImageView& imageView, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& formats, VkImageTiling tiling, VkFormatFeatureFlags features);
    void generateMipMaps(VkImage image, VkFormat format, uint32_t texWidth, uint32_t texHeight, uint32_t mipLevels);
    VkSampleCountFlagBits getMaxSampleCount();

    Mesh mesh;
    GLFWwindow* window;
    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice pDevice = VK_NULL_HANDLE;
    VkDevice device;
    QueueFamilies queueFamilies;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkQueue transferQueue;
    VkSwapchainKHR swapchain;
    VkExtent2D swapchainExtent;
    VkFormat swapchainFormat;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkPipeline graphicsPipeline;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayout meshDescriptorSetLayout;
    VkPipelineLayout layout;
    VkRenderPass renderpass;
    std::vector<VkFramebuffer> framebuffers;
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    uint32_t mipLevels;
    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;
    VkBuffer vertexBuffer;  
    VkDeviceMemory vertexBufferMemory;
    VkDeviceSize vertexBufferSize;
    VkBuffer meshletBuffer;
    VkDeviceMemory meshletBufferMemory;
    VkDeviceSize meshletBufferSize;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
    const uint32_t MAX_FRAMES_IN_FLIGHT = 3;
    uint32_t currFrame = 0;
    std::string MODEL_OBJ = "../viking_room.obj";
    std::string MODEL_TEX = "../viking_room.png";
};