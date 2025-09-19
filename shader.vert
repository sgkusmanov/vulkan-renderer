#version 450
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types: require

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

struct Vertex {
	float vx, vy, vz;
	uint8_t nx, ny, nz, nw;
	float tu, tv;
};
layout(set = 1, binding = 0) readonly buffer Vertices {
    Vertex vertices[];
};

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    Vertex vertex = vertices[gl_VertexIndex];
    vec3 inPosition = vec3(vertex.vx, vertex.vy, vertex.vz);
    vec3 inNormal = vec3(vertex.nx, vertex.ny, vertex.nz) / 255.0 * 2.0 - 1.0;
    vec2 inTexCoord = vec2(vertex.tu, vertex.tv);
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
}
