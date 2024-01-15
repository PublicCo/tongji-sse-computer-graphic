#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 Normal; // 法线
out vec3 FragPos; // 片段位置

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    TexCoords = aTexCoords;
    FragPos = vec3(model * vec4(aPos, 1.0)); // 计算世界空间中的位置
    Normal = mat3(transpose(inverse(model))) * aNormal; // 计算世界空间中的法线
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
