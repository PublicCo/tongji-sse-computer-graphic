#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 Normal;  // 法线
in vec3 FragPos; // 片段位置

uniform sampler2D texture_diffuse1;
uniform vec3 lightDir;  // 光源方向
uniform vec3 lightColor; // 光源颜色
uniform vec3 viewPos;  // 观察位置

void main()
{
    // 环境光
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir_normalized = normalize(-lightDir);
    float diff = max(dot(norm, lightDir_normalized), 0.0);
    vec3 diffuse = diff * lightColor;

    // 镜面反射
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir_normalized, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * texture(texture_diffuse1, TexCoords).rgb;
    FragColor = vec4(result, 1.0);
}
