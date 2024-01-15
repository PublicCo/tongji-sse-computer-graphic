#include <limits>
#include "model.h"
#include "our_gl.h"
#include <chrono>

const bool is_softshadow = 1;
const int width = 800; // 图像大小
const int height = 800;
const vec3 envlight{0, 1, 0}; // 光线方向
const float envlightstrenghth = 0.2;
// 定义点光源的位置
const vec3 light_pos{-1, 1, -1}; // 可以根据需要调整位置

// 定义光源衰减参数
const float constant_attenuation = 1.0;  // 常数衰减
const float linear_attenuation = 0.09;   // 线性衰减
const float quadratic_attenuation = 0.032; // 二次衰减

const vec3 eye{2, 0.5, 0}; // 相机位置
const vec3 center{0, 0, 0}; // 相机方向
const vec3 up{0, 1, 0}; // 相机向上的方向

extern mat<4, 4> ModelView;
extern mat<4, 4> Projection;

static std::vector<Model> models;

struct Shader : IShader {
    const Model &model;
    vec3 uniform_l;       // 视图坐标中的光线方向
    mat<2, 3> varying_uv;  // 由顶点着色器写入、片元着色器读取的三角形UV坐标
    mat<3, 3> varying_nrm; // 由片元着色器插值的每个顶点的法向量
    mat<3, 3> view_tri;    // 视图坐标中的三角形

    Shader(const Model &m) : model(m) {
        uniform_l = proj<3>((ModelView * embed<4>(envlight, 0.))).normalized(); // 将光矢量转换为视图坐标
        uniform_l=uniform_l*envlightstrenghth;
    }

    // 面序号，顶点序号，顶点位置
    virtual void vertex(const int iface, const int nthvert, vec4 &gl_Position) {
        //将第nthvert个顶点的纹理坐标设置为模型给出的纹理坐标
        varying_uv.set_col(nthvert, model.uv(iface, nthvert));
        //为顶点设置法线向量。注意，法线向量也需要经过变换从世界坐标转到当前坐标
        varying_nrm.set_col(nthvert,
                            proj<3>((ModelView).invert_transpose() * embed<4>(model.normal(iface, nthvert), 0.)));
        // 计算顶点位置，准备点光源相关的计算
        gl_Position = ModelView * embed<4>(model.vert(iface, nthvert));
        //给三角形设置好材质
        view_tri.set_col(nthvert, proj<3>(gl_Position));
        // 透视矫正
        gl_Position = Projection * gl_Position;
    }

    virtual bool fragment(const vec3 bar, TGAColor &gl_FragColor) {
        vec3 fragPos = interpolate(view_tri, bar); // 计算片元位置

        vec3 light_dir = (light_pos - fragPos).normalized(); // 计算光源方向


        float distance = (light_pos - fragPos).norm(); // 计算光源到片元的距离
        float attenuation = 1.0 / (constant_attenuation + linear_attenuation * distance +
                                   quadratic_attenuation * distance * distance);

        vec3 bn = interpolateNormal(bar); // 插值法线
        vec2 uv = interpolateUV(bar); // 插值UV坐标

        double envl=0.05;
        mat<3, 3> tangentSpaceMatrix = calculateTangentSpaceMatrix(bn); // 计算切线空间矩阵

        vec3 n = transformNormalToTangentSpace(uv, tangentSpaceMatrix); // 将纹理的法线转换到切线空间
        double diff = calculateDiffuseIntensity(bn, light_dir) * attenuation;
        double spec = calculateSpecularIntensity(bn, uv, light_dir) * attenuation;
        TGAColor c = sample2D(model.diffuse(), uv);
        applyLightingEffect(c, diff, spec,envl, gl_FragColor); // 应用光照效果

        if (is_softshadow) {
            float shadow = calculateSoftShadow(fragPos);
            shadow = std::min(shadow,0.2f);
            gl_FragColor = mix(gl_FragColor, shadow); // 根据阴影比例混合颜色

        } else {
            bool inShadow = isShadowed(fragPos, light_dir, models);
            if (inShadow) {
                diff *= 0.6; // 如果在阴影中，减少漫反射强度
            }
            applyLightingEffect(c, diff, spec,envl, gl_FragColor); // 应用光照效果
        }
        return false; // 像素不被丢弃
    }

    virtual float calculateSoftShadow(const vec3 &fragPos, const int numSamples = 16, const float radius = 0.05) {
        float shadow = 0.0;
        for (int i = 0; i < numSamples; ++i) {
            vec3 randomLightPos = light_pos + randomOffset(radius);
            vec3 lightDir = (randomLightPos - fragPos).normalized();
            float distanceToLight = (randomLightPos - fragPos).norm();
            bool inShadow = false;

            for (const auto &model: models) {
                for (int j = 0; j < model.nfaces(); j++) {
                    vec3 barycentricCoords;
                    vec3 v0 = model.vert(j, 0);
                    vec3 v1 = model.vert(j, 1);
                    vec3 v2 = model.vert(j, 2);

                    if (rayTriangleIntersect(fragPos + lightDir * 0.01, lightDir, v0, v1, v2, barycentricCoords)) {
                        float distanceToTriangle = (fragPos + barycentricCoords * distanceToLight - fragPos).norm();
                        if (distanceToTriangle < distanceToLight) {
                            inShadow = true;
                            break;
                        }
                    }
                }
                if (inShadow) break;
            }

            if (!inShadow) {
                shadow += 1.0;
            }
        }

        return 1-(shadow / static_cast<float>(numSamples));
    }


private:

    bool
    isShadowed(const vec3 &fragPos, const vec3 &lightDir, const std::vector<Model> &models, float shadowBias = 0.1) {
        if (is_softshadow) {
            vec3 currentPos = fragPos + lightDir * shadowBias; // 添加偏移以避免自阴影
            for (float t = shadowBias; t < 1.0; t += 0.05) { // 增量步进检测阴影
                currentPos = currentPos + lightDir * 0.05; // 沿光线方向移动
                for (const auto &model: models) {
                    // 遍历场景中的每个模型
                    for (int i = 0; i < model.nfaces(); i++) {
                        vec3 barycentricCoords;
                        vec3 v0 = model.vert(i, 0);
                        vec3 v1 = model.vert(i, 1);
                        vec3 v2 = model.vert(i, 2);
                        if (rayTriangleIntersect(currentPos, lightDir, v0, v1, v2, barycentricCoords)) {
                            return true; // 发现阻挡光线的物体
                        }
                    }
                }
            }
            return false; // 没有物体阻挡光线
        } else {
            vec3 currentPos = fragPos + lightDir * shadowBias; // 添加偏移以避免自阴影
            for (const auto &model: models) {
                // 遍历场景中的每个模型
                for (int i = 0; i < model.nfaces(); i++) {
                    vec3 barycentricCoords;
                    vec3 v0 = model.vert(i, 0);
                    vec3 v1 = model.vert(i, 1);
                    vec3 v2 = model.vert(i, 2);
                    if (rayTriangleIntersect(currentPos, lightDir, v0, v1, v2, barycentricCoords)) {
                        return true; // 发现阻挡光线的物体
                    }
                }
            }
            return false; // 没有物体阻挡光线
        }
    }


    TGAColor mix(TGAColor &color1,  float factor) {
        TGAColor result;

        result[0] = static_cast<std::uint8_t>(color1[0] * (1 - factor));// 蓝色分量
        result[1] = static_cast<std::uint8_t>(color1[1] * (1 - factor)); // 绿色分量
        result[2] = static_cast<std::uint8_t>(color1[2] * (1 - factor)); // 红色分量
        result[3] = 0; // Alpha通道，设置为不透明
        return result;
    }

    vec3 interpolateNormal(const vec3 &bar) {
        return (varying_nrm * bar).normalized();
    }

    vec2 interpolateUV(const vec3 &bar) {
        return varying_uv * bar;
    }

    mat<3, 3> calculateTangentSpaceMatrix(const vec3 &bn) {
        mat<3, 3> AI = mat<3, 3>{{view_tri.col(1) - view_tri.col(0), view_tri.col(2) - view_tri.col(0), bn}}.invert();
        vec3 i = AI * vec3{varying_uv[0][1] - varying_uv[0][0], varying_uv[0][2] - varying_uv[0][0], 0};
        vec3 j = AI * vec3{varying_uv[1][1] - varying_uv[1][0], varying_uv[1][2] - varying_uv[1][0], 0};
        return mat<3, 3>{{i.normalized(), j.normalized(), bn}}.transpose();
    }

    vec3 transformNormalToTangentSpace(const vec2 &uv, const mat<3, 3> &B) {
        return (B * model.normal(uv)).normalized();
    }

    double calculateDiffuseIntensity(const vec3 &n, const vec3 &lightDir) {
        float num = 1.5;
        return std::max(0., n * lightDir * num);
    }

    double calculateSpecularIntensity(const vec3 &n, const vec2 &uv, const vec3 &lightDir) {
        vec2 uv_non_const = uv; 
        vec3 r = (n * (n * lightDir) * 2 - lightDir).normalized();
        return std::pow(std::max(-r.z, 0.), 5 + sample2D(model.specular(), uv_non_const)[0]);
    }


    uint8_t getColorValue(const TGAColor &color, int index) {
        return color.bgra[index]; // 直接访问数组，避免使用operator[]
    }

    void applyLightingEffect(const TGAColor &c, double diff, double spec,double envl, TGAColor &gl_FragColor) {
        for (int i: {0, 1, 2}) {
            uint8_t colorValue = getColorValue(c, i); // 使用辅助函数获取颜色值
            gl_FragColor[i] = std::min<int>(10 + colorValue * (diff + spec+envl), 255);
        }
    }

    vec3 interpolate(const mat<3, 3> &tri, const vec3 &bar) {
        return vec3(tri[0][0] * bar.x + tri[1][0] * bar.y + tri[2][0] * bar.z,
                    tri[0][1] * bar.x + tri[1][1] * bar.y + tri[2][1] * bar.z,
                    tri[0][2] * bar.x + tri[1][2] * bar.y + tri[2][2] * bar.z);
    }

    vec3 randomOffset(float radius = 0.1) {
        float randX = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
        float randY = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
        float randZ = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
        return vec3(randX, randY, randZ).normalized() * radius;
    }


    bool rayTriangleIntersect(
            const vec3 &orig, const vec3 &dir,
            const vec3 &v0, const vec3 &v1, const vec3 &v2,
            vec3 &barycentricCoords) {

        const float EPSILON = 1e-7;
        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;
        vec3 pvec = cross(dir, edge2);
        float det = edge1 * pvec;

        // 如果行列式接近0，则射线与三角形平面平行
        if (fabs(det) < EPSILON) return false;

        float invDet = 1.0 / det;
        vec3 tvec = orig - v0;
        barycentricCoords[1] = (tvec * pvec) * invDet;
        if (barycentricCoords[1] < 0 || barycentricCoords[1] > 1) return false;

        vec3 qvec = cross(tvec, edge1);
        barycentricCoords[2] = (dir * qvec) * invDet;
        if (barycentricCoords[2] < 0 || barycentricCoords[1] + barycentricCoords[2] > 1) return false;

        barycentricCoords[0] = 1.0 - barycentricCoords[1] - barycentricCoords[2];
        return true;
    }


};

int main(int argc, char **argv) {
    if (2 > argc) {
        std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
        return 1;
    }
    TGAImage framebuffer(width, height, TGAImage::RGB); // the output image

    // 处理世界系到相机变换
    lookat(eye, center, up);

    //到图片位置变换
    viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);

    //归一化平面
    projection((eye - center).norm());
    std::vector<double> zbuffer(width * height, std::numeric_limits<double>::max());
    for (int m = 1; m < argc; m++) {
        Model model(argv[m]);
        models.push_back(model);
    }
    auto start = std::chrono::high_resolution_clock::now(); // 记录开始时间
    for (Model model: models) {
        Shader shader(model);
        // 遍历三角形
        for (int i = 0; i < model.nfaces(); i++) {
            vec4 clip_vert[3]; //顶点坐标
            for (int j: {0, 1, 2})
                shader.vertex(i, j, clip_vert[j]); //对三角形的三个顶点进行顶点着色器渲染
            triangle(clip_vert, shader, framebuffer, zbuffer);
        }
        std::cerr << "finish a model" << std::endl;
    }
    std::cerr << "finish" << std::endl;
    auto end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double, std::milli> duration = end - start; // 计算运行时间
    std::cerr << "Total time taken: " << duration.count() << " ms" << std::endl; // 打印运行时间
    framebuffer.write_tga_file(
            "C:\\Users\\leon\\Desktop\\ComputerGraphic\\finalhomework\\tinyrenderer\\framebuffer.tga");
    return 0;
}
