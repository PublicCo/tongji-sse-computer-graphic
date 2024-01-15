#include "our_gl.h"

mat<4, 4> ModelView;
mat<4, 4> Viewport;
mat<4, 4> Projection;

//将3D坐标映射到屏幕坐标
void viewport(const int x, const int y, const int w, const int h) {
    // 由于对于模型来说，他们都被装在[-1,1],[-1,1],[-1,1]的正方体里面，现在我们希望将模型映射到[x,x+width][y,y+height][0,d]上，因此执行下列矩阵变换
    // 首先将原点向右上平移：[1,0,0,1][0,1,0,1][0,0,1,1][0,0,0,1],这样就到[0,2]之间了
    //然后将坐标缩小至一半，乘[0.5,0,0,1][0,0.5,0,1][0,0,0.5,1][0,0,0,1]
    //然后对下x，y，z分别扩大w，h，d倍
    // 最后回到[x,x+width][y,y+height][0,d]的位置（加x加y）
    //于是有了magic matrix
    Viewport = {{
                        {w / 2., 0, 0, x + w / 2.},
                        {0, h / 2., 0, y + h / 2.},
                        {0, 0, 1, 0},
                        {0, 0, 0, 1}}};
}

// 定义的投影矩阵，入参为f为焦距
void projection(const double f) {
    // magic matrix，拿相似推的
    Projection = {{{1, 0, 0, 0},
                   {0, -1, 0, 0},
                   {0, 0, 1, 0},
                   {0, 0, -1 / f, 0}}};
}

//定义一个观测矩阵
// 入参：eye：相机位置，center：观察点，up：上方向
// 非常痛苦的数学推导
void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    //首先确定z方向(就是相机的方向)
    //z就是观察者面对目标点的方向（正视目标后目标的背面），因此相减就是向量，normalized就是化成单位向量有利于归一坐标系
    vec3 z = (center - eye).normalized();

    // 然后求左手。up定义了模型的上方向，而x既不是上也不是前，x应该与这两个方向垂直，因此就是叉乘即可
    vec3 x = cross(up, z).normalized();

    //up不一定是垂直z轴，因此只要根据坐标系定义叉乘求出y的坐标
    vec3 y = cross(z, x).normalized();
    //原来的参考系是认为世界坐标系是静止的，现在要转到相机的坐标系是静止的
    //要从世界坐标系变成相机坐标系，左上角的旋转矩阵就是[x,y,z],因此从相机坐标变成世界坐标，就是对原始旋转矩阵求逆，由于原始旋转矩阵正交，因此可以直接等同于求转置
    mat<4, 4> Minv = {{
        {x.x, x.y, x.z, 0},
        {y.x, y.y, y.z, 0},
        {z.x, z.y, z.z, 0},
        {0, 0, 0, 1}}};

    //从世界坐标系看，相机位于eye代表的方位，因此对于相机来说，世界坐标系就在-eye的位置
    // 注意最后需要归一到四元组的矩阵里（即升一维以归一旋转和平移）
    mat<4, 4> Tr = {{
                            {1, 0, 0, -eye.x},
                            {0, 1, 0, -eye.y},
                            {0, 0, 1, -eye.z},
                            {0, 0, 0, 1}}};
    ModelView = Minv * Tr;
}

// 求解重心坐标
// 简单来说就是求解方程P=(1-u-v)A+uB+vC有没有系数全为正的解.
//即求(1,u,v)*(PA,AB,AC)=0,由于是将ABC映射到xy坐标，因此就有两条方程。由于是求=0解，根据叉乘定义可以直接两条线叉乘
vec3 barycentric(const vec2 tri[3], const vec2 P) {
    vec2 A = tri[0];
    vec2 B = tri[1];
    vec2 C = tri[2];
    vec3 S1 = {B.y - A.y, C.y - A.y, A.y - P.y};
    vec3 S0 = {B.x - A.x, C.x - A.x, A.x - P.x};
    vec3 u = cross(S0, S1);
    // 如果三角形接近一条直线直接丢掉
    if (std::abs(u[2]) < 1e-2) {
        return vec3{-1, 1, 1};
    }
    return vec3{1.0 - (u.x + u.y) / u.z, u.x / u.z, u.y / u.z};
}

void triangle(const vec4 clip_verts[3], IShader &shader, TGAImage &image, std::vector<double> &zbuffer) {
    // 将点坐标转换到相机坐标
    vec4 clipSpacePoints[3] = {Viewport * clip_verts[0], Viewport * clip_verts[1], Viewport * clip_verts[2]};

    // 将裁剪空间坐标转换为屏幕空间坐标
    vec2 screenSpacePoints[3] = {proj<2>(clipSpacePoints[0] / clipSpacePoints[0][3]),
                                 proj<2>(clipSpacePoints[1] / clipSpacePoints[1][3]),
                                 proj<2>(clipSpacePoints[2] / clipSpacePoints[2][3])};
    // 计算三角形的bounding box
    int bboxmin[2] = {image.width() - 1, image.height() - 1};
    int bboxmax[2] = {0, 0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++) {
            bboxmin[j] = std::min(bboxmin[j], static_cast<int>(screenSpacePoints[i][j]));
            bboxmax[j] = std::max(bboxmax[j], static_cast<int>(screenSpacePoints[i][j]));
        }
#pragma omp parallel for
    // 检查bounding box里哪些点是可以填的
    for (int x = std::max(bboxmin[0], 0); x <= std::min(bboxmax[0], image.width() - 1); x++) {
        for (int y = std::max(bboxmin[1], 0); y <= std::min(bboxmax[1], image.height() - 1); y++) {
            // 计算像素点在三角形中的重心坐标
            vec3 barycentricCoord = barycentric(screenSpacePoints,
                                                vec2{static_cast<double>(x), static_cast<double>(y)});

            // 插值计算深度
            vec3 bcClip = {barycentricCoord.x / clipSpacePoints[0][3], barycentricCoord.y / clipSpacePoints[1][3],
                           barycentricCoord.z / clipSpacePoints[2][3]};
            // 求加权平均
            bcClip = bcClip / (bcClip.x + bcClip.y + bcClip.z);

            // 计算片元深度
            double fragmentDepth = vec3{clip_verts[0][2], clip_verts[1][2], clip_verts[2][2]} * bcClip;

            // 检查像素是否在三角形内部and深度测试
            if (barycentricCoord.x < 0 || barycentricCoord.y < 0 || barycentricCoord.z < 0 ||
                fragmentDepth > zbuffer[x + y * image.width()])
                continue;

            TGAColor color;

            if (shader.fragment(bcClip, color)) continue;

            // 更新深度缓冲区并设置像素颜色
            zbuffer[x + y * image.width()] = fragmentDepth;
            image.set(x, y, color);
        }
    }
}

