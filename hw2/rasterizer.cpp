// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector <Eigen::Vector3f> &positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector <Eigen::Vector3i> &indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector <Eigen::Vector3f> &cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

double cross(double v1[2], double v2[2]) {
    return v1[0] * v2[1] - v1[1] * v2[0];
}



static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f *v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
               (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() -
                v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
               (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() -
                v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
               (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() -
                v[1].x() * v[0].y());
    return {c1, c2, c3};
}
static bool insideTriangle(float x, float y, const Vector3f* _v)
{
    // 简单来说就是求解方程P=(1-u-v)A+uB+vC有没有系数全为正的解.
//即求(1,u,v)*(PA,AB,AC)=0,由于是将ABC映射到xy坐标，因此就有两条方程。由于是求=0解，根据叉乘定义可以直接两条线叉乘

    auto [alpha, beta, gamma] = computeBarycentric2D(x, y, _v);

    return alpha >= 0 && beta >= 0 && gamma >= 0;

}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type) {
    auto &buf = pos_buf[pos_buffer.pos_id];
    auto &ind = ind_buf[ind_buffer.ind_id];
    auto &col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto &i: ind) {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        // Homogeneous division
        for (auto &vec: v) {
            vec /= vec.w();
        }
        // Viewport transformation
        for (auto &vert: v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = -vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle_with_sample(t);

    }

    // 超采样之后，更新每个像素
    rasterize_triangle_with_pixel();
}

double getInterpolatedZ(double x, double y, const Triangle &t) {
    auto v = t.toVector4();

    auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    z_interpolated *= w_reciprocal;

    return z_interpolated;
}

Eigen::Vector3f getAverageColor(rst::sample_color_list scl) {
    Eigen::Vector3f color1, color2, color3, color4;
    color1 = scl.s1_color;
    color2 = scl.s2_color;
    color3 = scl.s3_color;
    color4 = scl.s4_color;

    double r = (color1.x() + color2.x() + color3.x() + color4.x()) / 4.0;
    double g = (color1.y() + color2.y() + color3.y() + color4.y()) / 4.0;
    double b = (color1.z() + color2.z() + color3.z() + color4.z()) / 4.0;
    return Eigen::Vector3f(r, g, b);
}

void rst::rasterizer::rasterize_triangle_with_pixel() {
    for (int ind = 0; ind < frame_buf.size(); ind++) {
        int x = ind % width;
        int y = -ind / width + height - 1;
        Eigen::Vector3f pixel_color = frame_buf[ind];

        set_pixel(Eigen::Vector3f(x, y, 0), pixel_color);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle_with_sample(const Triangle &t) {
    auto v = t.toVector4();

    // 先找出bounding box
    int xMin = floor(std::min(v[0].x(), std::min(v[1].x(), v[2].x())));
    int xMax = ceil(std::max(v[0].x(), std::max(v[1].x(), v[2].x())));
    int yMin = floor(std::min(v[0].y(), std::min(v[1].y(), v[2].y())));
    int yMax = ceil(std::max(v[0].y(), std::max(v[1].y(), v[2].y())));
    for (int i = xMin; i < xMax; i++){
        float x = static_cast<float>(i);
        for (int j = yMin; j < yMax; j++){
            float y = static_cast<float>(j);
            Vector2f x4[4] = {{0.25, 0.25},
                              {0.25, 0.75},
                              {0.75, 0.25},
                              {0.75, 0.75}};

            // 这个是根据zbuffer来判断需不需要更新的
            bool depth_test = false;
            for (int k = 0; k < 4; k++)
            {
                if (!insideTriangle(x + x4[k].x(), y + x4[k].y(), t.v))
                    continue;

                // 透视矫正后对重心进行插值
                auto [alpha, beta, gamma] = computeBarycentric2D(x + x4[k].x(), y + x4[k].y(), t.v);
                float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated =
                        alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                int index = get_subsample_index(i, j, k);
                // 根据zbuffer来判断是否更新
                if (z_interpolated < subsample_depth_buf[index])
                {
                    subsample_depth_buf[index] = z_interpolated;
                    subsample_color_buf[index] = t.getColor();
                    depth_test = true;
                }
            }

            // 如果需要更新就去set color
            if (depth_test)
                set_pixel(Vector3f(x, y, 0), get_sample_color(i, j));
        }
    }

}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) {
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) {
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) {
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        // 缓冲区也加一份
        std::fill(subsample_color_buf.begin(), subsample_color_buf.end(), Vector3f{0, 0, 0});
        for (auto &it: sampling_color_buf) {
            it.s1_color = Eigen::Vector3f{0, 0, 0};
            it.s2_color = Eigen::Vector3f{0, 0, 0};
            it.s3_color = Eigen::Vector3f{0, 0, 0};
            it.s4_color = Eigen::Vector3f{0, 0, 0};
        }

    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());

        // 缓冲区也加一份
        std::fill(subsample_depth_buf.begin(), subsample_depth_buf.end(), std::numeric_limits<float>::infinity());
        for (auto &it: sampling_depth_buf) {
            it.s1_depth = std::numeric_limits<float>::infinity();
            it.s2_depth = std::numeric_limits<float>::infinity();
            it.s3_depth = std::numeric_limits<float>::infinity();
            it.s4_depth = std::numeric_limits<float>::infinity();
        }
    }
}

rst::rasterizer::rasterizer(int
                            w, int
                            h) : width(w), height(h) {
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);

    sampling_color_buf.resize(w * h);
    sampling_depth_buf.resize(w * h);

    subsample_color_buf.resize(w * h * 4);
    subsample_depth_buf.resize(w * h * 4);

}

int rst::rasterizer::get_index(int x, int y) {
    return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f &point, const Eigen::Vector3f &color) {
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height - 1 - point.y()) * width + point.x();
    frame_buf[ind] = color;

}

// clang-format on