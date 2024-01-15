//
// Created by goksu on 4/6/19.
//

#pragma once

#include <eigen3/Eigen/Eigen>
#include <algorithm>
#include "global.hpp"
#include "Triangle.hpp"
using namespace Eigen;

namespace rst
{
    enum class Buffers
    {
        Color = 1,
        Depth = 2
    };

    inline Buffers operator|(Buffers a, Buffers b)
    {
        return Buffers((int)a | (int)b);
    }

    inline Buffers operator&(Buffers a, Buffers b)
    {
        return Buffers((int)a & (int)b);
    }

    enum class Primitive
    {
        Line,
        Triangle
    };

    /*
     * For the curious : The draw function takes two buffer id's as its arguments. These two structs
     * make sure that if you mix up with their orders, the compiler won't compile it.
     * Aka : Type safety
     * */
    struct pos_buf_id
    {
        int pos_id = 0;
    };

    struct ind_buf_id
    {
        int ind_id = 0;
    };

    struct col_buf_id
    {
        int col_id = 0;
    };

    //////////////////////////////////////
    // my definition: 1 pixel of 4 sample
    struct sample_depth_list {
        double s1_depth, s2_depth, s3_depth, s4_depth;   
    };

    struct sample_color_list {
        Vector3f s1_color, s2_color, s3_color, s4_color;
    };
    //////////////////////////////////////

    class rasterizer
    {
    public:
        rasterizer(int w, int h);
        pos_buf_id load_positions(const std::vector<Eigen::Vector3f>& positions);
        ind_buf_id load_indices(const std::vector<Eigen::Vector3i>& indices);
        col_buf_id load_colors(const std::vector<Eigen::Vector3f>& colors);

        void set_model(const Eigen::Matrix4f& m);
        void set_view(const Eigen::Matrix4f& v);
        void set_projection(const Eigen::Matrix4f& p);

        void set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color);

        void clear(Buffers buff);

        void draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type);

        std::vector<Eigen::Vector3f>& frame_buffer() { return frame_buf; }

    private:
        void draw_line(Eigen::Vector3f begin, Eigen::Vector3f end);

        void rasterize_triangle_with_sample(const Triangle& t);

        void rasterize_triangle_with_pixel();

        // 第y行第x列第k个子像素
        int get_subsample_index(int x, int y, int k) const{
            return (height - 1- y) * width * 4 + (width - 1 - x) * 4 + k;
        }

        // 根据buffer中的颜色记录来求出实际渲染的颜色（最终求平均）
        Vector3f get_sample_color(int x, int y) const
        {
            int index = get_subsample_index(x, y, 0);
            Vector3f sum{0, 0, 0};
            for (int i = 0; i < 4; i++)
                sum += subsample_color_buf[index + i];

            return sum / 4.0f;
        }
        // VERTEX SHADER -> MVP -> Clipping -> /.W -> VIEWPORT -> DRAWLINE/DRAWTRI -> FRAGSHADER

    private:
        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        std::map<int, std::vector<Eigen::Vector3f>> pos_buf;
        std::map<int, std::vector<Eigen::Vector3i>> ind_buf;
        std::map<int, std::vector<Eigen::Vector3f>> col_buf;

        std::vector<Eigen::Vector3f> frame_buf;

        std::vector<float> depth_buf;

        // MSAA中记录颜色与深度buffer，最后渲染用
        std::vector<float> subsample_depth_buf;
        std::vector<Eigen::Vector3f> subsample_color_buf;

        // add new property
        std::vector<sample_depth_list> sampling_depth_buf;
        std::vector<sample_color_list> sampling_color_buf;

        int get_index(int x, int y);

        int width, height;

        int next_id = 0;
        int get_next_id() { return next_id++; }
    };
}
