#include <iostream>
#include <sstream>
#include "model.h"

Model::Model(const std::string& filename) {
    if (!loadModel(filename)) {
        std::cerr << "模型加载失败：" << filename << std::endl;
    }
    loadTextures(filename);
}

bool Model::loadModel(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "文件打开失败：" << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        char i;
        if (line.rfind("v ", 0) == 0) {
            iss.get(i);
            parseVertex(iss);
        } else if (line.rfind("vn ", 0) == 0) {
            iss.get(i);
            iss.get(i);
            parseNormal(iss);
        } else if (line.rfind("vt ", 0) == 0) {
            iss.get(i);
            iss.get(i);
            parseTexture(iss);
        } else if (line.rfind("f ", 0) == 0) {
            iss.get(i);
            if (!parseFace(iss)) {
                std::cerr << "非法的面坐标：" << line << std::endl;
                return false;
            }
        }
    }
    return true;
}

void Model::parseVertex(std::istringstream& iss) {
    vec3 v;
    for (int i = 0; i < 3; ++i) iss >> v[i];
    verts.push_back(v);
}

void Model::parseNormal(std::istringstream& iss) {
    vec3 n;
    for (int i = 0; i < 3; ++i) iss >> n[i];
    norms.push_back(n.normalized());
}

void Model::parseTexture(std::istringstream& iss) {
    vec2 uv;
    for (int i = 0; i < 2; ++i) iss >> uv[i];
    tex_coord.push_back({uv.x, 1 - uv.y});
}

bool Model::parseFace(std::istringstream& iss) {
    int f, t, n;
    int cnt = 0;
    char trash;
    while (iss >> f >> trash >> t >> trash >> n) {
        facet_vrt.push_back(--f);
        facet_tex.push_back(--t);
        facet_nrm.push_back(--n);
        ++cnt;
    }
    return cnt == 3;
}

void Model::loadTextures(const std::string& filename) {
    load_texture(filename, "_diffuse.tga", diffusemap);
    load_texture(filename, "_nm_tangent.tga", normalmap);
    load_texture(filename, "_spec.tga", specularmap);
}
Model::~Model() {
}

int Model::nverts() const {
    return verts.size();
}

int Model::nfaces() const {
    return facet_vrt.size()/3;
}

vec3 Model::vert(const int i) const {
    return verts[i];
}

vec3 Model::vert(const int iface, const int nthvert) const {
    return verts[facet_vrt[iface*3+nthvert]];
}

void Model::load_texture(std::string filename, const std::string suffix, TGAImage &img) {
    size_t dot = filename.find_last_of(".");
    if (dot==std::string::npos) return;
    std::string texfile = filename.substr(0,dot) + suffix;
    std::cerr << "texture file " << texfile << " loading " << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;
}

vec3 Model::normal(const vec2 &uvf) const {
    TGAColor c = normalmap.get(uvf[0]*normalmap.width(), uvf[1]*normalmap.height());
    return vec3{(double)c[2],(double)c[1],(double)c[0]}*2./255. - vec3{1,1,1};
}

vec2 Model::uv(const int iface, const int nthvert) const {
    return tex_coord[facet_tex[iface*3+nthvert]];
}

vec3 Model::normal(const int iface, const int nthvert) const {
    return norms[facet_nrm[iface*3+nthvert]];
}
