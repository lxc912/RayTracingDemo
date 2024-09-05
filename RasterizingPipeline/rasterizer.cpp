#include "rasterizer.hpp"

#include <math.h>

#include <algorithm>
#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

#include "Shader.hpp"
#include "Triangle.hpp"

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f> &normals) {
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}

void rst::rasterizer::post_process_buffer() {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = get_index(x, y);
            for (int i = 0; i < 4; i++) {
                frame_buf[index] += ssaa_frame_buf[4 * index + i];
            }
            frame_buf[index] /= 4;
        }
    }
}

// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end) {
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

    dx = x2 - x1;
    dy = y2 - y1;
    dx1 = fabs(dx);
    dy1 = fabs(dy);
    px = 2 * dy1 - dx1;
    py = 2 * dx1 - dy1;

    if (dy1 <= dx1) {
        if (dx >= 0) {
            x = x1;
            y = y1;
            xe = x2;
        } else {
            x = x2;
            y = y2;
            xe = x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; x < xe; i++) {
            x = x + 1;
            if (px < 0) {
                px = px + 2 * dy1;
            } else {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
                    y = y + 1;
                } else {
                    y = y - 1;
                }
                px = px + 2 * (dy1 - dx1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    } else {
        if (dy >= 0) {
            x = x1;
            y = y1;
            ye = y2;
        } else {
            x = x2;
            y = y2;
            ye = y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; y < ye; i++) {
            y = y + 1;
            if (py <= 0) {
                py = py + 2 * dx1;
            } else {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) {
                    x = x + 1;
                } else {
                    x = x - 1;
                }
                py = py + 2 * (dx1 - dy1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector4f *_v) {
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f p(x, y, 1.);
    Vector3f f0, f1, f2;
    f0 = (p - v[0]).cross(v[1] - v[0]);
    f1 = (p - v[1]).cross(v[2] - v[1]);
    f2 = (p - v[2]).cross(v[0] - v[2]);
    if (f0.dot(f1) > 0 && f1.dot(f2) > 0)
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v) {
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
Eigen::Vector3f CameraToLightSpace(const Eigen::Vector3f& cameraPos, const Eigen::Matrix4f& cameraViewMatrix, const Eigen::Matrix4f& lightViewMatrix) {
    Eigen::Vector4f cameraPosHomogeneous(cameraPos[0], cameraPos[1], cameraPos[2], 1.0f);
    
    // from camera position to world postion
    Eigen::Matrix4f cameraViewMatrixInverse = cameraViewMatrix.inverse();
    Eigen::Vector4f worldPos = cameraViewMatrixInverse * cameraPosHomogeneous;
    
    // from world space to camera space
    Eigen::Vector4f lightSpacePos = lightViewMatrix * worldPos;
    
    // generate 3d coordinate
    Eigen::Vector3f lightSpaceCoord(lightSpacePos[0] / lightSpacePos[3], lightSpacePos[1] / lightSpacePos[3], lightSpacePos[2] / lightSpacePos[3]);
    
    return lightSpaceCoord;
}

Eigen::Matrix4f lookAt(Eigen::Vector3f eye_pos, Eigen::Vector3f target,
                        Eigen::Vector3f up = Eigen::Vector3f(0, 1, 0)) {
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Vector3f z = (eye_pos - target).normalized();
    Eigen::Vector3f x = up.cross(z).normalized();
    Eigen::Vector3f y = z.cross(x).normalized();

    Eigen::Matrix4f rotate;
    rotate << x.x(), x.y(), x.z(), 0, y.x(), y.y(), y.z(), 0, z.x(), z.y(), z.z(), 0, 0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 0, 1, 0, -eye_pos[1], 0, 0, 1, -eye_pos[2], 0, 0, 0, 1;

    view = rotate * translate * view;
    return view;
}

// TODO: Task1 Implement this function
void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type, bool culling, bool anti_aliasing) {
    //store vertice of boj buf[i] the ith vertice
    auto &buf = pos_buf[pos_buffer.pos_id];
    //store surface of obj ind[i]: a triangle with 3 index of vertice
    auto &ind = ind_buf[ind_buffer.ind_id];
    //the rgb color of each vertice
    auto &col = col_buf[col_buffer.col_id];
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto &i : ind) {
        Triangle t;
        //generate triangle with coordinate
        std::array<Eigen::Vector4f, 3> mm{view * model * to_vec4(buf[i[0]], 1.0f),
                                          view * model * to_vec4(buf[i[1]], 1.0f),
                                          view * model * to_vec4(buf[i[2]], 1.0f)};

        std::array<Eigen::Vector3f, 3> viewspace_pos;
        //transform each element of the mm array and store the result in the viewspace_pos array
        //viewspace_pos is used to store the coordinate of a triangle surface in viewspace
        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v) { return v.template head<3>(); });
        
        // TODO: Task1 Enable back face culling
        if (culling) {
            Eigen::Vector3f v01 = viewspace_pos[1] - viewspace_pos[0];
            Eigen::Vector3f v02 = viewspace_pos[2] - viewspace_pos[0];
            Eigen::Vector3f normal = v01.cross(v02); // Calculate triangle's normal

            // Determine if the triangle is back-facing
            if (normal.z() < 0) { // If normal.z() is negative, the triangle is facing away from the camera
                continue; // Skip this triangle
            }
        }

        Eigen::Vector4f v[] = {mvp * to_vec4(buf[i[0]], 1.0f), mvp * to_vec4(buf[i[1]], 1.0f),
                               mvp * to_vec4(buf[i[2]], 1.0f)};

        // Homogeneous division
        for (auto &vec : v) {
            vec /= vec.w();
        }
        // Viewport transformation
        for (auto &vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            t.setVertex(i, v[i]);
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);
/*************** update below *****************/
        rasterize_triangle(t, anti_aliasing);
    }
    if (anti_aliasing){
        post_process_buffer();
    }
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList, bool culling, rst::Shading shading, bool shadow) {
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;

    std::vector<light> viewspace_lights;
    for (const auto &l : lights) {
        light view_space_light;
        view_space_light.position = (view * to_vec4(l.position, 1.0f)).head<3>();
        view_space_light.intensity = l.intensity;
        viewspace_lights.push_back(view_space_light);
    }

    for (const auto &t : TriangleList) {
        Triangle newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{(view * model * t->v[0]), (view * model * t->v[1]), (view * model * t->v[2])};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v) { return v.template head<3>(); });

        // TODO: Task1 Enable back face culling
        if (culling) {
            Eigen::Vector3f v01 = viewspace_pos[1] - viewspace_pos[0];
            Eigen::Vector3f v02 = viewspace_pos[2] - viewspace_pos[0];
            Eigen::Vector3f normal = v01.cross(v02); // Calculate triangle's normal

            // Determine if the triangle is back-facing
            if (normal.z() < 0) { // If normal.z() is negative, the triangle is facing away from the camera
                continue; // Skip this triangle
            }
        }

        Eigen::Vector4f v[] = {mvp * t->v[0], mvp * t->v[1], mvp * t->v[2]};
        // Homogeneous division
        for (auto &vec : v) {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {inv_trans * to_vec4(t->normal[0], 0.0f), inv_trans * to_vec4(t->normal[1], 0.0f),
                               inv_trans * to_vec4(t->normal[2], 0.0f)};

        // Viewport transformation
        for (auto &vert : v) {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i) {
            // screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i) {
            // view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148, 121.0, 92.0);
        newtri.setColor(1, 148, 121.0, 92.0);
        newtri.setColor(2, 148, 121.0, 92.0);

        // Also pass view space vertice position
        rasterize_triangle(newtri, viewspace_pos, viewspace_lights, shading, shadow);
    }
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f &vert1,
                                   const Eigen::Vector3f &vert2, const Eigen::Vector3f &vert3, float weight) {
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f &vert1,
                                   const Eigen::Vector2f &vert2, const Eigen::Vector2f &vert3, float weight) {
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

// TODO: Task1 Implement this function
void rst::rasterizer::rasterize_triangle(const Triangle &t, bool anti_aliasing) {
    auto v = t.toVector4();
    //1. Find out the bounding box of the current triangle.
    int min_x = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    int max_x = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    int min_y = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    int max_y = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

    // Clip the bounding box to the screen dimensions
    min_x = std::max(0, min_x);
    max_x = std::min(width - 1, max_x);
    min_y = std::max(0, min_y);
    max_y = std::min(height - 1, max_y);
    // t.v is a homogenous coordinate
    // 2. Iterate through the pixel and find if the current pixel is inside the triangle
    // Subpixel sampling if do anti-aliasing
    for (int x = min_x; x <= max_x; ++x) {
        for (int y = min_y; y <= max_y; ++y) {
            int index = get_index(x, y);
            int subIndex = 0;
            // Sample the center of the pixel
            float sample_x = x + 0.5f;
            float sample_y = y + 0.5f;
            // Check if the current pixel is inside the triangle.
            if(anti_aliasing){//anti_aliasing sample
               for(int i = 0; i <= 1; i++){
                    for (int j = 0; j <= 1; j++)
                    {
                        float subPixel_x = x+0.25+0.5*i;
                        float subPixel_y = y+0.25+0.5*j;
                        if(insideTriangle((x+0.25+0.5*i), (y+0.25+0.5*j), t.v)){
                            auto [alpha, beta, gamma] = computeBarycentric2D(subPixel_x, subPixel_y, t.v);
                            float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w()
                             + gamma * v[2].z() / v[2].w();
                            z_interpolated *= w_reciprocal;
                            if (z_interpolated <= ssaa_depth_buf[4 * index + subIndex]) {
                                // Update depth buffer
                                ssaa_depth_buf[4 * index + subIndex] = z_interpolated;
                                // Set pixel color t.color: the rbg of the first triangle vertic 
                                //eg:t.color[0][0] is the red intensity of the first triangle
                                Vector3f interpolated_rgb = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2],1)*255;
                                ssaa_frame_buf[4 * index + subIndex] = interpolated_rgb;
                            }                            
                        }
                    subIndex++;
                    }                   
                }
            }
            else{//direct sampling
                if (insideTriangle(sample_x, sample_y, t.v))
                {    
                auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                //z-buffer depth iteration
                if (z_interpolated < depth_buf[get_index(x, y)]) {
                    // Update depth buffer
                    depth_buf[get_index(x, y)] = z_interpolated;
                    // Set pixel color t.color: the rbg of the first triangle vertic eg:t.color[0][0] is the red intensity of the first triangle
                    Vector3f interpolated_rgb = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2],1)*255;
                    set_pixel(Vector2i(x, y), interpolated_rgb);
                }
                }
                
            }
        }
    }
}

// TODO: Implement this function to test if point (x, y) is inside the triangle.
/**
 * @param _v: a pointer to the array of three triangle vertices. The coordiante of
 * vertices is homogenous.
 */

// TODO: Task2 Implement this function
void rst::rasterizer::rasterize_triangle(const Triangle &t, const std::array<Eigen::Vector3f, 3> &view_pos,
                                         const std::vector<light> &view_lights, rst::Shading shading, bool shadow) {
    auto v = t.toVector4();
    //1. Find out the bounding box of the current triangle.
    int min_x = std::min(v[0].x(), std::min(v[1].x(), v[2].x()));
    int max_x = std::max(v[0].x(), std::max(v[1].x(), v[2].x()));
    int min_y = std::min(v[0].y(), std::min(v[1].y(), v[2].y()));
    int max_y = std::max(v[0].y(), std::max(v[1].y(), v[2].y()));

    // Clip the bounding box to the screen dimensions
    min_x = std::max(0, min_x);
    max_x = std::min(width - 1, max_x);
    min_y = std::max(0, min_y);
    max_y = std::min(height - 1, max_y);
    if (shading == rst::Shading::Flat) {
        // Compute color at the center of the triangle
        auto center_color = (t.color[0] + t.color[1] + t.color[2]) / 3.0f *255;
        auto center_texture = (t.tex_coords[0] + t.tex_coords[1] + t.tex_coords[2]) / 3.0f;
        Vector3f normal_vector = (t.normal[0] + t.normal[1] + t.normal[2]) / 3.0f;
        fragment_shader_payload payload(center_color, normal_vector, center_texture, view_lights, texture ? &*texture : nullptr);
        payload.view_pos = (view_pos[0]+view_pos[1]+view_pos[2])/3.0f;
        auto pixel_color = fragment_shader(payload);
        for (int x = min_x; x <= max_x; x++) {
            for (int y = min_y; y <= max_y; y++) {
                if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                    int cur_index = get_index(x, y);
                    float sample_x = x + 0.5f;
                    float sample_y = y + 0.5f;
                    auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                        //before depth interplotation, The center of gravity coordinates are corrected by perspective
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
    
                    if (z_interpolated < depth_buf[cur_index]) {
                        depth_buf[cur_index] = z_interpolated;
                        Vector2i vertex;
                        vertex << x, y;
                        set_pixel(vertex, pixel_color);
                    }
                }
                }
        }
        if (shadow) {
            payload.view_pos = (view_pos[0]+view_pos[1]+view_pos[2])/3.0f;
            auto pixel_color = fragment_shader(payload);
            for (int x = min_x; x <= max_x; x++) {
                for (int y = min_y; y <= max_y; y++) {
            
                    if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                        int cur_index = get_index(x, y);
                        float sample_x = x + 0.5f;
                        float sample_y = y + 0.5f;
                        auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                            //before depth interplotation, The center of gravity coordinates are corrected by perspective
                        float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                        float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                        z_interpolated *= w_reciprocal;
        
                        if (z_interpolated < depth_buf[cur_index]) {
                            depth_buf[cur_index] = z_interpolated;
                            Vector2i vertex;
                            vertex << x, y;
                            set_pixel(vertex, pixel_color);
                        }
                    }
                    }
            }
            }
    } else if (shading == rst::Shading::Gouraud) {
        Eigen::Vector3f interpolated_color[3];
        Eigen::Vector3f interpolated_normal[3];
        Eigen::Vector2f interpolated_texcoords[3];
        Eigen::Vector3f interpolated_shadingcoords[3];
        for (int i = 0; i < 3; ++i) {
            interpolated_color[i] = t.color[i]*255;
            interpolated_normal[i] = t.normal[i];
            interpolated_texcoords[i] = t.tex_coords[i];
            interpolated_shadingcoords[i] = view_pos[i];
        }
        fragment_shader_payload payload[3];
        for (int i = 0; i < 3; ++i) {
            payload[i] = fragment_shader_payload(interpolated_color[i], interpolated_normal[i], interpolated_texcoords[i], view_lights, texture ? &*texture : nullptr);
            payload[i].view_pos = interpolated_shadingcoords[i];
        }
        Eigen::Vector3f pixel_color[3];
        for (int i = 0; i < 3; ++i) {
            pixel_color[i] = fragment_shader(payload[i]);
        }

        for (int x = min_x; x <= max_x; ++x) {
            for (int y = min_y; y <= max_y; ++y) {
                if (insideTriangle(x + 0.5f, y + 0.5f, t.v)) {
                    int cur_index= get_index(x,y);
                    // Compute barycentric coordinates
                    auto [alpha, beta, gamma] = computeBarycentric2D(x + 0.5f, y + 0.5f, t.v);

                    // Compute reciprocal of barycentric coordinates weighted by vertex w
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    auto mywatcher = v[0].z();
                    // Check if the interpolated depth is closer than the value in the depth buffer
                    if (z_interpolated < depth_buf[cur_index]) {
                        // Update the depth buffer
                        depth_buf[cur_index] = z_interpolated;
                        Eigen::Vector3f interpolated_color = interpolate(alpha, beta, gamma, pixel_color[0], pixel_color[1], pixel_color[2], 1.0f);

                        // Modify the color if shadow mapping is enabled

                        // Set the pixel color to the frame buffer
                        set_pixel(Vector2i(x, y), interpolated_color);
                    }
                }
            }
        }
        if (shadow) {}
            
    } else if (shading == rst::Shading::Phong) {
        // Find the bounding box of the triangle.

        // iterate through the pixel and find if the current pixel is inside the
        // triangle If so, use the following code to get the interpolated z value.
        // auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5, y+0.5, t.v);
        // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma /v[2].w());
        // float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() /
        // v[2].w(); z_interpolated *= w_reciprocal;

        // Check if the interpolated depth is closer than the value in the depth buffer.
        // If so, update the depth buffer
        // and calculate interpolated_color, interpolated_normal, interpolated_texcoords and interpolated_shadingcoords
        // float weight = alpha + beta + gamma;

        // pass them to the fragment_shader_payload
        // fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords,
        // view_lights, texture ? &*texture : nullptr); payload.view_pos = interpolated_shadingcoords;

        // Call the fragment shader to get the pixel color
        // auto pixel_color = fragment_shader(payload);

        // modify the color if do shadow mapping
        // Find the bounding box of the triangle.
        for (int x = min_x; x <= max_x; x++) {
            for (int y = min_y; y <= max_y; y++) {
                //判断是否在三角形内
                if (insideTriangle(x + 0.5, y + 0.5, t.v)) {
                    int cur_index = get_index(x, y);
                    float sample_x = x + 0.5f;
                    float sample_y = y + 0.5f;
                    auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                        //before depth interplotation, The center of gravity coordinates are corrected by perspective
                    float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    if (z_interpolated < depth_buf[cur_index]) {
                        depth_buf[cur_index] = z_interpolated;
    
                        auto interpolated_color = interpolate(alpha, beta, gamma, t.color[0], t.color[1], t.color[2], 1)*255;
                        auto interpolated_normal = interpolate(alpha, beta, gamma, t.normal[0], t.normal[1], t.normal[2], 1);
                        auto interpolated_texcoords = interpolate(alpha, beta, gamma, t.tex_coords[0], t.tex_coords[1], t.tex_coords[2], 1);
                        auto interpolated_shadingcoords = interpolate(alpha, beta, gamma, view_pos[0], view_pos[1], view_pos[2], 1);
                        //fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
                        fragment_shader_payload payload(interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, view_lights, texture ? &*texture : nullptr);
                        payload.view_pos = interpolated_shadingcoords;
                        auto pixel_color = fragment_shader(payload);
                        //作业2中的set_pixel输入的是vector3f点坐标，这次作业的是vector2i坐标，因此直接输入x，y坐标即可
                        Vector2i vertex;
                        vertex << x, y;
                        set_pixel(vertex, pixel_color);
                    }
                }
            }
    }
    if(shadow){
    Eigen::Vector3f cameraPos = {-1, 4.5, 14};
    Eigen::Vector3f lightPos = {-5, 6, 3};
    Eigen::Vector3f target = {0, 0, 0};

    Eigen::Matrix4f cameraViewMatrix = lookAt(cameraPos, target); 

    Eigen::Matrix4f lightViewMatrix = lookAt(lightPos, target); 
    Eigen::Vector4f lightCoord1;
    lightCoord1 << CameraToLightSpace(t.v[0].head<3>(), cameraViewMatrix, lightViewMatrix),1.0f;
    Eigen::Vector4f lightCoord2;
    lightCoord2 << CameraToLightSpace(t.v[1].head<3>(), cameraViewMatrix, lightViewMatrix), 1.0f;
    Eigen::Vector4f lightCoord3;
    lightCoord3 << CameraToLightSpace(t.v[2].head<3>(), cameraViewMatrix, lightViewMatrix),1.0f;
    Eigen::Vector4f lightCoords[3] = {lightCoord1, lightCoord2, lightCoord3};
    //the following is in light coordinate
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            if (insideTriangle(x + 0.5, y + 0.5, lightCoords)) {//judge w
                int cur_index = get_index(x, y);
                float sample_x = x + 0.5f;
                float sample_y = y + 0.5f;
                auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                    //before depth interplotation, The center of gravity coordinates are corrected by perspective
                float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                if (z_interpolated < depth_buf[cur_index]) {
                    shadow_buf[cur_index] = z_interpolated;
                }
            }
        }
    }
    //return to texture coordinate
    for (int x = min_x; x <= max_x; x++) {
        for (int y = min_y; y <= max_y; y++) {
            int cur_index = get_index(x, y);
            float sample_x = x + 0.5f;
            float sample_y = y + 0.5f;
            auto [alpha, beta, gamma] = computeBarycentric2D(sample_x, sample_y, t.v);
                //before depth interplotation, The center of gravity coordinates are corrected by perspective
            float w_reciprocal = 1.0f / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;//get z coornidate
            Eigen::Vector3f target = {sample_x, sample_y, z_interpolated};
            Eigen::Vector3f lightCoo = CameraToLightSpace(target,cameraViewMatrix, lightViewMatrix);
            if(lightCoo[2] > shadow_buf[get_index(lightCoo[0],lightCoo[1])]){//is shadow
                frame_buf[cur_index] = frame_buf[cur_index]*0.3;
            }
        }
    }
    }                        
        // set the pixel color to the frame buffer.
        // set_pixel(Vector2i(x, y), pixel_color);
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

void rst::rasterizer::set_lights(const std::vector<light> &l) {
    lights = l;
}

void rst::rasterizer::set_shadow_view(const Eigen::Matrix4f &v) {
    shadow_view = v;
}

void rst::rasterizer::set_shadow_buffer(const std::vector<float> &shadow_buffer) {
    std::copy(shadow_buffer.begin(), shadow_buffer.end(), this->shadow_buf.begin());
}

void rst::rasterizer::clear(rst::Buffers buff) {
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        std::fill(ssaa_frame_buf.begin(), ssaa_frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        std::fill(ssaa_depth_buf.begin(), ssaa_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    shadow_buf.resize(w * h);
    ssaa_frame_buf.resize(4 * w * h);
    ssaa_depth_buf.resize(4 * w * h);
    texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y) {
    return (height - y - 1) * width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color) {
    // old index: auto ind = point.y() + point.x() * width;
    int ind = (height - point.y() - 1) * width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader) {
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader) {
    fragment_shader = frag_shader;
}
