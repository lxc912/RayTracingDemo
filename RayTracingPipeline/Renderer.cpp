//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include <sstream>
#include "Scene.hpp"
#include "Renderer.hpp"
#include "Material.hpp"
#ifdef _OPENMP
    #include <omp.h>
#endif


inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 1e-2;

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    std::vector<Vector3f> framebuffer(scene.width * scene.height);

    float scale = tan(deg2rad(scene.fov * 0.5));
    float imageAspectRatio = scene.width / (float)scene.height;
    Vector3f eye_pos(278, 273, -800);

    std::cout << "SPP: " << scene.spp << "\n";

    float progress = 0.0f;

#pragma omp parallel for num_threads(8) // use multi-threading for speedup if openmp is available
    for (uint32_t j = 0; j < scene.height; ++j) {
        for (uint32_t i = 0; i < scene.width; ++i) {

            int m = i + j * scene.width;
            if(scene.spp==1){
                // TODO: task 1.2 pixel projection
                float ndc_x = -2 * (i + 0.5f) / scene.width + 1.0f;
                float ndc_y = 1.0f - 2 * (j + 0.5f) / scene.height;
                float x = ndc_x* imageAspectRatio * scale;
                float y = ndc_y * scale;
                Vector3f dir = normalize(Vector3f(x, y, 1));
                framebuffer[m] = scene.castRay(Ray(eye_pos, dir), 0);
            }else {
                // TODO: task 4 multi-sampling
                Vector3f pixelColor = Vector3f(0);
                for (int s = 0; s < scene.spp; s++) {
                    // Random offsets within the pixel [-0.5, 0.5] range
                    float offsetX = ((float)rand() / RAND_MAX - 0.5f);
                    float offsetY = ((float)rand() / RAND_MAX - 0.5f);

                    // Adjust ray to pass through a random point within the pixel
                    float ndc_x = -2 * (i + 0.5f + offsetX) / scene.width + 1.0f;
                    float ndc_y = 1.0f - 2 * (j + 0.5f + offsetY) / scene.height;
                    float x = ndc_x * imageAspectRatio * scale;
                    float y = ndc_y * scale;

                    Vector3f dir = normalize(Vector3f(x, y, 1));
                    pixelColor += scene.castRay(Ray(eye_pos, dir), 0);
                }
                pixelColor = pixelColor/scene.spp;  // Average the color
                framebuffer[m] = pixelColor;
            }
        }
        progress += 1.0f / (float)scene.height;
        UpdateProgress(progress);
    }
    UpdateProgress(1.f);

    // save framebuffer to file
    std::stringstream ss;
    ss << "binary_task" << TASK_N<<".ppm";
    std::string str = ss.str();
    const char* file_name = str.c_str();
    FILE* fp = fopen(file_name, "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);
}
