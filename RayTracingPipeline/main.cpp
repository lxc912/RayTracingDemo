#include "Renderer.hpp"
#include "Scene.hpp"
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Vector.hpp"
#include "global.hpp"
#include <chrono>

int TASK_N=3;  // 1, 2, 3, 4, 5

// In the main function of the program, we create the scene (create objects and
// lights) as well as set the options for the render (image width and height,
// maximum recursion depth, field-of-view, etc.). We then call the render
// function().
int main(int argc, char** argv)
{
    if (argc>=2)
        TASK_N=(int)atoi(argv[1]);
    // change the resolution for quick debugging if rendering is slow
    //Scene scene(512, 512);
    Scene scene(256, 256); // use this resolution for final rendering

    if(TASK_N>=4)
        scene.spp = 32; // number of samples per pixel
    else
        scene.spp = 1;
    Material* red = new Material(DIFFUSE, Vector3f(0.3f, 0.01f, 0.01f));
    Material* green = new Material(DIFFUSE, Vector3f(0.045, 0.22, 0.03));
    Material* white = new Material(DIFFUSE, Vector3f(0.48f, 0.45f, 0.4f));
    Material* light = new Material(EMIT, Vector3f(1));
    light->m_emission=100;

    MeshTriangle floor("../models/cornellbox/floor.obj", Vector3f(0), white);
    MeshTriangle shortbox("../models/cornellbox/shortbox.obj",Vector3f(0),  white);
    MeshTriangle tallbox("../models/cornellbox/tallbox.obj", Vector3f(0), white);
    MeshTriangle left("../models/cornellbox/left.obj", Vector3f(0), red);
    MeshTriangle right("../models/cornellbox/right.obj",Vector3f(0),  green);
    MeshTriangle light_("../models/cornellbox/light.obj",Vector3f(0,-5,0), light);

    scene.Add(&floor);
    scene.Add(&shortbox);
    scene.Add(&tallbox);
    scene.Add(&left);
    scene.Add(&right);
    scene.Add(&light_);

    scene.Add(new MeshTriangle("../models/bunny/bunny.obj", Vector3f(0),
                    new Material(GLASS, Vector3f(1))));

    scene.Add(new Sphere(Vector3f(450,60,100), 60,
                         new Material(GLASS, Vector3f(1))));

    scene.Add(new Sphere(Vector3f(370,30,150), 30,
                         new Material(DIFFUSE, Vector3f(0.7,0.5,1))));

    Vector3f verts[4] = {{0,0,0}, {552.8,0,0}, {549.6, 0,559.2}, {0,0,559.2}};
    Vector2f st[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    uint32_t vertIndex[6] = {0, 2, 1, 2,0,3};
    Material* mfloor=new Material(DIFFUSE, Vector3f(0));
    mfloor->textured=true;
    scene.Add(new MeshTriangle(verts, vertIndex, 2,st,mfloor));

    scene.Add(std::make_unique<PointLight>(Vector3f(-2000, 4000, -3000), 0.5));

    scene.buildBVH();

    Renderer r;

    auto start = std::chrono::system_clock::now();
    r.Render(scene);
    auto stop = std::chrono::system_clock::now();
    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";

    return 0;
}
