//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}


void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            pos.happened=true;  // area light that has emission exists
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}


// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const
{
    int max_depth = 6;
    if (depth > max_depth) {
        return Vector3f(0.0, 0.0, 0.0);
    }
    Vector3f hitColor = Vector3f(0);
    auto inter = intersect(ray);
    //std::cout<<this.<<" ";
    if (!inter.happened)return backgroundColor;

    Vector3f hitPoint = inter.coords;
    Vector3f N = inter.normal; // normal
    Vector2f st = inter.tcoords; // texture coordinates
    Vector3f dir = ray.direction;

    if (inter.material->m_type == EMIT) {
        return inter.material->m_emission;
    }else if (inter.material->m_type == DIFFUSE || TASK_N<3) {
        Vector3f lightAmt = 0, specularColor = 0;
        // sample area light
        int light_sample=4;
        for (int i = 0; i < light_sample && TASK_N >= 5; ++i) {//case soft shadow and diffuse material
            Intersection lightInter;
            float pdf_light = 0.0f;
            sampleLight(lightInter, pdf_light);  // sample a point on the area light
            //TODO: task 5 soft shadow
            Vector3f shadowPointOrig = (dotProduct(dir, N) < 0)
                    ? hitPoint + N * EPSILON
                    : hitPoint - N * EPSILON;
            //Vector3f shadowPointOrig = hitPoint + N * EPSILON;
            Vector3f light2Obj = lightInter.coords - shadowPointOrig; // Direction from hit point to light
            Vector3f lightDir = normalize(light2Obj);
            float cos = std::max(0.f, dotProduct(lightDir, N));
            // Cast a shadow ray to the sampled light point
            Ray shadowRay(shadowPointOrig, lightDir);
            Intersection shadowInter = intersect(shadowRay);
                
            if (shadowInter.happened && std::abs(shadowInter.tnear - light2Obj.norm())>EPSILON) {//in the shadow
                continue;
            }else{
                hitColor += inter.obj->evalDiffuseColor(st)*lightInter.material->getEmission()*lightInter.material->eval(lightDir, N)
                *cos/pdf_light/dotProduct(light2Obj, light2Obj);
            } 
        }
        hitColor = hitColor/(float)light_sample;
        // TODO: task 1.3 Basic shading
        Vector3f shadowPointOrig = (dotProduct(dir, N) < 0)
                                     ? hitPoint + N * EPSILON
                                     : hitPoint - N * EPSILON;
        for (auto &light : lights) {
            //ambient light
            Vector3f ambientColor = 0.005*light->intensity;
            Vector3f lightDir = light->position - shadowPointOrig;
            float lightDistance = lightDir.norm();
            lightDir = normalize(lightDir);
            float LdotN = std::max(0.f, dotProduct(lightDir, N));
            Ray shadowRay = Ray(shadowPointOrig, lightDir);
            auto shadow_res = intersect(shadowRay);
            if(shadow_res.tnear < lightDistance){
                continue;
            }
            else{
                lightAmt = light->intensity * LdotN;
            }
                
            Vector3f reflectionDirection = reflect(-lightDir, N);
            //diffuse color
            Vector3f diffuseColor = lightAmt * inter.obj->evalDiffuseColor(st) * inter.material->Kd;
            //specular color
            specularColor = powf(std::max(0.f, -dotProduct(reflectionDirection, dir)),inter.material->specularExponent) *light->intensity* inter.material->Ks;
            hitColor += ambientColor + lightAmt * inter.obj->evalDiffuseColor(st) * inter.material->Kd + specularColor;//add up three parts
        }
    } else if (inter.material->m_type == GLASS && TASK_N>=3) {
        // TODO: task 3 glass material
        Vector3f reflectionDirection = normalize(reflect(dir, N));
        Vector3f refractionDirection = normalize(refract(dir, N, inter.material->ior));
        Vector3f reflectionRayOrig = (dotProduct(reflectionDirection, N) < 0)
                                        ? hitPoint - N * EPSILON
                                        : hitPoint + N * EPSILON;
        Vector3f refractionRayOrig = (dotProduct(refractionDirection, N) < 0)
                                        ? hitPoint - N * EPSILON
                                        : hitPoint + N * EPSILON;
        Ray reflectionRay = Ray(reflectionRayOrig, reflectionDirection);
        Vector3f reflectionColor = castRay(reflectionRay, depth + 1);
        Ray refractionRay = Ray(refractionRayOrig, refractionDirection);
        Vector3f refractionColor = castRay(refractionRay, depth + 1);
        float kr = fresnel(dir, N, inter.material->ior);
        hitColor = reflectionColor * kr + refractionColor * (1 - kr);
    }


    return hitColor;
}
