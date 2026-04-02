#include "./camera.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

static const float ACCELERATOR = 0.0005f;
const float TURN_MAX_VELOCITY = 0.5f;
const float MAX_VELOCITY = 2.3f;

// TODO: Implement logic at the user level to allow for camera initialization AND toggling perspectives/acceleration or fixed speed

glm::mat4 Camera::getRotationMatrix()
    {
        glm::quat q = glm::quat(glm::vec3(pitch, -yaw, roll)); // No clue if this works, or if I need to calculate  the angleAxis per axis
        return glm::mat4_cast(q);
    }

glm::mat4 Camera::getViewMatrix()
    {
        glm::mat4 c_rotation = getRotationMatrix();
        glm::mat4 c_translation = glm::translate(-position - relative_camera_offset);
        return c_rotation * c_translation;
    }

glm::mat4 Camera::getProjectionMatrix(float fov, float aspect_ratio, float near, float far)
    { return glm::perspective(fov, aspect_ratio, near, far); }


void Camera::switchCamera() { perspective.togglePerspective(); }

static const float dampening(float velocity) { return velocity * 0.98f; }

void Camera::update()
    {
        // Need to check that the absolute value of the velocity + acceleration is less than the max velocity
        if (glm::length(velocity + acceleration) < MAX_VELOCITY)
            { velocity += acceleration; }
        
        if (glm::length(roll_velocity + roll_acceleration) < TURN_MAX_VELOCITY)
            { roll_velocity += roll_acceleration; }

        roll += roll_velocity;

        glm::mat4 c_rotation = getRotationMatrix();

        glm::vec3 forward = glm::vec3(c_rotation * glm::vec4(0.0f, 0.0f, -1.0f, 1.0f));
        glm::vec3 right = glm::vec3(c_rotation * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
        glm::vec3 direction = forward * velocity.z + right * velocity.x;
        position += direction; // Need to add the direction to the position to move relative to the camera's orientation

        // The relative camera offset has to be offset based on the perspective with respect to the camera direction
        relative_camera_offset = perspective.getOffset(c_rotation);

        velocity = glm::vec3(dampening(velocity.x), dampening(velocity.y), dampening(velocity.z));
        roll_velocity = dampening(roll_velocity);
    }

void Camera::rearView() { pitch = -pitch; yaw = -yaw; }

void Camera::processEvents(SDL_Event& e) 
    { 
        if (e.type == SDL_KEYDOWN) 
            {
                switch (e.key.keysym.sym) 
                    {
                        case SDLK_w: acceleration.z -= ACCELERATOR; break;
                        case SDLK_s: acceleration.z += ACCELERATOR; break;
                        case SDLK_a: acceleration.x -= ACCELERATOR; break;
                        case SDLK_d: acceleration.x += ACCELERATOR; break;
                        case SDLK_q: roll_acceleration -= ACCELERATOR; break;
                        case SDLK_e: roll_acceleration += ACCELERATOR; break;
                        case SDLK_1: switchCamera(); break;
                        case SDLK_2: rearView(); break;
                    }
            }
        else if (e.type == SDL_KEYUP) 
            {
                switch (e.key.keysym.sym) 
                    {
                        case SDLK_w: acceleration.x = 0.0f; break;
                        case SDLK_s: acceleration.x = 0.0f; break;
                        case SDLK_a: acceleration.z = 0.0f; break;
                        case SDLK_d: acceleration.z = 0.0f; break;
                        case SDLK_q: roll_acceleration = 0.0f; break;
                        case SDLK_e: roll_acceleration = 0.0f; break;
                        case SDLK_2: rearView(); break;
                    }
            }

        if (e.type == SDL_MOUSEMOTION) 
            {
                yaw += (float)e.motion.xrel / 200.f;
                pitch += (float)e.motion.yrel / 200.f;
            }
    }