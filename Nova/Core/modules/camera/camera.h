#pragma once
#include <glm/glm.hpp>
#include <SDL2/SDL_events.h>
#include "./perspective.h"

class Camera 
    {
        public:
            glm::mat4 getRotationMatrix();
            glm::mat4 getViewMatrix();
            glm::mat4 getProjectionMatrix(float fov, float aspect_ratio, float near, float far);
        
            void update();
            void processEvents(SDL_Event& event);
        
        private:
            Perspective perspective;
            void switchCamera();
            void rearView();

            glm::vec3 position{0.0f, -1.0f, 0.0f};
            glm::vec3 relative_camera_offset{0.0f, 0.0f, 0.0f};
            glm::vec3 velocity{0.0f, 0.0f, 0.0f};
            glm::vec3 acceleration{0.0f, 0.0f, 0.0f};
            float roll_velocity{0.0f};
            float roll_acceleration{0.0f};
            float pitch{0.0f};
            float yaw{0.0f};
            float roll{0.0f};
    };