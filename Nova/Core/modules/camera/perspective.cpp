#include "perspective.h"

Perspective::Perspective() 
    { perspective = CameraPerspective::FIRST_PERSON; }

Perspective::~Perspective() {}

void Perspective::togglePerspective() 
    {
        if (perspective == CameraPerspective::FIRST_PERSON) 
            { perspective = CameraPerspective::THIRD_PERSON_BEHIND; } 
        else if (perspective == CameraPerspective::THIRD_PERSON_BEHIND) 
            { perspective = CameraPerspective::THIRD_PERSON_ABOVE; } 
        else 
            { perspective = CameraPerspective::FIRST_PERSON; }
    }

glm::vec3 Perspective::getOffset(glm::mat4 rotation) 
    {
        glm::vec3 _perspective_offset = {0.0f, 0.0f, 0.0f};

        switch (perspective) 
            {
                case CameraPerspective::FIRST_PERSON: 
                    _perspective_offset = {0.0f, 0.0f, 0.0f};
                case CameraPerspective::THIRD_PERSON_BEHIND: 
                    _perspective_offset = {0.0f, 0.0f, -10.0f};
                case CameraPerspective::THIRD_PERSON_ABOVE: 
                    _perspective_offset = {0.0f, 15.0f, -20.0f};
            }
        
        return glm::vec3(glm::vec4(_perspective_offset, 1.0f) * rotation);
    }