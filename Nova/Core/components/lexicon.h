#pragma once
#include <vulkan/vulkan.h>
#include "./logger.h"
#include <stdio.h>
#include <cstdlib>

#define VK_TRY(x)                                                       \
    do {                                                                \
        VkResult err = x;                                               \
        if (err) {                                                      \
            fprintf(stderr, " [ERROR] Vulkan: VkResult=%d\n", err);      \
            abort();                                                    \
        }                                                               \
    } while (0)

