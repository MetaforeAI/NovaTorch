#pragma once
// Use new modular architecture (core_base, nova_compute, nova_graphics)
#include "./Core/core_base.h"
#include "./Core/nova_compute.h"
#include "./Core/nova_graphics.h"
#include "./Core/config.h"

// Backward compatibility: still include old core.h for legacy code
// TODO: Remove this once all code migrated to new architecture
#include "./Core/core.h"

#include <string>
#include <future>

// The goal of this layer of abstraction is to create a friendly user implementation for creating a graphics engine, for future projects.

// Forward declaration
class SMFTEngine;

// TODO: Cross Platform Support
class Nova {
    // Allow SMFTEngine to access private _architect for compute operations
    friend class SMFTEngine;

    public:
        bool initialized = false;

        Nova(NovaConfig);
        ~Nova();

        // TODO: Determine Default Initializers

        void illuminate();
        //void illuminate(fnManifest);

        // Mode detection
        enum class Mode { Compute, Graphics };
        Mode getMode() const { return _mode; }

        // Get appropriate interface
        NovaCore* getCore();  // Base interface (compute or graphics)
        NovaCompute* getCompute();  // Compute-only interface (nullptr if graphics mode)
        NovaGraphics* getGraphics();  // Graphics interface (nullptr if compute mode)

    private:
        NovaConfig _config;
        bool _suspended = false;
        struct SDL_Window* _window = nullptr;

        // Mode-specific instances
        Mode _mode;
        NovaCompute* _architect_compute = nullptr;
        NovaGraphics* _architect_graphics = nullptr;

        // Legacy compatibility
        NovaCore* _architect = nullptr;  // Points to either _architect_compute or _architect_graphics

        VkDebugUtilsMessengerEXT _debug_messenger;

        void _initFramework();
        void _initSwapChain(std::promise<void>&, std::future<void>&, std::promise<void>&);
        void _initPipeline(std::future<void>&, std::promise<void>&);
        void _initBuffers();
        void _initSyncStructures();
        void _resizeWindow();
        void _initGraphicsPipeline();
};
