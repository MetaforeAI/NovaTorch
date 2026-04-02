#include "./Nova.h"

#include <thread>
#include <chrono>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

    ///////////////////
    // INSTANTIATION //
    ///////////////////


Nova::Nova(NovaConfig config)
    {
        report(LOGGER::INFO, "Nova - Instantiating ..");

        setLogLevel(config.debug_level.c_str());
        _config = config;  // Store config for later use

        if (config.compute) {
            // Compute-only mode
            _mode = Mode::Compute;
            _window = nullptr;
            report(LOGGER::INFO, "Nova - Compute mode: creating NovaCompute instance ..");

            _architect_compute = new NovaCompute(config.debug_level);
            _architect = _architect_compute;  // Legacy compatibility

            report(LOGGER::INFO, "Nova - Compute mode initialized");
        } else {
            // Graphics mode
            _mode = Mode::Graphics;

            // Initialize SDL and create a window
            SDL_Init(SDL_INIT_VIDEO);
            SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

            _window = SDL_CreateWindow(
                config.name.c_str(),
                SDL_WINDOWPOS_UNDEFINED,
                SDL_WINDOWPOS_UNDEFINED,
                config.screen.width,
                config.screen.height,
                window_flags
            );

            if (_window == nullptr) {
                report(LOGGER::ERROR, "Nova - Failed to create SDL window ..");
                return;
            }

            // Create Vulkan surface from SDL window
            VkSurfaceKHR surface;
            if (!SDL_Vulkan_CreateSurface(_window, VK_NULL_HANDLE, &surface)) {
                report(LOGGER::ERROR, "Nova - Failed to create Vulkan surface");
                SDL_DestroyWindow(_window);
                SDL_Quit();
                return;
            }

            report(LOGGER::INFO, "Nova - Graphics mode: creating NovaGraphics instance ..");
            _architect_graphics = new NovaGraphics(config.screen, config.debug_level, surface);
            _architect = _architect_graphics;  // Legacy compatibility

            // Initialize graphics pipeline
            _initGraphicsPipeline();

            report(LOGGER::INFO, "Nova - Graphics mode initialized");
        }

        initialized = true;
        report(LOGGER::INFO, "Nova - Initialized ..");
    }

Nova::~Nova()
    {
        report(LOGGER::INFO, "Nova - Deconstructing ..");

        if (initialized) {
            if (_architect_compute) {
                vkDeviceWaitIdle(_architect_compute->getDevice());
            } else if (_architect_graphics) {
                vkDeviceWaitIdle(_architect_graphics->getDevice());
            }

            if (USE_VALIDATION_LAYERS) {
                report(LOGGER::VLINE, "\t .. Destroying Debug Messenger ..");
                VkInstance instance = _architect_compute ? _architect_compute->getInstance() : _architect_graphics->getInstance();
                destroyDebugUtilsMessengerEXT(instance, _debug_messenger, nullptr);
            }

            report(LOGGER::VLINE, "\t .. Destroying Nova architect ..");
            if (_architect_compute) {
                delete _architect_compute;
                _architect_compute = nullptr;
            }
            if (_architect_graphics) {
                delete _architect_graphics;
                _architect_graphics = nullptr;
            }
            _architect = nullptr;

            report(LOGGER::VLINE, "\t .. Destroying Window ..");
            if (_window != nullptr) {
                SDL_DestroyWindow(_window);
                SDL_Quit();
            }
        }

        report(LOGGER::INFO, "Nova - Destroyed ..");
    }

// Mode getters
NovaCore* Nova::getCore() {
    if (_mode == Mode::Compute) return _architect_compute;
    else return _architect_graphics;
}

NovaCompute* Nova::getCompute() {
    if (_mode != Mode::Compute) {
        report(LOGGER::ERROR, "Attempted to get compute interface in non-compute mode");
        return nullptr;
    }
    return _architect_compute;
}

NovaGraphics* Nova::getGraphics() {
    if (_mode != Mode::Graphics) {
        report(LOGGER::ERROR, "Attempted to get graphics interface in non-graphics mode");
        return nullptr;
    }
    return _architect_graphics;
}


    /////////////////////////
    // TOP LEVEL FUNCTIONS //
    /////////////////////////

void Nova::illuminate()
//void Nova::illuminate(fnManifest fnManifest)
    {
        report(LOGGER::INFO, "Nova - Illuminating ..");

        SDL_Event _e;
        bool _quit = false;

        while (!_quit) {
            while (SDL_PollEvent(&_e)) 
                {
                    if (_e.type == SDL_QUIT) { _quit = !_quit; }
                    
                    if (_e.type == SDL_KEYDOWN) 
                        { if (_e.key.keysym.sym == SDLK_ESCAPE) { _quit = !_quit; } }

                    if (_e.type == SDL_WINDOWEVENT) 
                        { 
                            switch (_e.window.event) 
                                {
                                    case SDL_WINDOWEVENT_MINIMIZED: _suspended = true; break;
                                    case SDL_WINDOWEVENT_RESTORED: _suspended = false; break;
                                    case SDL_WINDOWEVENT_RESIZED: _resizeWindow(); break;
                                }
                        }

                    if (_architect_graphics) {
                        _architect_graphics->player_camera.processEvents(_e);
                    }
                }
            
            if (_suspended)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
            else
                {
                    if (_architect_graphics) {
                        _architect_graphics->drawFrame();
                    }
                }

            //fnManifest();
        }

        return;
    }


    //////////////////
    // INITIALIZERS //
    //////////////////

void Nova::_initGraphicsPipeline()
    {
        report(LOGGER::INFO, "Nova - Initializing Graphics Pipeline ..");

        // NovaGraphics already initialized swapchain, render pass, etc.
        // This is for any additional graphics setup if needed

        // TODO: Move any remaining graphics initialization here
        // For now, NovaGraphics constructor handles everything

        report(LOGGER::INFO, "Nova - Graphics pipeline ready");
    }

// DEPRECATED: These methods are no longer needed
// NovaGraphics handles all initialization in its constructor
void Nova::_initSwapChain(std::promise<void>& startPipeline, std::future<void>& waitingForPipeline, std::promise<void>& waitForFrameBuffer)
    {
        report(LOGGER::INFO, "Nova - _initSwapChain deprecated (handled by NovaGraphics) ..");
        // Deprecated - NovaGraphics constructor handles this
        return;
    }

void Nova::_initPipeline(std::future<void>& startingPipeline, std::promise<void>& waitForPipeline)
    {
        report(LOGGER::INFO, "Nova - _initPipeline deprecated (handled by NovaGraphics) ..");
        // Deprecated - NovaGraphics constructor handles this
        return;
    }

void Nova::_initBuffers()
    {
        report(LOGGER::INFO, "Nova - _initBuffers deprecated (handled by NovaGraphics) ..");
        // Deprecated - NovaGraphics constructor handles this
        return;
    }

void Nova::_initSyncStructures()
    {
        report(LOGGER::INFO, "Nova - _initSyncStructures deprecated (handled by NovaGraphics) ..");
        // Deprecated - NovaGraphics constructor handles this
        return;
    }


    ///////////////////
    // RESIZE WINDOW //
    ///////////////////

inline void Nova::_resizeWindow()
    {
        report(LOGGER::VERBOSE, "Nova - Resizing Window ..");

        int w, h;
        SDL_Vulkan_GetDrawableSize(_window, &w, &h);

        _config.screen.width = w;
        _config.screen.height = h;

        if (_architect_graphics) {
            _architect_graphics->setWindowExtent(_config.screen);
            _architect_graphics->framebuffer_resized = true;
        }
        return;
    }   

