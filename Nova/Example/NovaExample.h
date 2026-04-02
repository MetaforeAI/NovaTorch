#pragma once
#include "../Nova.h"

class NovaExample {
    public:
        NovaExample();
        ~NovaExample();

        static NovaExample* manifest();   // Singleton

        static void materialize();      // Draw
        void actualize();               // Run
    
    private:
        NovaConfig _config;
        Nova* _actuality;

        NovaExample* realize();            // Init
};