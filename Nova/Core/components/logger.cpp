#include "logger.h"
#include <iostream>
#include <cstdarg>
#include "./debug_level.h"

static LOGGER getLogLevel (const char* dbg_lvl) {
    switch (getDebugLevel(dbg_lvl)) {
        case DEBUG_LEVEL::SILENT:
            return LOGGER::OFF;
        case DEBUG_LEVEL::RELEASE:
            return LOGGER::ERROR;
        case DEBUG_LEVEL::STAGING:
            return LOGGER::INFO;
        case DEBUG_LEVEL::DEV:
            return LOGGER::DEBUG;
        case DEBUG_LEVEL::LOUD:
            return LOGGER::VERBOSE;
        default:
            return LOGGER::OFF;
    }
};

static DEBUG_LEVEL getDebugLevel (LOGGER log_level) {
    switch (log_level) {
        case LOGGER::OFF:
            return DEBUG_LEVEL::SILENT;
        case LOGGER::ERROR:
            return DEBUG_LEVEL::RELEASE;
        case LOGGER::INFO:
            return DEBUG_LEVEL::STAGING;
        case LOGGER::DEBUG:
            return DEBUG_LEVEL::DEV;
        case LOGGER::VERBOSE:
            return DEBUG_LEVEL::LOUD;
        default:
            return DEBUG_LEVEL::INVALID;
    }

};

void setLogLevel(const char* debug_level) 
    { 
        LOG_LEVEL = getLogLevel(debug_level); 
        report(LOGGER::INFO, "Logger - Debug Level set to %s ..", debugString(getDebugLevel(LOG_LEVEL))); 
        return; 
    }


static bool letsGo(LOGGER level) 
    { return level <= LOG_LEVEL; }

void report(LOGGER log_level, const char* format, ...){
        std::va_list args;
        va_start(args, format);

        if (letsGo(log_level)) {
            switch(log_level) {
                case LOGGER::ILINE:
                case LOGGER::DLINE:
                case LOGGER::VLINE:
                    std::cout << " \t ";
                    break;
                case LOGGER::ERROR:
                    std::cout << " [ERROR]: ";
                    break;
                case LOGGER::INFO:
                    std::cout << " [INFO]: ";
                    break;
                case LOGGER::DEBUG:
                    std::cout << " [DEBUG]: ";
                    break;
                case LOGGER::VERBOSE:
                    std::cout << " [VERBOSE]: ";
                    break;
                default:
                    break;
            }

            vprintf(format, args);
            std::cout << std::endl;
        }
        va_end(args);

    return;
}