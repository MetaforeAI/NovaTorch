#pragma once
#include <string>

enum LOGGER {
    OFF,
    ERROR,
    ILINE,
    INFO,
    DLINE,
    DEBUG,
    VLINE,
    VERBOSE
};

inline LOGGER LOG_LEVEL = LOGGER::VERBOSE;

void report(LOGGER log_level, const char* message, ...);
void setLogLevel(const char* debug_level);
