#pragma once
#include <string>
#include <unordered_map>


enum DEBUG_LEVEL {
    SILENT,             // No output
    RELEASE,            // Would only be Critical User-facing errors
    STAGING,            // Would typically be status level updates for other testers
    DEV,                // Would be standard debug level information meant for other developers
    LOUD,               // Maximum debug level information for personal debugging and benchmarking
    INVALID             // Invalid debug level
};

const std::unordered_map<std::string, DEBUG_LEVEL> DEBUG_MAP = {
        { "none", DEBUG_LEVEL::SILENT },
        { "release", DEBUG_LEVEL::RELEASE },
        { "staging", DEBUG_LEVEL::STAGING },
        { "development", DEBUG_LEVEL::DEV },
        { "debug", DEBUG_LEVEL::LOUD }
};

// Adds a safety check to ensure that the debug level is set to a valid value.
DEBUG_LEVEL getDebugLevel (const char* dbg_lvl) {
    printf("Logger - Debug Level set to %s ..", dbg_lvl);
    if (DEBUG_MAP.find(dbg_lvl) != DEBUG_MAP.end()) 
        { return DEBUG_MAP.at(dbg_lvl); } 
    else 
        { return DEBUG_LEVEL::INVALID; }
};

// This is a helper function to convert the DEBUG_LEVEL enum to a string for output.
const char* debugString (DEBUG_LEVEL dbg_lvl) {
    switch (dbg_lvl) {
        case DEBUG_LEVEL::SILENT:
            return "none";
        case DEBUG_LEVEL::RELEASE:
            return "release";
        case DEBUG_LEVEL::STAGING:
            return "staging";
        case DEBUG_LEVEL::DEV:
            return "development";
        case DEBUG_LEVEL::LOUD:
            return "debug";
        default:
            return "invalid";
    }
};