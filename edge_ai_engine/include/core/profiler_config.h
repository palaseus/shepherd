/**
 * @file profiler_config.h
 * @brief Profiler configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ProfilerConfig class for profiler configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>

namespace edge_ai {

/**
 * @struct ProfilerConfig
 * @brief Configuration for profiler
 */
struct ProfilerConfig {
    // Profiling configuration
    bool enable_profiling = true;
    bool enable_detailed_profiling = false;
    bool enable_memory_profiling = true;
    bool enable_hardware_profiling = true;
    
    // Sampling configuration
    double sampling_rate = 1.0; // 100% sampling
    std::chrono::milliseconds sampling_interval{100};
    
    // Data collection configuration
    size_t max_data_points = 10000;
    bool enable_data_compression = true;
    bool enable_data_persistence = false;
    
    // Export configuration
    bool enable_auto_export = false;
    std::string export_format = "JSON";
    std::chrono::minutes export_interval{5};
    
    ProfilerConfig() = default;
};

} // namespace edge_ai
