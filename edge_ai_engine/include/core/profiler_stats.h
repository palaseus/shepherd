/**
 * @file profiler_stats.h
 * @brief Profiler statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ProfilerStats class for profiler statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>
#include <atomic>

namespace edge_ai {

/**
 * @struct ProfilerStats
 * @brief Statistics for profiler
 */
struct ProfilerStats {
    // Session statistics
    uint64_t total_sessions = 0;
    uint64_t active_sessions = 0;
    std::chrono::microseconds total_profiling_time{0};
    
    // Operation statistics
    uint64_t total_timing_operations = 0;
    uint64_t total_counter_operations = 0;
    uint64_t total_memory_operations = 0;
    uint64_t total_hardware_operations = 0;
    
    // Data statistics
    size_t total_data_points = 0;
    size_t memory_usage = 0;
    double data_compression_ratio = 0.0;
    
    ProfilerStats() = default;
};

} // namespace edge_ai
