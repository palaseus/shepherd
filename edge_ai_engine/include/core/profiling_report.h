/**
 * @file profiling_report.h
 * @brief Profiling report interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ProfilingReport class for profiling reports in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <chrono>

namespace edge_ai {

/**
 * @struct ProfilingReport
 * @brief Comprehensive profiling report
 */
struct ProfilingReport {
    // Report metadata
    std::string report_name;
    std::chrono::steady_clock::time_point generation_time;
    std::chrono::microseconds total_profiling_time;
    
    // Performance summary
    std::vector<TimingStats> top_operations;
    std::vector<CounterStats> top_counters;
    MemoryUsageStats memory_summary;
    std::vector<HardwareUtilizationStats> hardware_summary;
    
    // Recommendations
    std::vector<std::string> performance_recommendations;
    std::vector<std::string> optimization_suggestions;
    
    ProfilingReport() = default;
};

} // namespace edge_ai
