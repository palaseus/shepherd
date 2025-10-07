/**
 * @file timing_stats.h
 * @brief Timing statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the TimingStats class for timing statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>
#include <string>

namespace edge_ai {

/**
 * @struct TimingStats
 * @brief Statistics for timing operations
 */
struct TimingStats {
    std::string name;
    uint64_t call_count = 0;
    std::chrono::microseconds total_time{0};
    std::chrono::microseconds average_time{0};
    std::chrono::microseconds min_time{std::chrono::microseconds::max()};
    std::chrono::microseconds max_time{0};
    std::chrono::microseconds p95_time{0};
    std::chrono::microseconds p99_time{0};
    
    TimingStats() = default;
    explicit TimingStats(const std::string& n) : name(n) {}
};

} // namespace edge_ai
