/**
 * @file counter_stats.h
 * @brief Counter statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the CounterStats class for counter statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <string>

namespace edge_ai {

/**
 * @struct CounterStats
 * @brief Statistics for counter operations
 */
struct CounterStats {
    std::string name;
    int64_t total_value = 0;
    int64_t current_value = 0;
    int64_t min_value = 0;
    int64_t max_value = 0;
    double average_value = 0.0;
    uint64_t update_count = 0;
    
    CounterStats() = default;
    explicit CounterStats(const std::string& n) : name(n) {}
};

} // namespace edge_ai
