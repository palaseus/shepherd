/**
 * @file memory_usage_stats.h
 * @brief Memory usage statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryUsageStats class for memory usage statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"

namespace edge_ai {

/**
 * @struct MemoryUsageStats
 * @brief Statistics for memory usage
 */
struct MemoryUsageStats {
    size_t total_allocations = 0;
    size_t total_deallocations = 0;
    size_t current_memory_usage = 0;
    size_t peak_memory_usage = 0;
    size_t total_memory_allocated = 0;
    double average_allocation_size = 0.0;
    double memory_fragmentation_ratio = 0.0;
    
    MemoryUsageStats() = default;
};

} // namespace edge_ai
