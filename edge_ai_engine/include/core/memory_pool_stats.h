/**
 * @file memory_pool_stats.h
 * @brief Memory pool statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryPoolStats class for memory pool statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"

namespace edge_ai {

/**
 * @struct MemoryPoolStats
 * @brief Statistics for memory pool
 */
struct MemoryPoolStats {
    size_t pool_size = 0;
    size_t allocated_size = 0;
    size_t free_size = 0;
    uint64_t total_allocations = 0;
    uint64_t total_deallocations = 0;
    double utilization_ratio = 0.0;
    
    MemoryPoolStats() = default;
};

} // namespace edge_ai
