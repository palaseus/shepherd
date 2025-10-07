/**
 * @file memory_allocator_stats.h
 * @brief Memory allocator statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryAllocatorStats class for memory allocator statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"

namespace edge_ai {

/**
 * @struct MemoryAllocatorStats
 * @brief Statistics for memory allocator
 */
struct MemoryAllocatorStats {
    uint64_t total_allocations = 0;
    uint64_t total_deallocations = 0;
    size_t current_memory_usage = 0;
    size_t peak_memory_usage = 0;
    double average_allocation_time = 0.0;
    double average_deallocation_time = 0.0;
    
    MemoryAllocatorStats() = default;
};

} // namespace edge_ai
