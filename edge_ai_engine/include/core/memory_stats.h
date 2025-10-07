/**
 * @file memory_stats.h
 * @brief Memory statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryStats class for memory statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>
#include <atomic>

namespace edge_ai {

/**
 * @struct MemoryStats
 * @brief Statistics for memory manager
 */
struct MemoryStats {
    // Allocation statistics
    std::atomic<uint64_t> total_allocations{0};
    std::atomic<uint64_t> total_deallocations{0};
    std::atomic<size_t> current_memory_usage{0};
    std::atomic<size_t> peak_memory_usage{0};
    std::atomic<size_t> total_memory_allocated{0};
    
    // Performance statistics
    std::atomic<std::chrono::microseconds> total_allocation_time{0};
    std::atomic<std::chrono::microseconds> average_allocation_time{0};
    std::atomic<std::chrono::microseconds> total_deallocation_time{0};
    std::atomic<std::chrono::microseconds> average_deallocation_time{0};
    
    // Pool statistics
    std::atomic<uint64_t> pool_allocations{0};
    std::atomic<uint64_t> pool_deallocations{0};
    std::atomic<double> average_pool_utilization{0.0};
    
    MemoryStats() = default;
};

} // namespace edge_ai
