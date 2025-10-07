/**
 * @file scheduler_stats.h
 * @brief Scheduler statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the SchedulerStats class for scheduler statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>
#include <atomic>

namespace edge_ai {

/**
 * @struct SchedulerStats
 * @brief Statistics for the runtime scheduler
 */
struct SchedulerStats {
    // Task statistics
    std::atomic<uint64_t> total_tasks_submitted{0};
    std::atomic<uint64_t> total_tasks_completed{0};
    std::atomic<uint64_t> total_tasks_failed{0};
    std::atomic<uint64_t> total_tasks_cancelled{0};
    
    // Performance statistics
    std::atomic<std::chrono::microseconds> total_execution_time{0};
    std::atomic<std::chrono::microseconds> average_execution_time{0};
    std::atomic<std::chrono::microseconds> max_execution_time{0};
    std::atomic<std::chrono::microseconds> min_execution_time{std::chrono::microseconds::max()};
    
    // Queue statistics
    std::atomic<size_t> current_queue_size{0};
    std::atomic<size_t> max_queue_size{0};
    std::atomic<std::chrono::microseconds> average_queue_wait_time{0};
    
    // Device statistics
    std::atomic<uint64_t> total_device_utilization{0};
    std::atomic<double> average_device_utilization{0.0};
    
    SchedulerStats() = default;
};

} // namespace edge_ai
