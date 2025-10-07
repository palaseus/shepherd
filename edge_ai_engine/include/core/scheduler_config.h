/**
 * @file scheduler_config.h
 * @brief Scheduler configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the SchedulerConfig class for scheduler configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>

namespace edge_ai {

/**
 * @struct SchedulerConfig
 * @brief Configuration for the runtime scheduler
 */
struct SchedulerConfig {
    // Threading configuration
    int num_worker_threads = 0; // 0 = auto
    bool enable_work_stealing = true;
    int max_queue_size = 1000;
    
    // Task configuration
    std::chrono::milliseconds task_timeout{30000}; // 30 seconds
    bool enable_task_prioritization = true;
    bool enable_task_affinity = true;
    
    // Performance configuration
    bool enable_profiling = false;
    bool enable_optimization = true;
    
    // Device configuration
    bool enable_device_load_balancing = true;
    bool enable_device_failover = true;
    
    SchedulerConfig() = default;
};

} // namespace edge_ai
