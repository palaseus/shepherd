/**
 * @file batching_config.h
 * @brief Batching configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the BatchingConfig class for batching configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>

namespace edge_ai {

/**
 * @struct BatchingConfig
 * @brief Configuration for batching manager
 */
struct BatchingConfig {
    // Batching configuration
    bool enable_dynamic_batching = true;
    size_t max_batch_size = 32;
    size_t min_batch_size = 1;
    std::chrono::milliseconds batch_timeout{10};
    
    // Request configuration
    std::chrono::milliseconds request_timeout{5000};
    int max_queue_size = 1000;
    bool enable_request_prioritization = true;
    
    // Performance configuration
    bool enable_profiling = false;
    bool enable_optimization = true;
    
    // Latency configuration
    bool enable_latency_optimization = true;
    std::chrono::microseconds target_latency{10000}; // 10ms
    
    BatchingConfig() = default;
};

} // namespace edge_ai
