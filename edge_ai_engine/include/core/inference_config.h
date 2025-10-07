/**
 * @file inference_config.h
 * @brief Inference configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the InferenceConfig class for inference configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>

namespace edge_ai {

/**
 * @struct InferenceConfig
 * @brief Configuration for inference execution
 */
struct InferenceConfig {
    // Execution configuration
    bool enable_async_execution = true;
    int max_concurrent_requests = 10;
    std::chrono::milliseconds request_timeout{5000};
    
    // Memory configuration
    bool enable_memory_reuse = true;
    size_t max_memory_per_request = 100 * 1024 * 1024; // 100MB
    
    // Performance configuration
    bool enable_optimization = true;
    bool enable_hardware_acceleration = true;
    
    // Profiling configuration
    bool enable_profiling = false;
    bool enable_detailed_profiling = false;
    
    InferenceConfig() = default;
};

} // namespace edge_ai
