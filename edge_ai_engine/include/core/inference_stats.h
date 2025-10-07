/**
 * @file inference_stats.h
 * @brief Inference statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the InferenceStats class for inference statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>
#include <atomic>

namespace edge_ai {

/**
 * @struct InferenceStats
 * @brief Statistics for inference execution
 */
struct InferenceStats {
    // Request statistics
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<uint64_t> timeout_requests{0};
    
    // Latency statistics
    std::atomic<std::chrono::microseconds> total_latency{0};
    std::atomic<std::chrono::microseconds> min_latency{std::chrono::microseconds::max()};
    std::atomic<std::chrono::microseconds> max_latency{0};
    
    // Throughput statistics
    std::atomic<double> average_throughput{0.0};
    std::atomic<double> peak_throughput{0.0};
    
    // Memory statistics
    std::atomic<size_t> peak_memory_usage{0};
    std::atomic<size_t> total_memory_allocated{0};
    
    InferenceStats() = default;
};

} // namespace edge_ai
