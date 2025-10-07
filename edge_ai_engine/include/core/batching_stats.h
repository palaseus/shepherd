/**
 * @file batching_stats.h
 * @brief Batching statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the BatchingStats class for batching statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <chrono>
#include <atomic>

namespace edge_ai {

/**
 * @struct BatchingStats
 * @brief Statistics for batching manager
 */
struct BatchingStats {
    // Request statistics
    std::atomic<uint64_t> total_requests_submitted{0};
    std::atomic<uint64_t> total_requests_processed{0};
    std::atomic<uint64_t> total_requests_failed{0};
    std::atomic<uint64_t> total_requests_timed_out{0};
    
    // Batch statistics
    std::atomic<uint64_t> total_batches_formed{0};
    std::atomic<uint64_t> total_batches_processed{0};
    std::atomic<uint64_t> total_batches_failed{0};
    std::atomic<size_t> average_batch_size{0};
    std::atomic<size_t> max_batch_size{0};
    std::atomic<size_t> min_batch_size{0};
    
    // Latency statistics
    std::atomic<std::chrono::microseconds> total_latency{0};
    std::atomic<std::chrono::microseconds> average_latency{0};
    std::atomic<std::chrono::microseconds> min_latency{std::chrono::microseconds::max()};
    std::atomic<std::chrono::microseconds> max_latency{0};
    
    // Queue statistics
    std::atomic<size_t> current_queue_size{0};
    std::atomic<size_t> max_queue_size{0};
    std::atomic<std::chrono::microseconds> average_queue_wait_time{0};
    
    BatchingStats() = default;
};

} // namespace edge_ai
