/**
 * @file optimization_stats.h
 * @brief Optimization statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the OptimizationStats class for optimization statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"

namespace edge_ai {

/**
 * @struct OptimizationStats
 * @brief Statistics for optimization operations
 */
struct OptimizationStats {
    // Operation statistics
    uint64_t total_optimizations = 0;
    uint64_t successful_optimizations = 0;
    uint64_t failed_optimizations = 0;
    
    // Quantization statistics
    uint64_t quantization_operations = 0;
    double average_quantization_time = 0.0;
    double average_accuracy_loss = 0.0;
    
    // Pruning statistics
    uint64_t pruning_operations = 0;
    double average_pruning_time = 0.0;
    double average_compression_ratio = 0.0;
    
    // Graph optimization statistics
    uint64_t graph_optimizations = 0;
    double average_optimization_time = 0.0;
    double average_speedup = 0.0;
    
    OptimizationStats() = default;
};

} // namespace edge_ai
