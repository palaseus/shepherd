/**
 * @file pruning_config.h
 * @brief Pruning configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the PruningConfig class for pruning configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <vector>

namespace edge_ai {

/**
 * @enum PruningStrategy
 * @brief Pruning strategies
 */
enum class PruningStrategy {
    MAGNITUDE_BASED = 0,
    GRADIENT_BASED = 1,
    ACTIVATION_BASED = 2,
    RANDOM = 3
};

/**
 * @struct PruningConfig
 * @brief Configuration for model pruning
 */
struct PruningConfig {
    // Pruning type
    bool structured_pruning = true;
    bool unstructured_pruning = false;
    
    // Pruning ratio
    float pruning_ratio = 0.1f;
    std::vector<float> layer_pruning_ratios;
    
    // Pruning strategy
    PruningStrategy strategy = PruningStrategy::MAGNITUDE_BASED;
    bool iterative_pruning = true;
    int pruning_iterations = 3;
    
    // Recovery configuration
    bool enable_recovery = true;
    float recovery_learning_rate = 0.001f;
    int recovery_epochs = 10;
    
    PruningConfig() = default;
};

} // namespace edge_ai
