/**
 * @file graph_optimization_config.h
 * @brief Graph optimization configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the GraphOptimizationConfig class for graph optimization configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <string>

namespace edge_ai {

/**
 * @struct GraphOptimizationConfig
 * @brief Configuration for graph optimization
 */
struct GraphOptimizationConfig {
    // Optimization passes
    bool enable_operator_fusion = true;
    bool enable_constant_folding = true;
    bool enable_dead_code_elimination = true;
    bool enable_algebraic_simplification = true;
    
    // Memory optimization
    bool enable_memory_optimization = true;
    bool enable_inplace_operations = true;
    
    // Hardware-specific optimization
    bool enable_hardware_optimization = true;
    std::string target_hardware;
    
    GraphOptimizationConfig() = default;
};

} // namespace edge_ai
