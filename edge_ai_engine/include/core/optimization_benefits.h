/**
 * @file optimization_benefits.h
 * @brief Optimization benefits interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the OptimizationBenefits class for optimization benefits in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <vector>
#include <string>

namespace edge_ai {

/**
 * @struct OptimizationBenefits
 * @brief Estimation of optimization benefits
 */
struct OptimizationBenefits {
    // Size reduction
    double model_size_reduction = 0.0;
    size_t original_size = 0;
    size_t optimized_size = 0;
    
    // Performance improvement
    double inference_speedup = 0.0;
    double memory_reduction = 0.0;
    double power_reduction = 0.0;
    
    // Accuracy impact
    double accuracy_loss = 0.0;
    bool accuracy_acceptable = true;
    
    // Hardware compatibility
    bool hardware_compatible = true;
    std::vector<std::string> compatible_hardware;
    
    OptimizationBenefits() = default;
};

} // namespace edge_ai
