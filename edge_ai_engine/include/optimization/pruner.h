/**
 * @file pruner.h
 * @brief Model pruning interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Pruner class for model pruning in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>

namespace edge_ai {

// Forward declarations
class Model;

/**
 * @class Pruner
 * @brief Model pruning utility class
 * 
 * The Pruner class provides functionality for pruning models to reduce
 * their size and improve inference performance.
 */
class Pruner {
public:
    /**
     * @brief Constructor
     */
    Pruner();
    
    /**
     * @brief Destructor
     */
    ~Pruner();
    
    // Disable copy constructor and assignment operator
    Pruner(const Pruner&) = delete;
    Pruner& operator=(const Pruner&) = delete;
    
    /**
     * @brief Prune a model
     * @param input_model Input model to prune
     * @param output_model Pruned model (will be populated)
     * @param config Pruning configuration
     * @return Status indicating success or failure
     */
    Status PruneModel(std::shared_ptr<Model> input_model,
                     std::shared_ptr<Model>& output_model,
                     const PruningConfig& config);
};

} // namespace edge_ai
