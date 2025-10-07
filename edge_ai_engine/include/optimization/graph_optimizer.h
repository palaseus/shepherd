/**
 * @file graph_optimizer.h
 * @brief Graph optimization interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the GraphOptimizer class for graph optimization in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>

namespace edge_ai {

// Forward declarations
class Model;

/**
 * @class GraphOptimizer
 * @brief Graph optimization utility class
 * 
 * The GraphOptimizer class provides functionality for optimizing model graphs
 * to improve inference performance.
 */
class GraphOptimizer {
public:
    /**
     * @brief Constructor
     */
    GraphOptimizer();
    
    /**
     * @brief Destructor
     */
    ~GraphOptimizer();
    
    // Disable copy constructor and assignment operator
    GraphOptimizer(const GraphOptimizer&) = delete;
    GraphOptimizer& operator=(const GraphOptimizer&) = delete;
    
    /**
     * @brief Optimize model graph
     * @param input_model Input model to optimize
     * @param output_model Optimized model (will be populated)
     * @param config Graph optimization configuration
     * @return Status indicating success or failure
     */
    Status OptimizeGraph(std::shared_ptr<Model> input_model,
                        std::shared_ptr<Model>& output_model,
                        const GraphOptimizationConfig& config);
};

} // namespace edge_ai
