/**
 * @file optimizer.h
 * @brief Model optimization interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Optimizer class which handles model optimization
 * including quantization, pruning, and graph optimization.
 */

#pragma once

#include "../core/types.h"
#include <memory>
#include <vector>
#include <string>

namespace edge_ai {

// Forward declarations
class Model;
class Quantizer;
class Pruner;
class GraphOptimizer;

/**
 * @class Optimizer
 * @brief Handles model optimization including quantization, pruning, and graph optimization
 * 
 * The Optimizer class provides a unified interface for various model optimization
 * techniques to improve inference performance and reduce model size.
 */
class Optimizer {
public:
    /**
     * @brief Constructor
     * @param config Optimization configuration
     */
    explicit Optimizer(const OptimizationConfig& config = OptimizationConfig{});
    
    /**
     * @brief Destructor
     */
    ~Optimizer();
    
    // Disable copy constructor and assignment operator
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;
    
    /**
     * @brief Initialize the optimizer
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the optimizer
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Optimize a model
     * @param input_model Input model to optimize
     * @param output_model Optimized model (will be populated)
     * @param config Optimization configuration
     * @return Status indicating success or failure
     */
    Status OptimizeModel(std::shared_ptr<Model> input_model,
                        std::shared_ptr<Model>& output_model,
                        const OptimizationConfig& config);
    
    /**
     * @brief Quantize a model
     * @param input_model Input model to quantize
     * @param output_model Quantized model (will be populated)
     * @param quantization_config Quantization configuration
     * @return Status indicating success or failure
     */
    Status QuantizeModel(std::shared_ptr<Model> input_model,
                        std::shared_ptr<Model>& output_model,
                        const QuantizationConfig& quantization_config);
    
    /**
     * @brief Prune a model
     * @param input_model Input model to prune
     * @param output_model Pruned model (will be populated)
     * @param pruning_config Pruning configuration
     * @return Status indicating success or failure
     */
    Status PruneModel(std::shared_ptr<Model> input_model,
                     std::shared_ptr<Model>& output_model,
                     const PruningConfig& pruning_config);
    
    /**
     * @brief Optimize model graph
     * @param input_model Input model to optimize
     * @param output_model Optimized model (will be populated)
     * @param graph_config Graph optimization configuration
     * @return Status indicating success or failure
     */
    Status OptimizeGraph(std::shared_ptr<Model> input_model,
                        std::shared_ptr<Model>& output_model,
                        const GraphOptimizationConfig& graph_config);
    
            /**
             * @brief Get optimization statistics
             * @return Optimization statistics
             */
            OptimizationStats::Snapshot GetOptimizationStats() const;
    
    /**
     * @brief Set optimization configuration
     * @param config Optimization configuration
     * @return Status indicating success or failure
     */
    Status SetOptimizationConfig(const OptimizationConfig& config);
    
    /**
     * @brief Get current optimization configuration
     * @return Current optimization configuration
     */
    OptimizationConfig GetOptimizationConfig() const;
    
    /**
     * @brief Validate optimization configuration
     * @param config Configuration to validate
     * @return Status indicating success or failure
     */
    Status ValidateOptimizationConfig(const OptimizationConfig& config) const;
    
    /**
     * @brief Get supported optimization types
     * @return Vector of supported optimization types
     */
    std::vector<OptimizationType> GetSupportedOptimizationTypes() const;
    
    /**
     * @brief Estimate optimization benefits
     * @param model Model to analyze
     * @param config Optimization configuration
     * @return Optimization benefits estimation
     */
    OptimizationBenefits EstimateOptimizationBenefits(std::shared_ptr<Model> model,
                                                     const OptimizationConfig& config) const;

private:
    // Core components
    std::unique_ptr<Quantizer> quantizer_;
    std::unique_ptr<Pruner> pruner_;
    std::unique_ptr<GraphOptimizer> graph_optimizer_;
    
    // Configuration
    OptimizationConfig config_;
    
    // State
    bool initialized_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    OptimizationStats stats_;
    
    /**
     * @brief Initialize optimization components
     * @return Status indicating success or failure
     */
    Status InitializeComponents();
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};


} // namespace edge_ai
