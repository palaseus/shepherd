/**
 * @file edge_ai_engine.h
 * @brief Main Edge AI Inference Engine header
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the main Edge AI Inference Engine class and core interfaces.
 * The engine supports ONNX, TensorFlow Lite, and PyTorch Mobile models with
 * hardware acceleration, quantization, pruning, and dynamic batching.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <functional>

#include "model_loader.h"
#include "inference_engine.h"
#include "optimization/optimization_manager.h"
#include "runtime_scheduler.h"
#include "memory/memory_manager.h"
#include "batching/batching_manager.h"
#include "profiling/profiler.h"
#include "types.h"

namespace edge_ai {

/**
 * @class EdgeAIEngine
 * @brief Main Edge AI Inference Engine class
 * 
 * This is the primary interface for the Edge AI Inference Engine.
 * It orchestrates all components including model loading, optimization,
 * inference execution, and performance monitoring.
 */
class EdgeAIEngine {
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters for the engine
     */
    explicit EdgeAIEngine(const EngineConfig& config = EngineConfig{});
    
    /**
     * @brief Destructor
     */
    ~EdgeAIEngine();
    
    // Disable copy constructor and assignment operator
    EdgeAIEngine(const EdgeAIEngine&) = delete;
    EdgeAIEngine& operator=(const EdgeAIEngine&) = delete;
    
    /**
     * @brief Initialize the engine
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the engine
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Load a model from file
     * @param model_path Path to the model file
     * @param model_type Type of model (ONNX, TensorFlow Lite, PyTorch)
     * @return Status indicating success or failure
     */
    Status LoadModel(const std::string& model_path, ModelType model_type);
    
    /**
     * @brief Optimize the loaded model
     * @param optimization_config Configuration for optimization
     * @return Status indicating success or failure
     */
    Status OptimizeModel(const OptimizationConfig& optimization_config);
    
    /**
     * @brief Run inference on input data
     * @param inputs Input tensors
     * @param outputs Output tensors (will be populated)
     * @return Status indicating success or failure
     */
    Status RunInference(std::vector<Tensor> inputs, std::vector<Tensor>& outputs);
    
    /**
     * @brief Run inference asynchronously
     * @param inputs Input tensors
     * @param callback Callback function to handle results
     * @return Status indicating success or failure
     */
    Status RunInferenceAsync(std::vector<Tensor> inputs, 
                           std::function<void(Status, std::vector<Tensor>)> callback);
    
    /**
     * @brief Get engine statistics
     * @return Engine statistics
     */
    EngineStats GetStats() const;
    
    /**
     * @brief Get model information
     * @return Model information
     */
    ModelInfo GetModelInfo() const;
    
    /**
     * @brief Set performance monitoring
     * @param enable Enable or disable monitoring
     */
    void SetMonitoring(bool enable);
    
    /**
     * @brief Get performance metrics
     * @return Performance metrics
     */
    PerformanceMetrics GetMetrics() const;

private:
    // Core components
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<InferenceEngine> inference_engine_;
    std::unique_ptr<OptimizationManager> optimization_manager_;
    std::unique_ptr<RuntimeScheduler> scheduler_;
    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<BatchingManager> batching_manager_;
    Profiler* profiler_;
    
    // Configuration
    EngineConfig config_;
    
    // State
    bool initialized_;
    bool model_loaded_;
    bool optimized_;
    
    // Performance monitoring
    bool monitoring_enabled_;
    std::chrono::steady_clock::time_point start_time_;
    
    /**
     * @brief Validate configuration
     * @return Status indicating success or failure
     */
    Status ValidateConfig() const;
    
    /**
     * @brief Initialize components
     * @return Status indicating success or failure
     */
    Status InitializeComponents();
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};

} // namespace edge_ai
