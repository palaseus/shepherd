/**
 * @file inference_engine.h
 * @brief Core inference execution engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the InferenceEngine class which handles the execution
 * of AI model inference with hardware acceleration and optimization.
 */

#pragma once

#include "types.h"
#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace edge_ai {

// Forward declarations
class Model;
class Device;
class RuntimeScheduler;
class MemoryManager;
class Profiler;
class BatchingManager;

/**
 * @class InferenceEngine
 * @brief Core inference execution engine
 * 
 * The InferenceEngine class handles the execution of AI model inference,
 * including hardware acceleration, memory management, and performance optimization.
 */
class InferenceEngine {
public:
    /**
     * @brief Constructor
     * @param device Target device for inference
     * @param scheduler Runtime scheduler for task management
     * @param memory_manager Memory manager for tensor allocation
     * @param profiler Profiler for performance monitoring
     */
    InferenceEngine(std::shared_ptr<Device> device,
                   std::shared_ptr<RuntimeScheduler> scheduler,
                   std::shared_ptr<MemoryManager> memory_manager,
                   Profiler* profiler);
    
    /**
     * @brief Destructor
     */
    ~InferenceEngine();
    
    // Disable copy constructor and assignment operator
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    
    /**
     * @brief Initialize the inference engine
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the inference engine
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Set the model for inference
     * @param model Model to use for inference
     * @return Status indicating success or failure
     */
    Status SetModel(std::shared_ptr<Model> model);
    
    /**
     * @brief Run synchronous inference
     * @param inputs Input tensors
     * @param outputs Output tensors (will be populated)
     * @return Status indicating success or failure
     */
    Status RunInference(std::vector<Tensor> inputs, std::vector<Tensor>& outputs);
    
    /**
     * @brief Run asynchronous inference
     * @param inputs Input tensors
     * @param callback Callback function to handle results
     * @return Status indicating success or failure
     */
    Status RunInferenceAsync(std::vector<Tensor> inputs,
                           std::function<void(Status, std::vector<Tensor>)> callback);
    
    /**
     * @brief Run batch inference
     * @param batch_inputs Batch of input tensors
     * @param batch_outputs Batch of output tensors (will be populated)
     * @return Status indicating success or failure
     */
    Status RunBatchInference(std::vector<std::vector<Tensor>> batch_inputs,
                           std::vector<std::vector<Tensor>>& batch_outputs);
    
            /**
             * @brief Get inference statistics
             * @return Inference statistics
             */
            InferenceStats::Snapshot GetInferenceStats() const;
    
    /**
     * @brief Set inference configuration
     * @param config Inference configuration
     * @return Status indicating success or failure
     */
    Status SetInferenceConfig(const InferenceConfig& config);
    
    /**
     * @brief Get current inference configuration
     * @return Current inference configuration
     */
    InferenceConfig GetInferenceConfig() const;
    
    /**
     * @brief Enable or disable profiling
     * @param enable Enable profiling
     */
    void SetProfiling(bool enable);
    
    /**
     * @brief Check if profiling is enabled
     * @return True if profiling is enabled
     */
    bool IsProfilingEnabled() const;

private:
    // Core components
    std::shared_ptr<Device> device_;
    std::shared_ptr<RuntimeScheduler> scheduler_;
    std::shared_ptr<MemoryManager> memory_manager_;
    Profiler* profiler_;
    
    // Model
    std::shared_ptr<Model> model_;
    
    // Configuration
    InferenceConfig config_;
    
    // State
    bool initialized_;
    bool model_set_;
    bool profiling_enabled_;
    
    // Threading
    std::thread worker_thread_;
    std::queue<InferenceRequest> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_requested_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    InferenceStats stats_;
    
    /**
     * @brief Worker thread function for async inference
     */
    void WorkerThread();
    
    /**
     * @brief Process a single inference request
     * @param request Inference request to process
     * @return Inference result
     */
    InferenceResult ProcessInference(const InferenceRequest& request);
    
    /**
     * @brief Generate a unique request ID
     * @return Unique request identifier
     */
    std::string GenerateRequestId();
    
    /**
     * @brief Execute inference on device
     * @param inputs Input tensors
     * @param outputs Output tensors (will be populated)
     * @return Status indicating success or failure
     */
    Status ExecuteInference(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
    
    /**
     * @brief Validate input tensors
     * @param inputs Input tensors to validate
     * @return Status indicating success or failure
     */
    Status ValidateInputs(const std::vector<Tensor>& inputs) const;
    
    /**
     * @brief Prepare output tensors
     * @param outputs Output tensors to prepare
     * @return Status indicating success or failure
     */
    Status PrepareOutputs(std::vector<Tensor>& outputs) const;
    
    /**
     * @brief Update inference statistics
     * @param latency Inference latency
     * @param success Whether inference was successful
     */
    void UpdateStats(std::chrono::microseconds latency, bool success);
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};


} // namespace edge_ai
