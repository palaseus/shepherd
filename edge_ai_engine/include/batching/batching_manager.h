/**
 * @file batching_manager.h
 * @brief Dynamic batching and request management system
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the BatchingManager class which handles dynamic batching,
 * request queuing, and latency optimization for the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>
#include <unordered_map>

namespace edge_ai {

// Forward declarations
class InferenceRequest;
class InferenceResult;
class Profiler;

/**
 * @class BatchingManager
 * @brief Dynamic batching and request management system
 * 
 * The BatchingManager class handles dynamic batching of inference requests,
 * request queuing, and latency optimization for the Edge AI Engine.
 */
class BatchingManager {
public:
    /**
     * @brief Constructor
     * @param config Batching manager configuration
     */
    explicit BatchingManager(const BatchingConfig& config = BatchingConfig{});
    
    /**
     * @brief Destructor
     */
    ~BatchingManager();
    
    // Disable copy constructor and assignment operator
    BatchingManager(const BatchingManager&) = delete;
    BatchingManager& operator=(const BatchingManager&) = delete;
    
    /**
     * @brief Initialize the batching manager
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the batching manager
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Submit an inference request
     * @param request Inference request to submit
     * @return Status indicating success or failure
     */
    Status SubmitRequest(std::shared_ptr<InferenceRequest> request);
    
    /**
     * @brief Submit an inference request with callback
     * @param request Inference request to submit
     * @param callback Callback function for results
     * @return Status indicating success or failure
     */
    Status SubmitRequest(std::shared_ptr<InferenceRequest> request,
                        std::function<void(Status, std::shared_ptr<InferenceResult>)> callback);
    
    /**
     * @brief Process a batch of requests
     * @param batch_size Maximum batch size
     * @param timeout Maximum time to wait for batch formation
     * @return Vector of requests in the batch
     */
    std::vector<std::shared_ptr<InferenceRequest>> ProcessBatch(size_t batch_size,
                                                               std::chrono::milliseconds timeout);
    
    /**
     * @brief Complete a batch of requests
     * @param requests Requests in the batch
     * @param results Results for the requests
     * @return Status indicating success or failure
     */
    Status CompleteBatch(const std::vector<std::shared_ptr<InferenceRequest>>& requests,
                        const std::vector<std::shared_ptr<InferenceResult>>& results);
    
            /**
             * @brief Get batching statistics
             * @return Batching statistics
             */
            BatchingStats::Snapshot GetBatchingStats() const;
    
    /**
     * @brief Set batching configuration
     * @param config Batching configuration
     * @return Status indicating success or failure
     */
    Status SetBatchingConfig(const BatchingConfig& config);
    
    /**
     * @brief Get current batching configuration
     * @return Current batching configuration
     */
    BatchingConfig GetBatchingConfig() const;
    
    /**
     * @brief Enable or disable dynamic batching
     * @param enable Enable dynamic batching
     * @return Status indicating success or failure
     */
    Status SetDynamicBatching(bool enable);
    
    /**
     * @brief Check if dynamic batching is enabled
     * @return True if dynamic batching is enabled
     */
    bool IsDynamicBatchingEnabled() const;
    
    /**
     * @brief Set batch size
     * @param batch_size New batch size
     * @return Status indicating success or failure
     */
    Status SetBatchSize(size_t batch_size);
    
    /**
     * @brief Get current batch size
     * @return Current batch size
     */
    size_t GetBatchSize() const;
    
    /**
     * @brief Set batch timeout
     * @param timeout New batch timeout
     * @return Status indicating success or failure
     */
    Status SetBatchTimeout(std::chrono::milliseconds timeout);
    
    /**
     * @brief Get current batch timeout
     * @return Current batch timeout
     */
    std::chrono::milliseconds GetBatchTimeout() const;
    
    /**
     * @brief Get current queue size
     * @return Current queue size
     */
    size_t GetQueueSize() const;
    
    /**
     * @brief Clear all pending requests
     * @return Status indicating success or failure
     */
    Status ClearPendingRequests();
    
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
    // Configuration
    BatchingConfig config_;
    
    // State
    bool initialized_;
    bool dynamic_batching_enabled_;
    bool profiling_enabled_;
    std::atomic<bool> shutdown_requested_;
    
    // Request queue
    std::queue<std::shared_ptr<InferenceRequest>> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Batch processing
    std::thread batch_processor_thread_;
    std::atomic<size_t> current_batch_size_;
    std::chrono::steady_clock::time_point last_batch_time_;
    
    // Request tracking
    std::unordered_map<uint64_t, std::shared_ptr<InferenceRequest>> pending_requests_;
    std::mutex pending_requests_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    BatchingStats stats_;
    
    /**
     * @brief Batch processor thread function
     */
    void BatchProcessorThread();
    
    /**
     * @brief Form a batch from pending requests
     * @param max_batch_size Maximum batch size
     * @param timeout Maximum time to wait
     * @return Vector of requests in the batch
     */
    std::vector<std::shared_ptr<InferenceRequest>> FormBatch(size_t max_batch_size,
                                                            std::chrono::milliseconds timeout);
    
    /**
     * @brief Validate batch
     * @param requests Requests in the batch
     * @return Status indicating success or failure
     */
    Status ValidateBatch(const std::vector<std::shared_ptr<InferenceRequest>>& requests) const;
    
    /**
     * @brief Update batching statistics
     * @param batch_size Size of the batch
     * @param batch_time Time to form the batch
     * @param success Whether batch processing was successful
     */
    void UpdateBatchingStats(size_t batch_size, std::chrono::microseconds batch_time, bool success);
    
    /**
     * @brief Check if a request is compatible with existing batch
     * @param request Request to check
     * @param batch Existing batch
     * @return True if compatible, false otherwise
     */
    bool IsRequestCompatible(const std::shared_ptr<InferenceRequest>& request,
                           const std::vector<std::shared_ptr<InferenceRequest>>& batch) const;
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};


} // namespace edge_ai
