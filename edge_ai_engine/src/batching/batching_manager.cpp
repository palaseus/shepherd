/**
 * @file batching_manager.cpp
 * @brief Dynamic batching and request management system implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "batching/batching_manager.h"
#include "profiling/profiler.h"
#include <stdexcept>

namespace edge_ai {

BatchingManager::BatchingManager(const BatchingConfig& config)
    : config_(config)
    , initialized_(false)
    , dynamic_batching_enabled_(true)
    , profiling_enabled_(false)
    , shutdown_requested_(false)
    , current_batch_size_(0) {
}

BatchingManager::~BatchingManager() {
    Shutdown();
}

Status BatchingManager::Initialize() {
    try {
        if (initialized_) {
            return Status::ALREADY_INITIALIZED;
        }
        
        // Start batch processor thread
        if (dynamic_batching_enabled_) {
            batch_processor_thread_ = std::thread(&BatchingManager::BatchProcessorThread, this);
        }
        
        initialized_ = true;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status BatchingManager::Shutdown() {
    try {
        if (!initialized_) {
            return Status::SUCCESS;
        }
        
        shutdown_requested_ = true;
        queue_cv_.notify_all();
        
        // Wait for batch processor thread
        if (batch_processor_thread_.joinable()) {
            batch_processor_thread_.join();
        }
        
        Cleanup();
        initialized_ = false;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status BatchingManager::SubmitRequest(std::shared_ptr<InferenceRequest> request) {
    try {
        if (!initialized_) {
            return Status::NOT_INITIALIZED;
        }
        
        if (!request) {
            return Status::INVALID_ARGUMENT;
        }
        
        // Update statistics
        stats_.total_requests_batched.fetch_add(1);
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            request_queue_.push(request);
        }
        
        queue_cv_.notify_one();
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status BatchingManager::SubmitRequest(std::shared_ptr<InferenceRequest> request,
                                      [[maybe_unused]] std::function<void(Status, std::shared_ptr<InferenceResult>)> callback) {
    // For now, just submit the request normally
    // In practice, this would store the callback and call it when the request completes
    return SubmitRequest(request);
}

std::vector<std::shared_ptr<InferenceRequest>> BatchingManager::ProcessBatch(size_t batch_size,
                                                                           std::chrono::milliseconds timeout) {
    try {
        std::vector<std::shared_ptr<InferenceRequest>> batch;
        
        if (batch_size == 0) {
            return batch;
        }
        
        // Form batch from pending requests
        batch = FormBatch(batch_size, timeout);
        
        // Validate batch
        if (ValidateBatch(batch) != Status::SUCCESS) {
            batch.clear();
        }
        
        return batch;
    } catch (const std::exception& e) {
        return {};
    }
}

Status BatchingManager::CompleteBatch(const std::vector<std::shared_ptr<InferenceRequest>>& requests,
                                     const std::vector<std::shared_ptr<InferenceResult>>& results) {
    try {
        if (requests.size() != results.size()) {
            return Status::INVALID_ARGUMENT;
        }
        
        // Process results and update statistics
        for (size_t i = 0; i < requests.size(); ++i) {
            auto request = requests[i];
            auto result = results[i];
            
            if (request && result) {
                // Update individual request statistics
                if (result->status == Status::SUCCESS) {
                    stats_.total_batches_processed.fetch_add(1);
                } else {
                    stats_.total_batches_failed.fetch_add(1);
                }
            }
        }
        
        // Update batch-level statistics
        UpdateBatchingStats(requests.size(), std::chrono::microseconds(1000), true);
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

BatchingStats::Snapshot BatchingManager::GetBatchingStats() const {
    return stats_.GetSnapshot();
}

Status BatchingManager::SetBatchingConfig(const BatchingConfig& config) {
    try {
        config_ = config;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

BatchingConfig BatchingManager::GetBatchingConfig() const {
    return config_;
}

Status BatchingManager::SetDynamicBatching(bool enable) {
    dynamic_batching_enabled_ = enable;
    return Status::SUCCESS;
}

bool BatchingManager::IsDynamicBatchingEnabled() const {
    return dynamic_batching_enabled_;
}

Status BatchingManager::SetBatchSize(size_t batch_size) {
    if (batch_size == 0) {
        return Status::INVALID_ARGUMENT;
    }
    
    current_batch_size_ = batch_size;
    return Status::SUCCESS;
}

size_t BatchingManager::GetBatchSize() const {
    return current_batch_size_;
}

Status BatchingManager::SetBatchTimeout(std::chrono::milliseconds timeout) {
    config_.batch_timeout = timeout;
    return Status::SUCCESS;
}

std::chrono::milliseconds BatchingManager::GetBatchTimeout() const {
    return config_.batch_timeout;
}

size_t BatchingManager::GetQueueSize() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queue_mutex_));
    return request_queue_.size();
}

Status BatchingManager::ClearPendingRequests() {
    try {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!request_queue_.empty()) {
            request_queue_.pop();
        }
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

void BatchingManager::SetProfiling(bool enable) {
    profiling_enabled_ = enable;
}

bool BatchingManager::IsProfilingEnabled() const {
    return profiling_enabled_;
}

// Private methods
void BatchingManager::BatchProcessorThread() {
    while (!shutdown_requested_) {
        try {
            // Process batches
            auto batch = ProcessBatch(config_.max_batch_size, config_.batch_timeout);
            
            if (!batch.empty()) {
                // Process the batch with profiler tracking
                PROFILER_SCOPED_EVENT(0, "batch_execute");
                
                // This would typically call the inference engine
                
                // Update statistics
                UpdateBatchingStats(batch.size(), std::chrono::microseconds(1000), true);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } catch (const std::exception& e) {
            // Log error and continue
        }
    }
}

std::vector<std::shared_ptr<InferenceRequest>> BatchingManager::FormBatch(size_t max_batch_size,
                                                                         std::chrono::milliseconds timeout) {
    std::vector<std::shared_ptr<InferenceRequest>> batch;
    
    try {
        // Mark batch formation start
        PROFILER_MARK_EVENT(0, "batch_form_start");
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (batch.size() < max_batch_size) {
            std::shared_ptr<InferenceRequest> request;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                
                if (queue_cv_.wait_for(lock, timeout - elapsed, [this] { return !request_queue_.empty(); })) {
                    if (!request_queue_.empty()) {
                        request = request_queue_.front();
                        request_queue_.pop();
                    }
                } else {
                    // Timeout reached
                    break;
                }
            }
            
            if (request) {
                // Check if request is compatible with current batch
                if (IsRequestCompatible(request, batch)) {
                    batch.push_back(request);
                } else {
                    // If not compatible and batch is not empty, return current batch
                    if (!batch.empty()) {
                        // Put request back in queue for next batch
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        request_queue_.push(request);
                        break;
                    } else {
                        // If batch is empty, accept the request anyway
                        batch.push_back(request);
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        // Return empty batch on error
    }
    
    // Mark batch formation end
    PROFILER_MARK_EVENT(0, "batch_form_end");
    
    return batch;
}

Status BatchingManager::ValidateBatch(const std::vector<std::shared_ptr<InferenceRequest>>& requests) const {
    try {
        if (requests.empty()) {
            return Status::INVALID_ARGUMENT;
        }
        
        // Validate each request in the batch
        for (const auto& request : requests) {
            if (!request) {
                return Status::INVALID_ARGUMENT;
            }
            
            // Check for timeout (placeholder logic)
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - request->timestamp);
            if (elapsed > std::chrono::milliseconds(10000)) { // 10 second timeout
                return Status::FAILURE;
            }
        }
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

void BatchingManager::UpdateBatchingStats(size_t batch_size, std::chrono::microseconds batch_time, bool success) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (success) {
        stats_.total_batches_formed.fetch_add(1);
        stats_.total_batches_processed.fetch_add(1);
        auto current_avg = stats_.average_batch_size.load();
        stats_.average_batch_size.store((current_avg + batch_size) / 2);
        auto current_max = stats_.max_batch_size.load();
        if (batch_size > current_max) {
            stats_.max_batch_size.store(batch_size);
        }
        auto current_min = stats_.min_batch_size.load();
        if (batch_size < current_min) {
            stats_.min_batch_size.store(batch_size);
        }
    } else {
        stats_.total_batches_failed.fetch_add(1);
    }
    
    auto current_total = stats_.total_latency.load();
    stats_.total_latency.store(current_total + batch_time);
    auto total_batches = stats_.total_batches_formed.load();
    if (total_batches > 0) {
        stats_.average_latency.store(std::chrono::microseconds(
            stats_.total_latency.load().count() / total_batches));
    }
    auto current_min = stats_.min_latency.load();
    if (batch_time < current_min) {
        stats_.min_latency.store(batch_time);
    }
    auto current_max = stats_.max_latency.load();
    if (batch_time > current_max) {
        stats_.max_latency.store(batch_time);
    }
}

bool BatchingManager::IsRequestCompatible(const std::shared_ptr<InferenceRequest>& request,
                                         const std::vector<std::shared_ptr<InferenceRequest>>& batch) const {
    if (!request || batch.empty()) {
        return true; // Empty batch or null request is always compatible
    }
    
    // Get the first request in the batch as reference
    auto reference_request = batch[0];
    if (!reference_request) {
        return true;
    }
    
    // Check input tensor shape compatibility
    if (request->inputs.size() != reference_request->inputs.size()) {
        return false;
    }
    
    for (size_t i = 0; i < request->inputs.size(); ++i) {
        // Compare tensor shapes by dimensions
        const auto& shape1 = request->inputs[i].GetShape();
        const auto& shape2 = reference_request->inputs[i].GetShape();
        if (shape1.dimensions != shape2.dimensions) {
            return false;
        }
        if (request->inputs[i].GetDataType() != reference_request->inputs[i].GetDataType()) {
            return false;
        }
    }
    
    // Check priority compatibility (same priority level)
    if (request->priority != reference_request->priority) {
        return false;
    }
    
    return true;
}

void BatchingManager::Cleanup() {
    // Clear request queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!request_queue_.empty()) {
            request_queue_.pop();
        }
    }
    
    // Clear pending requests
    {
        std::lock_guard<std::mutex> lock(pending_requests_mutex_);
        pending_requests_.clear();
    }
}

// InferenceRequest and InferenceResult implementations have been moved to types.h as structs

} // namespace edge_ai