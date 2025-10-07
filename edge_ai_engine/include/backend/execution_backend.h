/**
 * @file execution_backend.h
 * @brief Execution backend interface for different hardware platforms
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ExecutionBackend interface which provides a unified
 * interface for executing inference on different hardware platforms.
 */

#pragma once

#include "../core/types.h"
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace edge_ai {

// Forward declarations
class Model;
class InferenceRequest;
class InferenceResult;
class Device;

/**
 * @enum BackendType
 * @brief Types of execution backends
 */
enum class BackendType {
    CPU = 0,
    GPU = 1,
    NPU = 2,
    TPU = 3,
    FPGA = 4
};

/**
 * @struct BackendCapabilities
 * @brief Capabilities of an execution backend
 */
struct BackendCapabilities {
    bool supports_batching{false};
    bool supports_quantization{false};
    bool supports_pruning{false};
    size_t max_batch_size{1};
    size_t max_memory_usage{0};
    std::vector<DataType> supported_data_types;
    std::vector<ModelType> supported_model_types;
    
    BackendCapabilities() = default;
};

/**
 * @struct BackendStats
 * @brief Statistics for execution backend
 */
struct BackendStats {
    std::atomic<uint64_t> total_executions{0};
    std::atomic<uint64_t> successful_executions{0};
    std::atomic<uint64_t> failed_executions{0};
    std::atomic<std::chrono::microseconds> total_execution_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> average_execution_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> max_execution_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> min_execution_time{std::chrono::microseconds::max()};
    std::atomic<size_t> total_memory_allocated{0};
    std::atomic<size_t> peak_memory_usage{0};
    
    BackendStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t total_executions;
        uint64_t successful_executions;
        uint64_t failed_executions;
        std::chrono::microseconds total_execution_time;
        std::chrono::microseconds average_execution_time;
        std::chrono::microseconds max_execution_time;
        std::chrono::microseconds min_execution_time;
        size_t total_memory_allocated;
        size_t peak_memory_usage;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_executions = total_executions.load();
        snapshot.successful_executions = successful_executions.load();
        snapshot.failed_executions = failed_executions.load();
        snapshot.total_execution_time = total_execution_time.load();
        snapshot.average_execution_time = average_execution_time.load();
        snapshot.max_execution_time = max_execution_time.load();
        snapshot.min_execution_time = min_execution_time.load();
        snapshot.total_memory_allocated = total_memory_allocated.load();
        snapshot.peak_memory_usage = peak_memory_usage.load();
        return snapshot;
    }
};

/**
 * @class ExecutionBackend
 * @brief Abstract base class for execution backends
 * 
 * The ExecutionBackend class provides a unified interface for executing
 * inference on different hardware platforms (CPU, GPU, NPU, etc.).
 */
class ExecutionBackend {
public:
    /**
     * @brief Constructor
     * @param backend_type Type of backend
     * @param device Device to use for execution
     */
    ExecutionBackend(BackendType backend_type, std::shared_ptr<Device> device);
    
    /**
     * @brief Destructor
     */
    virtual ~ExecutionBackend() = default;
    
    // Disable copy constructor and assignment operator
    ExecutionBackend(const ExecutionBackend&) = delete;
    ExecutionBackend& operator=(const ExecutionBackend&) = delete;
    
    /**
     * @brief Initialize the backend
     * @return Status indicating success or failure
     */
    virtual Status Initialize() = 0;
    
    /**
     * @brief Shutdown the backend
     * @return Status indicating success or failure
     */
    virtual Status Shutdown() = 0;
    
    /**
     * @brief Execute inference on a single request
     * @param model Model to execute
     * @param request Inference request
     * @param result Inference result
     * @return Status indicating success or failure
     */
    virtual Status Execute(const Model& model, 
                          const InferenceRequest& request, 
                          InferenceResult& result) = 0;
    
    /**
     * @brief Execute inference on a batch of requests
     * @param model Model to execute
     * @param requests Batch of inference requests
     * @param results Batch of inference results
     * @return Status indicating success or failure
     */
    virtual Status ExecuteBatch(const Model& model,
                               const std::vector<InferenceRequest>& requests,
                               std::vector<InferenceResult>& results) = 0;
    
    /**
     * @brief Get backend type
     * @return Backend type
     */
    BackendType GetBackendType() const { return backend_type_; }
    
    /**
     * @brief Get the unique identifier for this backend
     * @return Backend identifier
     */
    virtual std::string GetId() const = 0;
    
    /**
     * @brief Get backend capabilities
     * @return Backend capabilities
     */
    virtual BackendCapabilities GetCapabilities() const = 0;
    
    /**
     * @brief Get backend statistics
     * @return Backend statistics snapshot
     */
    BackendStats::Snapshot GetStats() const;
    
    /**
     * @brief Check if backend supports a specific model type
     * @param model_type Model type to check
     * @return True if supported, false otherwise
     */
    virtual bool SupportsModelType(ModelType model_type) const = 0;
    
    /**
     * @brief Check if backend supports a specific data type
     * @param data_type Data type to check
     * @return True if supported, false otherwise
     */
    virtual bool SupportsDataType(DataType data_type) const = 0;
    
    /**
     * @brief Get backend name
     * @return Backend name
     */
    virtual std::string GetName() const = 0;
    
    /**
     * @brief Get backend version
     * @return Backend version
     */
    virtual std::string GetVersion() const = 0;

protected:
    BackendType backend_type_;
    std::shared_ptr<Device> device_;
    BackendStats stats_;
    bool initialized_;
    
    /**
     * @brief Update execution statistics
     * @param execution_time Time taken for execution
     * @param success Whether execution was successful
     * @param memory_used Memory used during execution
     */
    void UpdateStats(std::chrono::microseconds execution_time, bool success, size_t memory_used = 0);
};

} // namespace edge_ai
