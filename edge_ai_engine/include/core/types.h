/**
 * @file types.h
 * @brief Core data types and structures for Edge AI Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains all the fundamental data types, enums, and structures
 * used throughout the Edge AI Inference Engine.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <cstdint>
#include <atomic>
#include <cstddef>

namespace edge_ai {

// Forward declarations
class Tensor;
class Model;
class Device;

/**
 * @enum Status
 * @brief Status codes for operations
 */
enum class Status {
    SUCCESS = 0,
    FAILURE = 1,
    INVALID_ARGUMENT = 2,
    OUT_OF_MEMORY = 3,
    NOT_IMPLEMENTED = 4,
    NOT_INITIALIZED = 5,
    ALREADY_INITIALIZED = 6,
    MODEL_NOT_LOADED = 7,
    MODEL_ALREADY_LOADED = 8,
    OPTIMIZATION_FAILED = 9,
    INFERENCE_FAILED = 10,
    HARDWARE_NOT_AVAILABLE = 11,
    INVALID_MODEL_FORMAT = 12,
    UNSUPPORTED_OPERATION = 13,
    NOT_FOUND = 14,
    ALREADY_RUNNING = 15,
    NOT_RUNNING = 16,
    TIMEOUT = 17,
    NETWORK_ERROR = 18,
    INVALID_STATE = 19,
    PERFORMANCE_DEGRADED = 20,
    NOT_CONNECTED = 21,
    ALREADY_EXISTS = 22,
    RESOURCE_EXHAUSTED = 23
};

/**
 * @enum ModelType
 * @brief Supported model formats
 */
enum class ModelType {
    ONNX = 0,
    TENSORFLOW_LITE = 1,
    PYTORCH_MOBILE = 2,
    UNKNOWN = 3
};

/**
 * @enum DeviceType
 * @brief Supported device types
 */
enum class DeviceType {
    CPU = 0,
    GPU = 1,
    NPU = 2,
    TPU = 3,
    FPGA = 4,
    AUTO = 5
};

/**
 * @enum DataType
 * @brief Supported data types
 */
enum class DataType {
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT32 = 2,
    INT16 = 3,
    INT8 = 4,
    UINT8 = 5,
    BOOL = 6,
    UNKNOWN = 7
};

/**
 * @enum OptimizationType
 * @brief Types of optimizations
 */
enum class OptimizationType {
    QUANTIZATION = 0,
    PRUNING = 1,
    GRAPH_OPTIMIZATION = 2,
    HARDWARE_ACCELERATION = 3,
    MEMORY_OPTIMIZATION = 4,
    BATCHING_OPTIMIZATION = 5
};

/**
 * @struct TensorShape
 * @brief Shape information for tensors
 */
struct TensorShape {
    std::vector<int64_t> dimensions;
    
    TensorShape() = default;
    explicit TensorShape(const std::vector<int64_t>& dims) : dimensions(dims) {}
    
    size_t GetTotalElements() const;
    bool IsValid() const;
    std::string ToString() const;
};

/**
 * @class Tensor
 * @brief Tensor data structure
 */
class Tensor {
public:
    Tensor() = default;
    Tensor(DataType type, const TensorShape& shape, void* data = nullptr);
    ~Tensor();
    
    // Disable copy constructor and assignment operator
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Move constructor and assignment operator
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    DataType GetDataType() const { return data_type_; }
    const TensorShape& GetShape() const { return shape_; }
    void* GetData() const { return data_; }
    size_t GetSize() const { return size_; }
    size_t GetDataSize() const { return size_; }
    size_t GetDataTypeSize() const;
    
    Status SetData(void* data, size_t size);
    Status Reshape(const TensorShape& new_shape);
    
    bool IsValid() const;
    std::string ToString() const;

private:
    DataType data_type_;
    TensorShape shape_;
    void* data_;
    size_t size_;
    bool owns_data_;
    
    void AllocateMemory();
    void DeallocateMemory();
};

/**
 * @struct EngineConfig
 * @brief Configuration for the Edge AI Engine
 */
struct EngineConfig {
    // Device configuration
    DeviceType device_type = DeviceType::AUTO;
    int device_id = 0;
    
    // Memory configuration
    size_t max_memory_usage = 1024 * 1024 * 1024; // 1GB
    bool enable_memory_pool = true;
    size_t memory_pool_size = 512 * 1024 * 1024; // 512MB
    
    // Performance configuration
    int num_threads = 0; // 0 = auto
    bool enable_optimization = true;
    bool enable_profiling = false;
    
    // Batching configuration
    size_t max_batch_size = 32;
    bool enable_dynamic_batching = true;
    std::chrono::milliseconds batch_timeout{10};
    
    // Logging configuration
    bool enable_logging = true;
    int log_level = 1; // 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG
    
    EngineConfig() = default;
};

/**
 * @struct OptimizationConfig
 * @brief Configuration for model optimization
 */
struct OptimizationConfig {
    // Quantization configuration
    bool enable_quantization = false;
    DataType quantization_type = DataType::INT8;
    bool per_channel_quantization = false;
    
    // Pruning configuration
    bool enable_pruning = false;
    float pruning_ratio = 0.1f;
    bool structured_pruning = true;
    
    // Graph optimization
    bool enable_graph_optimization = true;
    bool enable_operator_fusion = true;
    bool enable_constant_folding = true;
    
    // Hardware-specific optimization
    bool enable_hardware_acceleration = true;
    bool enable_simd = true;
    bool enable_gpu_acceleration = false;
    
    OptimizationConfig() = default;
};

/**
 * @struct ModelInfo
 * @brief Information about a loaded model
 */
struct ModelInfo {
    std::string name;
    ModelType type;
    std::string version;
    std::vector<TensorShape> input_shapes;
    std::vector<TensorShape> output_shapes;
    std::vector<DataType> input_types;
    std::vector<DataType> output_types;
    size_t model_size;
    bool is_optimized;
    
    ModelInfo() : type(ModelType::UNKNOWN), model_size(0), is_optimized(false) {}
};

/**
 * @struct EngineStats
 * @brief Statistics about the engine
 */
struct EngineStats {
    // Model statistics
    bool model_loaded;
    bool model_optimized;
    size_t model_size;
    
    // Performance statistics
    uint64_t total_inferences;
    uint64_t successful_inferences;
    uint64_t failed_inferences;
    std::chrono::milliseconds total_inference_time;
    std::chrono::milliseconds average_inference_time;
    
    // Memory statistics
    size_t current_memory_usage;
    size_t peak_memory_usage;
    size_t total_memory_allocated;
    
    // Batching statistics
    uint64_t total_batches;
    uint64_t average_batch_size;
    std::chrono::milliseconds average_batch_time;
    
    EngineStats() : model_loaded(false), model_optimized(false), model_size(0),
                   total_inferences(0), successful_inferences(0), failed_inferences(0),
                   total_inference_time(0), average_inference_time(0),
                   current_memory_usage(0), peak_memory_usage(0), total_memory_allocated(0),
                   total_batches(0), average_batch_size(0), average_batch_time(0) {}
};

/**
 * @struct PerformanceMetrics
 * @brief Detailed performance metrics
 */
struct PerformanceMetrics {
    // Latency metrics
    std::chrono::microseconds min_latency;
    std::chrono::microseconds max_latency;
    std::chrono::microseconds average_latency;
    std::chrono::microseconds p95_latency;
    std::chrono::microseconds p99_latency;
    
    // Throughput metrics
    double inferences_per_second;
    double batches_per_second;
    double data_processed_mb_per_second;
    
    // Resource utilization
    double cpu_utilization;
    double gpu_utilization;
    double memory_utilization;
    
    // Error rates
    double error_rate;
    double timeout_rate;
    
    PerformanceMetrics() : min_latency(0), max_latency(0), average_latency(0),
                          p95_latency(0), p99_latency(0), inferences_per_second(0.0),
                          batches_per_second(0.0), data_processed_mb_per_second(0.0),
                          cpu_utilization(0.0), gpu_utilization(0.0), memory_utilization(0.0),
                          error_rate(0.0), timeout_rate(0.0) {}
};

/**
 * @struct DeviceInfo
 * @brief Information about a device
 */
struct DeviceInfo {
    DeviceType type;
    std::string name;
    std::string vendor;
    std::string version;
    size_t memory_size;
    int compute_units;
    bool available;
    
    DeviceInfo() : type(DeviceType::CPU), memory_size(0), compute_units(0), available(false) {}
};

/**
 * @enum RequestPriority
 * @brief Request priority levels
 */
enum class RequestPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @struct InferenceRequest
 * @brief Request for inference
 */
struct InferenceRequest {
    std::vector<Tensor> inputs;
    std::function<void(Status, std::vector<Tensor>)> callback;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t request_id;
    RequestPriority priority;
    std::chrono::milliseconds timeout;
    
    InferenceRequest() : timestamp(std::chrono::steady_clock::now()), request_id(0), 
                        priority(RequestPriority::NORMAL), timeout(std::chrono::milliseconds(5000)) {}
};

/**
 * @struct InferenceResult
 * @brief Result of inference
 */
struct InferenceResult {
    Status status;
    std::vector<Tensor> outputs;
    std::chrono::microseconds latency;
    uint64_t request_id;
    
    InferenceResult() : status(Status::FAILURE), latency(0), request_id(0) {}
};

/**
 * @struct InferenceConfig
 * @brief Configuration for inference
 */
struct InferenceConfig {
    bool enable_profiling;
    bool enable_batching;
    size_t batch_size;
    std::chrono::milliseconds timeout;
    
    InferenceConfig() : enable_profiling(false), enable_batching(false), 
                       batch_size(1), timeout(std::chrono::milliseconds(1000)) {}
};

/**
 * @struct InferenceStats
 * @brief Statistics for inference
 */
struct InferenceStats {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> successful_requests{0};
    std::atomic<uint64_t> failed_requests{0};
    std::atomic<std::chrono::microseconds> total_latency{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> max_latency{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> min_latency{std::chrono::microseconds::max()};

    InferenceStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t total_requests;
        uint64_t successful_requests;
        uint64_t failed_requests;
        std::chrono::microseconds total_latency;
        std::chrono::microseconds max_latency;
        std::chrono::microseconds min_latency;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_requests = total_requests.load();
        snapshot.successful_requests = successful_requests.load();
        snapshot.failed_requests = failed_requests.load();
        snapshot.total_latency = total_latency.load();
        snapshot.max_latency = max_latency.load();
        snapshot.min_latency = min_latency.load();
        return snapshot;
    }
};

/**
 * @enum TaskType
 * @brief Types of tasks
 */
enum class TaskType {
    INFERENCE = 0,
    OPTIMIZATION = 1,
    MEMORY_ALLOCATION = 2,
    DATA_PREPROCESSING = 3,
    DATA_POSTPROCESSING = 4
};

/**
 * @enum TaskPriority
 * @brief Task priority levels
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

/**
 * @enum TaskStatus
 * @brief Task status
 */
enum class TaskStatus {
    PENDING = 0,
    RUNNING = 1,
    COMPLETED = 2,
    FAILED = 3,
    CANCELLED = 4
};

/**
 * @class Task
 * @brief Task wrapper for scheduler
 */
class Task {
public:
    Task(TaskType type, std::shared_ptr<void> data, TaskPriority priority = TaskPriority::NORMAL)
        : task_id_(GenerateTaskId())
        , type_(type)
        , data_(data)
        , priority_(priority)
        , status_(TaskStatus::PENDING)
        , created_time_(std::chrono::steady_clock::now())
        , started_time_()
        , completed_time_() {}
    
    // Getters
    uint64_t GetTaskId() const { return task_id_; }
    TaskType GetType() const { return type_; }
    TaskPriority GetPriority() const { return priority_; }
    TaskStatus GetStatus() const { return status_; }
    std::shared_ptr<void> GetData() const { return data_; }
    
    // Setters
    void SetStatus(TaskStatus status) { 
        status_ = status;
        if (status == TaskStatus::RUNNING) {
            started_time_ = std::chrono::steady_clock::now();
        } else if (status == TaskStatus::COMPLETED || status == TaskStatus::FAILED) {
            completed_time_ = std::chrono::steady_clock::now();
        }
    }
    
    void SetPriority(TaskPriority priority) { priority_ = priority; }
    
    // Timing
    std::chrono::steady_clock::time_point GetCreatedTime() const { return created_time_; }
    std::chrono::steady_clock::time_point GetStartedTime() const { return started_time_; }
    std::chrono::steady_clock::time_point GetCompletedTime() const { return completed_time_; }
    
    std::chrono::microseconds GetWaitTime() const {
        if (started_time_ == std::chrono::steady_clock::time_point{}) {
            return std::chrono::microseconds(0);
        }
        return std::chrono::duration_cast<std::chrono::microseconds>(started_time_ - created_time_);
    }
    
    std::chrono::microseconds GetExecutionTime() const {
        if (started_time_ == std::chrono::steady_clock::time_point{} || 
            completed_time_ == std::chrono::steady_clock::time_point{}) {
            return std::chrono::microseconds(0);
        }
        return std::chrono::duration_cast<std::chrono::microseconds>(completed_time_ - started_time_);
    }

private:
    uint64_t task_id_;
    TaskType type_;
    std::shared_ptr<void> data_;
    TaskPriority priority_;
    TaskStatus status_;
    std::chrono::steady_clock::time_point created_time_;
    std::chrono::steady_clock::time_point started_time_;
    std::chrono::steady_clock::time_point completed_time_;
    
    static uint64_t GenerateTaskId() {
        static std::atomic<uint64_t> next_id{1};
        return next_id.fetch_add(1);
    }
};

/**
 * @struct SchedulerConfig
 * @brief Configuration for runtime scheduler
 */
struct SchedulerConfig {
    // Threading configuration
    int num_worker_threads;
    bool enable_work_stealing;
    size_t max_queue_size;
    
    // Task configuration
    std::chrono::milliseconds task_timeout;
    bool enable_task_prioritization;
    bool enable_task_affinity;
    
    // Performance configuration
    bool enable_profiling;
    bool enable_optimization;
    
    // Device configuration
    bool enable_device_load_balancing;
    bool enable_device_failover;
    
    SchedulerConfig() : num_worker_threads(0), enable_work_stealing(true), max_queue_size(1000),
                       task_timeout(std::chrono::milliseconds(30000)), enable_task_prioritization(true),
                       enable_task_affinity(true), enable_profiling(false), enable_optimization(true),
                       enable_device_load_balancing(true), enable_device_failover(true) {}
};

/**
 * @struct SchedulerStats
 * @brief Statistics for scheduler
 */
struct SchedulerStats {
    // Task statistics
    std::atomic<uint64_t> total_tasks_submitted{0};
    std::atomic<uint64_t> total_tasks_completed{0};
    std::atomic<uint64_t> total_tasks_failed{0};
    std::atomic<uint64_t> total_tasks_cancelled{0};

    // Performance statistics
    std::atomic<std::chrono::microseconds> total_execution_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> average_execution_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> max_execution_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> min_execution_time{std::chrono::microseconds::max()};

    // Queue statistics
    std::atomic<size_t> current_queue_size{0};
    std::atomic<size_t> max_queue_size{0};
    std::atomic<std::chrono::microseconds> average_queue_wait_time{std::chrono::microseconds{0}};

    // Device statistics
    std::atomic<uint64_t> total_device_utilization{0};
    std::atomic<double> average_device_utilization{0.0};

    SchedulerStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t total_tasks_submitted;
        uint64_t total_tasks_completed;
        uint64_t total_tasks_failed;
        uint64_t total_tasks_cancelled;
        std::chrono::microseconds total_execution_time;
        std::chrono::microseconds average_execution_time;
        std::chrono::microseconds max_execution_time;
        std::chrono::microseconds min_execution_time;
        size_t current_queue_size;
        size_t max_queue_size;
        std::chrono::microseconds average_queue_wait_time;
        uint64_t total_device_utilization;
        double average_device_utilization;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_tasks_submitted = total_tasks_submitted.load();
        snapshot.total_tasks_completed = total_tasks_completed.load();
        snapshot.total_tasks_failed = total_tasks_failed.load();
        snapshot.total_tasks_cancelled = total_tasks_cancelled.load();
        snapshot.total_execution_time = total_execution_time.load();
        snapshot.average_execution_time = average_execution_time.load();
        snapshot.max_execution_time = max_execution_time.load();
        snapshot.min_execution_time = min_execution_time.load();
        snapshot.current_queue_size = current_queue_size.load();
        snapshot.max_queue_size = max_queue_size.load();
        snapshot.average_queue_wait_time = average_queue_wait_time.load();
        snapshot.total_device_utilization = total_device_utilization.load();
        snapshot.average_device_utilization = average_device_utilization.load();
        return snapshot;
    }
};

/**
 * @struct MemoryConfig
 * @brief Configuration for memory manager
 */
struct MemoryConfig {
    size_t max_memory_pool_size;
    size_t initial_memory_pool_size;
    size_t max_memory_usage;
    size_t memory_pool_size;
    int num_memory_pools;
    bool enable_memory_pooling;
    bool enable_memory_tracking;
    
    MemoryConfig() : max_memory_pool_size(1024ULL * 1024 * 1024), // 1GB
                     initial_memory_pool_size(256ULL * 1024 * 1024), // 256MB
                     max_memory_usage(2ULL * 1024 * 1024 * 1024), // 2GB
                     memory_pool_size(1024ULL * 1024 * 1024), // 1GB
                     num_memory_pools(4),
                     enable_memory_pooling(true),
                     enable_memory_tracking(true) {}
};

/**
 * @struct MemoryStats
 * @brief Statistics for memory usage
 */
struct MemoryStats {
    std::atomic<size_t> total_allocated{0};
    std::atomic<size_t> total_freed{0};
    std::atomic<size_t> peak_usage{0};
    std::atomic<size_t> current_usage{0};
    std::atomic<uint64_t> allocation_count{0};
    std::atomic<uint64_t> deallocation_count{0};
    std::atomic<uint64_t> total_allocations{0};
    std::atomic<uint64_t> total_deallocations{0};
    
    MemoryStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        size_t total_allocated;
        size_t total_freed;
        size_t peak_usage;
        size_t current_usage;
        uint64_t allocation_count;
        uint64_t deallocation_count;
        uint64_t total_allocations;
        uint64_t total_deallocations;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_allocated = total_allocated.load();
        snapshot.total_freed = total_freed.load();
        snapshot.peak_usage = peak_usage.load();
        snapshot.current_usage = current_usage.load();
        snapshot.allocation_count = allocation_count.load();
        snapshot.deallocation_count = deallocation_count.load();
        snapshot.total_allocations = total_allocations.load();
        snapshot.total_deallocations = total_deallocations.load();
        return snapshot;
    }
};

/**
 * @struct DeviceMemoryUsage
 * @brief Memory usage for a specific device
 */
struct DeviceMemoryUsage {
    size_t allocated_memory;
    size_t free_memory;
    size_t total_memory;
    double utilization_percentage;
    double utilization_ratio;
    
    DeviceMemoryUsage() : allocated_memory(0), free_memory(0), total_memory(0), utilization_percentage(0.0), utilization_ratio(0.0) {}
};

/**
 * @struct MemoryPoolStats
 * @brief Statistics for memory pool
 */
struct MemoryPoolStats {
    size_t pool_size;
    size_t allocated_size;
    size_t free_size;
    uint64_t allocation_count;
    uint64_t deallocation_count;
    uint64_t total_allocations;
    double utilization_ratio;
    
    MemoryPoolStats() : pool_size(0), allocated_size(0), free_size(0), 
                       allocation_count(0), deallocation_count(0), total_allocations(0), utilization_ratio(0.0) {}
};

/**
 * @struct MemoryAllocatorStats
 * @brief Statistics for memory allocator
 */
struct MemoryAllocatorStats {
    std::atomic<uint64_t> total_allocations{0};
    std::atomic<uint64_t> total_deallocations{0};
    std::atomic<size_t> total_allocated_bytes{0};
    std::atomic<size_t> total_freed_bytes{0};
    std::atomic<std::chrono::microseconds> total_allocation_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> average_allocation_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> total_deallocation_time{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> average_deallocation_time{std::chrono::microseconds{0}};
    
    MemoryAllocatorStats() = default;
};

/**
 * @struct BatchingConfig
 * @brief Configuration for batching manager
 */
struct BatchingConfig {
    size_t max_batch_size;
    std::chrono::milliseconds batch_timeout;
    bool enable_dynamic_batching;
    bool enable_priority_batching;
    
    BatchingConfig() : max_batch_size(32), batch_timeout(std::chrono::milliseconds(100)),
                      enable_dynamic_batching(true), enable_priority_batching(true) {}
};

/**
 * @struct BatchingStats
 * @brief Statistics for batching
 */
struct BatchingStats {
    std::atomic<uint64_t> total_batches_created{0};
    std::atomic<uint64_t> total_requests_batched{0};
    std::atomic<uint64_t> total_batches_processed{0};
    std::atomic<uint64_t> total_batches_failed{0};
    std::atomic<uint64_t> total_batches_formed{0};
    std::atomic<uint64_t> average_batch_size{0};
    std::atomic<uint64_t> max_batch_size{0};
    std::atomic<uint64_t> min_batch_size{0};
    std::atomic<std::chrono::microseconds> average_batch_latency{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> total_latency{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> average_latency{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> min_latency{std::chrono::microseconds{0}};
    std::atomic<std::chrono::microseconds> max_latency{std::chrono::microseconds{0}};
    
    BatchingStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t total_batches_created;
        uint64_t total_requests_batched;
        uint64_t total_batches_processed;
        uint64_t total_batches_failed;
        uint64_t total_batches_formed;
        uint64_t average_batch_size;
        uint64_t max_batch_size;
        uint64_t min_batch_size;
        std::chrono::microseconds average_batch_latency;
        std::chrono::microseconds total_latency;
        std::chrono::microseconds average_latency;
        std::chrono::microseconds min_latency;
        std::chrono::microseconds max_latency;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_batches_created = total_batches_created.load();
        snapshot.total_requests_batched = total_requests_batched.load();
        snapshot.total_batches_processed = total_batches_processed.load();
        snapshot.total_batches_failed = total_batches_failed.load();
        snapshot.total_batches_formed = total_batches_formed.load();
        snapshot.average_batch_size = average_batch_size.load();
        snapshot.max_batch_size = max_batch_size.load();
        snapshot.min_batch_size = min_batch_size.load();
        snapshot.average_batch_latency = average_batch_latency.load();
        snapshot.total_latency = total_latency.load();
        snapshot.average_latency = average_latency.load();
        snapshot.min_latency = min_latency.load();
        snapshot.max_latency = max_latency.load();
        return snapshot;
    }
};


/**
 * @enum PruningStrategy
 * @brief Pruning strategies
 */
enum class PruningStrategy {
    MAGNITUDE_BASED = 0,
    GRADIENT_BASED = 1,
    STRUCTURED = 2,
    UNSTRUCTURED = 3
};

/**
 * @struct OptimizationStats
 * @brief Statistics for optimization
 */
struct OptimizationStats {
    std::atomic<uint64_t> total_optimizations{0};
    std::atomic<uint64_t> successful_optimizations{0};
    std::atomic<uint64_t> failed_optimizations{0};
    std::atomic<std::chrono::microseconds> total_optimization_time{std::chrono::microseconds{0}};
    
    OptimizationStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t total_optimizations;
        uint64_t successful_optimizations;
        uint64_t failed_optimizations;
        std::chrono::microseconds total_optimization_time;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_optimizations = total_optimizations.load();
        snapshot.successful_optimizations = successful_optimizations.load();
        snapshot.failed_optimizations = failed_optimizations.load();
        snapshot.total_optimization_time = total_optimization_time.load();
        return snapshot;
    }
};

/**
 * @struct QuantizationConfig
 * @brief Configuration for quantization
 */
struct QuantizationConfig {
    bool enable_dynamic_quantization;
    bool enable_static_quantization;
    int quantization_bits;
    std::string calibration_dataset_path;
    
    QuantizationConfig() : enable_dynamic_quantization(false), enable_static_quantization(false),
                          quantization_bits(8), calibration_dataset_path("") {}
};

/**
 * @struct PruningConfig
 * @brief Configuration for pruning
 */
struct PruningConfig {
    PruningStrategy strategy;
    double pruning_ratio;
    bool enable_structured_pruning;
    bool enable_unstructured_pruning;
    
    PruningConfig() : strategy(PruningStrategy::MAGNITUDE_BASED), pruning_ratio(0.1),
                     enable_structured_pruning(false), enable_unstructured_pruning(true) {}
};

/**
 * @struct GraphOptimizationConfig
 * @brief Configuration for graph optimization
 */
struct GraphOptimizationConfig {
    bool enable_constant_folding;
    bool enable_dead_code_elimination;
    bool enable_operator_fusion;
    bool enable_memory_optimization;
    
    GraphOptimizationConfig() : enable_constant_folding(true), enable_dead_code_elimination(true),
                               enable_operator_fusion(true), enable_memory_optimization(true) {}
};

/**
 * @struct OptimizationBenefits
 * @brief Benefits of optimization
 */
struct OptimizationBenefits {
    double speedup_factor;
    double memory_reduction;
    double model_size_reduction;
    std::chrono::microseconds estimated_inference_time;
    double inference_speedup;
    double accuracy_loss;
    bool accuracy_acceptable;
    bool hardware_compatible;
    
    OptimizationBenefits() : speedup_factor(1.0), memory_reduction(0.0), model_size_reduction(0.0),
                            estimated_inference_time(std::chrono::microseconds{0}),
                            inference_speedup(1.0), accuracy_loss(0.0), accuracy_acceptable(true),
                            hardware_compatible(true) {}
};

/**
 * @struct ProfilerConfig
 * @brief Configuration for profiler
 */
struct ProfilerConfig {
    bool enable_timing;
    bool enable_memory_profiling;
    bool enable_hardware_profiling;
    std::string output_file;
    
    ProfilerConfig() : enable_timing(true), enable_memory_profiling(false), 
                      enable_hardware_profiling(false), output_file("") {}
};

/**
 * @struct ProfilerStats
 * @brief Statistics for profiler
 */
struct ProfilerStats {
    std::atomic<uint64_t> total_profiles{0};
    std::atomic<std::chrono::microseconds> total_profiling_time{std::chrono::microseconds{0}};
    std::atomic<size_t> peak_memory_usage{0};
    std::atomic<uint64_t> total_sessions{0};
    std::atomic<uint64_t> active_sessions{0};
    std::atomic<uint64_t> total_timing_operations{0};
    std::atomic<uint64_t> total_counter_operations{0};
    
    ProfilerStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t total_profiles;
        std::chrono::microseconds total_profiling_time;
        size_t peak_memory_usage;
        uint64_t total_sessions;
        uint64_t active_sessions;
        uint64_t total_timing_operations;
        uint64_t total_counter_operations;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_profiles = total_profiles.load();
        snapshot.total_profiling_time = total_profiling_time.load();
        snapshot.peak_memory_usage = peak_memory_usage.load();
        snapshot.total_sessions = total_sessions.load();
        snapshot.active_sessions = active_sessions.load();
        snapshot.total_timing_operations = total_timing_operations.load();
        snapshot.total_counter_operations = total_counter_operations.load();
        return snapshot;
    }
};

/**
 * @struct TimingStats
 * @brief Timing statistics
 */
struct TimingStats {
    std::chrono::microseconds total_time{0};
    std::chrono::microseconds min_time{std::chrono::microseconds::max()};
    std::chrono::microseconds max_time{0};
    std::chrono::microseconds average_time{0};
    uint64_t count{0};
    uint64_t call_count{0};
    
    TimingStats() = default;
};

/**
 * @struct CounterStats
 * @brief Counter statistics
 */
struct CounterStats {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total{0};
    std::atomic<uint64_t> min_value{0};
    std::atomic<uint64_t> max_value{0};
    std::atomic<int64_t> current_value{0};
    std::atomic<int64_t> total_value{0};
    std::atomic<uint64_t> update_count{0};
    std::atomic<double> average_value{0.0};
    std::string name;
    
    CounterStats() = default;
    CounterStats(const std::string& counter_name) : name(counter_name) {}
    
    // Non-atomic version for return values
    struct Snapshot {
        uint64_t count;
        uint64_t total;
        uint64_t min_value;
        uint64_t max_value;
        int64_t current_value;
        int64_t total_value;
        uint64_t update_count;
        double average_value;
        std::string name;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.count = count.load();
        snapshot.total = total.load();
        snapshot.min_value = min_value.load();
        snapshot.max_value = max_value.load();
        snapshot.current_value = current_value.load();
        snapshot.total_value = total_value.load();
        snapshot.update_count = update_count.load();
        snapshot.average_value = average_value.load();
        snapshot.name = name;
        return snapshot;
    }
};

/**
 * @struct MemoryUsageStats
 * @brief Memory usage statistics
 */
struct MemoryUsageStats {
    std::atomic<size_t> current_usage{0};
    std::atomic<size_t> peak_usage{0};
    std::atomic<size_t> total_allocated{0};
    std::atomic<size_t> total_freed{0};
    std::atomic<uint64_t> total_allocations{0};
    
    MemoryUsageStats() = default;
    
    // Non-atomic version for return values
    struct Snapshot {
        size_t current_usage;
        size_t peak_usage;
        size_t total_allocated;
        size_t total_freed;
        uint64_t total_allocations;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.current_usage = current_usage.load();
        snapshot.peak_usage = peak_usage.load();
        snapshot.total_allocated = total_allocated.load();
        snapshot.total_freed = total_freed.load();
        snapshot.total_allocations = total_allocations.load();
        return snapshot;
    }
};

/**
 * @struct HardwareUtilizationStats
 * @brief Hardware utilization statistics
 */
struct HardwareUtilizationStats {
    std::atomic<double> cpu_utilization{0.0};
    std::atomic<double> gpu_utilization{0.0};
    std::atomic<double> memory_utilization{0.0};
    std::atomic<uint64_t> active_threads{0};
    std::atomic<double> current_utilization{0.0};
    std::atomic<double> average_utilization{0.0};
    std::string device_name;
    std::atomic<double> peak_utilization{0.0};
    std::atomic<double> min_utilization{0.0};
    std::atomic<uint64_t> measurement_count{0};
    std::chrono::steady_clock::time_point last_measurement;
    
    HardwareUtilizationStats() : last_measurement(std::chrono::steady_clock::now()) {}
    
    // Non-atomic version for return values
    struct Snapshot {
        double cpu_utilization;
        double gpu_utilization;
        double memory_utilization;
        uint64_t active_threads;
        double current_utilization;
        double average_utilization;
        std::string device_name;
        double peak_utilization;
        double min_utilization;
        uint64_t measurement_count;
        std::chrono::steady_clock::time_point last_measurement;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.cpu_utilization = cpu_utilization.load();
        snapshot.gpu_utilization = gpu_utilization.load();
        snapshot.memory_utilization = memory_utilization.load();
        snapshot.active_threads = active_threads.load();
        snapshot.current_utilization = current_utilization.load();
        snapshot.average_utilization = average_utilization.load();
        snapshot.device_name = device_name;
        snapshot.peak_utilization = peak_utilization.load();
        snapshot.min_utilization = min_utilization.load();
        snapshot.measurement_count = measurement_count.load();
        snapshot.last_measurement = last_measurement;
        return snapshot;
    }
};

/**
 * @struct ProfilingReport
 * @brief Complete profiling report
 */
struct ProfilingReport {
    TimingStats inference_timing;
    MemoryUsageStats memory_usage;
    HardwareUtilizationStats hardware_utilization;
    std::unordered_map<std::string, TimingStats> operation_timing;
    std::string report_timestamp;
    std::vector<TimingStats> top_operations;
    MemoryUsageStats::Snapshot memory_summary;
    std::vector<HardwareUtilizationStats::Snapshot> hardware_summary;
    std::vector<std::string> performance_recommendations;
    std::vector<std::string> optimization_suggestions;
    std::string report_name;
    std::chrono::steady_clock::time_point generation_time;
    std::chrono::microseconds total_profiling_time;
    
    ProfilingReport() : report_timestamp(""), report_name(""), generation_time(std::chrono::steady_clock::now()), total_profiling_time(0) {}
    
    // Non-atomic version for return values
    struct Snapshot {
        TimingStats inference_timing;
        MemoryUsageStats::Snapshot memory_usage;
        HardwareUtilizationStats::Snapshot hardware_utilization;
        std::unordered_map<std::string, TimingStats> operation_timing;
        std::string report_timestamp;
        std::vector<TimingStats> top_operations;
        MemoryUsageStats::Snapshot memory_summary;
        std::vector<HardwareUtilizationStats::Snapshot> hardware_summary;
        std::vector<std::string> performance_recommendations;
        std::vector<std::string> optimization_suggestions;
        std::string report_name;
        std::chrono::steady_clock::time_point generation_time;
        std::chrono::microseconds total_profiling_time;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.inference_timing = inference_timing;
        snapshot.memory_usage = memory_usage.GetSnapshot();
        snapshot.hardware_utilization = hardware_utilization.GetSnapshot();
        snapshot.operation_timing = operation_timing;
        snapshot.report_timestamp = report_timestamp;
        snapshot.top_operations = top_operations;
        snapshot.memory_summary = memory_summary;
        snapshot.hardware_summary = hardware_summary;
        snapshot.performance_recommendations = performance_recommendations;
        snapshot.optimization_suggestions = optimization_suggestions;
        snapshot.report_name = report_name;
        snapshot.generation_time = generation_time;
        snapshot.total_profiling_time = total_profiling_time;
        return snapshot;
    }
};

} // namespace edge_ai
