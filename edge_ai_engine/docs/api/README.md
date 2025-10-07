# API Reference

This document provides comprehensive API documentation for the Edge AI Inference Engine.

## Table of Contents

- [Core Engine API](core_engine.md)
- [Model Loading API](model_loading.md)
- [Inference API](inference.md)
- [Optimization API](optimization.md)
- [Memory Management API](memory_management.md)
- [Profiling API](profiling.md)
- [Testing API](testing.md)
- [Distributed Runtime API](distributed_runtime.md)
- [Graph Execution API](graph_execution.md)
- [Hardware Acceleration API](hardware_acceleration.md)

## Core Engine

The main entry point for the Edge AI Inference Engine.

### EdgeAIEngine

```cpp
namespace edge_ai {

class EdgeAIEngine {
public:
    // Constructor
    explicit EdgeAIEngine(const EngineConfig& config);
    
    // Initialization and lifecycle
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Model management
    Status LoadModel(const std::string& model_path, ModelType type);
    Status UnloadModel();
    bool HasModel() const;
    ModelInfo GetModelInfo() const;
    
    // Optimization
    Status OptimizeModel(const OptimizationConfig& config);
    
    // Inference
    Status RunInference(const std::vector<Tensor>& inputs, 
                       std::vector<Tensor>& outputs);
    
    // Statistics and monitoring
    EngineStats GetStats() const;
    Metrics GetMetrics() const;
    void SetMonitoring(bool enabled);
};

} // namespace edge_ai
```

### EngineConfig

```cpp
struct EngineConfig {
    DeviceType device_type = DeviceType::CPU;
    size_t max_memory_usage = 1024 * 1024 * 1024; // 1GB
    bool enable_memory_pool = true;
    uint32_t num_threads = std::thread::hardware_concurrency();
    bool enable_optimization = true;
    bool enable_profiling = true;
    uint32_t max_batch_size = 32;
    bool enable_dynamic_batching = true;
    std::string log_level = "INFO";
};
```

## Model Loading

### ModelType

```cpp
enum class ModelType {
    ONNX,
    TENSORFLOW_LITE,
    PYTORCH_MOBILE,
    UNKNOWN
};
```

### ModelInfo

```cpp
struct ModelInfo {
    std::string name;
    std::string version;
    ModelType type;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    size_t model_size_bytes;
    std::map<std::string, std::string> metadata;
};
```

### TensorInfo

```cpp
struct TensorInfo {
    std::string name;
    DataType data_type;
    std::vector<int64_t> shape;
    size_t size_bytes;
};
```

## Inference

### Tensor

```cpp
class Tensor {
public:
    Tensor(DataType type, const std::vector<int64_t>& shape);
    Tensor(DataType type, const std::vector<int64_t>& shape, void* data);
    
    DataType GetDataType() const;
    const std::vector<int64_t>& GetShape() const;
    size_t GetSizeBytes() const;
    void* GetData();
    const void* GetData() const;
    
    template<typename T>
    T* GetDataAs();
    
    template<typename T>
    const T* GetDataAs() const;
};
```

### DataType

```cpp
enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT16,
    INT8,
    UINT32,
    UINT16,
    UINT8,
    BOOL
};
```

## Optimization

### OptimizationConfig

```cpp
struct OptimizationConfig {
    bool enable_quantization = false;
    DataType quantization_type = DataType::INT8;
    bool enable_pruning = false;
    float pruning_ratio = 0.1f;
    bool enable_graph_optimization = true;
    bool enable_hardware_acceleration = true;
    std::map<std::string, std::string> custom_options;
};
```

### OptimizationManager

```cpp
class OptimizationManager {
public:
    Status Initialize();
    Status Shutdown();
    
    Status RegisterComponents(const std::vector<OptimizationComponent>& components);
    Status StartOptimization();
    Status StopOptimization();
    
    Status UpdateMetrics(const Metrics& metrics);
    Status ApplyOptimization(const OptimizationDecision& decision);
    
    OptimizationStats GetStats() const;
    std::vector<OptimizationDecision> GetRecentDecisions() const;
    
    Status SetOptimizationPolicy(std::unique_ptr<OptimizationPolicy> policy);
    void SetOptimizationEnabled(bool enabled);
    
    Status ExportOptimizationTrace(const std::string& file_path) const;
};
```

## Memory Management

### MemoryManager

```cpp
class MemoryManager {
public:
    Status Initialize(const MemoryConfig& config);
    Status Shutdown();
    
    void* Allocate(size_t size, MemoryType type = MemoryType::GENERAL);
    void Deallocate(void* ptr);
    
    Status CreateMemoryPool(size_t pool_size, MemoryType type);
    void* AllocateFromPool(size_t size, MemoryType type);
    void DeallocateFromPool(void* ptr, MemoryType type);
    
    MemoryStats GetStats() const;
    bool IsMemoryAvailable(size_t size, MemoryType type) const;
};
```

### MemoryConfig

```cpp
struct MemoryConfig {
    size_t max_memory_usage = 1024 * 1024 * 1024; // 1GB
    bool enable_memory_pool = true;
    size_t pool_size = 512 * 1024 * 1024; // 512MB
    bool enable_zero_copy = true;
    uint32_t alignment = 64;
};
```

## Profiling

### Profiler

```cpp
class Profiler {
public:
    static Profiler& GetInstance();
    
    Status Initialize();
    Status Shutdown();
    
    Status StartGlobalSession(const std::string& session_name);
    Status StopGlobalSession();
    
    Status StartRequestSession(const std::string& request_id);
    Status StopRequestSession(const std::string& request_id);
    
    void MarkEvent(const std::string& request_id, const std::string& event_name);
    ScopedEvent CreateScopedEvent(const std::string& request_id, 
                                 const std::string& event_name);
    
    Status RegisterEventSource(const std::string& source_name);
    
    Status ExportSessionAsJson(const std::string& session_name, 
                              const std::string& file_path) const;
    
    ProfilerStats GetStats() const;
    bool IsEnabled() const;
    void SetEnabled(bool enabled);
};
```

### ScopedEvent

```cpp
class ScopedEvent {
public:
    ScopedEvent(const std::string& request_id, const std::string& event_name);
    ~ScopedEvent();
    
    // Non-copyable, movable
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;
    ScopedEvent(ScopedEvent&&) noexcept;
    ScopedEvent& operator=(ScopedEvent&&) noexcept;
};
```

## Testing Framework

### TestRunner

```cpp
class TestRunner {
public:
    Status Initialize(const RunnerConfiguration& config);
    Status Shutdown();
    
    Status RunAllTests();
    Status RunTestSuite(const std::string& suite_name);
    Status RunTestCase(const std::string& test_name);
    
    RunnerStatistics GetStatistics() const;
    std::vector<TestResult> GetResults() const;
    
    Status SetReporter(std::unique_ptr<TestReporter> reporter);
    Status SetDiscovery(std::unique_ptr<TestDiscovery> discovery);
};
```

### TestReporter

```cpp
class TestReporter {
public:
    virtual ~TestReporter() = default;
    
    virtual Status Initialize(const ReporterConfiguration& config) = 0;
    virtual Status Shutdown() = 0;
    
    virtual Status ReportTestStart(const TestCase& test_case) = 0;
    virtual Status ReportTestEnd(const TestResult& result) = 0;
    virtual Status ReportTestSuiteStart(const std::string& suite_name) = 0;
    virtual Status ReportTestSuiteEnd(const TestSuiteResult& result) = 0;
    
    virtual Status GenerateReport(const std::string& output_path) = 0;
    virtual TestStatistics GetStatistics() const = 0;
};
```

## Distributed Runtime

### DistributedRuntime

```cpp
class DistributedRuntime {
public:
    Status Initialize(const DistributedConfig& config);
    Status Shutdown();
    
    Status StartCluster();
    Status StopCluster();
    
    Status RegisterGraph(const Graph& graph);
    Status ExecuteGraph(const std::string& graph_id, 
                       const ExecutionContext& context);
    
    Status MigrateNode(const std::string& node_id, 
                      const std::string& target_device);
    
    ClusterStats GetClusterStats() const;
    std::vector<NodeInfo> GetNodeInfo() const;
};
```

### ClusterManager

```cpp
class ClusterManager {
public:
    Status Initialize(const ClusterConfig& config);
    Status Shutdown();
    
    Status AddNode(const NodeInfo& node_info);
    Status RemoveNode(const std::string& node_id);
    Status UpdateNodeStatus(const std::string& node_id, NodeStatus status);
    
    std::vector<NodeInfo> GetAvailableNodes() const;
    NodeInfo GetNodeInfo(const std::string& node_id) const;
    
    Status SetLoadBalancingPolicy(std::unique_ptr<LoadBalancingPolicy> policy);
    Status SetFaultTolerancePolicy(std::unique_ptr<FaultTolerancePolicy> policy);
};
```

## Graph Execution

### Graph

```cpp
class Graph {
public:
    Status AddNode(std::unique_ptr<Node> node);
    Status AddEdge(const std::string& from_node, const std::string& to_node);
    Status RemoveNode(const std::string& node_id);
    Status RemoveEdge(const std::string& from_node, const std::string& to_node);
    
    std::vector<std::string> GetNodes() const;
    std::vector<Edge> GetEdges() const;
    
    Status Compile();
    Status Execute(const ExecutionContext& context);
    
    GraphStats GetStats() const;
    bool IsCompiled() const;
};
```

### Node

```cpp
class Node {
public:
    virtual ~Node() = default;
    
    virtual Status Initialize() = 0;
    virtual Status Execute(const ExecutionContext& context) = 0;
    virtual Status Shutdown() = 0;
    
    virtual std::string GetId() const = 0;
    virtual NodeType GetType() const = 0;
    virtual std::vector<std::string> GetInputs() const = 0;
    virtual std::vector<std::string> GetOutputs() const = 0;
};
```

## Hardware Acceleration

### DeviceManager

```cpp
class DeviceManager {
public:
    Status Initialize();
    Status Shutdown();
    
    std::vector<DeviceInfo> GetAvailableDevices() const;
    DeviceInfo GetDeviceInfo(DeviceType type) const;
    
    Status SetDefaultDevice(DeviceType type);
    DeviceType GetDefaultDevice() const;
    
    Status EnableDevice(DeviceType type);
    Status DisableDevice(DeviceType type);
    bool IsDeviceEnabled(DeviceType type) const;
};
```

### DeviceInfo

```cpp
struct DeviceInfo {
    DeviceType type;
    std::string name;
    std::string vendor;
    std::string version;
    size_t memory_size;
    uint32_t compute_units;
    bool is_available;
    std::map<std::string, std::string> capabilities;
};
```

## Error Handling

### Status

```cpp
enum class Status {
    OK = 0,
    ERROR_INVALID_ARGUMENT = 1,
    ERROR_NOT_INITIALIZED = 2,
    ERROR_ALREADY_INITIALIZED = 3,
    ERROR_MODEL_NOT_LOADED = 4,
    ERROR_INFERENCE_FAILED = 5,
    ERROR_OPTIMIZATION_FAILED = 6,
    ERROR_MEMORY_ALLOCATION_FAILED = 7,
    ERROR_DEVICE_NOT_AVAILABLE = 8,
    ERROR_UNSUPPORTED_OPERATION = 9,
    ERROR_INTERNAL = 10
};
```

### Error Handling Utilities

```cpp
namespace edge_ai {

inline const char* StatusToString(Status status);
inline bool IsSuccess(Status status);
inline bool IsError(Status status);

} // namespace edge_ai
```

## Constants and Enums

### DeviceType

```cpp
enum class DeviceType {
    CPU,
    GPU,
    NPU,
    TPU,
    FPGA,
    AUTO
};
```

### MemoryType

```cpp
enum class MemoryType {
    GENERAL,
    DEVICE,
    HOST,
    UNIFIED
};
```

### NodeType

```cpp
enum class NodeType {
    INFERENCE,
    PREPROCESSING,
    POSTPROCESSING,
    CUSTOM
};
```

### NodeStatus

```cpp
enum class NodeStatus {
    IDLE,
    RUNNING,
    ERROR,
    OFFLINE
};
```
