/**
 * @file graph_types.h
 * @brief Core graph data structures for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the fundamental data structures for graph-based execution:
 * Graph, Node, Edge, and related metadata for multi-model, multi-device execution.
 */

#pragma once

#include "core/types.h"
#include "backend/execution_backend.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <atomic>
#include <chrono>
#include <variant>
#include <optional>
#include <any>
#include <limits>

namespace edge_ai {

/**
 * @enum NodeType
 * @brief Types of nodes in the execution graph
 */
enum class NodeType {
    MODEL_INFERENCE = 0,    // AI model inference node
    DATA_PROCESSING = 1,    // Data transformation/preprocessing
    AGGREGATION = 2,        // Data aggregation/merging
    FILTERING = 3,          // Data filtering/selection
    ROUTING = 4,            // Data routing/conditional execution
    STREAMING = 5,          // Streaming data processing
    REMOTE_CALL = 6,        // Remote execution node
    CUSTOM_OPERATOR = 7     // User-defined custom operator
};

/**
 * @enum EdgeType
 * @brief Types of edges in the execution graph
 */
enum class EdgeType {
    DATA_FLOW = 0,          // Standard data flow
    CONTROL_FLOW = 1,       // Control/conditional flow
    STREAMING = 2,          // Streaming data flow
    REMOTE = 3,             // Remote data transfer
    FEEDBACK = 4            // Feedback loop
};

/**
 * @enum ExecutionMode
 * @brief Execution modes for graph nodes
 */
enum class ExecutionMode {
    SEQUENTIAL = 0,         // Sequential execution
    PARALLEL = 1,           // Parallel execution
    PIPELINE = 2,           // Pipelined execution
    STREAMING = 3,          // Streaming execution
    CONDITIONAL = 4         // Conditional execution
};

/**
 * @enum NodeStatus
 * @brief Status of a graph node during execution
 */
enum class NodeStatus {
    PENDING = 0,            // Waiting to be executed
    READY = 1,              // Ready for execution
    RUNNING = 2,            // Currently executing
    COMPLETED = 3,          // Execution completed successfully
    FAILED = 4,             // Execution failed
    CANCELLED = 5,          // Execution was cancelled
    SKIPPED = 6             // Execution was skipped
};

/**
 * @struct NodeMetadata
 * @brief Metadata associated with a graph node
 */
struct NodeMetadata {
    std::string name;                           // Human-readable node name
    std::string description;                    // Node description
    std::string model_path;                     // Path to model file (for MODEL_INFERENCE)
    ModelType model_type;                       // Type of model
    std::vector<TensorShape> input_shapes;      // Expected input tensor shapes
    std::vector<TensorShape> output_shapes;     // Expected output tensor shapes
    std::unordered_map<std::string, std::string> parameters; // Node-specific parameters
    std::chrono::milliseconds timeout_ms;       // Execution timeout
    int priority;                              // Execution priority (higher = more important)
    bool is_critical;                          // Whether node failure should stop entire graph
    std::vector<std::string> tags;             // Tags for categorization
    
    NodeMetadata() : model_type(ModelType::ONNX), timeout_ms(std::chrono::milliseconds(5000)), 
                     priority(0), is_critical(false) {}
};

/**
 * @struct EdgeMetadata
 * @brief Metadata associated with a graph edge
 */
struct EdgeMetadata {
    std::string name;                           // Human-readable edge name
    std::string description;                    // Edge description
    std::vector<TensorShape> data_shapes;       // Data shapes flowing through edge
    DataType data_type;                         // Data type
    std::chrono::milliseconds transfer_timeout_ms; // Data transfer timeout
    size_t buffer_size;                         // Buffer size for data transfer
    bool is_buffered;                          // Whether to buffer data
    std::unordered_map<std::string, std::string> parameters; // Edge-specific parameters
    
    EdgeMetadata() : data_type(DataType::FLOAT32), transfer_timeout_ms(std::chrono::milliseconds(1000)),
                     buffer_size(1024 * 1024), is_buffered(true) {}
};

/**
 * @struct ExecutionContext
 * @brief Context for graph execution
 */
struct ExecutionContext {
    std::string session_id;                     // Unique session identifier
    std::string graph_id;                       // Graph identifier
    std::chrono::steady_clock::time_point start_time; // Execution start time
    std::atomic<bool> should_stop;              // Stop signal for execution
    std::atomic<int> active_nodes;              // Number of currently active nodes
    std::unordered_map<std::string, std::any> shared_data; // Shared data between nodes
    mutable std::mutex shared_data_mutex;       // Mutex for shared data access
    
    ExecutionContext() : should_stop(false), active_nodes(0) {
        start_time = std::chrono::steady_clock::now();
    }
    
    // Copy constructor
    ExecutionContext(const ExecutionContext& other) 
        : session_id(other.session_id), graph_id(other.graph_id), 
          start_time(other.start_time), should_stop(other.should_stop.load()),
          active_nodes(other.active_nodes.load()), shared_data(other.shared_data) {
    }
    
    // Move constructor
    ExecutionContext(ExecutionContext&& other) noexcept
        : session_id(std::move(other.session_id)), graph_id(std::move(other.graph_id)),
          start_time(other.start_time), should_stop(other.should_stop.load()),
          active_nodes(other.active_nodes.load()), shared_data(std::move(other.shared_data)) {
    }
    
    // Copy assignment operator
    ExecutionContext& operator=(const ExecutionContext& other) {
        if (this != &other) {
            session_id = other.session_id;
            graph_id = other.graph_id;
            start_time = other.start_time;
            should_stop.store(other.should_stop.load());
            active_nodes.store(other.active_nodes.load());
            shared_data = other.shared_data;
        }
        return *this;
    }
    
    // Move assignment operator
    ExecutionContext& operator=(ExecutionContext&& other) noexcept {
        if (this != &other) {
            session_id = std::move(other.session_id);
            graph_id = std::move(other.graph_id);
            start_time = other.start_time;
            should_stop.store(other.should_stop.load());
            active_nodes.store(other.active_nodes.load());
            shared_data = std::move(other.shared_data);
        }
        return *this;
    }
};

/**
 * @struct NodeExecutionResult
 * @brief Result of node execution
 */
struct NodeExecutionResult {
    NodeStatus status;                          // Final execution status
    std::chrono::milliseconds execution_time;   // Time taken to execute
    std::string error_message;                  // Error message if failed
    std::vector<Tensor> outputs;                // Output tensors
    std::unordered_map<std::string, std::any> metadata; // Additional metadata
    
    NodeExecutionResult() : status(NodeStatus::PENDING), execution_time(std::chrono::milliseconds(0)) {}
    
    // Move constructor
    NodeExecutionResult(NodeExecutionResult&& other) noexcept
        : status(other.status), execution_time(other.execution_time),
          error_message(std::move(other.error_message)), outputs(std::move(other.outputs)),
          metadata(std::move(other.metadata)) {}
    
    // Move assignment operator
    NodeExecutionResult& operator=(NodeExecutionResult&& other) noexcept {
        if (this != &other) {
            status = other.status;
            execution_time = other.execution_time;
            error_message = std::move(other.error_message);
            outputs = std::move(other.outputs);
            metadata = std::move(other.metadata);
        }
        return *this;
    }
    
    // Copy constructor - disabled due to Tensor non-copyability
    NodeExecutionResult(const NodeExecutionResult& other) = delete;
    NodeExecutionResult& operator=(const NodeExecutionResult& other) = delete;
};

/**
 * @struct GraphExecutionStats
 * @brief Statistics for graph execution
 */
struct GraphExecutionStats {
    std::atomic<uint64_t> total_nodes_executed;
    std::atomic<uint64_t> successful_nodes;
    std::atomic<uint64_t> failed_nodes;
    std::atomic<uint64_t> cancelled_nodes;
    std::atomic<uint64_t> total_execution_time_ms;
    std::atomic<uint64_t> max_node_execution_time_ms;
    std::atomic<uint64_t> min_node_execution_time_ms;
    std::atomic<size_t> peak_memory_usage;
    std::atomic<size_t> total_data_transferred;
    
    GraphExecutionStats() : total_nodes_executed(0), successful_nodes(0), failed_nodes(0),
                           cancelled_nodes(0), total_execution_time_ms(0),
                           max_node_execution_time_ms(0),
                           min_node_execution_time_ms(std::numeric_limits<uint64_t>::max()),
                           peak_memory_usage(0), total_data_transferred(0) {}
    
    // Move-only semantics
    GraphExecutionStats(const GraphExecutionStats&) = delete;
    GraphExecutionStats& operator=(const GraphExecutionStats&) = delete;
    
    GraphExecutionStats(GraphExecutionStats&& other) noexcept
        : total_nodes_executed(other.total_nodes_executed.load()),
          successful_nodes(other.successful_nodes.load()),
          failed_nodes(other.failed_nodes.load()),
          cancelled_nodes(other.cancelled_nodes.load()),
          total_execution_time_ms(other.total_execution_time_ms.load()),
          max_node_execution_time_ms(other.max_node_execution_time_ms.load()),
          min_node_execution_time_ms(other.min_node_execution_time_ms.load()),
          peak_memory_usage(other.peak_memory_usage.load()),
          total_data_transferred(other.total_data_transferred.load()) {}
    
    GraphExecutionStats& operator=(GraphExecutionStats&& other) noexcept {
        if (this != &other) {
            total_nodes_executed.store(other.total_nodes_executed.load());
            successful_nodes.store(other.successful_nodes.load());
            failed_nodes.store(other.failed_nodes.load());
            cancelled_nodes.store(other.cancelled_nodes.load());
            total_execution_time_ms.store(other.total_execution_time_ms.load());
            max_node_execution_time_ms.store(other.max_node_execution_time_ms.load());
            min_node_execution_time_ms.store(other.min_node_execution_time_ms.load());
            peak_memory_usage.store(other.peak_memory_usage.load());
            total_data_transferred.store(other.total_data_transferred.load());
        }
        return *this;
    }
    
    /**
     * @brief Get a snapshot of current stats
     */
    struct Snapshot {
        uint64_t total_nodes_executed;
        uint64_t successful_nodes;
        uint64_t failed_nodes;
        uint64_t cancelled_nodes;
        std::chrono::milliseconds total_execution_time;
        std::chrono::milliseconds max_node_execution_time;
        std::chrono::milliseconds min_node_execution_time;
        size_t peak_memory_usage;
        size_t total_data_transferred;
        double success_rate;
        double average_node_time_ms;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_nodes_executed = total_nodes_executed.load();
        snapshot.successful_nodes = successful_nodes.load();
        snapshot.failed_nodes = failed_nodes.load();
        snapshot.cancelled_nodes = cancelled_nodes.load();
        snapshot.total_execution_time = std::chrono::milliseconds(total_execution_time_ms.load());
        snapshot.max_node_execution_time = std::chrono::milliseconds(max_node_execution_time_ms.load());
        snapshot.min_node_execution_time = std::chrono::milliseconds(min_node_execution_time_ms.load());
        snapshot.peak_memory_usage = peak_memory_usage.load();
        snapshot.total_data_transferred = total_data_transferred.load();
        
        if (snapshot.total_nodes_executed > 0) {
            snapshot.success_rate = static_cast<double>(snapshot.successful_nodes) / snapshot.total_nodes_executed;
            snapshot.average_node_time_ms = static_cast<double>(snapshot.total_execution_time.count()) / snapshot.total_nodes_executed;
        } else {
            snapshot.success_rate = 0.0;
            snapshot.average_node_time_ms = 0.0;
        }
        
        return snapshot;
    }
};

/**
 * @struct GraphConfig
 * @brief Configuration for graph execution
 */
struct GraphConfig {
    ExecutionMode default_execution_mode;       // Default execution mode
    int max_parallel_nodes;                     // Maximum parallel nodes
    std::chrono::milliseconds global_timeout;   // Global execution timeout
    bool enable_profiling;                      // Enable profiling
    bool enable_optimization;                   // Enable optimization
    bool enable_distributed_execution;          // Enable distributed execution
    std::string optimization_policy;            // Optimization policy to use
    size_t max_memory_usage;                    // Maximum memory usage
    bool fail_fast;                            // Stop on first node failure
    
    GraphConfig() : default_execution_mode(ExecutionMode::PARALLEL), max_parallel_nodes(4),
                   global_timeout(std::chrono::milliseconds(30000)), enable_profiling(true),
                   enable_optimization(true), enable_distributed_execution(false),
                   optimization_policy("RuleBasedPolicy"), max_memory_usage(1024 * 1024 * 1024),
                   fail_fast(false) {}
};

} // namespace edge_ai
