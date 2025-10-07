/**
 * @file graph_scheduler.h
 * @brief Graph scheduler for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the GraphScheduler class that orchestrates execution
 * across threads, devices, and nodes, respecting dependencies and exploiting parallelism.
 */

#pragma once

#include "graph_types.h"
#include "graph.h"
#include "core/types.h"
#include "backend/execution_backend.h"
#include "optimization/optimization_manager.h"
#include "profiling/profiler.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <future>
#include <chrono>
#include <limits>
#include <functional>

namespace edge_ai {

/**
 * @struct SchedulerConfig
 * @brief Configuration for graph scheduler
 */
struct GraphSchedulerConfig {
    int num_worker_threads;                     // Number of worker threads
    int max_parallel_nodes;                     // Maximum parallel nodes per execution
    std::chrono::milliseconds node_timeout;     // Default node execution timeout
    std::chrono::milliseconds scheduling_interval; // Scheduling interval
    bool enable_dynamic_scheduling;             // Enable dynamic scheduling
    bool enable_load_balancing;                 // Enable load balancing
    bool enable_fault_tolerance;                // Enable fault tolerance
    bool enable_profiling;                      // Enable scheduling profiling
    std::string optimization_policy;            // Optimization policy to use
    
    GraphSchedulerConfig() : num_worker_threads(4), max_parallel_nodes(4),
                            node_timeout(std::chrono::milliseconds(5000)),
                            scheduling_interval(std::chrono::milliseconds(100)),
                            enable_dynamic_scheduling(true), enable_load_balancing(true),
                            enable_fault_tolerance(true), enable_profiling(true),
                            optimization_policy("RuleBasedPolicy") {}
};

/**
 * @struct NodeExecutionTask
 * @brief Task for node execution
 */
struct NodeExecutionTask {
    std::string node_id;
    std::string graph_id;
    std::string session_id;
    std::vector<Tensor> inputs;
    std::shared_ptr<std::promise<NodeExecutionResult>> result_promise;
    std::chrono::steady_clock::time_point submit_time;
    int priority;
    bool is_critical;
    
    NodeExecutionTask() : priority(0), is_critical(false) {
        submit_time = std::chrono::steady_clock::now();
        result_promise = std::make_shared<std::promise<NodeExecutionResult>>();
    }
    
    // Copy constructor - disabled due to Tensor non-copyability
    NodeExecutionTask(const NodeExecutionTask& other) = delete;
    
    // Move constructor
    NodeExecutionTask(NodeExecutionTask&& other) noexcept
        : node_id(std::move(other.node_id)), graph_id(std::move(other.graph_id)), 
          session_id(std::move(other.session_id)), inputs(std::move(other.inputs)),
          result_promise(std::move(other.result_promise)), submit_time(other.submit_time),
          priority(other.priority), is_critical(other.is_critical) {
    }
    
    // Copy assignment operator - disabled due to Tensor non-copyability
    NodeExecutionTask& operator=(const NodeExecutionTask& other) = delete;
    
    // Move assignment operator
    NodeExecutionTask& operator=(NodeExecutionTask&& other) noexcept {
        if (this != &other) {
            node_id = std::move(other.node_id);
            graph_id = std::move(other.graph_id);
            session_id = std::move(other.session_id);
            inputs = std::move(other.inputs);
            result_promise = std::move(other.result_promise);
            submit_time = other.submit_time;
            priority = other.priority;
            is_critical = other.is_critical;
        }
        return *this;
    }
};

/**
 * @struct SchedulerStats
 * @brief Statistics for graph scheduler
 */
struct GraphSchedulerStats {
    std::atomic<uint64_t> total_tasks_submitted;
    std::atomic<uint64_t> total_tasks_completed;
    std::atomic<uint64_t> total_tasks_failed;
    std::atomic<uint64_t> total_tasks_cancelled;
    std::atomic<uint64_t> total_scheduling_time_ms;
    std::atomic<uint64_t> total_execution_time_ms;
    std::atomic<uint64_t> max_task_execution_time_ms;
    std::atomic<uint64_t> min_task_execution_time_ms;
    std::atomic<size_t> peak_queue_size;
    std::atomic<int> active_workers;
    
    GraphSchedulerStats() : total_tasks_submitted(0), total_tasks_completed(0),
                           total_tasks_failed(0), total_tasks_cancelled(0),
                           total_scheduling_time_ms(0),
                           total_execution_time_ms(0),
                           max_task_execution_time_ms(0),
                           min_task_execution_time_ms(std::numeric_limits<uint64_t>::max()),
                           peak_queue_size(0), active_workers(0) {}
    
    /**
     * @brief Get a snapshot of current stats
     */
    struct Snapshot {
        uint64_t total_tasks_submitted;
        uint64_t total_tasks_completed;
        uint64_t total_tasks_failed;
        uint64_t total_tasks_cancelled;
        std::chrono::milliseconds total_scheduling_time;
        std::chrono::milliseconds total_execution_time;
        std::chrono::milliseconds max_task_execution_time;
        std::chrono::milliseconds min_task_execution_time;
        size_t peak_queue_size;
        int active_workers;
        double success_rate;
        double average_execution_time_ms;
        double average_scheduling_time_ms;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_tasks_submitted = total_tasks_submitted.load();
        snapshot.total_tasks_completed = total_tasks_completed.load();
        snapshot.total_tasks_failed = total_tasks_failed.load();
        snapshot.total_tasks_cancelled = total_tasks_cancelled.load();
        snapshot.total_scheduling_time = std::chrono::milliseconds(total_scheduling_time_ms.load());
        snapshot.total_execution_time = std::chrono::milliseconds(total_execution_time_ms.load());
        snapshot.max_task_execution_time = std::chrono::milliseconds(max_task_execution_time_ms.load());
        snapshot.min_task_execution_time = std::chrono::milliseconds(min_task_execution_time_ms.load());
        snapshot.peak_queue_size = peak_queue_size.load();
        snapshot.active_workers = active_workers.load();
        
        if (snapshot.total_tasks_submitted > 0) {
            snapshot.success_rate = static_cast<double>(snapshot.total_tasks_completed) / snapshot.total_tasks_submitted;
            snapshot.average_execution_time_ms = static_cast<double>(snapshot.total_execution_time.count()) / snapshot.total_tasks_completed;
            snapshot.average_scheduling_time_ms = static_cast<double>(snapshot.total_scheduling_time.count()) / snapshot.total_tasks_submitted;
        } else {
            snapshot.success_rate = 0.0;
            snapshot.average_execution_time_ms = 0.0;
            snapshot.average_scheduling_time_ms = 0.0;
        }
        
        return snapshot;
    }
};

/**
 * @class GraphScheduler
 * @brief Orchestrates graph execution across threads, devices, and nodes
 */
class GraphScheduler {
public:
    GraphScheduler(const GraphSchedulerConfig& config = GraphSchedulerConfig{});
    ~GraphScheduler();
    
    // Lifecycle management
    bool Initialize();
    void Shutdown();
    bool IsInitialized() const { return initialized_.load(); }
    
    // Graph execution
    std::future<GraphExecutionStats::Snapshot> ExecuteGraphAsync(std::shared_ptr<Graph> graph,
                                                               const ExecutionContext& context);
    GraphExecutionStats::Snapshot ExecuteGraph(std::shared_ptr<Graph> graph,
                                              const ExecutionContext& context);
    
    // Task management
    bool SubmitTask(const NodeExecutionTask& task);
    bool CancelTask(const std::string& task_id);
    bool CancelAllTasks();
    
    // Scheduling control
    void PauseScheduling();
    void ResumeScheduling();
    bool IsSchedulingPaused() const { return scheduling_paused_.load(); }
    
    // Statistics and monitoring
    GraphSchedulerStats::Snapshot GetStatsSnapshot() const;
    void ResetStats();
    
    // Configuration
    void UpdateConfig(const GraphSchedulerConfig& config);
    GraphSchedulerConfig GetConfig() const { return config_; }
    
    // Backend management
    bool RegisterBackend(std::shared_ptr<ExecutionBackend> backend);
    bool UnregisterBackend(const std::string& backend_id);
    std::vector<std::string> GetAvailableBackends() const;
    
    // Optimization integration
    void SetOptimizationManager(std::shared_ptr<OptimizationManager> optimization_manager);
    void SetProfiler(Profiler* profiler);
    
private:
    // Worker thread management
    void WorkerThreadMain(int worker_id);
    void SchedulerThreadMain();
    
    // Task scheduling
    std::vector<NodeExecutionTask> GetReadyTasks(std::shared_ptr<Graph> graph);
    void ScheduleTasks(const std::vector<NodeExecutionTask>& tasks);
    void UpdateNodeDependencies(std::shared_ptr<Graph> graph, const std::string& completed_node);
    
    // Task execution
    NodeExecutionResult ExecuteNode(const NodeExecutionTask& task);
    std::shared_ptr<ExecutionBackend> SelectBackend(const NodeExecutionTask& task);
    
    // Load balancing
    int SelectWorker(const NodeExecutionTask& task);
    void UpdateWorkerLoads();
    
    // Fault tolerance
    void HandleTaskFailure(const NodeExecutionTask& task, const std::string& error);
    bool ShouldRetryTask(const NodeExecutionTask& task);
    
    // Statistics updates
    void UpdateStats(const NodeExecutionResult& result, std::chrono::milliseconds execution_time);
    
    // Configuration
    GraphSchedulerConfig config_;
    std::atomic<bool> initialized_;
    std::atomic<bool> shutdown_requested_;
    std::atomic<bool> scheduling_paused_;
    
    // Threading
    std::vector<std::thread> worker_threads_;
    std::thread scheduler_thread_;
    std::atomic<int> active_workers_;
    
    // Task queues
    std::vector<std::unique_ptr<NodeExecutionTask>> task_queue_;
    std::mutex task_queue_mutex_;
    std::condition_variable task_queue_cv_;
    
    // Active tasks
    std::unordered_map<std::string, std::shared_future<NodeExecutionResult>> active_tasks_;
    std::mutex active_tasks_mutex_;
    
    // Backend management
    std::unordered_map<std::string, std::shared_ptr<ExecutionBackend>> backends_;
    mutable std::mutex backends_mutex_;
    
    // Worker load tracking
    std::vector<int> worker_loads_;
    std::vector<bool> worker_busy_;
    mutable std::mutex worker_loads_mutex_;
    mutable std::mutex worker_busy_mutex_;
    
    // Statistics
    GraphSchedulerStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Optimization and profiling
    std::shared_ptr<OptimizationManager> optimization_manager_;
    Profiler* profiler_;
    
    // Task comparison for heap operations
    static bool TaskPriorityComparator(const std::unique_ptr<NodeExecutionTask>& a, const std::unique_ptr<NodeExecutionTask>& b) {
        if (a->is_critical != b->is_critical) {
            return a->is_critical < b->is_critical; // Critical tasks first
        }
        return a->priority < b->priority; // Higher priority first
    }
};

/**
 * @class GraphExecutionOrchestrator
 * @brief High-level orchestrator for complex graph execution scenarios
 */
class GraphExecutionOrchestrator {
public:
    GraphExecutionOrchestrator(std::shared_ptr<GraphScheduler> scheduler);
    ~GraphExecutionOrchestrator() = default;
    
    // Execution orchestration
    std::future<GraphExecutionStats::Snapshot> ExecutePipelineAsync(
        const std::vector<std::shared_ptr<Graph>>& graphs,
        const ExecutionContext& context);
    
    std::future<GraphExecutionStats::Snapshot> ExecuteParallelAsync(
        const std::vector<std::shared_ptr<Graph>>& graphs,
        const ExecutionContext& context);
    
    std::future<GraphExecutionStats::Snapshot> ExecuteConditionalAsync(
        const std::vector<std::shared_ptr<Graph>>& graphs,
        const std::function<int(const ExecutionContext&)>& condition_func,
        const ExecutionContext& context);
    
    // Streaming execution
    std::future<void> ExecuteStreamingAsync(
        std::shared_ptr<Graph> graph,
        const ExecutionContext& context,
        std::function<void(const std::vector<Tensor>&)> output_callback);
    
    // Monitoring and control
    void PauseExecution();
    void ResumeExecution();
    void CancelExecution();
    bool IsExecutionActive() const;
    
private:
    std::shared_ptr<GraphScheduler> scheduler_;
    std::atomic<bool> execution_active_;
    std::mutex execution_mutex_;
};

} // namespace edge_ai
