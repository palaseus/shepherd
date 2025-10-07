/**
 * @file distributed_scheduler.h
 * @brief Distributed scheduling and orchestration for multi-node execution
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include "graph/graph_types.h"
#include "distributed/cluster_types.h"
#include "distributed/graph_partitioner.h"
#include "core/types.h"

namespace edge_ai {

/**
 * @brief Distributed execution task for cross-node coordination
 */
struct DistributedTask {
    std::string task_id;
    std::string partition_id;
    std::string source_node_id;
    std::string target_node_id;
    std::shared_ptr<GraphPartition> partition;
    
    // Task metadata
    TaskPriority priority{TaskPriority::NORMAL};
    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point completion_time;
    
    // Execution state
    std::atomic<TaskStatus> status{TaskStatus::PENDING};
    std::atomic<bool> is_critical{false};
    std::atomic<uint32_t> retry_count{0};
    std::atomic<uint32_t> max_retries{3};
    
    // Dependencies
    std::set<std::string> depends_on_tasks;
    std::set<std::string> dependent_tasks;
    
    // Data transfer
    std::map<std::string, std::vector<uint8_t>> input_data;
    std::map<std::string, std::vector<uint8_t>> output_data;
    std::atomic<uint64_t> data_size_bytes{0};
    
    // Error handling
    std::string error_message;
    std::string failure_reason;
    
    DistributedTask() = default;
    DistributedTask(const std::string& id, const std::string& part_id, 
                   const std::string& source, const std::string& target)
        : task_id(id), partition_id(part_id), source_node_id(source), target_node_id(target) {
        submit_time = std::chrono::steady_clock::now();
    }
    
    // Disable copy, enable move
    DistributedTask(const DistributedTask&) = delete;
    DistributedTask& operator=(const DistributedTask&) = delete;
    DistributedTask(DistributedTask&&) = default;
    DistributedTask& operator=(DistributedTask&&) = default;
};

/**
 * @brief Distributed execution result
 */
struct DistributedExecutionResult {
    std::string task_id;
    std::string partition_id;
    Status execution_status{Status::NOT_INITIALIZED};
    std::chrono::milliseconds execution_time_ms{0};
    std::chrono::milliseconds data_transfer_time_ms{0};
    std::chrono::milliseconds total_time_ms{0};
    
    // Results
    std::map<std::string, std::vector<uint8_t>> output_data;
    std::map<std::string, std::string> metadata;
    
    // Performance metrics
    double cpu_usage_percent{0.0};
    double memory_usage_percent{0.0};
    double gpu_usage_percent{0.0};
    uint64_t memory_used_bytes{0};
    
    // Error information
    std::string error_message;
    std::string stack_trace;
    
    DistributedExecutionResult() = default;
};

/**
 * @brief Distributed scheduler configuration
 */
struct DistributedSchedulerConfig {
    // Scheduling parameters
    uint32_t max_concurrent_tasks{100};
    uint32_t max_tasks_per_node{10};
    uint32_t task_timeout_ms{30000};
    uint32_t retry_delay_ms{1000};
    uint32_t max_retries{3};
    
    // Load balancing
    bool enable_load_balancing{true};
    double load_balance_threshold{0.2};
    uint32_t rebalancing_interval_ms{5000};
    
    // Data transfer
    bool enable_data_compression{true};
    uint32_t max_data_size_mb{100};
    uint32_t data_transfer_timeout_ms{10000};
    
    // Fault tolerance
    bool enable_fault_tolerance{true};
    uint32_t node_failure_timeout_ms{5000};
    bool enable_automatic_retry{true};
    
    // Performance monitoring
    bool enable_performance_monitoring{true};
    uint32_t metrics_collection_interval_ms{1000};
    
    DistributedSchedulerConfig() = default;
};

/**
 * @brief Distributed scheduler statistics
 */
struct DistributedSchedulerStats {
    std::atomic<uint32_t> total_tasks_submitted{0};
    std::atomic<uint32_t> total_tasks_completed{0};
    std::atomic<uint32_t> total_tasks_failed{0};
    std::atomic<uint32_t> total_tasks_retried{0};
    std::atomic<uint32_t> total_tasks_cancelled{0};
    
    // Performance metrics
    std::atomic<double> avg_execution_time_ms{0.0};
    std::atomic<double> avg_data_transfer_time_ms{0.0};
    std::atomic<double> avg_total_time_ms{0.0};
    std::atomic<double> throughput_tasks_per_sec{0.0};
    
    // Load balancing metrics
    std::atomic<uint32_t> rebalancing_events{0};
    std::atomic<uint32_t> migration_events{0};
    std::atomic<uint64_t> total_migration_time_ms{0};
    
    // Fault tolerance metrics
    std::atomic<uint32_t> node_failures_detected{0};
    std::atomic<uint32_t> tasks_rerouted{0};
    std::atomic<uint32_t> recovery_events{0};
    
    DistributedSchedulerStats() = default;
    
    struct Snapshot {
        uint32_t total_tasks_submitted;
        uint32_t total_tasks_completed;
        uint32_t total_tasks_failed;
        uint32_t total_tasks_retried;
        uint32_t total_tasks_cancelled;
        double avg_execution_time_ms;
        double avg_data_transfer_time_ms;
        double avg_total_time_ms;
        double throughput_tasks_per_sec;
        uint32_t rebalancing_events;
        uint32_t migration_events;
        uint64_t total_migration_time_ms;
        uint32_t node_failures_detected;
        uint32_t tasks_rerouted;
        uint32_t recovery_events;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_tasks_submitted = total_tasks_submitted.load();
        snapshot.total_tasks_completed = total_tasks_completed.load();
        snapshot.total_tasks_failed = total_tasks_failed.load();
        snapshot.total_tasks_retried = total_tasks_retried.load();
        snapshot.total_tasks_cancelled = total_tasks_cancelled.load();
        snapshot.avg_execution_time_ms = avg_execution_time_ms.load();
        snapshot.avg_data_transfer_time_ms = avg_data_transfer_time_ms.load();
        snapshot.avg_total_time_ms = avg_total_time_ms.load();
        snapshot.throughput_tasks_per_sec = throughput_tasks_per_sec.load();
        snapshot.rebalancing_events = rebalancing_events.load();
        snapshot.migration_events = migration_events.load();
        snapshot.total_migration_time_ms = total_migration_time_ms.load();
        snapshot.node_failures_detected = node_failures_detected.load();
        snapshot.tasks_rerouted = tasks_rerouted.load();
        snapshot.recovery_events = recovery_events.load();
        return snapshot;
    }
};

/**
 * @brief Distributed scheduler for multi-node orchestration
 */
class DistributedScheduler {
public:
    using TaskCompletionCallback = std::function<void(const DistributedExecutionResult&)>;
    using NodeFailureCallback = std::function<void(const std::string&, const std::string&)>;
    
    /**
     * @brief Constructor
     * @param config Scheduler configuration
     */
    explicit DistributedScheduler(const DistributedSchedulerConfig& config);
    
    /**
     * @brief Destructor
     */
    ~DistributedScheduler();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Task scheduling
    Status SubmitTask(std::unique_ptr<DistributedTask> task);
    Status SubmitPartition(std::shared_ptr<GraphPartition> partition, 
                          const std::string& target_node_id);
    Status CancelTask(const std::string& task_id);
    Status CancelAllTasks();
    
    // Execution management
    Status ExecutePartitioningResult(std::unique_ptr<PartitioningResult> result);
    Status WaitForCompletion(const std::string& task_id, uint32_t timeout_ms = 0);
    Status WaitForAllTasks(uint32_t timeout_ms = 0);
    
    // Node management
    Status RegisterNode(const std::string& node_id, const NodeCapabilities& capabilities);
    Status UnregisterNode(const std::string& node_id);
    Status UpdateNodeStatus(const std::string& node_id, ClusterNodeStatus status);
    Status MarkNodeFailed(const std::string& node_id, const std::string& reason);
    
    // Load balancing and migration
    Status RebalanceTasks();
    Status MigrateTask(const std::string& task_id, const std::string& target_node_id);
    Status MigratePartition(const std::string& partition_id, const std::string& target_node_id);
    
    // Data transfer
    Status TransferData(const std::string& from_node_id, const std::string& to_node_id,
                       const std::map<std::string, std::vector<uint8_t>>& data);
    Status CompressData(const std::vector<uint8_t>& input, std::vector<uint8_t>& output);
    Status DecompressData(const std::vector<uint8_t>& input, std::vector<uint8_t>& output);
    
    // Monitoring and statistics
    DistributedSchedulerStats::Snapshot GetStats() const;
    void ResetStats();
    std::vector<std::string> GetActiveTasks() const;
    std::vector<std::string> GetFailedTasks() const;
    std::map<std::string, uint32_t> GetNodeTaskCounts() const;
    
    // Event handling
    Status RegisterTaskCompletionCallback(TaskCompletionCallback callback);
    Status RegisterNodeFailureCallback(NodeFailureCallback callback);
    
    // Configuration
    DistributedSchedulerConfig GetConfig() const;
    Status UpdateConfig(const DistributedSchedulerConfig& config);

private:
    // Internal scheduling methods
    void SchedulerThreadMain();
    void TaskExecutionThreadMain();
    void LoadBalancingThreadMain();
    void MonitoringThreadMain();
    
    // Task management
    std::unique_ptr<DistributedTask> GetNextTask();
    Status ExecuteTask(std::unique_ptr<DistributedTask> task);
    Status RetryTask(std::unique_ptr<DistributedTask> task);
    void CompleteTask(std::unique_ptr<DistributedTask> task, const DistributedExecutionResult& result);
    
    // Node selection and load balancing
    std::string SelectNodeForTask(const DistributedTask& task) const;
    std::string SelectNodeForMigration(const std::string& current_node_id) const;
    bool ShouldRebalance() const;
    void PerformRebalancing();
    
    // Fault tolerance
    void HandleNodeFailure(const std::string& node_id, const std::string& reason);
    Status RerouteTask(const std::string& task_id, const std::string& new_node_id);
    Status RecoverFailedTasks(const std::string& failed_node_id);
    
    // Data management
    Status SerializeTaskData(const DistributedTask& task, std::vector<uint8_t>& data);
    Status DeserializeTaskData(const std::vector<uint8_t>& data, DistributedTask& task);
    
    // Configuration
    DistributedSchedulerConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Task management
    mutable std::mutex task_queue_mutex_;
    std::priority_queue<std::unique_ptr<DistributedTask>, 
                       std::vector<std::unique_ptr<DistributedTask>>,
                       std::function<bool(const std::unique_ptr<DistributedTask>&,
                                        const std::unique_ptr<DistributedTask>&)>> task_queue_;
    std::condition_variable task_queue_cv_;
    
    mutable std::mutex active_tasks_mutex_;
    std::map<std::string, std::unique_ptr<DistributedTask>> active_tasks_;
    
    mutable std::mutex completed_tasks_mutex_;
    std::map<std::string, DistributedExecutionResult> completed_tasks_;
    
    // Node management
    mutable std::mutex nodes_mutex_;
    std::map<std::string, NodeCapabilities> node_capabilities_;
    std::map<std::string, ClusterNodeStatus> node_status_;
    std::map<std::string, uint32_t> node_task_counts_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    DistributedSchedulerStats stats_;
    
    // Threading
    std::thread scheduler_thread_;
    std::thread task_execution_thread_;
    std::thread load_balancing_thread_;
    std::thread monitoring_thread_;
    
    // Event callbacks
    mutable std::mutex callbacks_mutex_;
    std::vector<TaskCompletionCallback> task_completion_callbacks_;
    std::vector<NodeFailureCallback> node_failure_callbacks_;
    
    // Load balancing state
    std::chrono::steady_clock::time_point last_rebalancing_time_;
    std::atomic<bool> rebalancing_in_progress_{false};
};

} // namespace edge_ai
