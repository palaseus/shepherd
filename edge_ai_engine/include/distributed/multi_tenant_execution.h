/**
 * @file multi_tenant_execution.h
 * @brief Multi-tenant priority-aware execution system with resource quotas and SLA enforcement
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <functional>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <queue>
#include <deque>
#include <unordered_map>
#include "core/types.h"
#include "distributed/cluster_types.h"
#include "graph/graph.h"

// Forward declarations
namespace edge_ai {
    class ClusterManager;
}

namespace edge_ai {

/**
 * @brief Tenant resource quota
 */
struct TenantResourceQuota {
    std::string tenant_id;
    std::string tenant_name;
    
    // Resource limits
    uint32_t max_nodes{10};
    uint64_t max_memory_mb{16384};
    uint64_t max_gpu_memory_mb{8192};
    uint32_t max_cpu_cores{32};
    uint32_t max_gpu_cores{8};
    uint64_t max_bandwidth_mbps{1000};
    
    // Cost limits
    double max_cost_per_hour{10.0};
    double max_cost_per_day{100.0};
    double max_cost_per_month{1000.0};
    
    // Performance limits
    std::chrono::milliseconds max_latency_ms{1000};
    double min_throughput_ops_per_sec{10.0};
    double max_energy_consumption_watts{1000.0};
    
    // Priority and SLA
    uint32_t priority_level{1};  // 1 = highest, 10 = lowest
    double sla_availability_target{0.99};
    std::chrono::milliseconds sla_response_time_ms{100};
    std::chrono::milliseconds sla_recovery_time_ms{300000};  // 5 minutes
    
    // Current usage
    uint32_t current_nodes{0};
    uint64_t current_memory_mb{0};
    uint64_t current_gpu_memory_mb{0};
    uint32_t current_cpu_cores{0};
    uint32_t current_gpu_cores{0};
    double current_cost_per_hour{0.0};
    
    // Usage statistics
    std::chrono::steady_clock::time_point last_usage_update;
    std::vector<double> hourly_usage_history;
    std::vector<double> daily_cost_history;
    
    TenantResourceQuota() {
        last_usage_update = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Tenant execution policy
 */
struct TenantExecutionPolicy {
    std::string tenant_id;
    std::string policy_name;
    
    // Execution preferences
    enum class ExecutionPreference {
        PERFORMANCE_OPTIMIZED,
        COST_OPTIMIZED,
        ENERGY_OPTIMIZED,
        BALANCED,
        CUSTOM
    } execution_preference{ExecutionPreference::BALANCED};
    
    // Resource allocation strategy
    enum class AllocationStrategy {
        STATIC,
        DYNAMIC,
        ON_DEMAND,
        RESERVED,
        SPOT
    } allocation_strategy{AllocationStrategy::DYNAMIC};
    
    // Scheduling preferences
    bool allow_preemption{true};
    bool allow_migration{true};
    bool allow_scaling{true};
    uint32_t max_concurrent_tasks{100};
    std::chrono::milliseconds task_timeout_ms{300000};  // 5 minutes
    
    // Quality of Service
    uint32_t min_guaranteed_resources{1};
    double max_resource_contention{0.8};
    bool enforce_isolation{true};
    bool enable_monitoring{true};
    
    // Custom parameters
    std::map<std::string, std::string> custom_parameters;
    std::vector<std::string> allowed_node_types;
    std::vector<std::string> restricted_node_types;
    
    TenantExecutionPolicy() = default;
};

/**
 * @brief Priority-based task scheduling
 */
struct PriorityTask {
    std::string task_id;
    std::string tenant_id;
    std::string graph_id;
    
    // Priority information
    uint32_t priority_level{5};  // 1 = highest, 10 = lowest
    uint32_t tenant_priority{5};
    uint32_t task_priority{5};
    double calculated_priority{0.0};
    
    // Task characteristics
    std::chrono::milliseconds estimated_duration{0};
    uint64_t estimated_memory_usage{0};
    uint32_t estimated_cpu_cores{1};
    bool requires_gpu{false};
    bool requires_low_latency{false};
    
    // SLA requirements
    std::chrono::steady_clock::time_point deadline;
    std::chrono::milliseconds max_latency_ms{1000};
    double min_throughput_ops_per_sec{1.0};
    
    // Scheduling metadata
    std::chrono::steady_clock::time_point submission_time;
    std::chrono::steady_clock::time_point scheduled_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point completion_time;
    
    // Execution status
    enum class TaskStatus {
        PENDING,
        SCHEDULED,
        RUNNING,
        COMPLETED,
        FAILED,
        CANCELLED,
        PREEMPTED
    } status{TaskStatus::PENDING};
    
    // Resource allocation
    std::string assigned_node_id;
    std::vector<std::string> allocated_resources;
    double resource_utilization{0.0};
    
    PriorityTask() {
        submission_time = std::chrono::steady_clock::now();
        deadline = submission_time + std::chrono::hours(1);
    }
};

/**
 * @brief Resource contention resolution
 */
struct ResourceContentionResolution {
    std::string contention_id;
    std::string resource_id;
    std::string resource_type;  // "node", "memory", "gpu", "bandwidth"
    
    // Contending tenants
    std::vector<std::string> contending_tenants;
    std::map<std::string, double> tenant_demands;
    std::map<std::string, uint32_t> tenant_priorities;
    
    // Resolution strategy
    enum class ResolutionStrategy {
        PRIORITY_BASED,
        FAIR_SHARING,
        PROPORTIONAL_SHARING,
        AUCTION_BASED,
        NEGOTIATION_BASED,
        CUSTOM_ALGORITHM
    } resolution_strategy{ResolutionStrategy::PRIORITY_BASED};
    
    // Resolution result
    std::map<std::string, double> allocated_resources;
    std::map<std::string, std::chrono::milliseconds> allocation_duration;
    std::vector<std::string> preempted_tasks;
    std::vector<std::string> migrated_tasks;
    
    // Resolution metadata
    std::chrono::steady_clock::time_point resolution_time;
    std::chrono::milliseconds resolution_duration{0};
    double resolution_efficiency{0.0};
    std::string resolution_algorithm;
    
    ResourceContentionResolution() {
        resolution_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief SLA monitoring and enforcement
 */
struct SLAMonitoring {
    std::string tenant_id;
    std::string sla_id;
    
    // SLA metrics
    double availability_percentage{0.0};
    std::chrono::milliseconds avg_response_time{0};
    std::chrono::milliseconds max_response_time{0};
    double throughput_ops_per_sec{0.0};
    double error_rate{0.0};
    
    // SLA violations
    uint64_t total_violations{0};
    uint64_t availability_violations{0};
    uint64_t response_time_violations{0};
    uint64_t throughput_violations{0};
    uint64_t error_rate_violations{0};
    
    // Violation details
    std::vector<std::chrono::steady_clock::time_point> violation_times;
    std::vector<std::string> violation_reasons;
    std::vector<double> violation_severities;
    
    // Compliance tracking
    double sla_compliance_rate{0.0};
    std::chrono::steady_clock::time_point last_compliance_check;
    std::vector<double> compliance_history;
    
    // Penalty tracking
    double total_penalties{0.0};
    double current_penalty_rate{0.0};
    std::vector<double> penalty_history;
    
    SLAMonitoring() {
        last_compliance_check = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Multi-tenant execution statistics
 */
struct MultiTenantExecutionStats {
    std::atomic<uint64_t> total_tenants{0};
    std::atomic<uint64_t> active_tenants{0};
    std::atomic<uint64_t> total_tasks_scheduled{0};
    std::atomic<uint64_t> total_tasks_completed{0};
    std::atomic<uint64_t> total_tasks_failed{0};
    std::atomic<uint64_t> total_tasks_preempted{0};
    
    // Resource utilization
    std::atomic<double> avg_cpu_utilization{0.0};
    std::atomic<double> avg_memory_utilization{0.0};
    std::atomic<double> avg_gpu_utilization{0.0};
    std::atomic<double> avg_bandwidth_utilization{0.0};
    
    // Performance metrics
    std::atomic<double> avg_task_completion_time_ms{0.0};
    std::atomic<double> avg_task_waiting_time_ms{0.0};
    std::atomic<double> avg_task_throughput{0.0};
    std::atomic<double> avg_task_success_rate{0.0};
    
    // SLA compliance
    std::atomic<uint64_t> total_sla_violations{0};
    std::atomic<double> avg_sla_compliance_rate{0.0};
    std::atomic<double> avg_availability{0.0};
    std::atomic<double> avg_response_time_ms{0.0};
    
    // Resource contention
    std::atomic<uint64_t> total_contention_events{0};
    std::atomic<uint64_t> resolved_contention_events{0};
    std::atomic<double> avg_contention_resolution_time_ms{0.0};
    std::atomic<double> contention_resolution_success_rate{0.0};
    
    // Cost and efficiency
    std::atomic<double> total_cost{0.0};
    std::atomic<double> avg_cost_per_task{0.0};
    std::atomic<double> resource_efficiency{0.0};
    std::atomic<double> energy_efficiency{0.0};
    
    MultiTenantExecutionStats() = default;
    
    struct Snapshot {
        uint64_t total_tenants;
        uint64_t active_tenants;
        uint64_t total_tasks_scheduled;
        uint64_t total_tasks_completed;
        uint64_t total_tasks_failed;
        uint64_t total_tasks_preempted;
        double avg_cpu_utilization;
        double avg_memory_utilization;
        double avg_gpu_utilization;
        double avg_bandwidth_utilization;
        double avg_task_completion_time_ms;
        double avg_task_waiting_time_ms;
        double avg_task_throughput;
        double avg_task_success_rate;
        uint64_t total_sla_violations;
        double avg_sla_compliance_rate;
        double avg_availability;
        double avg_response_time_ms;
        uint64_t total_contention_events;
        uint64_t resolved_contention_events;
        double avg_contention_resolution_time_ms;
        double contention_resolution_success_rate;
        double total_cost;
        double avg_cost_per_task;
        double resource_efficiency;
        double energy_efficiency;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_tenants = total_tenants.load();
        snapshot.active_tenants = active_tenants.load();
        snapshot.total_tasks_scheduled = total_tasks_scheduled.load();
        snapshot.total_tasks_completed = total_tasks_completed.load();
        snapshot.total_tasks_failed = total_tasks_failed.load();
        snapshot.total_tasks_preempted = total_tasks_preempted.load();
        snapshot.avg_cpu_utilization = avg_cpu_utilization.load();
        snapshot.avg_memory_utilization = avg_memory_utilization.load();
        snapshot.avg_gpu_utilization = avg_gpu_utilization.load();
        snapshot.avg_bandwidth_utilization = avg_bandwidth_utilization.load();
        snapshot.avg_task_completion_time_ms = avg_task_completion_time_ms.load();
        snapshot.avg_task_waiting_time_ms = avg_task_waiting_time_ms.load();
        snapshot.avg_task_throughput = avg_task_throughput.load();
        snapshot.avg_task_success_rate = avg_task_success_rate.load();
        snapshot.total_sla_violations = total_sla_violations.load();
        snapshot.avg_sla_compliance_rate = avg_sla_compliance_rate.load();
        snapshot.avg_availability = avg_availability.load();
        snapshot.avg_response_time_ms = avg_response_time_ms.load();
        snapshot.total_contention_events = total_contention_events.load();
        snapshot.resolved_contention_events = resolved_contention_events.load();
        snapshot.avg_contention_resolution_time_ms = avg_contention_resolution_time_ms.load();
        snapshot.contention_resolution_success_rate = contention_resolution_success_rate.load();
        snapshot.total_cost = total_cost.load();
        snapshot.avg_cost_per_task = avg_cost_per_task.load();
        snapshot.resource_efficiency = resource_efficiency.load();
        snapshot.energy_efficiency = energy_efficiency.load();
        return snapshot;
    }
};

/**
 * @brief Multi-tenant execution system
 */
class MultiTenantExecution {
public:
    /**
     * @brief Constructor
     * @param cluster_manager Cluster manager for node information
     */
    explicit MultiTenantExecution(std::shared_ptr<ClusterManager> cluster_manager);
    
    /**
     * @brief Destructor
     */
    ~MultiTenantExecution();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Tenant management
    Status RegisterTenant(const std::string& tenant_id, const TenantResourceQuota& quota, 
                         const TenantExecutionPolicy& policy);
    Status UnregisterTenant(const std::string& tenant_id);
    Status UpdateTenantQuota(const std::string& tenant_id, const TenantResourceQuota& quota);
    Status UpdateTenantPolicy(const std::string& tenant_id, const TenantExecutionPolicy& policy);
    
    // Task scheduling
    Status ScheduleTask(const PriorityTask& task);
    Status ScheduleBatchTasks(const std::vector<PriorityTask>& tasks);
    Status CancelTask(const std::string& task_id);
    Status PreemptTask(const std::string& task_id, const std::string& reason);
    Status GetTaskStatus(const std::string& task_id, PriorityTask& task);
    
    // Resource allocation
    Status AllocateResources(const std::string& tenant_id, const std::map<std::string, double>& requirements);
    Status DeallocateResources(const std::string& tenant_id, const std::map<std::string, double>& resources);
    Status CheckResourceAvailability(const std::string& tenant_id, 
                                   const std::map<std::string, double>& requirements);
    
    // Resource contention resolution
    Status DetectResourceContention();
    Status ResolveResourceContention(const ResourceContentionResolution& resolution);
    Status NegotiateResourceSharing(const std::vector<std::string>& tenant_ids, 
                                  const std::string& resource_id);
    
    // SLA monitoring and enforcement
    Status MonitorSLACompliance(const std::string& tenant_id);
    Status EnforceSLAViolations(const std::string& tenant_id);
    Status CalculateSLAPenalties(const std::string& tenant_id, double& penalty);
    Status GenerateSLAReport(const std::string& tenant_id, SLAMonitoring& report);
    
    // Priority management
    Status UpdateTaskPriority(const std::string& task_id, uint32_t new_priority);
    Status UpdateTenantPriority(const std::string& tenant_id, uint32_t new_priority);
    Status CalculateDynamicPriority(const std::string& task_id, double& priority);
    
    // Resource isolation
    Status EnforceResourceIsolation(const std::string& tenant_id);
    Status ValidateResourceIsolation(const std::string& tenant_id);
    Status ImplementResourceQuotas(const std::string& tenant_id);
    
    // Cost management
    Status CalculateResourceCost(const std::string& tenant_id, const std::map<std::string, double>& resources, 
                                double& cost);
    Status TrackResourceUsage(const std::string& tenant_id, const std::map<std::string, double>& usage);
    Status GenerateCostReport(const std::string& tenant_id, std::map<std::string, double>& costs);
    
    // Performance optimization
    Status OptimizeResourceAllocation();
    Status OptimizeTaskScheduling();
    Status OptimizeTenantPlacement();
    Status OptimizeCostEfficiency();
    
    // Statistics and monitoring
    MultiTenantExecutionStats::Snapshot GetStats() const;
    void ResetStats();
    Status GenerateExecutionReport();
    
    // Configuration
    void SetSchedulingEnabled(bool enabled);
    void SetResourceContentionResolutionEnabled(bool enabled);
    void SetSLAEnforcementEnabled(bool enabled);
    void SetCostTrackingEnabled(bool enabled);
    void SetIsolationEnforcementEnabled(bool enabled);

private:
    // Internal scheduling methods
    Status CalculateTaskPriority(const PriorityTask& task, double& priority);
    Status SelectOptimalNode(const PriorityTask& task, std::string& node_id);
    Status AllocateNodeResources(const std::string& node_id, const PriorityTask& task);
    Status ExecuteTask(const PriorityTask& task);
    
    // Resource management algorithms
    Status ImplementFairSharing(const std::vector<std::string>& tenant_ids, 
                               const std::string& resource_id);
    Status ImplementProportionalSharing(const std::vector<std::string>& tenant_ids, 
                                       const std::string& resource_id);
    Status ImplementAuctionBasedAllocation(const std::vector<std::string>& tenant_ids, 
                                          const std::string& resource_id);
    
    // SLA enforcement algorithms
    Status DetectSLAViolations(const std::string& tenant_id, std::vector<std::string>& violations);
    Status ApplySLAViolationPenalties(const std::string& tenant_id, const std::vector<std::string>& violations);
    Status ImplementSLAComplianceMeasures(const std::string& tenant_id);
    
    // Cost calculation algorithms
    double CalculateNodeCost(const std::string& node_id, const std::map<std::string, double>& resources);
    double CalculateNetworkCost(const std::string& source_node_id, const std::string& target_node_id, 
                               uint64_t data_size);
    double CalculateEnergyCost(const std::string& node_id, double energy_consumption);
    
    // Threading and synchronization
    void TaskSchedulingThread();
    void ResourceManagementThread();
    void SLAMonitoringThread();
    void CostTrackingThread();
    void ContentionResolutionThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> scheduling_enabled_{true};
    std::atomic<bool> resource_contention_resolution_enabled_{true};
    std::atomic<bool> sla_enforcement_enabled_{true};
    std::atomic<bool> cost_tracking_enabled_{true};
    std::atomic<bool> isolation_enforcement_enabled_{true};
    
    // Dependencies
    std::shared_ptr<ClusterManager> cluster_manager_;
    
    // Tenant management
    mutable std::mutex tenant_mutex_;
    std::map<std::string, TenantResourceQuota> tenant_quotas_;
    std::map<std::string, TenantExecutionPolicy> tenant_policies_;
    std::map<std::string, SLAMonitoring> sla_monitoring_;
    
    // Task scheduling
    mutable std::mutex scheduling_mutex_;
    std::priority_queue<PriorityTask, std::vector<PriorityTask>, 
                       std::function<bool(const PriorityTask&, const PriorityTask&)>> task_queue_;
    std::map<std::string, PriorityTask> active_tasks_;
    std::map<std::string, PriorityTask> completed_tasks_;
    
    // Resource allocation
    mutable std::mutex resource_mutex_;
    std::map<std::string, std::map<std::string, double>> tenant_resource_allocations_;
    std::map<std::string, std::map<std::string, double>> node_resource_allocations_;
    std::map<std::string, std::vector<std::string>> tenant_allocated_nodes_;
    
    // Resource contention
    mutable std::mutex contention_mutex_;
    std::queue<ResourceContentionResolution> pending_contention_resolutions_;
    std::map<std::string, ResourceContentionResolution> active_contention_resolutions_;
    std::vector<ResourceContentionResolution> contention_history_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    MultiTenantExecutionStats stats_;
    
    // Threading
    std::thread task_scheduling_thread_;
    std::thread resource_management_thread_;
    std::thread sla_monitoring_thread_;
    std::thread cost_tracking_thread_;
    std::thread contention_resolution_thread_;
    
    std::condition_variable scheduling_cv_;
    std::condition_variable resource_cv_;
    std::condition_variable sla_cv_;
    std::condition_variable cost_cv_;
    std::condition_variable contention_cv_;
    
    mutable std::mutex scheduling_cv_mutex_;
    mutable std::mutex resource_cv_mutex_;
    mutable std::mutex sla_cv_mutex_;
    mutable std::mutex cost_cv_mutex_;
    mutable std::mutex contention_cv_mutex_;
};

} // namespace edge_ai
