/**
 * @file reactive_scheduler.h
 * @brief Reactive cluster scheduler with ML-driven orchestration and QoS guarantees
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
#include "core/types.h"
#include "distributed/cluster_types.h"
#include "optimization/optimization_manager.h"
#include "ml_policy/ml_based_policy.h"
#include "graph/graph.h"
#include "distributed/cluster_manager.h"

namespace edge_ai {

/**
 * @brief QoS requirements for graph execution
 */
struct QoSRequirements {
    std::chrono::milliseconds max_latency_ms{1000};
    double min_throughput_ops_per_sec{1.0};
    double max_energy_consumption_watts{100.0};
    uint32_t priority{0};  // Higher = more critical
    bool requires_gpu{false};
    bool requires_low_latency{false};
    bool allows_migration{true};
    
    // SLA constraints
    double availability_target{0.99};  // 99% uptime
    std::chrono::milliseconds recovery_time_ms{5000};
    
    QoSRequirements() = default;
};

/**
 * @brief Reactive scheduling decision
 */
struct ReactiveSchedulingDecision {
    std::string task_id;
    std::string assigned_node_id;
    std::string backup_node_id;
    uint32_t estimated_batch_size{1};
    std::chrono::milliseconds estimated_latency_ms{0};
    double confidence_score{0.0};
    std::string reasoning;
    std::chrono::steady_clock::time_point decision_time;
    
    ReactiveSchedulingDecision() = default;
};

/**
 * @brief Cluster telemetry snapshot
 */
struct ClusterTelemetrySnapshot {
    std::map<std::string, double> node_cpu_usage;
    std::map<std::string, double> node_memory_usage;
    std::map<std::string, double> node_gpu_usage;
    std::map<std::string, double> node_energy_consumption;
    std::map<std::string, double> node_network_latency;
    std::map<std::string, double> node_throughput;
    std::map<std::string, uint32_t> node_active_tasks;
    std::map<std::string, double> node_queue_depth;
    
    std::chrono::steady_clock::time_point timestamp;
    
    ClusterTelemetrySnapshot() {
        timestamp = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Reactive scheduler statistics
 */
struct ReactiveSchedulerStats {
    std::atomic<uint64_t> total_decisions{0};
    std::atomic<uint64_t> successful_decisions{0};
    std::atomic<uint64_t> failed_decisions{0};
    std::atomic<uint64_t> qos_violations{0};
    std::atomic<uint64_t> sla_violations{0};
    std::atomic<uint64_t> predictive_migrations{0};
    std::atomic<uint64_t> emergency_migrations{0};
    
    // Performance metrics
    std::atomic<double> avg_decision_time_ms{0.0};
    std::atomic<double> avg_confidence_score{0.0};
    std::atomic<double> avg_latency_improvement_percent{0.0};
    std::atomic<double> avg_throughput_improvement_percent{0.0};
    std::atomic<double> avg_energy_efficiency_percent{0.0};
    
    // Load balancing metrics
    std::atomic<uint64_t> load_rebalancing_events{0};
    std::atomic<double> avg_load_balance_score{0.0};
    std::atomic<uint64_t> auto_scaling_events{0};
    
    ReactiveSchedulerStats() = default;
    
    struct Snapshot {
        uint64_t total_decisions;
        uint64_t successful_decisions;
        uint64_t failed_decisions;
        uint64_t qos_violations;
        uint64_t sla_violations;
        uint64_t predictive_migrations;
        uint64_t emergency_migrations;
        double avg_decision_time_ms;
        double avg_confidence_score;
        double avg_latency_improvement_percent;
        double avg_throughput_improvement_percent;
        double avg_energy_efficiency_percent;
        uint64_t load_rebalancing_events;
        double avg_load_balance_score;
        uint64_t auto_scaling_events;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_decisions = total_decisions.load();
        snapshot.successful_decisions = successful_decisions.load();
        snapshot.failed_decisions = failed_decisions.load();
        snapshot.qos_violations = qos_violations.load();
        snapshot.sla_violations = sla_violations.load();
        snapshot.predictive_migrations = predictive_migrations.load();
        snapshot.emergency_migrations = emergency_migrations.load();
        snapshot.avg_decision_time_ms = avg_decision_time_ms.load();
        snapshot.avg_confidence_score = avg_confidence_score.load();
        snapshot.avg_latency_improvement_percent = avg_latency_improvement_percent.load();
        snapshot.avg_throughput_improvement_percent = avg_throughput_improvement_percent.load();
        snapshot.avg_energy_efficiency_percent = avg_energy_efficiency_percent.load();
        snapshot.load_rebalancing_events = load_rebalancing_events.load();
        snapshot.avg_load_balance_score = avg_load_balance_score.load();
        snapshot.auto_scaling_events = auto_scaling_events.load();
        return snapshot;
    }
};

/**
 * @brief Reactive cluster scheduler with ML-driven orchestration
 */
class ReactiveScheduler {
public:
    /**
     * @brief Constructor
     * @param optimization_manager Optimization manager for ML policies
     * @param cluster_manager Cluster manager for node information
     */
    ReactiveScheduler(std::shared_ptr<OptimizationManager> optimization_manager,
                     std::shared_ptr<ClusterManager> cluster_manager);
    
    /**
     * @brief Destructor
     */
    ~ReactiveScheduler();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Reactive scheduling
    Status ScheduleTask(const std::string& task_id, 
                       const QoSRequirements& qos_requirements,
                       std::shared_ptr<Graph> graph,
                       ReactiveSchedulingDecision& decision);
    
    Status ScheduleBatch(const std::vector<std::string>& task_ids,
                        const QoSRequirements& qos_requirements,
                        std::shared_ptr<Graph> graph,
                        std::vector<ReactiveSchedulingDecision>& decisions);
    
    // Predictive scheduling
    Status PredictOptimalPlacement(const std::string& task_id,
                                  const QoSRequirements& qos_requirements,
                                  std::shared_ptr<Graph> graph,
                                  std::vector<ReactiveSchedulingDecision>& candidates);
    
    // Load balancing and auto-scaling
    Status RebalanceCluster();
    Status TriggerAutoScaling();
    Status EvaluateScalingNeeds();
    
    // QoS and SLA management
    Status EnforceQoSConstraints();
    Status MonitorSLACompliance();
    Status HandleQoSViolation(const std::string& task_id, const std::string& reason);
    
    // Telemetry and monitoring
    Status UpdateClusterTelemetry();
    ClusterTelemetrySnapshot GetClusterTelemetry() const;
    Status RegisterTelemetryCallback(std::function<void(const ClusterTelemetrySnapshot&)> callback);
    
    // ML policy integration
    Status UpdateMLPolicy();
    Status TrainMLModel(const std::vector<MLFeatureVector>& features,
                        const std::vector<MLDecision>& targets);
    
    // Statistics and monitoring
    ReactiveSchedulerStats::Snapshot GetStats() const;
    void ResetStats();
    
    // Configuration
    void SetReactiveMode(bool enabled);
    void SetMLPolicyEnabled(bool enabled);
    void SetQoSEnforcementEnabled(bool enabled);
    void SetAutoScalingEnabled(bool enabled);

private:
    // Internal scheduling methods
    std::string SelectOptimalNode(const std::string& task_id,
                                 const QoSRequirements& qos_requirements,
                                 std::shared_ptr<Graph> graph);
    
    double CalculateNodeScore(const std::string& node_id,
                             const QoSRequirements& qos_requirements,
                             std::shared_ptr<Graph> graph);
    
    bool ValidateQoSConstraints(const std::string& node_id,
                               const QoSRequirements& qos_requirements);
    
    // ML-driven decision making
    ReactiveSchedulingDecision MakeMLDecision(const std::string& task_id,
                                             const QoSRequirements& qos_requirements,
                                             std::shared_ptr<Graph> graph);
    
    ReactiveSchedulingDecision MakeRuleBasedDecision(const std::string& task_id,
                                                    const QoSRequirements& qos_requirements,
                                                    std::shared_ptr<Graph> graph);
    
    // Predictive analysis
    bool PredictNodeFailure(const std::string& node_id);
    bool PredictQoSViolation(const std::string& task_id, const std::string& node_id);
    std::vector<std::string> GetOptimalMigrationTargets(const std::string& source_node_id);
    
    // Load balancing algorithms
    Status ApplyLoadBalancing();
    Status ApplyPredictiveLoadBalancing();
    double CalculateLoadBalanceScore();
    
    // Auto-scaling logic
    Status ScaleUpCluster();
    Status ScaleDownCluster();
    bool ShouldScaleUp();
    bool ShouldScaleDown();
    
    // Telemetry processing
    void ProcessTelemetryUpdate(const ClusterTelemetrySnapshot& telemetry);
    void UpdateNodeHealthPredictions();
    void UpdatePerformancePredictions();
    
    // Threading and synchronization
    void ReactiveSchedulingThread();
    void TelemetryProcessingThread();
    void QoSMonitoringThread();
    void AutoScalingThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> reactive_mode_enabled_{true};
    std::atomic<bool> ml_policy_enabled_{true};
    std::atomic<bool> qos_enforcement_enabled_{true};
    std::atomic<bool> auto_scaling_enabled_{true};
    
    // Dependencies
    std::shared_ptr<OptimizationManager> optimization_manager_;
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<MLBasedPolicy> ml_policy_;
    
    // Scheduling state
    mutable std::mutex scheduling_mutex_;
    std::map<std::string, ReactiveSchedulingDecision> active_decisions_;
    std::map<std::string, QoSRequirements> task_qos_requirements_;
    std::queue<std::string> pending_tasks_;
    
    // Telemetry and monitoring
    mutable std::mutex telemetry_mutex_;
    ClusterTelemetrySnapshot current_telemetry_;
    std::vector<std::function<void(const ClusterTelemetrySnapshot&)>> telemetry_callbacks_;
    std::map<std::string, std::vector<double>> node_health_history_;
    std::map<std::string, std::vector<double>> node_performance_history_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    ReactiveSchedulerStats stats_;
    
    // Threading
    std::thread reactive_scheduling_thread_;
    std::thread telemetry_processing_thread_;
    std::thread qos_monitoring_thread_;
    std::thread auto_scaling_thread_;
    
    std::condition_variable scheduling_cv_;
    std::condition_variable telemetry_cv_;
    std::condition_variable qos_cv_;
    std::condition_variable auto_scaling_cv_;
    
    mutable std::mutex scheduling_cv_mutex_;
    mutable std::mutex telemetry_cv_mutex_;
    mutable std::mutex qos_cv_mutex_;
    mutable std::mutex auto_scaling_cv_mutex_;
};

} // namespace edge_ai
