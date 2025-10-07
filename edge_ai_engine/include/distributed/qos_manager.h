/**
 * @file qos_manager.h
 * @brief QoS and SLA enforcement manager for distributed inference
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

namespace edge_ai {

/**
 * @brief SLA (Service Level Agreement) definition
 */
struct SLADefinition {
    std::string sla_id;
    std::string description;
    
    // Performance targets
    std::chrono::milliseconds max_latency_ms{1000};
    double min_throughput_ops_per_sec{1.0};
    double availability_target{0.99};  // 99% uptime
    
    // Energy and cost constraints
    double max_energy_consumption_watts{100.0};
    double max_cost_per_inference{0.01};
    
    // Priority and criticality
    uint32_t priority{0};  // Higher = more critical
    bool is_critical{false};
    bool allows_degradation{true};
    
    // Penalty definitions
    double latency_penalty_per_ms{0.1};
    double throughput_penalty_per_op{0.01};
    double availability_penalty_per_percent{1.0};
    
    SLADefinition() = default;
};

/**
 * @brief QoS violation event
 */
struct QoSViolationEvent {
    std::string violation_id;
    std::string task_id;
    std::string node_id;
    std::string sla_id;
    
    enum class ViolationType {
        LATENCY_EXCEEDED,
        THROUGHPUT_BELOW_TARGET,
        AVAILABILITY_DEGRADED,
        ENERGY_EXCEEDED,
        COST_EXCEEDED,
        PRIORITY_VIOLATION
    } violation_type;
    
    double severity_score{0.0};  // 0.0 to 1.0
    std::string description;
    std::chrono::steady_clock::time_point timestamp;
    
    // Corrective actions taken
    std::vector<std::string> actions_taken;
    bool resolved{false};
    
    QoSViolationEvent() {
        timestamp = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Backpressure control mechanism
 */
struct BackpressureControl {
    std::string task_id;
    std::string node_id;
    
    enum class ControlType {
        RATE_LIMITING,
        QUEUE_DEPTH_LIMIT,
        PRIORITY_ADJUSTMENT,
        RESOURCE_THROTTLING,
        TASK_MIGRATION
    } control_type;
    
    double control_factor{1.0};  // 0.0 to 1.0
    std::chrono::milliseconds duration_ms{1000};
    bool is_active{false};
    
    BackpressureControl() = default;
};

/**
 * @brief QoS manager statistics
 */
struct QoSManagerStats {
    std::atomic<uint64_t> total_slas_monitored{0};
    std::atomic<uint64_t> total_violations_detected{0};
    std::atomic<uint64_t> total_violations_resolved{0};
    std::atomic<uint64_t> total_backpressure_events{0};
    std::atomic<uint64_t> total_corrective_actions{0};
    
    // Performance metrics
    std::atomic<double> avg_violation_resolution_time_ms{0.0};
    std::atomic<double> avg_backpressure_effectiveness{0.0};
    std::atomic<double> sla_compliance_rate{0.0};
    std::atomic<double> qos_satisfaction_score{0.0};
    
    // Cost and efficiency metrics
    std::atomic<double> total_penalty_cost{0.0};
    std::atomic<double> avg_energy_efficiency{0.0};
    std::atomic<double> avg_cost_per_inference{0.0};
    
    QoSManagerStats() = default;
    
    struct Snapshot {
        uint64_t total_slas_monitored;
        uint64_t total_violations_detected;
        uint64_t total_violations_resolved;
        uint64_t total_backpressure_events;
        uint64_t total_corrective_actions;
        double avg_violation_resolution_time_ms;
        double avg_backpressure_effectiveness;
        double sla_compliance_rate;
        double qos_satisfaction_score;
        double total_penalty_cost;
        double avg_energy_efficiency;
        double avg_cost_per_inference;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_slas_monitored = total_slas_monitored.load();
        snapshot.total_violations_detected = total_violations_detected.load();
        snapshot.total_violations_resolved = total_violations_resolved.load();
        snapshot.total_backpressure_events = total_backpressure_events.load();
        snapshot.total_corrective_actions = total_corrective_actions.load();
        snapshot.avg_violation_resolution_time_ms = avg_violation_resolution_time_ms.load();
        snapshot.avg_backpressure_effectiveness = avg_backpressure_effectiveness.load();
        snapshot.sla_compliance_rate = sla_compliance_rate.load();
        snapshot.qos_satisfaction_score = qos_satisfaction_score.load();
        snapshot.total_penalty_cost = total_penalty_cost.load();
        snapshot.avg_energy_efficiency = avg_energy_efficiency.load();
        snapshot.avg_cost_per_inference = avg_cost_per_inference.load();
        return snapshot;
    }
};

/**
 * @brief QoS and SLA enforcement manager
 */
class QoSManager {
public:
    /**
     * @brief Constructor
     */
    QoSManager();
    
    /**
     * @brief Destructor
     */
    ~QoSManager();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // SLA management
    Status RegisterSLA(const SLADefinition& sla);
    Status UnregisterSLA(const std::string& sla_id);
    Status UpdateSLA(const std::string& sla_id, const SLADefinition& updated_sla);
    std::vector<SLADefinition> GetRegisteredSLAs() const;
    
    // QoS monitoring
    Status StartQoSMonitoring();
    Status StopQoSMonitoring();
    Status MonitorTaskQoS(const std::string& task_id, const std::string& sla_id);
    Status StopMonitoringTask(const std::string& task_id);
    
    // Violation handling
    Status DetectViolations();
    Status HandleViolation(const QoSViolationEvent& violation);
    Status ResolveViolation(const std::string& violation_id);
    std::vector<QoSViolationEvent> GetActiveViolations() const;
    std::vector<QoSViolationEvent> GetViolationHistory() const;
    
    // Backpressure control
    Status ApplyBackpressure(const BackpressureControl& control);
    Status RemoveBackpressure(const std::string& task_id);
    Status AdjustBackpressure(const std::string& task_id, double new_factor);
    std::vector<BackpressureControl> GetActiveBackpressureControls() const;
    
    // Corrective actions
    Status MigrateTask(const std::string& task_id, const std::string& target_node_id);
    Status AdjustTaskPriority(const std::string& task_id, uint32_t new_priority);
    Status ThrottleNode(const std::string& node_id, double throttle_factor);
    Status AllocateAdditionalResources(const std::string& node_id);
    
    // Metrics and reporting
    Status CalculateSLACompliance();
    Status GenerateQoSReport();
    QoSManagerStats::Snapshot GetStats() const;
    void ResetStats();
    
    // Configuration
    void SetViolationDetectionEnabled(bool enabled);
    void SetBackpressureEnabled(bool enabled);
    void SetCorrectiveActionsEnabled(bool enabled);
    void SetMonitoringInterval(std::chrono::milliseconds interval);

private:
    // Internal monitoring methods
    bool CheckLatencyCompliance(const std::string& task_id, const SLADefinition& sla);
    bool CheckThroughputCompliance(const std::string& task_id, const SLADefinition& sla);
    bool CheckAvailabilityCompliance(const std::string& task_id, const SLADefinition& sla);
    bool CheckEnergyCompliance(const std::string& task_id, const SLADefinition& sla);
    bool CheckCostCompliance(const std::string& task_id, const SLADefinition& sla);
    
    // Violation analysis
    double CalculateViolationSeverity(const QoSViolationEvent& violation);
    std::vector<std::string> DetermineCorrectiveActions(const QoSViolationEvent& violation);
    Status ExecuteCorrectiveAction(const std::string& action, const QoSViolationEvent& violation);
    
    // Backpressure management
    Status CalculateBackpressureFactor(const std::string& task_id, double& factor);
    Status ApplyRateLimiting(const std::string& task_id, double rate_limit);
    Status ApplyQueueDepthLimit(const std::string& task_id, uint32_t max_depth);
    Status ApplyResourceThrottling(const std::string& node_id, double throttle_factor);
    
    // Threading and synchronization
    void QoSMonitoringThread();
    void ViolationProcessingThread();
    void BackpressureControlThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> violation_detection_enabled_{true};
    std::atomic<bool> backpressure_enabled_{true};
    std::atomic<bool> corrective_actions_enabled_{true};
    std::atomic<std::chrono::milliseconds> monitoring_interval_{1000};
    
    // SLA and QoS state
    mutable std::mutex sla_mutex_;
    std::map<std::string, SLADefinition> registered_slas_;
    std::map<std::string, std::string> task_sla_mapping_;  // task_id -> sla_id
    
    // Violation tracking
    mutable std::mutex violations_mutex_;
    std::map<std::string, QoSViolationEvent> active_violations_;
    std::vector<QoSViolationEvent> violation_history_;
    
    // Backpressure controls
    mutable std::mutex backpressure_mutex_;
    std::map<std::string, BackpressureControl> active_backpressure_controls_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    QoSManagerStats stats_;
    
    // Threading
    std::thread qos_monitoring_thread_;
    std::thread violation_processing_thread_;
    std::thread backpressure_control_thread_;
    
    std::condition_variable qos_cv_;
    std::condition_variable violation_cv_;
    std::condition_variable backpressure_cv_;
    
    mutable std::mutex qos_cv_mutex_;
    mutable std::mutex violation_cv_mutex_;
    mutable std::mutex backpressure_cv_mutex_;
};

} // namespace edge_ai
