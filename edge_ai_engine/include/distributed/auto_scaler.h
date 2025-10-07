/**
 * @file auto_scaler.h
 * @brief Edge-aware auto-scaling system for distributed inference clusters
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
 * @brief Scaling trigger conditions
 */
struct ScalingTrigger {
    std::string trigger_id;
    std::string description;
    
    enum class TriggerType {
        CPU_UTILIZATION,
        MEMORY_UTILIZATION,
        GPU_UTILIZATION,
        QUEUE_DEPTH,
        LATENCY_THRESHOLD,
        THROUGHPUT_DEGRADATION,
        ENERGY_EFFICIENCY,
        COST_OPTIMIZATION,
        NODE_FAILURE,
        PREDICTIVE_LOAD
    } trigger_type;
    
    double threshold_value{0.0};
    std::chrono::milliseconds evaluation_window_ms{60000};  // 1 minute
    std::chrono::milliseconds cooldown_period_ms{300000};   // 5 minutes
    
    // Scaling parameters
    uint32_t min_nodes{1};
    uint32_t max_nodes{10};
    uint32_t scale_up_step{1};
    uint32_t scale_down_step{1};
    
    bool is_enabled{true};
    double confidence_threshold{0.8};
    
    ScalingTrigger() = default;
};

/**
 * @brief Node template for auto-scaling
 */
struct NodeTemplate {
    std::string template_id;
    std::string description;
    
    // Hardware specifications
    uint32_t cpu_cores{4};
    uint64_t memory_mb{8192};
    uint64_t gpu_memory_mb{4096};
    bool has_gpu{false};
    bool has_npu{false};
    
    // Performance characteristics
    double compute_efficiency{1.0};
    double memory_efficiency{1.0};
    uint32_t bandwidth_mbps{1000};
    uint32_t latency_ms{1};
    
    // Cost and energy
    double cost_per_hour{0.1};
    double energy_consumption_watts{100.0};
    
    // Supported models and backends
    std::set<ModelType> supported_models;
    std::set<BackendType> supported_backends;
    
    // Scaling preferences
    uint32_t preferred_count{2};
    double priority_score{1.0};
    bool auto_scaling_enabled{true};
    
    NodeTemplate() = default;
};

/**
 * @brief Scaling decision
 */
struct ScalingDecision {
    std::string decision_id;
    std::string trigger_id;
    
    enum class DecisionType {
        SCALE_UP,
        SCALE_DOWN,
        NO_ACTION,
        EMERGENCY_SCALE_UP,
        PREVENTIVE_SCALE_UP
    } decision_type;
    
    std::string node_template_id;
    uint32_t target_node_count{0};
    uint32_t current_node_count{0};
    
    double confidence_score{0.0};
    std::string reasoning;
    std::chrono::steady_clock::time_point decision_time;
    
    // Cost and efficiency impact
    double estimated_cost_impact{0.0};
    double estimated_performance_impact{0.0};
    double estimated_energy_impact{0.0};
    
    // Execution status
    bool executed{false};
    std::chrono::steady_clock::time_point execution_time;
    Status execution_status{Status::NOT_INITIALIZED};
    
    ScalingDecision() {
        decision_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Multi-tenant resource allocation
 */
struct TenantResourceAllocation {
    std::string tenant_id;
    std::string tenant_name;
    
    // Resource quotas
    uint32_t max_nodes{5};
    uint32_t min_nodes{1};
    uint64_t max_memory_mb{16384};
    uint64_t max_gpu_memory_mb{8192};
    
    // Cost constraints
    double max_cost_per_hour{1.0};
    double current_cost_per_hour{0.0};
    
    // Priority and SLA
    uint32_t priority{1};
    double sla_availability_target{0.99};
    
    // Current allocation
    std::set<std::string> allocated_nodes;
    uint32_t current_node_count{0};
    
    TenantResourceAllocation() = default;
};

/**
 * @brief Auto-scaler statistics
 */
struct AutoScalerStats {
    std::atomic<uint64_t> total_scaling_events{0};
    std::atomic<uint64_t> scale_up_events{0};
    std::atomic<uint64_t> scale_down_events{0};
    std::atomic<uint64_t> emergency_scaling_events{0};
    std::atomic<uint64_t> predictive_scaling_events{0};
    
    // Performance metrics
    std::atomic<double> avg_scaling_time_ms{0.0};
    std::atomic<double> avg_decision_confidence{0.0};
    std::atomic<double> scaling_accuracy{0.0};
    std::atomic<double> cost_efficiency{0.0};
    
    // Resource utilization
    std::atomic<double> avg_cluster_utilization{0.0};
    std::atomic<double> avg_node_utilization{0.0};
    std::atomic<uint32_t> peak_node_count{0};
    std::atomic<uint32_t> current_node_count{0};
    
    // Cost and energy metrics
    std::atomic<double> total_scaling_cost{0.0};
    std::atomic<double> avg_energy_efficiency{0.0};
    std::atomic<double> cost_savings_percent{0.0};
    
    AutoScalerStats() = default;
    
    struct Snapshot {
        uint64_t total_scaling_events;
        uint64_t scale_up_events;
        uint64_t scale_down_events;
        uint64_t emergency_scaling_events;
        uint64_t predictive_scaling_events;
        double avg_scaling_time_ms;
        double avg_decision_confidence;
        double scaling_accuracy;
        double cost_efficiency;
        double avg_cluster_utilization;
        double avg_node_utilization;
        uint32_t peak_node_count;
        uint32_t current_node_count;
        double total_scaling_cost;
        double avg_energy_efficiency;
        double cost_savings_percent;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_scaling_events = total_scaling_events.load();
        snapshot.scale_up_events = scale_up_events.load();
        snapshot.scale_down_events = scale_down_events.load();
        snapshot.emergency_scaling_events = emergency_scaling_events.load();
        snapshot.predictive_scaling_events = predictive_scaling_events.load();
        snapshot.avg_scaling_time_ms = avg_scaling_time_ms.load();
        snapshot.avg_decision_confidence = avg_decision_confidence.load();
        snapshot.scaling_accuracy = scaling_accuracy.load();
        snapshot.cost_efficiency = cost_efficiency.load();
        snapshot.avg_cluster_utilization = avg_cluster_utilization.load();
        snapshot.avg_node_utilization = avg_node_utilization.load();
        snapshot.peak_node_count = peak_node_count.load();
        snapshot.current_node_count = current_node_count.load();
        snapshot.total_scaling_cost = total_scaling_cost.load();
        snapshot.avg_energy_efficiency = avg_energy_efficiency.load();
        snapshot.cost_savings_percent = cost_savings_percent.load();
        return snapshot;
    }
};

/**
 * @brief Edge-aware auto-scaler
 */
class AutoScaler {
public:
    /**
     * @brief Constructor
     * @param cluster_manager Cluster manager for node operations
     */
    explicit AutoScaler(std::shared_ptr<ClusterManager> cluster_manager);
    
    /**
     * @brief Destructor
     */
    ~AutoScaler();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Scaling trigger management
    Status RegisterScalingTrigger(const ScalingTrigger& trigger);
    Status UnregisterScalingTrigger(const std::string& trigger_id);
    Status UpdateScalingTrigger(const std::string& trigger_id, const ScalingTrigger& updated_trigger);
    std::vector<ScalingTrigger> GetRegisteredTriggers() const;
    
    // Node template management
    Status RegisterNodeTemplate(const NodeTemplate& template_def);
    Status UnregisterNodeTemplate(const std::string& template_id);
    Status UpdateNodeTemplate(const std::string& template_id, const NodeTemplate& updated_template);
    std::vector<NodeTemplate> GetRegisteredTemplates() const;
    
    // Multi-tenant management
    Status RegisterTenant(const TenantResourceAllocation& tenant);
    Status UnregisterTenant(const std::string& tenant_id);
    Status UpdateTenantAllocation(const std::string& tenant_id, const TenantResourceAllocation& updated_allocation);
    std::vector<TenantResourceAllocation> GetRegisteredTenants() const;
    
    // Scaling operations
    Status EvaluateScalingNeeds();
    Status ExecuteScalingDecision(const ScalingDecision& decision);
    Status ScaleUp(const std::string& node_template_id, uint32_t count = 1);
    Status ScaleDown(const std::string& node_template_id, uint32_t count = 1);
    Status EmergencyScaleUp(const std::string& node_template_id, uint32_t count = 1);
    
    // Predictive scaling
    Status PredictScalingNeeds(std::chrono::minutes prediction_horizon = std::chrono::minutes(30));
    Status ApplyPredictiveScaling();
    Status UpdateScalingPredictions();
    
    // Cost optimization
    Status OptimizeCosts();
    Status CalculateCostImpact(const ScalingDecision& decision, double& cost_impact);
    Status EvaluateCostEfficiency();
    
    // Energy efficiency
    Status OptimizeEnergyConsumption();
    Status CalculateEnergyImpact(const ScalingDecision& decision, double& energy_impact);
    Status EvaluateEnergyEfficiency();
    
    // Monitoring and metrics
    Status UpdateClusterMetrics();
    Status CalculateClusterUtilization();
    Status MonitorNodeHealth();
    AutoScalerStats::Snapshot GetStats() const;
    void ResetStats();
    
    // Configuration
    void SetAutoScalingEnabled(bool enabled);
    void SetPredictiveScalingEnabled(bool enabled);
    void SetCostOptimizationEnabled(bool enabled);
    void SetEnergyOptimizationEnabled(bool enabled);
    void SetEvaluationInterval(std::chrono::milliseconds interval);

private:
    // Internal scaling logic
    ScalingDecision MakeScalingDecision(const ScalingTrigger& trigger);
    bool ShouldScaleUp(const ScalingTrigger& trigger);
    bool ShouldScaleDown(const ScalingTrigger& trigger);
    std::string SelectOptimalNodeTemplate(const ScalingTrigger& trigger);
    
    // Cost and efficiency calculations
    double CalculateScalingCost(const ScalingDecision& decision);
    double CalculateEnergyImpact(const ScalingDecision& decision);
    double CalculatePerformanceImpact(const ScalingDecision& decision);
    
    // Multi-tenant resource management
    bool ValidateTenantConstraints(const std::string& tenant_id, const ScalingDecision& decision);
    Status AllocateResourcesToTenant(const std::string& tenant_id, const std::string& node_id);
    Status DeallocateResourcesFromTenant(const std::string& tenant_id, const std::string& node_id);
    
    // Predictive analysis
    double PredictFutureLoad(const std::string& node_id, std::chrono::minutes horizon);
    bool PredictNodeFailure(const std::string& node_id);
    std::vector<std::string> GetOptimalScalingTargets(const ScalingTrigger& trigger);
    
    // Threading and synchronization
    void ScalingEvaluationThread();
    void PredictiveScalingThread();
    void CostOptimizationThread();
    void EnergyOptimizationThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> auto_scaling_enabled_{true};
    std::atomic<bool> predictive_scaling_enabled_{true};
    std::atomic<bool> cost_optimization_enabled_{true};
    std::atomic<bool> energy_optimization_enabled_{true};
    std::atomic<std::chrono::milliseconds> evaluation_interval_{30000};  // 30 seconds
    
    // Dependencies
    std::shared_ptr<ClusterManager> cluster_manager_;
    
    // Scaling configuration
    mutable std::mutex triggers_mutex_;
    std::map<std::string, ScalingTrigger> scaling_triggers_;
    
    mutable std::mutex templates_mutex_;
    std::map<std::string, NodeTemplate> node_templates_;
    
    mutable std::mutex tenants_mutex_;
    std::map<std::string, TenantResourceAllocation> tenants_;
    
    // Scaling state
    mutable std::mutex scaling_mutex_;
    std::queue<ScalingDecision> pending_decisions_;
    std::vector<ScalingDecision> scaling_history_;
    std::map<std::string, std::chrono::steady_clock::time_point> last_scaling_times_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    AutoScalerStats stats_;
    
    // Threading
    std::thread scaling_evaluation_thread_;
    std::thread predictive_scaling_thread_;
    std::thread cost_optimization_thread_;
    std::thread energy_optimization_thread_;
    
    std::condition_variable scaling_cv_;
    std::condition_variable predictive_cv_;
    std::condition_variable cost_cv_;
    std::condition_variable energy_cv_;
    
    mutable std::mutex scaling_cv_mutex_;
    mutable std::mutex predictive_cv_mutex_;
    mutable std::mutex cost_cv_mutex_;
    mutable std::mutex energy_cv_mutex_;
};

} // namespace edge_ai
