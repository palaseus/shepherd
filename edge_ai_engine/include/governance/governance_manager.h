/**
 * @file governance_manager.h
 * @brief Autonomous governance layer for continuous auditing and policy enforcement
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
#include "optimization/optimization_manager.h"
#include "profiling/profiler.h"

// Forward declarations
namespace edge_ai {
    class ClusterManager;
    class MLBasedPolicy;
    class OptimizationManager;
}

namespace edge_ai {

/**
 * @brief Governance policy types
 */
enum class GovernancePolicyType {
    PERFORMANCE_SLA = 0,
    RESOURCE_UTILIZATION = 1,
    SECURITY_COMPLIANCE = 2,
    COST_OPTIMIZATION = 3,
    AVAILABILITY_TARGET = 4,
    LATENCY_SLA = 5,
    THROUGHPUT_SLA = 6,
    ENERGY_EFFICIENCY = 7
};

/**
 * @brief Governance policy definition
 */
struct GovernancePolicy {
    std::string policy_id;
    std::string name;
    GovernancePolicyType type;
    std::string description;
    
    // Policy thresholds and targets
    double target_value;
    double warning_threshold;
    double critical_threshold;
    double min_value;
    double max_value;
    
    // Enforcement parameters
    bool auto_remediation_enabled;
    std::vector<std::string> remediation_actions;
    std::chrono::milliseconds evaluation_interval;
    std::chrono::milliseconds remediation_timeout;
    
    // Policy metadata
    std::string cluster_id;
    std::string tenant_id;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_updated;
    bool enabled;
    uint32_t priority;
};

/**
 * @brief Governance audit result
 */
struct GovernanceAuditResult {
    std::string audit_id;
    std::string policy_id;
    std::string cluster_id;
    std::string node_id;
    
    // Audit metrics
    double current_value;
    double target_value;
    double deviation_percent;
    bool policy_violated;
    bool warning_triggered;
    bool critical_triggered;
    
    // Recommendations
    std::vector<std::string> recommended_actions;
    std::vector<std::string> auto_remediation_actions;
    double confidence_score;
    
    // Timing
    std::chrono::steady_clock::time_point audit_time;
    std::chrono::milliseconds audit_duration;
    
    // Context
    std::map<std::string, double> context_metrics;
    std::string violation_reason;
    std::string remediation_status;
};

/**
 * @brief Cross-cluster governance coordination
 */
struct CrossClusterGovernance {
    std::string coordination_id;
    std::vector<std::string> participating_clusters;
    std::string coordination_strategy;
    
    // Shared policies
    std::vector<GovernancePolicy> shared_policies;
    std::map<std::string, double> global_metrics;
    
    // Federation parameters
    bool federated_learning_enabled;
    std::chrono::milliseconds sync_interval;
    std::string model_sharing_protocol;
    
    // Coordination state
    std::chrono::steady_clock::time_point last_sync;
    uint32_t sync_failures{0};
    bool coordination_active{false};
};

/**
 * @brief Governance statistics
 */
struct GovernanceStats {
    // Policy enforcement
    std::atomic<uint64_t> total_audits{0};
    std::atomic<uint64_t> policy_violations{0};
    std::atomic<uint64_t> warnings_triggered{0};
    std::atomic<uint64_t> critical_violations{0};
    std::atomic<uint64_t> auto_remediations{0};
    std::atomic<uint64_t> failed_remediations{0};
    
    // Performance metrics
    std::atomic<double> avg_audit_time_ms{0.0};
    std::atomic<double> avg_remediation_time_ms{0.0};
    std::atomic<double> policy_compliance_rate{0.0};
    std::atomic<double> auto_remediation_success_rate{0.0};
    
    // Cross-cluster coordination
    std::atomic<uint64_t> cross_cluster_syncs{0};
    std::atomic<uint64_t> cross_cluster_failures{0};
    std::atomic<double> federation_effectiveness{0.0};
    
    // System health
    std::atomic<double> overall_governance_score{0.0};
    std::atomic<uint64_t> active_policies{0};
    std::atomic<uint64_t> monitored_clusters{0};
    
    /**
     * @brief Get a snapshot of current statistics
     */
    struct Snapshot {
        uint64_t total_audits;
        uint64_t policy_violations;
        uint64_t warnings_triggered;
        uint64_t critical_violations;
        uint64_t auto_remediations;
        uint64_t failed_remediations;
        double avg_audit_time_ms;
        double avg_remediation_time_ms;
        double policy_compliance_rate;
        double auto_remediation_success_rate;
        uint64_t cross_cluster_syncs;
        uint64_t cross_cluster_failures;
        double federation_effectiveness;
        double overall_governance_score;
        uint64_t active_policies;
        uint64_t monitored_clusters;
    };
    
    Snapshot GetSnapshot() const {
        return {
            total_audits.load(),
            policy_violations.load(),
            warnings_triggered.load(),
            critical_violations.load(),
            auto_remediations.load(),
            failed_remediations.load(),
            avg_audit_time_ms.load(),
            avg_remediation_time_ms.load(),
            policy_compliance_rate.load(),
            auto_remediation_success_rate.load(),
            cross_cluster_syncs.load(),
            cross_cluster_failures.load(),
            federation_effectiveness.load(),
            overall_governance_score.load(),
            active_policies.load(),
            monitored_clusters.load()
        };
    }
};

/**
 * @class GovernanceManager
 * @brief Autonomous governance layer for continuous auditing and policy enforcement
 */
class GovernanceManager {
public:
    explicit GovernanceManager(std::shared_ptr<ClusterManager> cluster_manager,
                             std::shared_ptr<MLBasedPolicy> ml_policy,
                             std::shared_ptr<OptimizationManager> optimization_manager);
    virtual ~GovernanceManager();
    
    /**
     * @brief Initialize the governance manager
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the governance manager
     */
    Status Shutdown();
    
    /**
     * @brief Check if the governance manager is initialized
     */
    bool IsInitialized() const;
    
    // Policy Management
    
    /**
     * @brief Register a governance policy
     */
    Status RegisterPolicy(const GovernancePolicy& policy);
    
    /**
     * @brief Unregister a governance policy
     */
    Status UnregisterPolicy(const std::string& policy_id);
    
    /**
     * @brief Update an existing policy
     */
    Status UpdatePolicy(const GovernancePolicy& policy);
    
    /**
     * @brief Get all registered policies
     */
    std::vector<GovernancePolicy> GetPolicies() const;
    
    /**
     * @brief Get policies for a specific cluster
     */
    std::vector<GovernancePolicy> GetClusterPolicies(const std::string& cluster_id) const;
    
    /**
     * @brief Get policies by type
     */
    std::vector<GovernancePolicy> GetPoliciesByType(GovernancePolicyType type) const;
    
    // Audit and Enforcement
    
    /**
     * @brief Perform governance audit
     */
    Status PerformAudit(const std::string& policy_id, const std::string& cluster_id, 
                       const std::string& node_id, GovernanceAuditResult& result);
    
    /**
     * @brief Perform comprehensive cluster audit
     */
    Status PerformClusterAudit(const std::string& cluster_id, 
                              std::vector<GovernanceAuditResult>& results);
    
    /**
     * @brief Perform cross-cluster audit
     */
    Status PerformCrossClusterAudit(const std::vector<std::string>& cluster_ids,
                                   std::vector<GovernanceAuditResult>& results);
    
    /**
     * @brief Execute auto-remediation for a policy violation
     */
    Status ExecuteAutoRemediation(const GovernanceAuditResult& audit_result);
    
    /**
     * @brief Get audit history for a policy
     */
    std::vector<GovernanceAuditResult> GetAuditHistory(const std::string& policy_id,
                                                      std::chrono::hours lookback_hours = std::chrono::hours(24)) const;
    
    // Cross-Cluster Coordination
    
    /**
     * @brief Initialize cross-cluster governance coordination
     */
    Status InitializeCrossClusterCoordination(const CrossClusterGovernance& coordination);
    
    /**
     * @brief Synchronize governance policies across clusters
     */
    Status SynchronizeCrossClusterPolicies();
    
    /**
     * @brief Share governance insights with other clusters
     */
    Status ShareGovernanceInsights(const std::string& target_cluster_id,
                                  const std::vector<GovernanceAuditResult>& insights);
    
    /**
     * @brief Receive governance insights from other clusters
     */
    Status ReceiveGovernanceInsights(const std::string& source_cluster_id,
                                    const std::vector<GovernanceAuditResult>& insights);
    
    /**
     * @brief Coordinate global load balancing
     */
    Status CoordinateGlobalLoadBalancing(const std::vector<std::string>& cluster_ids,
                                        std::map<std::string, double>& load_distribution);
    
    // Self-Optimization
    
    /**
     * @brief Optimize governance policies based on historical data
     */
    Status OptimizeGovernancePolicies();
    
    /**
     * @brief Tune auto-remediation parameters
     */
    Status TuneAutoRemediationParameters(const std::string& policy_id);
    
    /**
     * @brief Evolve governance strategies using ML
     */
    Status EvolveGovernanceStrategies();
    
    /**
     * @brief Predict governance policy effectiveness
     */
    Status PredictPolicyEffectiveness(const GovernancePolicy& policy, double& effectiveness_score);
    
    // Analytics and Reporting
    
    /**
     * @brief Generate governance compliance report
     */
    Status GenerateComplianceReport(const std::string& cluster_id,
                                   std::map<std::string, double>& compliance_metrics);
    
    /**
     * @brief Generate cross-cluster federation report
     */
    Status GenerateFederationReport(std::map<std::string, double>& federation_metrics);
    
    /**
     * @brief Get governance statistics
     */
    GovernanceStats::Snapshot GetStats() const;
    
    /**
     * @brief Reset governance statistics
     */
    void ResetStats();
    
    /**
     * @brief Generate governance insights
     */
    Status GenerateGovernanceInsights(std::vector<std::string>& insights);
    
    // Configuration
    
    /**
     * @brief Set governance evaluation interval
     */
    void SetEvaluationInterval(std::chrono::milliseconds interval);
    
    /**
     * @brief Enable/disable auto-remediation
     */
    void SetAutoRemediationEnabled(bool enabled);
    
    /**
     * @brief Enable/disable cross-cluster coordination
     */
    void SetCrossClusterCoordinationEnabled(bool enabled);
    
    /**
     * @brief Set governance strictness level
     */
    void SetGovernanceStrictness(double strictness_level);

private:
    // Core components
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<MLBasedPolicy> ml_policy_;
    std::shared_ptr<OptimizationManager> optimization_manager_;
    
    // State management
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> auto_remediation_enabled_{true};
    std::atomic<bool> cross_cluster_coordination_enabled_{true};
    std::atomic<double> governance_strictness_{0.8};
    std::atomic<std::chrono::milliseconds> evaluation_interval_{std::chrono::milliseconds(30000)};
    
    // Policy storage
    mutable std::mutex policies_mutex_;
    std::map<std::string, GovernancePolicy> policies_;
    std::map<std::string, std::vector<std::string>> cluster_policies_;
    std::map<GovernancePolicyType, std::vector<std::string>> type_policies_;
    
    // Audit storage
    mutable std::mutex audit_mutex_;
    std::deque<GovernanceAuditResult> audit_history_;
    std::map<std::string, std::deque<GovernanceAuditResult>> policy_audit_history_;
    
    // Cross-cluster coordination
    mutable std::mutex coordination_mutex_;
    std::map<std::string, CrossClusterGovernance> cross_cluster_coordinations_;
    std::map<std::string, std::vector<GovernanceAuditResult>> received_insights_;
    
    // Background threads
    std::thread audit_thread_;
    std::thread remediation_thread_;
    std::thread coordination_thread_;
    std::thread optimization_thread_;
    
    // Condition variables
    std::mutex audit_cv_mutex_;
    std::condition_variable audit_cv_;
    std::mutex remediation_cv_mutex_;
    std::condition_variable remediation_cv_;
    std::mutex coordination_cv_mutex_;
    std::condition_variable coordination_cv_;
    std::mutex optimization_cv_mutex_;
    std::condition_variable optimization_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    GovernanceStats stats_;
    
    // Private methods
    
    /**
     * @brief Background audit thread
     */
    void AuditThread();
    
    /**
     * @brief Background remediation thread
     */
    void RemediationThread();
    
    /**
     * @brief Background coordination thread
     */
    void CoordinationThread();
    
    /**
     * @brief Background optimization thread
     */
    void OptimizationThread();
    
    /**
     * @brief Evaluate a single policy
     */
    Status EvaluatePolicy(const GovernancePolicy& policy, const std::string& cluster_id,
                         const std::string& node_id, GovernanceAuditResult& result);
    
    /**
     * @brief Calculate policy compliance score
     */
    double CalculateComplianceScore(const GovernancePolicy& policy, double current_value) const;
    
    /**
     * @brief Generate remediation recommendations
     */
    std::vector<std::string> GenerateRemediationRecommendations(const GovernanceAuditResult& audit_result) const;
    
    /**
     * @brief Execute remediation action
     */
    Status ExecuteRemediationAction(const std::string& action, const GovernanceAuditResult& audit_result);
    
    /**
     * @brief Update governance statistics
     */
    void UpdateStats(const GovernanceAuditResult& audit_result);
    
    /**
     * @brief Calculate overall governance score
     */
    double CalculateOverallGovernanceScore() const;
    
    /**
     * @brief Optimize policy thresholds using ML
     */
    Status OptimizePolicyThresholds(const std::string& policy_id);
    
    /**
     * @brief Learn from cross-cluster insights
     */
    Status LearnFromCrossClusterInsights();
    
    /**
     * @brief Predict cluster performance
     */
    Status PredictClusterPerformance(const std::string& cluster_id, double& performance_score);
    
    /**
     * @brief Coordinate resource allocation across clusters
     */
    Status CoordinateResourceAllocation(const std::vector<std::string>& cluster_ids,
                                       std::map<std::string, double>& allocation);
};

} // namespace edge_ai
