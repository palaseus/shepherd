/**
 * @file autonomous_optimizer.h
 * @brief Autonomous self-optimization feedback loop for continuous DAG and architecture evolution
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
#include <random>
#include <algorithm>
#include <thread>
#include <nlohmann/json.hpp>
#include "core/types.h"
#include "graph/graph_types.h"
#include "distributed/cluster_types.h"
#include "evolution/evolution_manager.h"
#include "analytics/telemetry_analytics.h"
#include "governance/governance_manager.h"
#include "autonomous/dag_generator.h"
#include "autonomous/synthetic_testbed.h"
#include "profiling/profiler.h"

// Forward declarations
namespace edge_ai {
    class DAGGenerator;
    class SyntheticTestbed;
    class EvolutionManager;
    class TelemetryAnalytics;
    class GovernanceManager;
}

namespace edge_ai {

/**
 * @brief Optimization feedback event
 */
struct OptimizationFeedbackEvent {
    std::string event_id;
    std::string event_type; // performance_regression, load_spike, failure, improvement
    std::chrono::steady_clock::time_point timestamp;
    
    // Event context
    std::string cluster_id;
    std::string node_id;
    std::string dag_id;
    std::string workload_id;
    
    // Event data
    std::map<std::string, double> metrics;
    std::map<std::string, std::string> context;
    double severity_score;
    double confidence_score;
    
    // Impact assessment
    std::vector<std::string> affected_services;
    std::vector<std::string> affected_tenants;
    double estimated_impact;
    std::string impact_description;
    
    // Recommendations
    std::vector<std::string> suggested_actions;
    std::vector<std::string> optimization_opportunities;
    double priority_score;
};

/**
 * @brief Autonomous optimization action
 */
struct AutonomousOptimizationAction {
    std::string action_id;
    std::string action_type; // dag_restructure, resource_rebalance, model_swap, parameter_tune
    std::chrono::steady_clock::time_point scheduled_time;
    
    // Action parameters
    std::map<std::string, std::string> parameters;
    std::map<std::string, double> expected_improvements;
    double confidence_score;
    double risk_score;
    
    // Execution context
    std::string target_cluster_id;
    std::string target_node_id;
    std::string target_dag_id;
    std::vector<std::string> dependencies;
    
    // Safety constraints
    std::vector<std::string> safety_constraints;
    double max_rollback_time_ms;
    bool requires_approval;
    std::string approval_authority;
    
    // Execution status
    std::string status; // pending, executing, completed, failed, rolled_back
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::map<std::string, double> actual_improvements;
    std::vector<std::string> execution_log;
};

/**
 * @brief Autonomous optimization policy
 */
struct AutonomousOptimizationPolicy {
    std::string policy_id;
    std::string policy_name;
    std::string policy_description;
    
    // Policy rules
    std::vector<std::string> trigger_conditions;
    std::vector<std::string> action_types;
    std::map<std::string, double> thresholds;
    std::map<std::string, double> weights;
    
    // Safety constraints
    std::vector<std::string> safety_rules;
    double max_risk_tolerance;
    double min_confidence_threshold;
    std::chrono::seconds max_execution_time;
    
    // Performance targets
    std::map<std::string, double> performance_targets;
    std::map<std::string, double> improvement_thresholds;
    std::chrono::seconds evaluation_window;
    
    // Policy metadata
    std::string policy_version;
    std::chrono::steady_clock::time_point created_time;
    std::chrono::steady_clock::time_point last_updated;
    bool is_active;
    std::string created_by;
};

/**
 * @brief Optimization session
 */
struct OptimizationSession {
    std::string session_id;
    std::string session_name;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    
    // Session context
    std::string cluster_id;
    std::vector<std::string> target_dags;
    std::vector<std::string> target_workloads;
    std::map<std::string, double> initial_metrics;
    std::map<std::string, double> target_metrics;
    
    // Optimization actions
    std::vector<AutonomousOptimizationAction> actions_planned;
    std::vector<AutonomousOptimizationAction> actions_executed;
    std::vector<AutonomousOptimizationAction> actions_rolled_back;
    
    // Results
    std::map<std::string, double> final_metrics;
    std::map<std::string, double> improvements_achieved;
    std::map<std::string, double> costs_incurred;
    double overall_success_score;
    bool meets_targets;
    
    // Learning data
    std::vector<OptimizationFeedbackEvent> feedback_events;
    std::map<std::string, double> learned_parameters;
    std::vector<std::string> insights_generated;
};

/**
 * @brief Autonomous optimization configuration
 */
struct AutonomousOptimizationConfig {
    // Feedback loop parameters
    std::chrono::milliseconds feedback_collection_interval;
    std::chrono::milliseconds optimization_trigger_interval;
    std::chrono::milliseconds action_execution_interval;
    std::chrono::seconds session_evaluation_window;
    
    // Optimization thresholds
    double performance_regression_threshold;
    double improvement_threshold;
    double confidence_threshold;
    double risk_tolerance_threshold;
    
    // Safety constraints
    bool enable_automatic_rollback;
    std::chrono::seconds max_rollback_time;
    uint32_t max_concurrent_optimizations;
    bool require_human_approval;
    
    // Learning parameters
    bool enable_online_learning;
    double learning_rate;
    uint32_t experience_buffer_size;
    std::chrono::seconds model_update_interval;
    
    // Performance targets
    std::map<std::string, double> global_performance_targets;
    std::map<std::string, double> optimization_weights;
    std::vector<std::string> priority_metrics;
    
    // Integration settings
    bool enable_dag_optimization;
    bool enable_architecture_evolution;
    bool enable_resource_optimization;
    bool enable_cross_cluster_optimization;
};

/**
 * @brief Autonomous optimization statistics
 */
struct AutonomousOptimizationStats {
    // Session metrics
    std::atomic<uint32_t> total_sessions{0};
    std::atomic<uint32_t> successful_sessions{0};
    std::atomic<uint32_t> failed_sessions{0};
    std::atomic<uint32_t> sessions_with_rollback{0};
    
    // Action metrics
    std::atomic<uint32_t> total_actions_planned{0};
    std::atomic<uint32_t> total_actions_executed{0};
    std::atomic<uint32_t> total_actions_rolled_back{0};
    std::atomic<uint32_t> total_actions_failed{0};
    
    // Performance metrics
    std::atomic<double> avg_improvement_percentage{0.0};
    std::atomic<double> avg_optimization_time_ms{0.0};
    std::atomic<double> avg_rollback_time_ms{0.0};
    std::atomic<double> success_rate{0.0};
    
    // Learning metrics
    std::atomic<uint32_t> total_feedback_events{0};
    std::atomic<uint32_t> total_insights_generated{0};
    std::atomic<double> avg_confidence_score{0.0};
    std::atomic<double> avg_risk_score{0.0};
    
    // Resource utilization
    std::atomic<double> avg_cpu_overhead_percent{0.0};
    std::atomic<double> avg_memory_overhead_mb{0.0};
    std::atomic<double> avg_network_overhead_mbps{0.0};
    
    // Timestamps
    std::chrono::steady_clock::time_point last_optimization_time;
    std::chrono::steady_clock::time_point last_learning_update;
    std::chrono::steady_clock::time_point last_policy_update;
    
    // Snapshot for safe copying
    struct Snapshot {
        uint32_t total_sessions;
        uint32_t successful_sessions;
        uint32_t failed_sessions;
        uint32_t sessions_with_rollback;
        uint32_t total_actions_planned;
        uint32_t total_actions_executed;
        uint32_t total_actions_rolled_back;
        uint32_t total_actions_failed;
        double avg_improvement_percentage;
        double avg_optimization_time_ms;
        double avg_rollback_time_ms;
        double success_rate;
        uint32_t total_feedback_events;
        uint32_t total_insights_generated;
        double avg_confidence_score;
        double avg_risk_score;
        double avg_cpu_overhead_percent;
        double avg_memory_overhead_mb;
        double avg_network_overhead_mbps;
        std::chrono::steady_clock::time_point last_optimization_time;
        std::chrono::steady_clock::time_point last_learning_update;
        std::chrono::steady_clock::time_point last_policy_update;
    };
    
    Snapshot GetSnapshot() const {
        return Snapshot{
            total_sessions.load(),
            successful_sessions.load(),
            failed_sessions.load(),
            sessions_with_rollback.load(),
            total_actions_planned.load(),
            total_actions_executed.load(),
            total_actions_rolled_back.load(),
            total_actions_failed.load(),
            avg_improvement_percentage.load(),
            avg_optimization_time_ms.load(),
            avg_rollback_time_ms.load(),
            success_rate.load(),
            total_feedback_events.load(),
            total_insights_generated.load(),
            avg_confidence_score.load(),
            avg_risk_score.load(),
            avg_cpu_overhead_percent.load(),
            avg_memory_overhead_mb.load(),
            avg_network_overhead_mbps.load(),
            last_optimization_time,
            last_learning_update,
            last_policy_update
        };
    }
};

/**
 * @brief Continuous optimization feedback loop configuration
 */
struct ContinuousOptimizationConfig {
    bool enable_continuous_optimization{true};
    std::chrono::milliseconds optimization_interval{5000}; // 5 seconds
    std::chrono::milliseconds feedback_collection_interval{1000}; // 1 second
    std::chrono::milliseconds learning_update_interval{10000}; // 10 seconds
    
    // Feedback thresholds
    double performance_improvement_threshold{0.05}; // 5% improvement
    double performance_degradation_threshold{0.02}; // 2% degradation
    double confidence_threshold{0.7}; // 70% confidence
    double risk_threshold{0.3}; // 30% risk
    
    // Learning parameters
    double learning_rate{0.01};
    double exploration_rate{0.1};
    double exploitation_rate{0.9};
    uint32_t experience_buffer_size{1000};
    uint32_t batch_size{32};
    
    // Safety constraints
    double max_optimization_overhead_percent{10.0};
    double max_rollback_frequency{0.1}; // 10% of optimizations
    std::chrono::milliseconds max_optimization_time{30000}; // 30 seconds
    uint32_t max_concurrent_optimizations{3};
    
    // Adaptation parameters
    bool enable_adaptive_parameters{true};
    double adaptation_sensitivity{0.1};
    std::chrono::milliseconds adaptation_window{60000}; // 1 minute
    double parameter_decay_rate{0.95};
};

/**
 * @brief Real-time performance feedback
 */
struct PerformanceFeedback {
    std::string feedback_id;
    std::chrono::steady_clock::time_point timestamp;
    
    // Performance metrics
    double latency_ms{0.0};
    double throughput_rps{0.0};
    double memory_usage_mb{0.0};
    double cpu_usage_percent{0.0};
    double power_consumption_watts{0.0};
    double accuracy{0.0};
    
    // System state
    std::map<std::string, double> system_metrics;
    std::vector<std::string> active_optimizations;
    std::vector<std::string> recent_actions;
    
    // Context information
    std::string workload_type;
    std::string cluster_state;
    std::string network_condition;
    uint32_t concurrent_requests;
    
    // Quality indicators
    double data_quality_score{1.0};
    double measurement_confidence{1.0};
    bool is_anomaly{false};
    std::string anomaly_type;
};

/**
 * @brief Optimization insight and learning
 */
struct OptimizationInsight {
    std::string insight_id;
    std::chrono::steady_clock::time_point generated_time;
    
    // Insight type
    std::string insight_type; // "performance_pattern", "resource_utilization", "failure_prediction", "optimization_opportunity"
    std::string insight_category; // "latency", "throughput", "memory", "power", "accuracy"
    
    // Insight content
    std::string description;
    std::map<std::string, double> metrics;
    std::vector<std::string> affected_components;
    std::vector<std::string> recommended_actions;
    
    // Confidence and impact
    double confidence_score{0.0};
    double impact_score{0.0};
    double urgency_score{0.0};
    
    // Learning metadata
    std::vector<std::string> supporting_evidence;
    std::vector<std::string> conflicting_evidence;
    uint32_t validation_count{0};
    bool is_validated{false};
    
    // Actionability
    bool is_actionable{false};
    std::vector<AutonomousOptimizationAction> suggested_actions;
    double estimated_improvement{0.0};
    double estimated_risk{0.0};
};

/**
 * @brief Continuous learning state
 */
struct ContinuousLearningState {
    // Experience buffer
    std::deque<PerformanceFeedback> experience_buffer_;
    std::deque<OptimizationInsight> insight_buffer_;
    
    // Learning models
    std::map<std::string, double> performance_models_;
    std::map<std::string, double> optimization_models_;
    std::map<std::string, double> risk_models_;
    
    // Adaptation state
    std::map<std::string, double> adaptive_parameters_;
    std::chrono::steady_clock::time_point last_adaptation_time_;
    uint32_t adaptation_count_{0};
    
    // Learning metrics
    double learning_progress_{0.0};
    double model_accuracy_{0.0};
    double prediction_confidence_{0.0};
    uint32_t total_insights_generated_{0};
    uint32_t validated_insights_{0};
    
    // Performance tracking
    std::vector<double> performance_history_;
    std::vector<double> improvement_history_;
    std::vector<double> risk_history_;
    double baseline_performance_{0.0};
    double current_performance_{0.0};
    double best_performance_{0.0};
};

/**
 * @brief Autonomous Self-Optimization Feedback Loop
 */
class AutonomousOptimizer {
public:
    AutonomousOptimizer();
    ~AutonomousOptimizer();

    // Initialization and lifecycle
    Status Initialize(const AutonomousOptimizationConfig& config);
    Status Shutdown();
    bool IsInitialized() const;

    // Optimization session management
    Status StartOptimizationSession(const std::string& session_name,
                                   const std::string& cluster_id,
                                   const std::vector<std::string>& target_dags,
                                   std::string& session_id);
    Status StopOptimizationSession(const std::string& session_id);
    Status GetOptimizationSession(const std::string& session_id, OptimizationSession& session);
    Status ListOptimizationSessions(std::vector<std::string>& session_ids);

    // Feedback collection and processing
    Status CollectFeedback(const OptimizationFeedbackEvent& event);
    Status ProcessFeedbackEvents();
    Status AnalyzePerformanceTrends(std::map<std::string, double>& trends);
    Status DetectOptimizationOpportunities(std::vector<AutonomousOptimizationAction>& opportunities);

    // Optimization action management
    Status PlanOptimizationAction(const OptimizationFeedbackEvent& event,
                                 AutonomousOptimizationAction& action);
    Status ExecuteOptimizationAction(const AutonomousOptimizationAction& action);
    Status RollbackOptimizationAction(const std::string& action_id);
    Status GetActionStatus(const std::string& action_id, std::string& status);

    // Policy management
    Status CreateOptimizationPolicy(const AutonomousOptimizationPolicy& policy);
    Status UpdateOptimizationPolicy(const std::string& policy_id, const AutonomousOptimizationPolicy& policy);
    Status DeleteOptimizationPolicy(const std::string& policy_id);
    Status GetOptimizationPolicy(const std::string& policy_id, AutonomousOptimizationPolicy& policy);
    Status ListOptimizationPolicies(std::vector<std::string>& policy_ids);

    // Continuous learning
    Status UpdateOptimizationModel(const OptimizationSession& session);
    Status GenerateOptimizationInsights(const OptimizationSession& session,
                                       std::vector<std::string>& insights);
    Status AdaptOptimizationStrategy(const std::string& session_id,
                                    const std::map<std::string, double>& feedback);

    // Safety and rollback
    Status ValidateOptimizationAction(const AutonomousOptimizationAction& action, bool& is_valid);
    Status ExecuteSafetyChecks(const OptimizationAction& action, bool& is_safe);
    Status RollbackToPreviousState(const std::string& session_id);
    Status RestoreSystemState(const std::string& checkpoint_id);

    // Performance monitoring
    Status MonitorOptimizationPerformance(const std::string& session_id,
                                         std::map<std::string, double>& performance_metrics);
    Status EvaluateOptimizationSuccess(const OptimizationSession& session,
                                      double& success_score);
    Status GenerateOptimizationReport(const std::string& session_id,
                                     const std::string& report_file);

    // Configuration and control
    Status UpdateOptimizationConfig(const AutonomousOptimizationConfig& config);
    Status SetOptimizationEnabled(bool enabled);
    Status SetLearningEnabled(bool enabled);
    Status SetSafetyMode(const std::string& safety_level);

    // Statistics and monitoring
    AutonomousOptimizationStats::Snapshot GetStats() const;
    Status GetOptimizationHistory(std::vector<OptimizationSession>& history);
    Status GetActiveOptimizations(std::vector<std::string>& session_ids);

    // Integration with other systems
    Status SetDAGGenerator(std::shared_ptr<DAGGenerator> dag_generator);
    Status SetSyntheticTestbed(std::shared_ptr<SyntheticTestbed> testbed);
    Status SetEvolutionManager(std::shared_ptr<EvolutionManager> evolution_manager);
    Status SetTelemetryAnalytics(std::shared_ptr<TelemetryAnalytics> analytics);
    Status SetGovernanceManager(std::shared_ptr<GovernanceManager> governance);

    // Continuous Self-Optimization Feedback Loop
    Status InitializeContinuousOptimization(const ContinuousOptimizationConfig& config);
    Status StartContinuousOptimization();
    Status StopContinuousOptimization();
    Status PauseContinuousOptimization();
    Status ResumeContinuousOptimization();
    bool IsContinuousOptimizationActive() const;
    
    // Real-time feedback collection
    Status CollectPerformanceFeedback(PerformanceFeedback& feedback);
    Status ProcessPerformanceFeedback(const PerformanceFeedback& feedback);
    Status UpdateLearningModels(const PerformanceFeedback& feedback);
    Status GenerateOptimizationInsights(std::vector<OptimizationInsight>& insights);
    
    // Continuous learning and adaptation
    Status UpdateContinuousLearning(const std::vector<PerformanceFeedback>& feedback_batch);
    Status AdaptOptimizationParameters(const std::vector<OptimizationInsight>& insights);
    Status UpdatePerformanceModels(const std::vector<PerformanceFeedback>& feedback);
    Status UpdateRiskModels(const std::vector<OptimizationInsight>& insights);
    
    // Automated optimization decision making
    Status EvaluateOptimizationOpportunities(const std::vector<OptimizationInsight>& insights,
                                            std::vector<AutonomousOptimizationAction>& actions);
    Status PlanContinuousOptimizationActions(const PerformanceFeedback& feedback,
                                            std::vector<AutonomousOptimizationAction>& actions);
    Status ExecuteContinuousOptimizationActions(const std::vector<AutonomousOptimizationAction>& actions);
    Status MonitorOptimizationProgress(const std::string& session_id, double& progress);
    
    // Safety and rollback mechanisms
    Status ValidateOptimizationSafety(const AutonomousOptimizationAction& action, bool& is_safe);
    Status ImplementSafetyConstraints(const std::vector<AutonomousOptimizationAction>& actions,
                                     std::vector<AutonomousOptimizationAction>& safe_actions);
    Status ExecuteAutomaticRollback(const std::string& session_id, const std::string& reason);
    Status AssessRollbackNecessity(const PerformanceFeedback& feedback, bool& rollback_needed);
    
    // Performance monitoring and alerting
    Status DetectPerformanceAnomalies(const PerformanceFeedback& feedback,
                                     std::vector<std::string>& anomalies);
    Status GenerateOptimizationAlerts(const std::vector<OptimizationInsight>& insights,
                                     std::vector<std::string>& alerts);
    Status AssessOptimizationImpact(const std::string& session_id, double& impact_score);
    
    // Learning and model management
    Status SaveLearningState(const std::string& state_file);
    Status LoadLearningState(const std::string& state_file);
    Status ResetLearningState();
    Status ExportLearningInsights(const std::string& export_file);
    Status ImportLearningInsights(const std::string& import_file);
    
    // Advanced analytics and reporting
    Status GenerateContinuousOptimizationReport(const std::string& session_id,
                                               nlohmann::json& report);
    Status AnalyzeOptimizationTrends(const std::vector<OptimizationSession>& sessions,
                                    std::map<std::string, double>& trends);
    Status PredictOptimizationOutcomes(const std::vector<AutonomousOptimizationAction>& actions,
                                      std::map<std::string, double>& predictions);
    Status EvaluateLearningEffectiveness(double& effectiveness_score);

private:
    // Core optimization logic
    Status ProcessOptimizationTrigger(const OptimizationFeedbackEvent& event);
    Status GenerateOptimizationPlan(const OptimizationSession& session,
                                   std::vector<AutonomousOptimizationAction>& actions);
    Status ExecuteOptimizationPlan(const std::vector<AutonomousOptimizationAction>& actions,
                                  const std::string& session_id);
    Status EvaluateOptimizationResults(const OptimizationSession& session);

    // Feedback processing
    Status AnalyzeFeedbackEvent(const OptimizationFeedbackEvent& event,
                               std::vector<AutonomousOptimizationAction>& actions);
    Status CorrelateFeedbackEvents(const std::vector<OptimizationFeedbackEvent>& events,
                                  std::map<std::string, double>& correlations);
    Status PredictOptimizationImpact(const AutonomousOptimizationAction& action,
                                    std::map<std::string, double>& predicted_impact);

    // Learning and adaptation
    Status UpdateLearningModel(const OptimizationSession& session);
    Status ExtractLearningFeatures(const OptimizationSession& session,
                                  std::map<std::string, double>& features);
    Status ApplyLearnedInsights(const std::string& session_id,
                               const std::map<std::string, double>& insights);

    // Safety and validation
    Status ValidateActionSafety(const AutonomousOptimizationAction& action, bool& is_safe);
    Status CheckSystemConstraints(const AutonomousOptimizationAction& action, bool& is_valid);
    Status EstimateActionRisk(const AutonomousOptimizationAction& action, double& risk_score);
    Status CreateSystemCheckpoint(const std::string& session_id, std::string& checkpoint_id);

    // Utility functions
    std::string GenerateSessionId();
    std::string GenerateActionId();
    std::string GeneratePolicyId();
    Status LoadDefaultPolicies();
    void CleanupExpiredData();

    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> optimization_enabled_{true};
    std::atomic<bool> learning_enabled_{true};
    AutonomousOptimizationConfig config_;
    
    // External dependencies
    std::shared_ptr<DAGGenerator> dag_generator_;
    std::shared_ptr<SyntheticTestbed> testbed_;
    std::shared_ptr<EvolutionManager> evolution_manager_;
    std::shared_ptr<TelemetryAnalytics> analytics_;
    std::shared_ptr<GovernanceManager> governance_;
    
    // Optimization state
    std::map<std::string, OptimizationSession> active_sessions_;
    std::map<std::string, AutonomousOptimizationAction> pending_actions_;
    std::map<std::string, AutonomousOptimizationPolicy> optimization_policies_;
    std::vector<OptimizationFeedbackEvent> feedback_queue_;
    std::mutex sessions_mutex_;
    std::mutex actions_mutex_;
    std::mutex policies_mutex_;
    std::mutex feedback_mutex_;
    
    // Background threads
    std::thread feedback_processing_thread_;
    std::thread optimization_thread_;
    std::thread learning_thread_;
    std::atomic<bool> shutdown_requested_{false};
    
    // Learning state
    std::map<std::string, double> learned_parameters_;
    std::vector<OptimizationSession> learning_history_;
    std::mutex learning_mutex_;
    
    // Random number generation
    std::random_device rd_;
    mutable std::mt19937 gen_;
    mutable std::uniform_real_distribution<double> uniform_dist_;
    mutable std::normal_distribution<double> normal_dist_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    AutonomousOptimizationStats stats_;
    
    // Profiler integration
    mutable std::mutex profiler_mutex_;
    std::map<std::string, std::chrono::steady_clock::time_point> profiler_timers_;
    
    // Continuous optimization state
    ContinuousOptimizationConfig continuous_config_;
    ContinuousLearningState learning_state_;
    std::atomic<bool> continuous_optimization_active_{false};
    std::atomic<bool> continuous_optimization_paused_{false};
    
    // Continuous optimization threads
    std::thread continuous_optimization_thread_;
    std::thread feedback_collection_thread_;
    std::thread learning_update_thread_;
    std::thread monitoring_thread_;
    
    // Real-time feedback and insights
    std::deque<PerformanceFeedback> performance_feedback_queue_;
    std::deque<OptimizationInsight> optimization_insights_queue_;
    std::mutex feedback_queue_mutex_;
    std::mutex insights_queue_mutex_;
    std::condition_variable feedback_cv_;
    std::condition_variable insights_cv_;
    
    // Safety and monitoring
    std::map<std::string, std::chrono::steady_clock::time_point> optimization_start_times_;
    std::map<std::string, double> optimization_progress_;
    std::vector<std::string> performance_anomalies_;
    std::vector<std::string> optimization_alerts_;
    std::mutex safety_mutex_;
    std::mutex monitoring_mutex_;
    
    // Learning model state
    std::map<std::string, std::vector<double>> performance_model_weights_;
    std::map<std::string, std::vector<double>> optimization_model_weights_;
    std::map<std::string, std::vector<double>> risk_model_weights_;
    std::mutex models_mutex_;
    
    // Continuous optimization statistics
    std::atomic<uint32_t> continuous_optimization_cycles_{0};
    std::atomic<uint32_t> feedback_events_processed_{0};
    std::atomic<uint32_t> insights_generated_{0};
    std::atomic<uint32_t> automatic_rollbacks_{0};
    std::atomic<double> continuous_improvement_percentage_{0.0};
};

} // namespace edge_ai
