/**
 * @file ml_orchestrator.h
 * @brief ML-driven orchestration system for distributed inference optimization
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
#include "ml_policy/ml_based_policy.h"
#include "optimization/optimization_manager.h"

namespace edge_ai {

/**
 * @brief ML orchestration strategy
 */
enum class MLOrchestrationStrategy {
    RULE_BASED = 0,
    ML_DRIVEN = 1,
    HYBRID = 2,
    ADAPTIVE = 3
};

/**
 * @brief Cross-node optimization features
 */
struct CrossNodeOptimizationFeatures {
    // Node characteristics
    std::string node_id;
    double cpu_utilization{0.0};
    double memory_utilization{0.0};
    double gpu_utilization{0.0};
    double energy_consumption{0.0};
    double network_latency{0.0};
    double bandwidth_utilization{0.0};
    
    // Task characteristics
    std::string task_type;
    uint32_t task_priority{0};
    std::chrono::milliseconds estimated_duration{0};
    uint64_t estimated_memory_usage{0};
    bool requires_gpu{false};
    bool requires_low_latency{false};
    
    // Historical performance
    double avg_execution_time{0.0};
    double success_rate{1.0};
    double energy_efficiency{1.0};
    double cost_per_inference{0.0};
    
    // Context features
    std::chrono::steady_clock::time_point timestamp;
    uint32_t cluster_load{0};
    uint32_t queue_depth{0};
    double temperature{25.0};  // Node temperature
    
    CrossNodeOptimizationFeatures() {
        timestamp = std::chrono::steady_clock::now();
    }
};

/**
 * @brief ML orchestration decision
 */
struct MLOrchestrationDecision {
    std::string decision_id;
    std::string task_id;
    
    // Placement decision
    std::string optimal_node_id;
    std::string backup_node_id;
    std::vector<std::string> alternative_nodes;
    
    // Optimization parameters
    uint32_t optimal_batch_size{1};
    std::chrono::milliseconds estimated_latency{0};
    double estimated_throughput{0.0};
    double estimated_energy_consumption{0.0};
    double estimated_cost{0.0};
    
    // ML model outputs
    double confidence_score{0.0};
    double performance_prediction{0.0};
    double energy_efficiency_prediction{0.0};
    double cost_efficiency_prediction{0.0};
    
    // Decision metadata
    MLOrchestrationStrategy strategy_used{MLOrchestrationStrategy::ML_DRIVEN};
    std::string model_version;
    std::string reasoning;
    std::chrono::steady_clock::time_point decision_time;
    
    // Learning feedback
    bool feedback_received{false};
    double actual_performance{0.0};
    double actual_energy_consumption{0.0};
    double actual_cost{0.0};
    std::chrono::steady_clock::time_point feedback_time;
    
    MLOrchestrationDecision() {
        decision_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Federated learning configuration
 */
struct FederatedLearningConfig {
    bool enabled{false};
    std::chrono::minutes aggregation_interval{60};
    uint32_t min_nodes_for_aggregation{3};
    double learning_rate{0.01};
    uint32_t max_rounds{100};
    double convergence_threshold{0.001};
    
    // Privacy and security
    bool differential_privacy_enabled{false};
    double privacy_budget{1.0};
    bool secure_aggregation_enabled{false};
    
    FederatedLearningConfig() = default;
};

/**
 * @brief ML orchestrator statistics
 */
struct MLOrchestratorStats {
    std::atomic<uint64_t> total_decisions{0};
    std::atomic<uint64_t> ml_decisions{0};
    std::atomic<uint64_t> rule_based_decisions{0};
    std::atomic<uint64_t> hybrid_decisions{0};
    std::atomic<uint64_t> adaptive_decisions{0};
    
    // Performance metrics
    std::atomic<double> avg_decision_time_ms{0.0};
    std::atomic<double> avg_confidence_score{0.0};
    std::atomic<double> avg_performance_improvement{0.0};
    std::atomic<double> avg_energy_efficiency_improvement{0.0};
    std::atomic<double> avg_cost_efficiency_improvement{0.0};
    
    // Learning metrics
    std::atomic<uint64_t> total_training_examples{0};
    std::atomic<uint64_t> model_updates{0};
    std::atomic<uint64_t> federated_rounds{0};
    std::atomic<double> model_accuracy{0.0};
    std::atomic<double> convergence_rate{0.0};
    
    // Cross-node optimization
    std::atomic<uint64_t> cross_node_optimizations{0};
    std::atomic<double> avg_load_balance_improvement{0.0};
    std::atomic<double> avg_latency_reduction{0.0};
    std::atomic<double> avg_throughput_improvement{0.0};
    
    MLOrchestratorStats() = default;
    
    struct Snapshot {
        uint64_t total_decisions;
        uint64_t ml_decisions;
        uint64_t rule_based_decisions;
        uint64_t hybrid_decisions;
        uint64_t adaptive_decisions;
        double avg_decision_time_ms;
        double avg_confidence_score;
        double avg_performance_improvement;
        double avg_energy_efficiency_improvement;
        double avg_cost_efficiency_improvement;
        uint64_t total_training_examples;
        uint64_t model_updates;
        uint64_t federated_rounds;
        double model_accuracy;
        double convergence_rate;
        uint64_t cross_node_optimizations;
        double avg_load_balance_improvement;
        double avg_latency_reduction;
        double avg_throughput_improvement;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_decisions = total_decisions.load();
        snapshot.ml_decisions = ml_decisions.load();
        snapshot.rule_based_decisions = rule_based_decisions.load();
        snapshot.hybrid_decisions = hybrid_decisions.load();
        snapshot.adaptive_decisions = adaptive_decisions.load();
        snapshot.avg_decision_time_ms = avg_decision_time_ms.load();
        snapshot.avg_confidence_score = avg_confidence_score.load();
        snapshot.avg_performance_improvement = avg_performance_improvement.load();
        snapshot.avg_energy_efficiency_improvement = avg_energy_efficiency_improvement.load();
        snapshot.avg_cost_efficiency_improvement = avg_cost_efficiency_improvement.load();
        snapshot.total_training_examples = total_training_examples.load();
        snapshot.model_updates = model_updates.load();
        snapshot.federated_rounds = federated_rounds.load();
        snapshot.model_accuracy = model_accuracy.load();
        snapshot.convergence_rate = convergence_rate.load();
        snapshot.cross_node_optimizations = cross_node_optimizations.load();
        snapshot.avg_load_balance_improvement = avg_load_balance_improvement.load();
        snapshot.avg_latency_reduction = avg_latency_reduction.load();
        snapshot.avg_throughput_improvement = avg_throughput_improvement.load();
        return snapshot;
    }
};

/**
 * @brief ML-driven orchestration system
 */
class MLOrchestrator {
public:
    /**
     * @brief Constructor
     * @param optimization_manager Optimization manager for ML policies
     * @param cluster_manager Cluster manager for node information
     */
    MLOrchestrator(std::shared_ptr<OptimizationManager> optimization_manager,
                  std::shared_ptr<ClusterManager> cluster_manager);
    
    /**
     * @brief Destructor
     */
    ~MLOrchestrator();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // ML orchestration
    Status MakeOrchestrationDecision(const std::string& task_id,
                                   const CrossNodeOptimizationFeatures& features,
                                   MLOrchestrationDecision& decision);
    
    Status MakeBatchOrchestrationDecision(const std::vector<std::string>& task_ids,
                                        const std::vector<CrossNodeOptimizationFeatures>& features,
                                        std::vector<MLOrchestrationDecision>& decisions);
    
    // Strategy management
    Status SetOrchestrationStrategy(MLOrchestrationStrategy strategy);
    MLOrchestrationStrategy GetCurrentStrategy() const;
    Status EnableAdaptiveStrategy();
    Status DisableAdaptiveStrategy();
    
    // Learning and training
    Status ProvideFeedback(const std::string& decision_id,
                          double actual_performance,
                          double actual_energy_consumption,
                          double actual_cost);
    
    Status TrainModel(const std::vector<CrossNodeOptimizationFeatures>& features,
                     const std::vector<MLOrchestrationDecision>& decisions);
    
    Status UpdateModel();
    Status RetrainModel();
    
    // Federated learning
    Status EnableFederatedLearning(const FederatedLearningConfig& config);
    Status DisableFederatedLearning();
    Status ParticipateInFederatedRound();
    Status AggregateFederatedModels();
    
    // Cross-node optimization
    Status OptimizeCrossNodePlacement(const std::vector<std::string>& task_ids);
    Status OptimizeLoadBalancing();
    Status OptimizeEnergyEfficiency();
    Status OptimizeCostEfficiency();
    
    // Predictive optimization
    Status PredictOptimalPlacements(const std::vector<std::string>& task_ids,
                                   std::chrono::minutes prediction_horizon);
    Status PredictNodePerformance(const std::string& node_id,
                                 const CrossNodeOptimizationFeatures& features);
    
    // Model management
    Status SaveModel(const std::string& model_path);
    Status LoadModel(const std::string& model_path);
    Status ExportModelWeights(std::vector<double>& weights);
    Status ImportModelWeights(const std::vector<double>& weights);
    
    // Statistics and monitoring
    MLOrchestratorStats::Snapshot GetStats() const;
    void ResetStats();
    Status GenerateOrchestrationReport();
    
    // Configuration
    void SetMLPolicyEnabled(bool enabled);
    void SetLearningEnabled(bool enabled);
    void SetFederatedLearningEnabled(bool enabled);
    void SetCrossNodeOptimizationEnabled(bool enabled);

private:
    // Internal orchestration methods
    MLOrchestrationDecision MakeMLDecision(const std::string& task_id,
                                          const CrossNodeOptimizationFeatures& features);
    
    MLOrchestrationDecision MakeRuleBasedDecision(const std::string& task_id,
                                                 const CrossNodeOptimizationFeatures& features);
    
    MLOrchestrationDecision MakeHybridDecision(const std::string& task_id,
                                              const CrossNodeOptimizationFeatures& features);
    
    MLOrchestrationDecision MakeAdaptiveDecision(const std::string& task_id,
                                                const CrossNodeOptimizationFeatures& features);
    
    // Learning and adaptation
    Status UpdateMLModel(const std::vector<CrossNodeOptimizationFeatures>& features,
                        const std::vector<MLOrchestrationDecision>& decisions);
    
    Status AdaptStrategy(const std::vector<MLOrchestrationDecision>& recent_decisions);
    double EvaluateStrategyPerformance(MLOrchestrationStrategy strategy);
    
    // Cross-node optimization algorithms
    Status ApplyGeneticAlgorithm(const std::vector<std::string>& task_ids);
    Status ApplySimulatedAnnealing(const std::vector<std::string>& task_ids);
    Status ApplyParticleSwarmOptimization(const std::vector<std::string>& task_ids);
    
    // Predictive modeling
    double PredictNodeLatency(const std::string& node_id, const CrossNodeOptimizationFeatures& features);
    double PredictNodeThroughput(const std::string& node_id, const CrossNodeOptimizationFeatures& features);
    double PredictNodeEnergyConsumption(const std::string& node_id, const CrossNodeOptimizationFeatures& features);
    double PredictNodeCost(const std::string& node_id, const CrossNodeOptimizationFeatures& features);
    
    // Federated learning implementation
    Status CollectLocalModelUpdates();
    Status AggregateModelUpdates();
    Status DistributeAggregatedModel();
    Status ValidateModelConvergence();
    
    // Threading and synchronization
    void LearningThread();
    void FederatedLearningThread();
    void CrossNodeOptimizationThread();
    void ModelUpdateThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> ml_policy_enabled_{true};
    std::atomic<bool> learning_enabled_{true};
    std::atomic<bool> federated_learning_enabled_{false};
    std::atomic<bool> cross_node_optimization_enabled_{true};
    
    // Current strategy
    std::atomic<MLOrchestrationStrategy> current_strategy_{MLOrchestrationStrategy::ML_DRIVEN};
    std::atomic<bool> adaptive_strategy_enabled_{false};
    
    // Dependencies
    std::shared_ptr<OptimizationManager> optimization_manager_;
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<MLBasedPolicy> ml_policy_;
    
    // Learning state
    mutable std::mutex learning_mutex_;
    std::vector<CrossNodeOptimizationFeatures> training_features_;
    std::vector<MLOrchestrationDecision> training_decisions_;
    std::map<std::string, MLOrchestrationDecision> pending_feedback_;
    
    // Federated learning
    mutable std::mutex federated_mutex_;
    FederatedLearningConfig federated_config_;
    std::vector<double> local_model_weights_;
    std::vector<double> aggregated_model_weights_;
    std::chrono::steady_clock::time_point last_aggregation_time_;
    
    // Cross-node optimization
    mutable std::mutex optimization_mutex_;
    std::queue<std::vector<std::string>> pending_optimizations_;
    std::map<std::string, double> node_performance_predictions_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    MLOrchestratorStats stats_;
    
    // Threading
    std::thread learning_thread_;
    std::thread federated_learning_thread_;
    std::thread cross_node_optimization_thread_;
    std::thread model_update_thread_;
    
    std::condition_variable learning_cv_;
    std::condition_variable federated_cv_;
    std::condition_variable optimization_cv_;
    std::condition_variable model_update_cv_;
    
    mutable std::mutex learning_cv_mutex_;
    mutable std::mutex federated_cv_mutex_;
    mutable std::mutex optimization_cv_mutex_;
    mutable std::mutex model_update_cv_mutex_;
};

} // namespace edge_ai
