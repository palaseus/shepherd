/**
 * @file evolution_manager.h
 * @brief Self-optimization and continuous evolution manager
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
#include "core/types.h"
#include "distributed/cluster_types.h"
#include "ml_policy/ml_based_policy.h"
#include "governance/governance_manager.h"
#include "federation/federation_manager.h"

// Forward declarations
namespace edge_ai {
    class ClusterManager;
    class MLBasedPolicy;
    class GovernanceManager;
    class FederationManager;
}

namespace edge_ai {

/**
 * @brief Evolution strategy types
 */
enum class EvolutionStrategy {
    GENETIC_ALGORITHM = 0,
    PARTICLE_SWARM_OPTIMIZATION = 1,
    SIMULATED_ANNEALING = 2,
    REINFORCEMENT_LEARNING = 3,
    NEURAL_ARCHITECTURE_SEARCH = 4,
    HYPERPARAMETER_OPTIMIZATION = 5,
    MULTI_OBJECTIVE_OPTIMIZATION = 6,
    ADAPTIVE_EVOLUTION = 7
};

/**
 * @brief Evolution objective types
 */
enum class EvolutionObjective {
    MINIMIZE_LATENCY = 0,
    MAXIMIZE_THROUGHPUT = 1,
    MINIMIZE_RESOURCE_USAGE = 2,
    MAXIMIZE_ENERGY_EFFICIENCY = 3,
    MINIMIZE_COST = 4,
    MAXIMIZE_ACCURACY = 5,
    MINIMIZE_MEMORY_USAGE = 6,
    MAXIMIZE_AVAILABILITY = 7,
    MULTI_OBJECTIVE = 8
};

/**
 * @brief Evolution individual (candidate solution)
 */
struct EvolutionIndividual {
    std::string individual_id;
    std::string generation_id;
    
    // Genetic representation
    std::vector<double> genes;
    std::map<std::string, double> hyperparameters;
    std::map<std::string, std::string> configuration;
    
    // Fitness evaluation
    double fitness_score;
    std::map<EvolutionObjective, double> objective_scores;
    bool is_evaluated;
    std::chrono::steady_clock::time_point evaluation_time;
    
    // Evolution metadata
    std::string parent_ids;
    std::string mutation_type;
    double mutation_strength;
    uint32_t generation_number;
    bool is_elite;
    
    // Performance metrics
    double latency_ms;
    double throughput_ops_per_sec;
    double resource_utilization;
    double energy_consumption;
    double cost_per_operation;
    double accuracy_score;
    double memory_usage_mb;
    double availability_percent;
};

/**
 * @brief Evolution population
 */
struct EvolutionPopulation {
    std::string population_id;
    std::string evolution_task_id;
    EvolutionStrategy strategy;
    std::vector<EvolutionObjective> objectives;
    
    // Population state
    std::vector<EvolutionIndividual> individuals;
    uint32_t generation_number;
    uint32_t max_generations;
    uint32_t population_size;
    uint32_t elite_size;
    
    // Evolution parameters
    double mutation_rate;
    double crossover_rate;
    double selection_pressure;
    double diversity_threshold;
    
    // Performance tracking
    double best_fitness;
    double avg_fitness;
    double fitness_std_dev;
    double diversity_score;
    bool has_converged;
    
    // Timing
    std::chrono::steady_clock::time_point creation_time;
    std::chrono::steady_clock::time_point last_update_time;
    std::chrono::milliseconds total_evolution_time;
};

/**
 * @brief Evolution task configuration
 */
struct EvolutionTask {
    std::string task_id;
    std::string name;
    std::string description;
    
    // Task parameters
    EvolutionStrategy strategy;
    std::vector<EvolutionObjective> objectives;
    std::map<std::string, double> objective_weights;
    
    // Search space
    std::map<std::string, std::pair<double, double>> parameter_bounds;
    std::map<std::string, std::vector<std::string>> discrete_parameters;
    std::vector<std::string> parameter_names;
    
    // Evolution configuration
    uint32_t population_size;
    uint32_t max_generations;
    uint32_t elite_size;
    double mutation_rate;
    double crossover_rate;
    double convergence_threshold;
    
    // Evaluation configuration
    std::chrono::milliseconds evaluation_timeout;
    uint32_t evaluation_samples;
    std::string evaluation_environment;
    bool use_simulation;
    
    // Status
    bool is_active;
    bool is_completed;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::string status_message;
    double progress_percent;
};

/**
 * @brief Evolution result
 */
struct EvolutionResult {
    std::string result_id;
    std::string task_id;
    std::string best_individual_id;
    
    // Best solution
    EvolutionIndividual best_individual;
    std::map<EvolutionObjective, double> best_objective_scores;
    double best_fitness_score;
    
    // Evolution statistics
    uint32_t total_generations;
    uint32_t total_evaluations;
    std::chrono::milliseconds total_time;
    double convergence_rate;
    double diversity_maintained;
    
    // Performance improvements
    std::map<std::string, double> performance_improvements;
    double overall_improvement_percent;
    bool meets_objectives;
    
    // Recommendations
    std::vector<std::string> recommended_actions;
    std::vector<std::string> implementation_steps;
    double confidence_score;
    
    // Validation
    bool is_validated;
    std::string validation_status;
    std::vector<std::string> validation_errors;
};

/**
 * @brief Continuous learning configuration
 */
struct ContinuousLearningConfig {
    std::string config_id;
    std::string model_type;
    
    // Learning parameters
    double learning_rate;
    double adaptation_rate;
    uint32_t adaptation_frequency;
    std::chrono::milliseconds learning_interval;
    
    // Online learning
    bool online_learning_enabled;
    uint32_t online_batch_size;
    double online_learning_rate;
    double forgetting_factor;
    
    // Meta-learning
    bool meta_learning_enabled;
    std::string meta_learning_strategy;
    uint32_t meta_learning_episodes;
    
    // Evolution integration
    bool evolution_integration_enabled;
    std::string evolution_trigger_condition;
    double evolution_trigger_threshold;
    
    // Performance monitoring
    bool performance_monitoring_enabled;
    std::vector<std::string> monitored_metrics;
    double performance_degradation_threshold;
};

/**
 * @brief Evolution statistics
 */
struct EvolutionStats {
    // Task metrics
    std::atomic<uint64_t> total_tasks{0};
    std::atomic<uint64_t> completed_tasks{0};
    std::atomic<uint64_t> failed_tasks{0};
    std::atomic<uint64_t> active_tasks{0};
    
    // Population metrics
    std::atomic<uint64_t> total_individuals_evaluated{0};
    std::atomic<uint64_t> total_generations{0};
    std::atomic<double> avg_generations_per_task{0.0};
    std::atomic<double> avg_evaluation_time_ms{0.0};
    
    // Performance metrics
    std::atomic<double> avg_fitness_improvement{0.0};
    std::atomic<double> avg_convergence_rate{0.0};
    std::atomic<double> avg_diversity_score{0.0};
    std::atomic<double> evolution_effectiveness{0.0};
    
    // Continuous learning
    std::atomic<uint64_t> continuous_learning_updates{0};
    std::atomic<double> avg_learning_improvement{0.0};
    std::atomic<double> adaptation_success_rate{0.0};
    std::atomic<double> meta_learning_effectiveness{0.0};
    
    // Resource usage
    std::atomic<double> avg_cpu_usage_percent{0.0};
    std::atomic<double> avg_memory_usage_mb{0.0};
    std::atomic<double> avg_energy_consumption{0.0};
    std::atomic<double> evolution_efficiency{0.0};
    
    /**
     * @brief Get a snapshot of current statistics
     */
    struct Snapshot {
        uint64_t total_tasks;
        uint64_t completed_tasks;
        uint64_t failed_tasks;
        uint64_t active_tasks;
        uint64_t total_individuals_evaluated;
        uint64_t total_generations;
        double avg_generations_per_task;
        double avg_evaluation_time_ms;
        double avg_fitness_improvement;
        double avg_convergence_rate;
        double avg_diversity_score;
        double evolution_effectiveness;
        uint64_t continuous_learning_updates;
        double avg_learning_improvement;
        double adaptation_success_rate;
        double meta_learning_effectiveness;
        double avg_cpu_usage_percent;
        double avg_memory_usage_mb;
        double avg_energy_consumption;
        double evolution_efficiency;
    };
    
    Snapshot GetSnapshot() const {
        return {
            total_tasks.load(),
            completed_tasks.load(),
            failed_tasks.load(),
            active_tasks.load(),
            total_individuals_evaluated.load(),
            total_generations.load(),
            avg_generations_per_task.load(),
            avg_evaluation_time_ms.load(),
            avg_fitness_improvement.load(),
            avg_convergence_rate.load(),
            avg_diversity_score.load(),
            evolution_effectiveness.load(),
            continuous_learning_updates.load(),
            avg_learning_improvement.load(),
            adaptation_success_rate.load(),
            meta_learning_effectiveness.load(),
            avg_cpu_usage_percent.load(),
            avg_memory_usage_mb.load(),
            avg_energy_consumption.load(),
            evolution_efficiency.load()
        };
    }
};

/**
 * @class EvolutionManager
 * @brief Self-optimization and continuous evolution manager
 */
class EvolutionManager {
public:
    explicit EvolutionManager(std::shared_ptr<ClusterManager> cluster_manager,
                            std::shared_ptr<MLBasedPolicy> ml_policy,
                            std::shared_ptr<GovernanceManager> governance_manager,
                            std::shared_ptr<FederationManager> federation_manager);
    virtual ~EvolutionManager();
    
    /**
     * @brief Initialize the evolution manager
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the evolution manager
     */
    Status Shutdown();
    
    /**
     * @brief Check if the evolution manager is initialized
     */
    bool IsInitialized() const;
    
    // Evolution Task Management
    
    /**
     * @brief Create evolution task
     */
    Status CreateEvolutionTask(const EvolutionTask& task);
    
    /**
     * @brief Start evolution task
     */
    Status StartEvolutionTask(const std::string& task_id);
    
    /**
     * @brief Stop evolution task
     */
    Status StopEvolutionTask(const std::string& task_id);
    
    /**
     * @brief Get evolution task status
     */
    Status GetEvolutionTaskStatus(const std::string& task_id, EvolutionTask& task) const;
    
    /**
     * @brief Get evolution task results
     */
    Status GetEvolutionTaskResults(const std::string& task_id, EvolutionResult& result) const;
    
    /**
     * @brief Get all evolution tasks
     */
    std::vector<EvolutionTask> GetEvolutionTasks() const;
    
    /**
     * @brief Delete evolution task
     */
    Status DeleteEvolutionTask(const std::string& task_id);
    
    // Population Management
    
    /**
     * @brief Create initial population
     */
    Status CreateInitialPopulation(const std::string& task_id, EvolutionPopulation& population);
    
    /**
     * @brief Evolve population for one generation
     */
    Status EvolvePopulation(const std::string& population_id, EvolutionPopulation& evolved_population);
    
    /**
     * @brief Evaluate individual fitness
     */
    Status EvaluateIndividual(const std::string& task_id, EvolutionIndividual& individual);
    
    /**
     * @brief Get population statistics
     */
    Status GetPopulationStatistics(const std::string& population_id, 
                                 std::map<std::string, double>& statistics) const;
    
    // Continuous Learning
    
    /**
     * @brief Configure continuous learning
     */
    Status ConfigureContinuousLearning(const ContinuousLearningConfig& config);
    
    /**
     * @brief Update model with new data
     */
    Status UpdateModelWithNewData(const std::string& model_id, 
                                const std::vector<std::vector<double>>& features,
                                const std::vector<double>& targets);
    
    /**
     * @brief Perform online adaptation
     */
    Status PerformOnlineAdaptation(const std::string& model_id);
    
    /**
     * @brief Perform meta-learning
     */
    Status PerformMetaLearning(const std::vector<std::string>& task_ids,
                             std::map<std::string, double>& meta_learned_params);
    
    // Hyperparameter Optimization
    
    /**
     * @brief Optimize hyperparameters
     */
    Status OptimizeHyperparameters(const std::string& model_id,
                                 const std::map<std::string, std::pair<double, double>>& parameter_bounds,
                                 std::map<std::string, double>& optimized_params);
    
    /**
     * @brief Tune model parameters
     */
    Status TuneModelParameters(const std::string& model_id,
                             const std::vector<std::string>& parameter_names,
                             std::map<std::string, double>& tuned_params);
    
    // Neural Architecture Search
    
    /**
     * @brief Perform neural architecture search
     */
    Status PerformNeuralArchitectureSearch(const std::string& task_id,
                                         const std::map<std::string, std::vector<int>>& search_space,
                                         std::map<std::string, int>& optimal_architecture);
    
    /**
     * @brief Evolve neural architecture
     */
    Status EvolveNeuralArchitecture(const std::string& base_architecture_id,
                                  const std::vector<std::string>& evolution_operations,
                                  std::string& evolved_architecture_id);
    
    // Multi-Objective Optimization
    
    /**
     * @brief Perform multi-objective optimization
     */
    Status PerformMultiObjectiveOptimization(const std::string& task_id,
                                           const std::vector<EvolutionObjective>& objectives,
                                           std::vector<EvolutionIndividual>& pareto_front);
    
    /**
     * @brief Get pareto front solutions
     */
    Status GetParetoFrontSolutions(const std::string& task_id,
                                 std::vector<EvolutionIndividual>& pareto_solutions);
    
    // Predictive Simulation
    
    /**
     * @brief Create predictive simulation environment
     */
    Status CreatePredictiveSimulation(const std::string& simulation_id,
                                    const std::map<std::string, double>& simulation_params);
    
    /**
     * @brief Run predictive simulation
     */
    Status RunPredictiveSimulation(const std::string& simulation_id,
                                 const EvolutionIndividual& candidate_solution,
                                 std::map<std::string, double>& simulation_results);
    
    /**
     * @brief Validate solution in simulation
     */
    Status ValidateSolutionInSimulation(const std::string& simulation_id,
                                      const EvolutionIndividual& solution,
                                      bool& is_valid);
    
    // Analytics and Reporting
    
    /**
     * @brief Generate evolution progress report
     */
    Status GenerateEvolutionProgressReport(const std::string& task_id,
                                         std::map<std::string, double>& progress_metrics);
    
    /**
     * @brief Generate continuous learning report
     */
    Status GenerateContinuousLearningReport(std::map<std::string, double>& learning_metrics);
    
    /**
     * @brief Get evolution statistics
     */
    EvolutionStats::Snapshot GetStats() const;
    
    /**
     * @brief Reset evolution statistics
     */
    void ResetStats();
    
    /**
     * @brief Generate evolution insights
     */
    Status GenerateEvolutionInsights(std::vector<std::string>& insights);

private:
    // Core components
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<MLBasedPolicy> ml_policy_;
    std::shared_ptr<GovernanceManager> governance_manager_;
    std::shared_ptr<FederationManager> federation_manager_;
    
    // State management
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Evolution storage
    mutable std::mutex tasks_mutex_;
    std::map<std::string, EvolutionTask> evolution_tasks_;
    std::map<std::string, EvolutionResult> evolution_results_;
    std::map<std::string, EvolutionPopulation> populations_;
    
    // Continuous learning storage
    mutable std::mutex learning_mutex_;
    std::map<std::string, ContinuousLearningConfig> learning_configs_;
    std::map<std::string, std::vector<std::vector<double>>> model_training_data_;
    std::map<std::string, std::vector<double>> model_targets_;
    
    // Simulation storage
    mutable std::mutex simulation_mutex_;
    std::map<std::string, std::map<std::string, double>> simulation_environments_;
    std::map<std::string, std::map<std::string, double>> simulation_results_;
    
    // Background threads
    std::thread evolution_thread_;
    std::thread learning_thread_;
    std::thread simulation_thread_;
    
    // Condition variables
    std::mutex evolution_cv_mutex_;
    std::condition_variable evolution_cv_;
    std::mutex learning_cv_mutex_;
    std::condition_variable learning_cv_;
    std::mutex simulation_cv_mutex_;
    std::condition_variable simulation_cv_;
    
    // Random number generation
    std::random_device rd_;
    mutable std::mt19937 gen_;
    mutable std::uniform_real_distribution<double> uniform_dist_;
    mutable std::normal_distribution<double> normal_dist_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    EvolutionStats stats_;
    
    // Private methods
    
    /**
     * @brief Background evolution thread
     */
    void EvolutionThread();
    
    /**
     * @brief Background continuous learning thread
     */
    void LearningThread();
    
    /**
     * @brief Background simulation thread
     */
    void SimulationThread();
    
    /**
     * @brief Initialize random number generators
     */
    void InitializeRandomGenerators();
    
    /**
     * @brief Generate random individual
     */
    EvolutionIndividual GenerateRandomIndividual(const EvolutionTask& task) const;
    
    /**
     * @brief Perform crossover operation
     */
    Status PerformCrossover(const EvolutionIndividual& parent1, const EvolutionIndividual& parent2,
                          EvolutionIndividual& offspring) const;
    
    /**
     * @brief Perform mutation operation
     */
    Status PerformMutation(EvolutionIndividual& individual, const EvolutionTask& task) const;
    
    /**
     * @brief Select individuals for reproduction
     */
    std::vector<EvolutionIndividual> SelectIndividuals(const EvolutionPopulation& population,
                                                      uint32_t num_selected) const;
    
    /**
     * @brief Calculate fitness score
     */
    double CalculateFitnessScore(const EvolutionIndividual& individual,
                               const std::vector<EvolutionObjective>& objectives,
                               const std::map<std::string, double>& objective_weights) const;
    
    /**
     * @brief Check convergence criteria
     */
    bool CheckConvergence(const EvolutionPopulation& population, double threshold) const;
    
    /**
     * @brief Calculate population diversity
     */
    double CalculatePopulationDiversity(const EvolutionPopulation& population) const;
    
    /**
     * @brief Update evolution statistics
     */
    void UpdateStats(const EvolutionResult& result);
    
    /**
     * @brief Perform online learning update
     */
    Status PerformOnlineLearningUpdate(const std::string& model_id,
                                     const std::vector<double>& features,
                                     double target);
    
    /**
     * @brief Perform meta-learning episode
     */
    Status PerformMetaLearningEpisode(const std::string& task_id,
                                    std::map<std::string, double>& meta_params);
    
    /**
     * @brief Simulate individual performance
     */
    Status SimulateIndividualPerformance(const EvolutionIndividual& individual,
                                       const std::string& simulation_id,
                                       std::map<std::string, double>& performance_metrics);
    
    /**
     * @brief Validate evolution constraints
     */
    bool ValidateEvolutionConstraints(const EvolutionIndividual& individual,
                                    const EvolutionTask& task) const;
    
    /**
     * @brief Archive elite individuals
     */
    Status ArchiveEliteIndividuals(const EvolutionPopulation& population);
    
    /**
     * @brief Restore from archive
     */
    Status RestoreFromArchive(const std::string& task_id, EvolutionPopulation& population);
    
    /**
     * @brief Calculate evolution efficiency
     */
    double CalculateEvolutionEfficiency(const EvolutionResult& result) const;
    
    /**
     * @brief Generate evolution recommendations
     */
    std::vector<std::string> GenerateEvolutionRecommendations(const EvolutionResult& result) const;
    
    /**
     * @brief Optimize evolution parameters
     */
    Status OptimizeEvolutionParameters(const std::string& task_id,
                                     std::map<std::string, double>& optimized_params);
    
    /**
     * @brief Adapt evolution strategy
     */
    Status AdaptEvolutionStrategy(const std::string& task_id, EvolutionStrategy& new_strategy);
    
    /**
     * @brief Handle evolution task timeout
     */
    Status HandleEvolutionTaskTimeout(const std::string& task_id);
    
    /**
     * @brief Cleanup completed tasks
     */
    Status CleanupCompletedTasks();
    
    /**
     * @brief Backup evolution state
     */
    Status BackupEvolutionState(const std::string& backup_id);
    
    /**
     * @brief Restore evolution state
     */
    Status RestoreEvolutionState(const std::string& backup_id);
};

} // namespace edge_ai
