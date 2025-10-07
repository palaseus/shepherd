/**
 * @file optimization_manager.h
 * @brief Adaptive Optimization & Self-Tuning Subsystem for Edge AI Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the OptimizationManager class that provides real-time
 * feedback-driven optimization of engine parameters based on profiler data.
 */

#pragma once

#include "core/types.h"
#include "profiling/profiler.h"
#include <memory>
#include <vector>
#include <map>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <functional>

namespace edge_ai {

// Forward declarations
class BatchingManager;
class RuntimeScheduler;
class InferenceEngine;

/**
 * @enum OptimizationAction
 * @brief Types of optimization actions that can be applied
 */
enum class OptimizationAction {
    ADJUST_BATCH_SIZE = 0,
    SWITCH_BACKEND = 1,
    MODIFY_SCHEDULER_POLICY = 2,
    ADJUST_MEMORY_POOL = 3,
    ENABLE_DISABLE_FEATURE = 4,
    THROTTLE_REQUESTS = 5
};

/**
 * @enum OptimizationTrigger
 * @brief Conditions that trigger optimization decisions
 */
enum class OptimizationTrigger {
    LATENCY_THRESHOLD_EXCEEDED = 0,
    THROUGHPUT_DEGRADATION = 1,
    MEMORY_PRESSURE = 2,
    BACKEND_PERFORMANCE_CHANGE = 3,
    QUEUE_OVERFLOW = 4,
    PERIODIC_TUNE = 5
};

/**
 * @struct OptimizationDecision
 * @brief Represents a single optimization decision
 */
struct OptimizationDecision {
    OptimizationAction action;
    OptimizationTrigger trigger;
    std::string parameter_name;
    std::string old_value;
    std::string new_value;
    double expected_improvement;
    uint64_t timestamp_ns;
    uint64_t request_id;
    
    OptimizationDecision() : action(OptimizationAction::ADJUST_BATCH_SIZE),
                           trigger(OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED),
                           expected_improvement(0.0), timestamp_ns(0), request_id(0) {}
};

/**
 * @struct OptimizationMetrics
 * @brief Metrics used for optimization decisions
 */
struct OptimizationMetrics {
    double avg_latency_ms;
    double p99_latency_ms;
    double throughput_ops_per_sec;
    double memory_usage_percent;
    double cpu_utilization_percent;
    double queue_depth;
    double batch_efficiency;
    uint64_t total_requests;
    uint64_t failed_requests;
    
    OptimizationMetrics() : avg_latency_ms(0.0), p99_latency_ms(0.0), 
                          throughput_ops_per_sec(0.0), memory_usage_percent(0.0),
                          cpu_utilization_percent(0.0), queue_depth(0.0),
                          batch_efficiency(0.0), total_requests(0), failed_requests(0) {}
};

/**
 * @struct AdaptiveOptimizationConfig
 * @brief Configuration for the adaptive optimization manager
 */
struct AdaptiveOptimizationConfig {
    bool enable_adaptive_batching;
    bool enable_backend_switching;
    bool enable_scheduler_tuning;
    bool enable_memory_optimization;
    
    double latency_threshold_ms;
    double throughput_degradation_threshold;
    double memory_pressure_threshold;
    
    std::chrono::milliseconds optimization_interval;
    std::chrono::milliseconds convergence_timeout;
    
    int max_optimization_attempts;
    double improvement_threshold;
    
    AdaptiveOptimizationConfig() : enable_adaptive_batching(true), enable_backend_switching(true),
                                 enable_scheduler_tuning(true), enable_memory_optimization(true),
                                 latency_threshold_ms(100.0), throughput_degradation_threshold(0.2),
                                 memory_pressure_threshold(0.8), optimization_interval(std::chrono::milliseconds(1000)),
                                 convergence_timeout(std::chrono::milliseconds(5000)), max_optimization_attempts(10),
                                 improvement_threshold(0.05) {}
};

/**
 * @struct AdaptiveOptimizationStats
 * @brief Statistics about adaptive optimization performance
 */
struct AdaptiveOptimizationStats {
    std::atomic<uint64_t> total_decisions{0};
    std::atomic<uint64_t> successful_optimizations{0};
    std::atomic<uint64_t> failed_optimizations{0};
    std::atomic<uint64_t> latency_improvements{0};
    std::atomic<uint64_t> throughput_improvements{0};
    std::atomic<double> avg_improvement_percent{0.0};
    std::atomic<double> convergence_time_ms{0.0};
    
    struct Snapshot {
        uint64_t total_decisions;
        uint64_t successful_optimizations;
        uint64_t failed_optimizations;
        uint64_t latency_improvements;
        uint64_t throughput_improvements;
        double avg_improvement_percent;
        double convergence_time_ms;
        
        Snapshot() : total_decisions(0), successful_optimizations(0), failed_optimizations(0),
                    latency_improvements(0), throughput_improvements(0), avg_improvement_percent(0.0),
                    convergence_time_ms(0.0) {}
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_decisions = total_decisions.load();
        snapshot.successful_optimizations = successful_optimizations.load();
        snapshot.failed_optimizations = failed_optimizations.load();
        snapshot.latency_improvements = latency_improvements.load();
        snapshot.throughput_improvements = throughput_improvements.load();
        snapshot.avg_improvement_percent = avg_improvement_percent.load();
        snapshot.convergence_time_ms = convergence_time_ms.load();
        return snapshot;
    }
};

/**
 * @class OptimizationPolicy
 * @brief Abstract base class for optimization policies
 */
class OptimizationPolicy {
public:
    virtual ~OptimizationPolicy() = default;
    
    /**
     * @brief Analyze metrics and generate optimization decisions
     * @param metrics Current system metrics
     * @param config Optimization configuration
     * @return Vector of optimization decisions
     */
    virtual std::vector<OptimizationDecision> AnalyzeAndDecide(
        const OptimizationMetrics& metrics,
        const AdaptiveOptimizationConfig& config) = 0;
    
    /**
     * @brief Get policy name
     * @return Policy name string
     */
    virtual std::string GetName() const = 0;
    
    /**
     * @brief Check if policy is applicable for current conditions
     * @param metrics Current system metrics
     * @return True if policy should be applied
     */
    virtual bool IsApplicable(const OptimizationMetrics& metrics) const = 0;
};

/**
 * @class RuleBasedPolicy
 * @brief Rule-based optimization policy implementation
 */
class RuleBasedPolicy : public OptimizationPolicy {
public:
    RuleBasedPolicy();
    virtual ~RuleBasedPolicy() = default;
    
    std::vector<OptimizationDecision> AnalyzeAndDecide(
        const OptimizationMetrics& metrics,
        const AdaptiveOptimizationConfig& config) override;
    
    std::string GetName() const override { return "RuleBasedPolicy"; }
    bool IsApplicable(const OptimizationMetrics& metrics) const override;
    
private:
    OptimizationDecision CreateBatchSizeDecision(const OptimizationMetrics& metrics, 
                                                const AdaptiveOptimizationConfig& config);
    OptimizationDecision CreateBackendSwitchDecision(const OptimizationMetrics& metrics,
                                                    const AdaptiveOptimizationConfig& config);
    OptimizationDecision CreateSchedulerTuningDecision(const OptimizationMetrics& metrics,
                                                      const AdaptiveOptimizationConfig& config);
    OptimizationDecision CreateMemoryOptimizationDecision(const OptimizationMetrics& metrics,
                                                         const AdaptiveOptimizationConfig& config);
};

/**
 * @class OptimizationManager
 * @brief Main optimization manager that coordinates real-time tuning
 */
class OptimizationManager {
public:
    /**
     * @brief Constructor
     * @param config Optimization configuration
     */
    explicit OptimizationManager(const AdaptiveOptimizationConfig& config = AdaptiveOptimizationConfig{});
    
    /**
     * @brief Destructor
     */
    ~OptimizationManager();
    
    /**
     * @brief Initialize the optimization manager
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the optimization manager
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Register components for optimization
     * @param batching_manager Batching manager to optimize
     * @param scheduler Runtime scheduler to optimize
     * @param inference_engine Inference engine to optimize
     * @return Status indicating success or failure
     */
    Status RegisterComponents(std::shared_ptr<BatchingManager> batching_manager,
                            std::shared_ptr<RuntimeScheduler> scheduler,
                            std::shared_ptr<InferenceEngine> inference_engine);
    
    /**
     * @brief Start optimization monitoring
     * @return Status indicating success or failure
     */
    Status StartOptimization();
    
    /**
     * @brief Stop optimization monitoring
     * @return Status indicating success or failure
     */
    Status StopOptimization();
    
    /**
     * @brief Update metrics from profiler data
     * @param metrics Current system metrics
     * @return Status indicating success or failure
     */
    Status UpdateMetrics(const OptimizationMetrics& metrics);
    
    /**
     * @brief Apply optimization decision
     * @param decision Optimization decision to apply
     * @return Status indicating success or failure
     */
    Status ApplyOptimization(const OptimizationDecision& decision);
    
    /**
     * @brief Get optimization statistics
     * @return Optimization statistics snapshot
     */
    AdaptiveOptimizationStats::Snapshot GetStats() const;
    
    /**
     * @brief Get recent optimization decisions
     * @param max_decisions Maximum number of decisions to return
     * @return Vector of recent optimization decisions
     */
    std::vector<OptimizationDecision> GetRecentDecisions(size_t max_decisions = 100) const;
    
    /**
     * @brief Set optimization policy
     * @param policy Optimization policy to use
     * @return Status indicating success or failure
     */
    Status SetOptimizationPolicy(std::unique_ptr<OptimizationPolicy> policy);
    
    /**
     * @brief Enable or disable optimization
     * @param enabled True to enable optimization
     */
    void SetOptimizationEnabled(bool enabled);
    
    /**
     * @brief Check if optimization is enabled
     * @return True if optimization is enabled
     */
    bool IsOptimizationEnabled() const;
    
    /**
     * @brief Export optimization trace to JSON
     * @param session_name Session name for export
     * @param file_path Output file path
     * @return Status indicating success or failure
     */
    Status ExportOptimizationTrace(const std::string& session_name, 
                                  const std::string& file_path) const;

private:
    /**
     * @brief Optimization monitoring thread
     */
    void OptimizationThread();
    
    /**
     * @brief Collect metrics from profiler and components
     * @return Current optimization metrics
     */
    OptimizationMetrics CollectMetrics();
    
    /**
     * @brief Process optimization decisions
     * @param decisions Vector of optimization decisions
     */
    void ProcessDecisions(const std::vector<OptimizationDecision>& decisions);
    
    /**
     * @brief Log optimization decision
     * @param decision Decision to log
     */
    void LogDecision(const OptimizationDecision& decision);
    
    // Configuration
    AdaptiveOptimizationConfig config_;
    
    // Component references
    std::shared_ptr<BatchingManager> batching_manager_;
    std::shared_ptr<RuntimeScheduler> scheduler_;
    std::shared_ptr<InferenceEngine> inference_engine_;
    
    // Optimization policy
    std::unique_ptr<OptimizationPolicy> policy_;
    
    // State management
    std::atomic<bool> initialized_{false};
    std::atomic<bool> optimization_enabled_{true};
    std::atomic<bool> shutdown_requested_{false};
    
    // Threading
    std::unique_ptr<std::thread> optimization_thread_;
    mutable std::mutex metrics_mutex_;
    mutable std::mutex decisions_mutex_;
    
    // Metrics and decisions
    OptimizationMetrics current_metrics_;
    std::vector<OptimizationDecision> recent_decisions_;
    AdaptiveOptimizationStats stats_;
    
    // Profiler integration
    Profiler* profiler_;
};

} // namespace edge_ai
