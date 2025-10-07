/**
 * @file ml_based_policy.h
 * @brief ML-Driven Optimization Policy for Edge AI Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MLBasedPolicy class that learns from profiler traces
 * and runtime metrics to make intelligent optimization decisions.
 */

#pragma once

#include "optimization/optimization_manager.h"
#include <vector>
#include <deque>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>
#include <chrono>
#include <fstream>

namespace edge_ai {

/**
 * @struct MLFeatureVector
 * @brief Feature vector for ML model input
 */
struct MLFeatureVector {
    // System metrics
    double avg_latency_ms;
    double p99_latency_ms;
    double throughput_ops_per_sec;
    double memory_usage_percent;
    double cpu_utilization_percent;
    double queue_depth;
    double batch_efficiency;
    
    // Request characteristics
    double request_size_mb;
    int input_tensor_count;
    int output_tensor_count;
    double model_complexity_score;
    
    // Historical performance
    double recent_cpu_performance;
    double recent_gpu_performance;
    double recent_batch_performance;
    
    // Time-based features
    double time_of_day_normalized;  // 0.0 to 1.0
    double load_trend;              // -1.0 to 1.0 (decreasing to increasing)
    
    MLFeatureVector() : avg_latency_ms(0.0), p99_latency_ms(0.0), throughput_ops_per_sec(0.0),
                       memory_usage_percent(0.0), cpu_utilization_percent(0.0), queue_depth(0.0),
                       batch_efficiency(0.0), request_size_mb(0.0), input_tensor_count(0),
                       output_tensor_count(0), model_complexity_score(0.0), recent_cpu_performance(0.0),
                       recent_gpu_performance(0.0), recent_batch_performance(0.0),
                       time_of_day_normalized(0.0), load_trend(0.0) {}
    
    /**
     * @brief Convert to vector for ML model
     */
    std::vector<double> ToVector() const {
        return {
            avg_latency_ms, p99_latency_ms, throughput_ops_per_sec, memory_usage_percent,
            cpu_utilization_percent, queue_depth, batch_efficiency, request_size_mb,
            static_cast<double>(input_tensor_count), static_cast<double>(output_tensor_count), model_complexity_score,
            recent_cpu_performance, recent_gpu_performance, recent_batch_performance,
            time_of_day_normalized, load_trend
        };
    }
    
    /**
     * @brief Get feature count
     */
    static constexpr size_t GetFeatureCount() { return 16; }
};

/**
 * @struct MLDecision
 * @brief ML model decision output
 */
struct MLDecision {
    OptimizationAction action;
    std::string parameter_name;
    std::string new_value;
    double confidence_score;
    double expected_improvement;
    
    MLDecision() : action(OptimizationAction::ADJUST_BATCH_SIZE), confidence_score(0.0), expected_improvement(0.0) {}
};

/**
 * @struct TrainingExample
 * @brief Training example for online learning
 */
struct TrainingExample {
    MLFeatureVector features;
    OptimizationDecision decision;
    double actual_improvement;
    uint64_t timestamp_ns;
    
    TrainingExample() : actual_improvement(0.0), timestamp_ns(0) {}
};

/**
 * @struct MLModelState
 * @brief State of the ML model for persistence
 */
struct MLModelState {
    std::vector<double> weights;
    double learning_rate;
    double bias;
    uint64_t training_examples_count;
    double total_loss;
    std::chrono::steady_clock::time_point last_training_time;
    
    MLModelState() : learning_rate(0.01), bias(0.0), training_examples_count(0), total_loss(0.0) {}
};

/**
 * @class SimpleLinearModel
 * @brief Simple linear regression model for optimization decisions
 */
class SimpleLinearModel {
public:
    SimpleLinearModel(double learning_rate = 0.01);
    
    /**
     * @brief Predict optimization decision
     */
    MLDecision Predict(const MLFeatureVector& features, OptimizationTrigger trigger);
    
    /**
     * @brief Train on a single example
     */
    void Train(const TrainingExample& example);
    
    /**
     * @brief Get model state
     */
    MLModelState GetState() const;
    
    /**
     * @brief Set model state
     */
    void SetState(const MLModelState& state);
    
    /**
     * @brief Get feature importance
     */
    std::vector<double> GetFeatureImportance() const;
    
    /**
     * @brief Save model to file
     */
    bool SaveToFile(const std::string& filepath) const;
    
    /**
     * @brief Load model from file
     */
    bool LoadFromFile(const std::string& filepath);

private:
    std::vector<double> weights_;
    double learning_rate_;
    double bias_;
    uint64_t training_count_;
    double total_loss_;
    mutable std::mutex model_mutex_;
    
    /**
     * @brief Compute prediction score
     */
    double ComputeScore(const MLFeatureVector& features) const;
    
    /**
     * @brief Convert score to decision
     */
    MLDecision ScoreToDecision(double score, OptimizationTrigger trigger) const;
    
    /**
     * @brief Update weights using gradient descent
     */
    void UpdateWeights(const MLFeatureVector& features, double target, double prediction);
};

/**
 * @class ReplayBuffer
 * @brief Replay buffer for off-policy learning
 */
class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t max_size = 1000);
    
    /**
     * @brief Add training example
     */
    void AddExample(const TrainingExample& example);
    
    /**
     * @brief Sample random batch
     */
    std::vector<TrainingExample> SampleBatch(size_t batch_size) const;
    
    /**
     * @brief Get buffer size
     */
    size_t Size() const;
    
    /**
     * @brief Clear buffer
     */
    void Clear();

private:
    std::deque<TrainingExample> buffer_;
    size_t max_size_;
    mutable std::mutex buffer_mutex_;
    mutable std::mt19937 rng_;
};

/**
 * @class TelemetryAggregator
 * @brief Aggregates telemetry data for ML training
 */
class TelemetryAggregator {
public:
    TelemetryAggregator();
    
    /**
     * @brief Add system metrics
     */
    void AddMetrics(const OptimizationMetrics& metrics);
    
    /**
     * @brief Add optimization decision
     */
    void AddDecision(const OptimizationDecision& decision);
    
    /**
     * @brief Generate training example
     */
    TrainingExample GenerateTrainingExample() const;
    
    /**
     * @brief Get current feature vector
     */
    MLFeatureVector GetCurrentFeatures() const;
    
    /**
     * @brief Update performance tracking
     */
    void UpdatePerformance(const std::string& backend, double latency_ms, double throughput);

private:
    mutable std::mutex telemetry_mutex_;
    
    // Current metrics
    OptimizationMetrics current_metrics_;
    
    // Historical data
    std::deque<OptimizationMetrics> metrics_history_;
    std::deque<OptimizationDecision> decisions_history_;
    
    // Performance tracking
    std::map<std::string, std::deque<double>> backend_latency_history_;
    std::map<std::string, std::deque<double>> backend_throughput_history_;
    
    // Time tracking
    std::chrono::steady_clock::time_point start_time_;
    
    /**
     * @brief Calculate load trend
     */
    double CalculateLoadTrend() const;
    
    /**
     * @brief Calculate time of day normalized
     */
    double CalculateTimeOfDay() const;
    
    /**
     * @brief Calculate recent performance metrics
     */
    double CalculateRecentPerformance(const std::string& backend, const std::string& metric) const;
};

/**
 * @class MLBasedPolicy
 * @brief ML-driven optimization policy
 */
class MLBasedPolicy : public OptimizationPolicy {
public:
    explicit MLBasedPolicy(const std::string& model_path = "");
    virtual ~MLBasedPolicy() = default;
    
    /**
     * @brief Analyze metrics and generate optimization decisions
     */
    std::vector<OptimizationDecision> AnalyzeAndDecide(
        const OptimizationMetrics& metrics,
        const AdaptiveOptimizationConfig& config) override;
    
    /**
     * @brief Get policy name
     */
    std::string GetName() const override { return "MLBasedPolicy"; }
    
    /**
     * @brief Check if policy is applicable
     */
    bool IsApplicable(const OptimizationMetrics& metrics) const override;
    
    /**
     * @brief Train the model on new data
     */
    void Train(const TrainingExample& example);
    
    /**
     * @brief Get model confidence
     */
    double GetConfidence() const;
    
    /**
     * @brief Get feature importance
     */
    std::vector<double> GetFeatureImportance() const;
    
    /**
     * @brief Save model state
     */
    bool SaveModel(const std::string& filepath) const;
    
    /**
     * @brief Load model state
     */
    bool LoadModel(const std::string& filepath);
    
    /**
     * @brief Get training statistics
     */
    struct TrainingStats {
        uint64_t total_examples;
        double average_loss;
        double current_confidence;
        std::chrono::milliseconds last_training_time;
        
        TrainingStats() : total_examples(0), average_loss(0.0), current_confidence(0.0) {}
    };
    
    TrainingStats GetTrainingStats() const;
    
    /**
     * @brief Enable/disable online learning
     */
    void SetOnlineLearningEnabled(bool enabled);
    
    /**
     * @brief Check if online learning is enabled
     */
    bool IsOnlineLearningEnabled() const;

private:
    std::unique_ptr<SimpleLinearModel> model_;
    std::unique_ptr<TelemetryAggregator> telemetry_;
    std::unique_ptr<ReplayBuffer> replay_buffer_;
    
    std::atomic<bool> online_learning_enabled_;
    std::atomic<uint64_t> decision_count_;
    std::atomic<double> total_confidence_;
    
    mutable std::mutex policy_mutex_;
    
    /**
     * @brief Generate ML decision
     */
    MLDecision GenerateMLDecision(const MLFeatureVector& features, OptimizationTrigger trigger);
    
    /**
     * @brief Fallback to rule-based decision
     */
    OptimizationDecision FallbackToRuleBased(const OptimizationMetrics& metrics, 
                                            const AdaptiveOptimizationConfig& config);
    
    /**
     * @brief Update confidence tracking
     */
    void UpdateConfidence(double confidence);
    
    /**
     * @brief Perform batch training from replay buffer
     */
    void PerformBatchTraining();
};

} // namespace edge_ai
