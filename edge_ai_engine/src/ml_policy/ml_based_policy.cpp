/**
 * @file ml_based_policy.cpp
 * @brief Implementation of ML-Driven Optimization Policy
 * @author AI Co-Developer
 * @date 2024
 */

#include "ml_policy/ml_based_policy.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace edge_ai {

// ============================================================================
// SimpleLinearModel Implementation
// ============================================================================

SimpleLinearModel::SimpleLinearModel(double learning_rate)
    : learning_rate_(learning_rate), bias_(0.0), training_count_(0), total_loss_(0.0) {
    // Initialize weights with small random values
    weights_.resize(MLFeatureVector::GetFeatureCount(), 0.01);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);
    
    for (auto& weight : weights_) {
        weight = dist(gen);
    }
}

MLDecision SimpleLinearModel::Predict(const MLFeatureVector& features, OptimizationTrigger trigger) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    double score = ComputeScore(features);
    return ScoreToDecision(score, trigger);
}

void SimpleLinearModel::Train(const TrainingExample& example) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    double prediction = ComputeScore(example.features);
    double target = example.actual_improvement;
    
    // Calculate loss (mean squared error)
    double loss = (prediction - target) * (prediction - target);
    total_loss_ += loss;
    training_count_++;
    
    // Update weights using gradient descent
    UpdateWeights(example.features, target, prediction);
}

MLModelState SimpleLinearModel::GetState() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    MLModelState state;
    state.weights = weights_;
    state.learning_rate = learning_rate_;
    state.bias = bias_;
    state.training_examples_count = training_count_;
    state.total_loss = total_loss_;
    state.last_training_time = std::chrono::steady_clock::now();
    
    return state;
}

void SimpleLinearModel::SetState(const MLModelState& state) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    weights_ = state.weights;
    learning_rate_ = state.learning_rate;
    bias_ = state.bias;
    training_count_ = state.training_examples_count;
    total_loss_ = state.total_loss;
}

std::vector<double> SimpleLinearModel::GetFeatureImportance() const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    // Feature importance is the absolute value of weights
    std::vector<double> importance = weights_;
    for (auto& imp : importance) {
        imp = std::abs(imp);
    }
    
    return importance;
}

bool SimpleLinearModel::SaveToFile(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    // Save model state as JSON
    file << "{\n";
    file << "  \"weights\": [";
    for (size_t i = 0; i < weights_.size(); ++i) {
        file << std::fixed << std::setprecision(6) << weights_[i];
        if (i < weights_.size() - 1) file << ", ";
    }
    file << "],\n";
    file << "  \"bias\": " << std::fixed << std::setprecision(6) << bias_ << ",\n";
    file << "  \"learning_rate\": " << std::fixed << std::setprecision(6) << learning_rate_ << ",\n";
    file << "  \"training_count\": " << training_count_ << ",\n";
    file << "  \"total_loss\": " << std::fixed << std::setprecision(6) << total_loss_ << "\n";
    file << "}\n";
    
    file.close();
    return true;
}

bool SimpleLinearModel::LoadFromFile(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    // Simple JSON parsing (in production, use a proper JSON library)
    std::string line;
    std::string content;
    while (std::getline(file, line)) {
        content += line;
    }
    
    // Extract weights (simplified parsing)
    size_t weights_start = content.find("\"weights\": [");
    if (weights_start != std::string::npos) {
        weights_start = content.find("[", weights_start);
        size_t weights_end = content.find("]", weights_start);
        std::string weights_str = content.substr(weights_start + 1, weights_end - weights_start - 1);
        
        std::istringstream iss(weights_str);
        std::string weight_str;
        weights_.clear();
        
        while (std::getline(iss, weight_str, ',')) {
            // Remove whitespace
            weight_str.erase(std::remove_if(weight_str.begin(), weight_str.end(), ::isspace), weight_str.end());
            if (!weight_str.empty()) {
                weights_.push_back(std::stod(weight_str));
            }
        }
    }
    
    file.close();
    return !weights_.empty();
}

double SimpleLinearModel::ComputeScore(const MLFeatureVector& features) const {
    std::vector<double> feature_vector = features.ToVector();
    
    if (feature_vector.size() != weights_.size()) {
        return 0.0;
    }
    
    double score = bias_;
    for (size_t i = 0; i < feature_vector.size(); ++i) {
        score += weights_[i] * feature_vector[i];
    }
    
    // Apply sigmoid activation
    return 1.0 / (1.0 + std::exp(-score));
}

MLDecision SimpleLinearModel::ScoreToDecision(double score, OptimizationTrigger trigger) const {
    MLDecision decision;
    decision.confidence_score = score;
    
    // Map score to optimization action based on trigger
    if (trigger == OptimizationTrigger::QUEUE_OVERFLOW) {
        // For queue overflow, prioritize throughput optimizations
        if (score < 0.5) {
            decision.action = OptimizationAction::ADJUST_BATCH_SIZE;
            decision.parameter_name = "max_batch_size";
            decision.new_value = "8";
            decision.expected_improvement = 0.2;
        } else {
            decision.action = OptimizationAction::MODIFY_SCHEDULER_POLICY;
            decision.parameter_name = "scheduler_policy";
            decision.new_value = "PRIORITY";
            decision.expected_improvement = 0.15;
        }
    } else if (trigger == OptimizationTrigger::MEMORY_PRESSURE) {
        // For memory pressure, prioritize memory optimizations
        decision.action = OptimizationAction::ADJUST_BATCH_SIZE;
        decision.parameter_name = "max_batch_size";
        decision.new_value = "2";
        decision.expected_improvement = 0.1;
    } else if (trigger == OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED) {
        // For latency issues, try backend switching
        decision.action = OptimizationAction::SWITCH_BACKEND;
        decision.parameter_name = "preferred_backend";
        decision.new_value = "GPU";
        decision.expected_improvement = 0.2;
    } else {
        // Default behavior based on score
        if (score < 0.3) {
            decision.action = OptimizationAction::ADJUST_BATCH_SIZE;
            decision.parameter_name = "max_batch_size";
            decision.new_value = "4";
            decision.expected_improvement = 0.1;
        } else if (score < 0.6) {
            decision.action = OptimizationAction::SWITCH_BACKEND;
            decision.parameter_name = "preferred_backend";
            decision.new_value = "GPU";
            decision.expected_improvement = 0.2;
        } else {
            decision.action = OptimizationAction::MODIFY_SCHEDULER_POLICY;
            decision.parameter_name = "scheduler_policy";
            decision.new_value = "PRIORITY";
            decision.expected_improvement = 0.15;
        }
    }
    
    return decision;
}

void SimpleLinearModel::UpdateWeights(const MLFeatureVector& features, double target, double prediction) {
    std::vector<double> feature_vector = features.ToVector();
    
    if (feature_vector.size() != weights_.size()) {
        return;
    }
    
    double error = target - prediction;
    
    // Update bias
    bias_ += learning_rate_ * error;
    
    // Update weights
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] += learning_rate_ * error * feature_vector[i];
    }
}

// ============================================================================
// ReplayBuffer Implementation
// ============================================================================

ReplayBuffer::ReplayBuffer(size_t max_size) : max_size_(max_size), rng_(std::random_device{}()) {}

void ReplayBuffer::AddExample(const TrainingExample& example) {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    buffer_.push_back(example);
    
    if (buffer_.size() > max_size_) {
        buffer_.pop_front();
    }
}

std::vector<TrainingExample> ReplayBuffer::SampleBatch(size_t batch_size) const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    
    std::vector<TrainingExample> batch;
    if (buffer_.empty()) {
        return batch;
    }
    
    size_t actual_batch_size = std::min(batch_size, buffer_.size());
    batch.reserve(actual_batch_size);
    
    std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);
    
    for (size_t i = 0; i < actual_batch_size; ++i) {
        size_t idx = dist(rng_);
        batch.push_back(buffer_[idx]);
    }
    
    return batch;
}

size_t ReplayBuffer::Size() const {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return buffer_.size();
}

void ReplayBuffer::Clear() {
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    buffer_.clear();
}

// ============================================================================
// TelemetryAggregator Implementation
// ============================================================================

TelemetryAggregator::TelemetryAggregator() : start_time_(std::chrono::steady_clock::now()) {}

void TelemetryAggregator::AddMetrics(const OptimizationMetrics& metrics) {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    
    current_metrics_ = metrics;
    metrics_history_.push_back(metrics);
    
    // Keep only recent history
    if (metrics_history_.size() > 100) {
        metrics_history_.pop_front();
    }
}

void TelemetryAggregator::AddDecision(const OptimizationDecision& decision) {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    
    decisions_history_.push_back(decision);
    
    // Keep only recent history
    if (decisions_history_.size() > 100) {
        decisions_history_.pop_front();
    }
}

TrainingExample TelemetryAggregator::GenerateTrainingExample() const {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    
    TrainingExample example;
    example.features = GetCurrentFeatures();
    example.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // Calculate actual improvement (simplified)
    if (metrics_history_.size() >= 2) {
        const auto& current = metrics_history_.back();
        const auto& previous = metrics_history_[metrics_history_.size() - 2];
        
        double latency_improvement = (previous.avg_latency_ms - current.avg_latency_ms) / previous.avg_latency_ms;
        double throughput_improvement = (current.throughput_ops_per_sec - previous.throughput_ops_per_sec) / previous.throughput_ops_per_sec;
        
        example.actual_improvement = (latency_improvement + throughput_improvement) / 2.0;
    }
    
    return example;
}

MLFeatureVector TelemetryAggregator::GetCurrentFeatures() const {
    MLFeatureVector features;
    
    // System metrics
    features.avg_latency_ms = current_metrics_.avg_latency_ms;
    features.p99_latency_ms = current_metrics_.p99_latency_ms;
    features.throughput_ops_per_sec = current_metrics_.throughput_ops_per_sec;
    features.memory_usage_percent = current_metrics_.memory_usage_percent;
    features.cpu_utilization_percent = current_metrics_.cpu_utilization_percent;
    features.queue_depth = current_metrics_.queue_depth;
    features.batch_efficiency = current_metrics_.batch_efficiency;
    
    // Request characteristics (simplified)
    features.request_size_mb = 1.0; // Placeholder
    features.input_tensor_count = 3; // Placeholder
    features.output_tensor_count = 1; // Placeholder
    features.model_complexity_score = 0.5; // Placeholder
    
    // Historical performance
    features.recent_cpu_performance = CalculateRecentPerformance("CPU", "latency");
    features.recent_gpu_performance = CalculateRecentPerformance("GPU", "latency");
    features.recent_batch_performance = current_metrics_.batch_efficiency;
    
    // Time-based features
    features.time_of_day_normalized = CalculateTimeOfDay();
    features.load_trend = CalculateLoadTrend();
    
    return features;
}

void TelemetryAggregator::UpdatePerformance(const std::string& backend, double latency_ms, double throughput) {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    
    backend_latency_history_[backend].push_back(latency_ms);
    backend_throughput_history_[backend].push_back(throughput);
    
    // Keep only recent history
    if (backend_latency_history_[backend].size() > 50) {
        backend_latency_history_[backend].pop_front();
    }
    if (backend_throughput_history_[backend].size() > 50) {
        backend_throughput_history_[backend].pop_front();
    }
}

double TelemetryAggregator::CalculateLoadTrend() const {
    if (metrics_history_.size() < 5) {
        return 0.0;
    }
    
    // Calculate trend using linear regression on recent throughput
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    size_t n = std::min(metrics_history_.size(), size_t(10));
    
    for (size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        double y = metrics_history_[metrics_history_.size() - n + i].throughput_ops_per_sec;
        
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return std::tanh(slope / 100.0); // Normalize to [-1, 1]
}

double TelemetryAggregator::CalculateTimeOfDay() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    double seconds_in_day = tm.tm_hour * 3600.0 + tm.tm_min * 60.0 + tm.tm_sec;
    return seconds_in_day / (24.0 * 3600.0); // Normalize to [0, 1]
}

double TelemetryAggregator::CalculateRecentPerformance(const std::string& backend, const std::string& metric) const {
    if (metric == "latency") {
        auto it = backend_latency_history_.find(backend);
        if (it != backend_latency_history_.end() && !it->second.empty()) {
            return std::accumulate(it->second.begin(), it->second.end(), 0.0) / it->second.size();
        }
    } else if (metric == "throughput") {
        auto it = backend_throughput_history_.find(backend);
        if (it != backend_throughput_history_.end() && !it->second.empty()) {
            return std::accumulate(it->second.begin(), it->second.end(), 0.0) / it->second.size();
        }
    }
    
    return 0.0;
}

// ============================================================================
// MLBasedPolicy Implementation
// ============================================================================

MLBasedPolicy::MLBasedPolicy(const std::string& model_path)
    : online_learning_enabled_(true), decision_count_(0), total_confidence_(0.0) {
    
    model_ = std::make_unique<SimpleLinearModel>();
    telemetry_ = std::make_unique<TelemetryAggregator>();
    replay_buffer_ = std::make_unique<ReplayBuffer>(1000);
    
    // Load model if path provided
    if (!model_path.empty()) {
        LoadModel(model_path);
    }
}

std::vector<OptimizationDecision> MLBasedPolicy::AnalyzeAndDecide(
    const OptimizationMetrics& metrics,
    const AdaptiveOptimizationConfig& config) {
    
    std::lock_guard<std::mutex> lock(policy_mutex_);
    
    // Update telemetry
    telemetry_->AddMetrics(metrics);
    
    std::vector<OptimizationDecision> decisions;
    
    try {
        // Generate ML-based decision
        MLFeatureVector features = telemetry_->GetCurrentFeatures();
        
        // Determine trigger based on metrics
        OptimizationTrigger trigger = OptimizationTrigger::PERIODIC_TUNE;
        if (metrics.avg_latency_ms > config.latency_threshold_ms) {
            trigger = OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED;
        } else if (metrics.memory_usage_percent > config.memory_pressure_threshold) {
            trigger = OptimizationTrigger::MEMORY_PRESSURE;
        } else if (metrics.queue_depth > 10) {
            trigger = OptimizationTrigger::QUEUE_OVERFLOW;
        }
        
        MLDecision ml_decision = GenerateMLDecision(features, trigger);
        
        // Convert to OptimizationDecision
        OptimizationDecision decision;
        decision.action = ml_decision.action;
        decision.trigger = trigger;
        decision.parameter_name = ml_decision.parameter_name;
        decision.old_value = "unknown"; // Would need to track current values
        decision.new_value = ml_decision.new_value;
        decision.expected_improvement = ml_decision.expected_improvement;
        decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        decision.request_id = decision_count_.fetch_add(1);
        
        decisions.push_back(decision);
        
        // Update confidence tracking
        UpdateConfidence(ml_decision.confidence_score);
        
        // Add to telemetry for future training
        telemetry_->AddDecision(decision);
        
    } catch (const std::exception& e) {
        // Fallback to rule-based policy on error
        OptimizationDecision fallback = FallbackToRuleBased(metrics, config);
        decisions.push_back(fallback);
    }
    
    return decisions;
}

bool MLBasedPolicy::IsApplicable(const OptimizationMetrics& metrics) const {
    [[maybe_unused]] auto metrics_ref = metrics;
    // ML policy is always applicable
    return true;
}

void MLBasedPolicy::Train(const TrainingExample& example) {
    if (!online_learning_enabled_.load()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(policy_mutex_);
    
    // Add to replay buffer
    replay_buffer_->AddExample(example);
    
    // Train model
    model_->Train(example);
    
    // Perform batch training periodically
    if (replay_buffer_->Size() >= 10) {
        PerformBatchTraining();
    }
}

double MLBasedPolicy::GetConfidence() const {
    uint64_t count = decision_count_.load();
    if (count == 0) return 0.0;
    
    return total_confidence_.load() / count;
}

std::vector<double> MLBasedPolicy::GetFeatureImportance() const {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    return model_->GetFeatureImportance();
}

bool MLBasedPolicy::SaveModel(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    return model_->SaveToFile(filepath);
}

bool MLBasedPolicy::LoadModel(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    return model_->LoadFromFile(filepath);
}

MLBasedPolicy::TrainingStats MLBasedPolicy::GetTrainingStats() const {
    std::lock_guard<std::mutex> lock(policy_mutex_);
    
    TrainingStats stats;
    MLModelState state = model_->GetState();
    
    stats.total_examples = state.training_examples_count;
    stats.average_loss = state.training_examples_count > 0 ? 
                       state.total_loss / state.training_examples_count : 0.0;
    stats.current_confidence = GetConfidence();
    stats.last_training_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - state.last_training_time);
    
    return stats;
}

void MLBasedPolicy::SetOnlineLearningEnabled(bool enabled) {
    online_learning_enabled_.store(enabled);
}

bool MLBasedPolicy::IsOnlineLearningEnabled() const {
    return online_learning_enabled_.load();
}

MLDecision MLBasedPolicy::GenerateMLDecision(const MLFeatureVector& features, OptimizationTrigger trigger) {
    return model_->Predict(features, trigger);
}

OptimizationDecision MLBasedPolicy::FallbackToRuleBased(const OptimizationMetrics& metrics,
                                                       const AdaptiveOptimizationConfig& config) {
    [[maybe_unused]] auto metrics_ref = metrics;
    [[maybe_unused]] auto config_ref = config;
    // Simple rule-based fallback
    OptimizationDecision decision;
    decision.action = OptimizationAction::ADJUST_BATCH_SIZE;
    decision.trigger = OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED;
    decision.parameter_name = "max_batch_size";
    decision.old_value = "8";
    decision.new_value = "4";
    decision.expected_improvement = 0.1;
    decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    decision.request_id = decision_count_.fetch_add(1);
    
    return decision;
}

void MLBasedPolicy::UpdateConfidence(double confidence) {
    total_confidence_.fetch_add(confidence);
}

void MLBasedPolicy::PerformBatchTraining() {
    auto batch = replay_buffer_->SampleBatch(5);
    
    for (const auto& example : batch) {
        model_->Train(example);
    }
}

} // namespace edge_ai
