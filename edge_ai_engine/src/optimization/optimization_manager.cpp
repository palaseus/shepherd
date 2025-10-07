/**
 * @file optimization_manager.cpp
 * @brief Implementation of the OptimizationManager class
 * @author AI Co-Developer
 * @date 2024
 */

#include "optimization/optimization_manager.h"
#include "batching/batching_manager.h"
#include "core/runtime_scheduler.h"
#include "core/inference_engine.h"
#include "profiling/profiler.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

namespace edge_ai {

// ============================================================================
// RuleBasedPolicy Implementation
// ============================================================================

RuleBasedPolicy::RuleBasedPolicy() {
    // Initialize rule-based policy
}

std::vector<OptimizationDecision> RuleBasedPolicy::AnalyzeAndDecide(
    const OptimizationMetrics& metrics,
    const AdaptiveOptimizationConfig& config) {
    
    std::vector<OptimizationDecision> decisions;
    
    // Rule 1: If latency exceeds threshold, try to reduce batch size or switch backend
    if (metrics.avg_latency_ms > config.latency_threshold_ms) {
        if (config.enable_adaptive_batching) {
            decisions.push_back(CreateBatchSizeDecision(metrics, config));
        }
        if (config.enable_backend_switching) {
            decisions.push_back(CreateBackendSwitchDecision(metrics, config));
        }
    }
    
    // Rule 2: If throughput is degrading, optimize scheduler
    if (metrics.throughput_ops_per_sec < (metrics.throughput_ops_per_sec * (1.0 - config.throughput_degradation_threshold))) {
        if (config.enable_scheduler_tuning) {
            decisions.push_back(CreateSchedulerTuningDecision(metrics, config));
        }
    }
    
    // Rule 3: If memory pressure is high, optimize memory usage
    if (metrics.memory_usage_percent > config.memory_pressure_threshold) {
        if (config.enable_memory_optimization) {
            decisions.push_back(CreateMemoryOptimizationDecision(metrics, config));
        }
    }
    
    // Rule 4: If queue depth is too high, throttle requests
    if (metrics.queue_depth > 100) {
        OptimizationDecision throttle_decision;
        throttle_decision.action = OptimizationAction::THROTTLE_REQUESTS;
        throttle_decision.trigger = OptimizationTrigger::QUEUE_OVERFLOW;
        throttle_decision.parameter_name = "request_throttle_rate";
        throttle_decision.old_value = "1.0";
        throttle_decision.new_value = "0.8";
        throttle_decision.expected_improvement = 0.15;
        throttle_decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        decisions.push_back(throttle_decision);
    }
    
    return decisions;
}

bool RuleBasedPolicy::IsApplicable([[maybe_unused]] const OptimizationMetrics& metrics) const {
    // Rule-based policy is always applicable
    return true;
}

OptimizationDecision RuleBasedPolicy::CreateBatchSizeDecision(
    [[maybe_unused]] const OptimizationMetrics& metrics,
    [[maybe_unused]] const AdaptiveOptimizationConfig& config) {
    
    OptimizationDecision decision;
    decision.action = OptimizationAction::ADJUST_BATCH_SIZE;
    decision.trigger = OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED;
    decision.parameter_name = "max_batch_size";
    
    // Reduce batch size if latency is high
    int current_batch_size = 8; // Default assumption
    int new_batch_size = std::max(1, current_batch_size / 2);
    
    decision.old_value = std::to_string(current_batch_size);
    decision.new_value = std::to_string(new_batch_size);
    decision.expected_improvement = 0.2; // 20% latency improvement expected
    decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    return decision;
}

OptimizationDecision RuleBasedPolicy::CreateBackendSwitchDecision(
    [[maybe_unused]] const OptimizationMetrics& metrics,
    [[maybe_unused]] const AdaptiveOptimizationConfig& config) {
    
    OptimizationDecision decision;
    decision.action = OptimizationAction::SWITCH_BACKEND;
    decision.trigger = OptimizationTrigger::BACKEND_PERFORMANCE_CHANGE;
    decision.parameter_name = "preferred_backend";
    decision.old_value = "CPU";
    decision.new_value = "GPU";
    decision.expected_improvement = 0.3; // 30% performance improvement expected
    decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    return decision;
}

OptimizationDecision RuleBasedPolicy::CreateSchedulerTuningDecision(
    [[maybe_unused]] const OptimizationMetrics& metrics,
    [[maybe_unused]] const AdaptiveOptimizationConfig& config) {
    
    OptimizationDecision decision;
    decision.action = OptimizationAction::MODIFY_SCHEDULER_POLICY;
    decision.trigger = OptimizationTrigger::THROUGHPUT_DEGRADATION;
    decision.parameter_name = "scheduler_policy";
    decision.old_value = "FIFO";
    decision.new_value = "PRIORITY";
    decision.expected_improvement = 0.15; // 15% throughput improvement expected
    decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    return decision;
}

OptimizationDecision RuleBasedPolicy::CreateMemoryOptimizationDecision(
    [[maybe_unused]] const OptimizationMetrics& metrics,
    [[maybe_unused]] const AdaptiveOptimizationConfig& config) {
    
    OptimizationDecision decision;
    decision.action = OptimizationAction::ADJUST_MEMORY_POOL;
    decision.trigger = OptimizationTrigger::MEMORY_PRESSURE;
    decision.parameter_name = "memory_pool_size";
    decision.old_value = "1024MB";
    decision.new_value = "2048MB";
    decision.expected_improvement = 0.1; // 10% memory efficiency improvement expected
    decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    return decision;
}

// ============================================================================
// OptimizationManager Implementation
// ============================================================================

OptimizationManager::OptimizationManager(const AdaptiveOptimizationConfig& config)
    : config_(config), profiler_(&Profiler::GetInstance()) {
}

OptimizationManager::~OptimizationManager() {
    Shutdown();
}

Status OptimizationManager::Initialize() {
    if (initialized_.load()) {
        return Status::ALREADY_INITIALIZED;
    }
    
    // Initialize default policy if none set
    if (!policy_) {
        policy_ = std::make_unique<RuleBasedPolicy>();
    }
    
    // Initialize profiler if not already done
    if (profiler_) {
        profiler_->Initialize();
    }
    
    initialized_.store(true);
    return Status::SUCCESS;
}

Status OptimizationManager::Shutdown() {
    if (!initialized_.load()) {
        return Status::NOT_INITIALIZED;
    }
    
    // Stop optimization thread
    StopOptimization();
    
    initialized_.store(false);
    return Status::SUCCESS;
}

Status OptimizationManager::RegisterComponents(
    std::shared_ptr<BatchingManager> batching_manager,
    std::shared_ptr<RuntimeScheduler> scheduler,
    std::shared_ptr<InferenceEngine> inference_engine) {
    
    if (!initialized_.load()) {
        return Status::NOT_INITIALIZED;
    }
    
    batching_manager_ = batching_manager;
    scheduler_ = scheduler;
    inference_engine_ = inference_engine;
    
    return Status::SUCCESS;
}

Status OptimizationManager::StartOptimization() {
    if (!initialized_.load()) {
        return Status::NOT_INITIALIZED;
    }
    
    if (optimization_thread_) {
        return Status::ALREADY_INITIALIZED;
    }
    
    shutdown_requested_.store(false);
    optimization_thread_ = std::make_unique<std::thread>(&OptimizationManager::OptimizationThread, this);
    
    return Status::SUCCESS;
}

Status OptimizationManager::StopOptimization() {
    if (!optimization_thread_) {
        return Status::SUCCESS;
    }
    
    shutdown_requested_.store(true);
    if (optimization_thread_->joinable()) {
        optimization_thread_->join();
    }
    optimization_thread_.reset();
    
    return Status::SUCCESS;
}

Status OptimizationManager::UpdateMetrics(const OptimizationMetrics& metrics) {
    if (!initialized_.load()) {
        return Status::NOT_INITIALIZED;
    }
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_ = metrics;
    
    return Status::SUCCESS;
}

Status OptimizationManager::ApplyOptimization(const OptimizationDecision& decision) {
    if (!initialized_.load()) {
        return Status::NOT_INITIALIZED;
    }
    
    // Log the decision
    LogDecision(decision);
    
    // Apply the optimization based on action type
    switch (decision.action) {
        case OptimizationAction::ADJUST_BATCH_SIZE:
            // Apply batch size adjustment
            // This would interface with BatchingManager
            break;
            
        case OptimizationAction::SWITCH_BACKEND:
            // Apply backend switching
            // This would interface with InferenceEngine
            break;
            
        case OptimizationAction::MODIFY_SCHEDULER_POLICY:
            // Apply scheduler policy changes
            // This would interface with RuntimeScheduler
            break;
            
        case OptimizationAction::ADJUST_MEMORY_POOL:
            // Apply memory pool adjustments
            // This would interface with MemoryManager
            break;
            
        case OptimizationAction::ENABLE_DISABLE_FEATURE:
            // Enable/disable features
            break;
            
        case OptimizationAction::THROTTLE_REQUESTS:
            // Apply request throttling
            break;
            
        default:
            return Status::INVALID_ARGUMENT;
    }
    
    // Update statistics
    stats_.total_decisions.fetch_add(1);
    stats_.successful_optimizations.fetch_add(1);
    
    return Status::SUCCESS;
}

AdaptiveOptimizationStats::Snapshot OptimizationManager::GetStats() const {
    return stats_.GetSnapshot();
}

std::vector<OptimizationDecision> OptimizationManager::GetRecentDecisions(size_t max_decisions) const {
    std::lock_guard<std::mutex> lock(decisions_mutex_);
    
    size_t start_idx = (recent_decisions_.size() > max_decisions) ? 
                      (recent_decisions_.size() - max_decisions) : 0;
    
    return std::vector<OptimizationDecision>(recent_decisions_.begin() + start_idx, 
                                           recent_decisions_.end());
}

Status OptimizationManager::SetOptimizationPolicy(std::unique_ptr<OptimizationPolicy> policy) {
    if (!policy) {
        return Status::INVALID_ARGUMENT;
    }
    
    policy_ = std::move(policy);
    return Status::SUCCESS;
}

void OptimizationManager::SetOptimizationEnabled(bool enabled) {
    optimization_enabled_.store(enabled);
}

bool OptimizationManager::IsOptimizationEnabled() const {
    return optimization_enabled_.load();
}

Status OptimizationManager::ExportOptimizationTrace(
    const std::string& session_name,
    const std::string& file_path) const {
    
    if (!initialized_.load()) {
        return Status::NOT_INITIALIZED;
    }
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Export optimization decisions as JSON
    file << "{\n";
    file << "  \"session_name\": \"" << session_name << "\",\n";
    file << "  \"timestamp\": " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << ",\n";
    file << "  \"optimization_stats\": {\n";
    
    auto stats = GetStats();
    file << "    \"total_decisions\": " << stats.total_decisions << ",\n";
    file << "    \"successful_optimizations\": " << stats.successful_optimizations << ",\n";
    file << "    \"failed_optimizations\": " << stats.failed_optimizations << ",\n";
    file << "    \"avg_improvement_percent\": " << std::fixed << std::setprecision(2) 
         << stats.avg_improvement_percent << "\n";
    file << "  },\n";
    
    file << "  \"decisions\": [\n";
    
    auto decisions = GetRecentDecisions(1000);
    for (size_t i = 0; i < decisions.size(); ++i) {
        const auto& decision = decisions[i];
        file << "    {\n";
        file << "      \"action\": " << static_cast<int>(decision.action) << ",\n";
        file << "      \"trigger\": " << static_cast<int>(decision.trigger) << ",\n";
        file << "      \"parameter_name\": \"" << decision.parameter_name << "\",\n";
        file << "      \"old_value\": \"" << decision.old_value << "\",\n";
        file << "      \"new_value\": \"" << decision.new_value << "\",\n";
        file << "      \"expected_improvement\": " << decision.expected_improvement << ",\n";
        file << "      \"timestamp_ns\": " << decision.timestamp_ns << ",\n";
        file << "      \"request_id\": " << decision.request_id << "\n";
        file << "    }";
        if (i < decisions.size() - 1) {
            file << ",";
        }
        file << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
    
    file.close();
    return Status::SUCCESS;
}

void OptimizationManager::OptimizationThread() {
    auto last_optimization = std::chrono::steady_clock::now();
    
    while (!shutdown_requested_.load()) {
        if (!optimization_enabled_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        auto now = std::chrono::steady_clock::now();
        if (now - last_optimization >= config_.optimization_interval) {
            // Collect current metrics
            OptimizationMetrics metrics = CollectMetrics();
            
            // Update internal metrics
            UpdateMetrics(metrics);
            
            // Generate optimization decisions
            if (policy_ && policy_->IsApplicable(metrics)) {
                auto decisions = policy_->AnalyzeAndDecide(metrics, config_);
                ProcessDecisions(decisions);
            }
            
            last_optimization = now;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

OptimizationMetrics OptimizationManager::CollectMetrics() {
    OptimizationMetrics metrics;
    
    // Collect metrics from profiler and components
    // This is a simplified version - in a real implementation, we would
    // collect actual metrics from the profiler and component statistics
    
    // Simulate metric collection
    metrics.avg_latency_ms = 50.0 + (rand() % 100); // 50-150ms
    metrics.p99_latency_ms = metrics.avg_latency_ms * 2.0;
    metrics.throughput_ops_per_sec = 1000.0 - (rand() % 200); // 800-1000 ops/sec
    metrics.memory_usage_percent = 0.3 + (rand() % 50) / 100.0; // 30-80%
    metrics.cpu_utilization_percent = 0.4 + (rand() % 40) / 100.0; // 40-80%
    metrics.queue_depth = rand() % 50; // 0-50
    metrics.batch_efficiency = 0.7 + (rand() % 20) / 100.0; // 70-90%
    metrics.total_requests = 10000 + rand() % 5000; // 10k-15k
    metrics.failed_requests = rand() % 100; // 0-100
    
    return metrics;
}

void OptimizationManager::ProcessDecisions(const std::vector<OptimizationDecision>& decisions) {
    for (const auto& decision : decisions) {
        Status status = ApplyOptimization(decision);
        if (status != Status::SUCCESS) {
            stats_.failed_optimizations.fetch_add(1);
        }
    }
}

void OptimizationManager::LogDecision(const OptimizationDecision& decision) {
    std::lock_guard<std::mutex> lock(decisions_mutex_);
    
    // Add to recent decisions
    recent_decisions_.push_back(decision);
    
    // Keep only recent decisions (limit to 1000)
    if (recent_decisions_.size() > 1000) {
        recent_decisions_.erase(recent_decisions_.begin(), 
                              recent_decisions_.begin() + (recent_decisions_.size() - 1000));
    }
    
    // Log to profiler if available
    if (profiler_) {
        std::string event_name = "optimization_decision_" + decision.parameter_name;
        profiler_->MarkEvent(decision.request_id, event_name.c_str());
    }
}

} // namespace edge_ai
