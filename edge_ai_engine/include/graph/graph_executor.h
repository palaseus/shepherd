/**
 * @file graph_executor.h
 * @brief Graph executor for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the GraphExecutor class that executes compiled graphs
 * with hooks into Profiler and OptimizationManager.
 */

#pragma once

#include "graph_types.h"
#include "graph.h"
#include "graph_scheduler.h"
#include "core/types.h"
#include "backend/execution_backend.h"
#include "optimization/optimization_manager.h"
#include "profiling/profiler.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>
#include <future>
#include <chrono>
#include <any>
#include <functional>
#include <limits>

namespace edge_ai {

/**
 * @struct ExecutionPlan
 * @brief Optimized execution plan for a graph
 */
struct ExecutionPlan {
    std::vector<std::vector<std::string>> execution_stages; // Stages of parallel execution
    std::unordered_map<std::string, BackendType> node_backend_mapping; // Node to backend mapping
    std::unordered_map<std::string, int> node_priorities; // Node execution priorities
    std::chrono::milliseconds estimated_execution_time; // Estimated total execution time
    size_t estimated_memory_usage; // Estimated memory usage
    std::vector<std::string> critical_path; // Critical path for execution
    
    ExecutionPlan() : estimated_execution_time(std::chrono::milliseconds(0)), 
                     estimated_memory_usage(0) {}
};

/**
 * @struct ExecutionResult
 * @brief Result of graph execution
 */
struct ExecutionResult {
    bool success;
    std::string error_message;
    GraphExecutionStats::Snapshot stats;
    std::chrono::milliseconds total_execution_time;
    std::vector<Tensor> final_outputs;
    std::unordered_map<std::string, std::shared_ptr<NodeExecutionResult>> node_results;
    std::unordered_map<std::string, std::any> execution_metadata;
    
    ExecutionResult() : success(false), total_execution_time(std::chrono::milliseconds(0)) {}
    
    // Move-only semantics
    ExecutionResult(const ExecutionResult&) = delete;
    ExecutionResult& operator=(const ExecutionResult&) = delete;
    ExecutionResult(ExecutionResult&&) = default;
    ExecutionResult& operator=(ExecutionResult&&) = default;
};

/**
 * @struct ExecutorConfig
 * @brief Configuration for graph executor
 */
struct ExecutorConfig {
    bool enable_optimization;                   // Enable execution optimization
    bool enable_profiling;                      // Enable execution profiling
    bool enable_caching;                        // Enable result caching
    bool enable_adaptive_scheduling;            // Enable adaptive scheduling
    std::chrono::milliseconds execution_timeout; // Execution timeout
    size_t max_memory_usage;                    // Maximum memory usage
    int max_retry_attempts;                     // Maximum retry attempts for failed nodes
    std::string optimization_policy;            // Optimization policy to use
    bool fail_fast;                            // Stop on first node failure
    
    ExecutorConfig() : enable_optimization(true), enable_profiling(true),
                      enable_caching(false), enable_adaptive_scheduling(true),
                      execution_timeout(std::chrono::milliseconds(30000)),
                      max_memory_usage(1024 * 1024 * 1024), max_retry_attempts(3),
                      optimization_policy("RuleBasedPolicy"), fail_fast(false) {}
};

/**
 * @class GraphExecutor
 * @brief Executes compiled graphs with optimization and profiling hooks
 */
class GraphExecutor {
public:
    GraphExecutor(const ExecutorConfig& config = ExecutorConfig{});
    ~GraphExecutor();
    
    // Lifecycle management
    bool Initialize();
    void Shutdown();
    bool IsInitialized() const { return initialized_.load(); }
    
    // Graph execution
    std::future<ExecutionResult> ExecuteAsync(std::shared_ptr<Graph> graph,
                                            const ExecutionContext& context);
    ExecutionResult Execute(std::shared_ptr<Graph> graph,
                          const ExecutionContext& context);
    
    // Execution planning
    ExecutionPlan CreateExecutionPlan(std::shared_ptr<Graph> graph,
                                    const ExecutionContext& context);
    bool OptimizeExecutionPlan(ExecutionPlan& plan, std::shared_ptr<Graph> graph);
    
    // Configuration
    void UpdateConfig(const ExecutorConfig& config);
    ExecutorConfig GetConfig() const { return config_; }
    
    // Component integration
    void SetScheduler(std::shared_ptr<GraphScheduler> scheduler);
    void SetOptimizationManager(std::shared_ptr<OptimizationManager> optimization_manager);
    void SetProfiler(Profiler* profiler);
    
    // Backend management
    bool RegisterBackend(std::shared_ptr<ExecutionBackend> backend);
    bool UnregisterBackend(const std::string& backend_id);
    std::vector<std::string> GetAvailableBackends() const;
    
    // Caching
    void EnableCaching(bool enable) { config_.enable_caching = enable; }
    void ClearCache();
    size_t GetCacheSize() const;
    
    // Statistics and monitoring
    struct ExecutorStats {
        std::atomic<uint64_t> total_executions;
        std::atomic<uint64_t> successful_executions;
        std::atomic<uint64_t> failed_executions;
        std::atomic<uint64_t> total_execution_time_ms;
        std::atomic<uint64_t> max_execution_time_ms;
        std::atomic<uint64_t> min_execution_time_ms;
        std::atomic<size_t> cache_hits;
        std::atomic<size_t> cache_misses;
        
        ExecutorStats() : total_executions(0), successful_executions(0), failed_executions(0),
                         total_execution_time_ms(0),
                         max_execution_time_ms(0),
                         min_execution_time_ms(std::numeric_limits<uint64_t>::max()),
                         cache_hits(0), cache_misses(0) {}
        
        struct Snapshot {
            uint64_t total_executions;
            uint64_t successful_executions;
            uint64_t failed_executions;
            std::chrono::milliseconds total_execution_time;
            std::chrono::milliseconds max_execution_time;
            std::chrono::milliseconds min_execution_time;
            size_t cache_hits;
            size_t cache_misses;
            double success_rate;
            double average_execution_time_ms;
            double cache_hit_rate;
        };
        
        Snapshot GetSnapshot() const {
            Snapshot snapshot;
            snapshot.total_executions = total_executions.load();
            snapshot.successful_executions = successful_executions.load();
            snapshot.failed_executions = failed_executions.load();
            snapshot.total_execution_time = std::chrono::milliseconds(total_execution_time_ms.load());
            snapshot.max_execution_time = std::chrono::milliseconds(max_execution_time_ms.load());
            snapshot.min_execution_time = std::chrono::milliseconds(min_execution_time_ms.load());
            snapshot.cache_hits = cache_hits.load();
            snapshot.cache_misses = cache_misses.load();
            
            if (snapshot.total_executions > 0) {
                snapshot.success_rate = static_cast<double>(snapshot.successful_executions) / snapshot.total_executions;
                snapshot.average_execution_time_ms = static_cast<double>(snapshot.total_execution_time.count()) / snapshot.total_executions;
            } else {
                snapshot.success_rate = 0.0;
                snapshot.average_execution_time_ms = 0.0;
            }
            
            size_t total_cache_requests = snapshot.cache_hits + snapshot.cache_misses;
            if (total_cache_requests > 0) {
                snapshot.cache_hit_rate = static_cast<double>(snapshot.cache_hits) / total_cache_requests;
            } else {
                snapshot.cache_hit_rate = 0.0;
            }
            
            return snapshot;
        }
    };
    
    ExecutorStats::Snapshot GetStatsSnapshot() const;
    void ResetStats();
    
private:
    // Execution phases
    ExecutionResult ExecuteGraphInternal(std::shared_ptr<Graph> graph,
                                       const ExecutionContext& context);
    bool PreExecuteValidation(std::shared_ptr<Graph> graph, const ExecutionContext& context);
    bool PostExecuteValidation(const ExecutionResult& result);
    
    // Execution planning
    std::vector<std::vector<std::string>> CreateExecutionStages(std::shared_ptr<Graph> graph);
    std::unordered_map<std::string, BackendType> OptimizeBackendSelection(std::shared_ptr<Graph> graph);
    std::vector<std::string> FindCriticalPath(std::shared_ptr<Graph> graph);
    
    // Node execution
    NodeExecutionResult ExecuteNode(std::shared_ptr<Graph> graph, const std::string& node_id,
                                  const ExecutionContext& context);
    NodeExecutionResult ExecuteModelNode(std::shared_ptr<Graph> graph, const std::string& node_id,
                                        const ExecutionContext& context);
    NodeExecutionResult ExecuteOperatorNode(std::shared_ptr<Graph> graph, const std::string& node_id,
                                           const ExecutionContext& context);
    NodeExecutionResult ExecuteGenericNode(std::shared_ptr<Graph> graph, const std::string& node_id,
                                          const ExecutionContext& context);
    bool PrepareNodeInputs(std::shared_ptr<Graph> graph, const std::string& node_id);
    bool ProcessNodeOutputs(std::shared_ptr<Graph> graph, const std::string& node_id);
    bool ExecuteGraphDirectly(std::shared_ptr<Graph> graph, const ExecutionContext& context);
    
    // Optimization integration
    void ApplyOptimizations(std::shared_ptr<Graph> graph, const ExecutionContext& context);
    OptimizationMetrics CollectExecutionMetrics(std::shared_ptr<Graph> graph);
    
    // Profiling integration
    void StartProfilingSession(const std::string& session_id);
    void EndProfilingSession(const std::string& session_id);
    void ProfileNodeExecution(const std::string& node_id, const std::chrono::milliseconds& duration);
    
    // Caching
    std::string GenerateCacheKey(std::shared_ptr<Graph> graph, const ExecutionContext& context);
    bool GetCachedResult(const std::string& cache_key, ExecutionResult& result);
    void CacheResult(const std::string& cache_key, ExecutionResult&& result);
    
    // Error handling
    void HandleExecutionError(const std::string& error, ExecutionResult& result);
    bool ShouldRetryExecution(const ExecutionResult& result);
    
    // Statistics updates
    void UpdateStats(const ExecutionResult& result, std::chrono::milliseconds execution_time);
    
    // Configuration
    ExecutorConfig config_;
    std::atomic<bool> initialized_;
    std::atomic<bool> shutdown_requested_;
    
    // Component dependencies
    std::shared_ptr<GraphScheduler> scheduler_;
    std::shared_ptr<OptimizationManager> optimization_manager_;
    Profiler* profiler_;
    
    // Backend management
    std::unordered_map<std::string, std::shared_ptr<ExecutionBackend>> backends_;
    mutable std::mutex backends_mutex_;
    
    // Caching
    std::unordered_map<std::string, ExecutionResult> result_cache_;
    mutable std::mutex cache_mutex_;
    
    // Statistics
    ExecutorStats stats_;
    mutable std::mutex stats_mutex_;
    
    // Execution state
    std::unordered_map<std::string, std::shared_ptr<Graph>> active_executions_;
    std::mutex active_executions_mutex_;
};

/**
 * @class GraphExecutionPipeline
 * @brief Pipeline for executing multiple graphs in sequence or parallel
 */
class GraphExecutionPipeline {
public:
    GraphExecutionPipeline(std::shared_ptr<GraphExecutor> executor);
    ~GraphExecutionPipeline() = default;
    
    // Pipeline execution
    std::future<std::vector<ExecutionResult>> ExecuteSequentialAsync(
        const std::vector<std::shared_ptr<Graph>>& graphs,
        const ExecutionContext& context);
    
    std::future<std::vector<ExecutionResult>> ExecuteParallelAsync(
        const std::vector<std::shared_ptr<Graph>>& graphs,
        const ExecutionContext& context);
    
    std::future<std::vector<ExecutionResult>> ExecuteConditionalAsync(
        const std::vector<std::shared_ptr<Graph>>& graphs,
        const std::function<int(const ExecutionContext&)>& condition_func,
        const ExecutionContext& context);
    
    // Pipeline configuration
    void SetMaxParallelGraphs(int max_parallel);
    void SetPipelineTimeout(std::chrono::milliseconds timeout);
    void SetFailFast(bool fail_fast);
    
private:
    std::shared_ptr<GraphExecutor> executor_;
    int max_parallel_graphs_;
    std::chrono::milliseconds pipeline_timeout_;
    bool fail_fast_;
    std::mutex config_mutex_;
};

} // namespace edge_ai
