/**
 * @file graph_executor.cpp
 * @brief Implementation of GraphExecutor for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 */

#include "graph/graph_executor.h"
#include "profiling/profiler.h"
#include <algorithm>
#include <chrono>
#include <functional>

namespace edge_ai {

GraphExecutor::GraphExecutor(const ExecutorConfig& config)
    : config_(config), initialized_(false), shutdown_requested_(false) {
}

GraphExecutor::~GraphExecutor() {
    Shutdown();
}

bool GraphExecutor::Initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    initialized_.store(true);
    return true;
}

void GraphExecutor::Shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    shutdown_requested_.store(true);
    initialized_.store(false);
}

std::future<ExecutionResult> GraphExecutor::ExecuteAsync(std::shared_ptr<Graph> graph,
                                                       const ExecutionContext& context) {
    return std::async(std::launch::async, [this, graph, context]() {
        return Execute(graph, context);
    });
}

ExecutionResult GraphExecutor::Execute(std::shared_ptr<Graph> graph, const ExecutionContext& context) {
    ExecutionResult result;
    
    if (!graph || !initialized_.load()) {
        result.success = false;
        result.error_message = "Graph executor not initialized or invalid graph";
        return result;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Pre-execution validation
    if (!PreExecuteValidation(graph, context)) {
        result.success = false;
        result.error_message = "Pre-execution validation failed";
        return result;
    }
    
    // Create execution plan
    ExecutionPlan plan = CreateExecutionPlan(graph, context);
    
    // Optimize execution plan if enabled
    if (config_.enable_optimization) {
        OptimizeExecutionPlan(plan, graph);
    }
    
    // Start profiling session if enabled
    if (config_.enable_profiling && profiler_) {
        StartProfilingSession(context.session_id);
    }
    
    // Execute graph
    result = ExecuteGraphInternal(graph, context);
    
    // End profiling session if enabled
    if (config_.enable_profiling && profiler_) {
        EndProfilingSession(context.session_id);
    }
    
    // Post-execution validation
    if (!PostExecuteValidation(result)) {
        result.success = false;
        result.error_message = "Post-execution validation failed";
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.total_execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Update statistics
    UpdateStats(result, result.total_execution_time);
    
    return result;
}

ExecutionPlan GraphExecutor::CreateExecutionPlan(std::shared_ptr<Graph> graph,
                                                const ExecutionContext& context) {
    (void)context;
    ExecutionPlan plan;
    
    // Create execution stages
    plan.execution_stages = CreateExecutionStages(graph);
    
    // Optimize backend selection
    plan.node_backend_mapping = OptimizeBackendSelection(graph);
    
    // Find critical path
    plan.critical_path = FindCriticalPath(graph);
    
    // Estimate execution time and memory usage
    plan.estimated_execution_time = std::chrono::milliseconds(1000); // TODO: Calculate actual estimate
    plan.estimated_memory_usage = 1024 * 1024; // TODO: Calculate actual estimate
    
    return plan;
}

bool GraphExecutor::OptimizeExecutionPlan(ExecutionPlan& plan, std::shared_ptr<Graph> graph) {
    (void)plan;
    (void)graph;
    // TODO: Implement execution plan optimization
    return true;
}

void GraphExecutor::UpdateConfig(const ExecutorConfig& config) {
    config_ = config;
}

void GraphExecutor::SetScheduler(std::shared_ptr<GraphScheduler> scheduler) {
    scheduler_ = scheduler;
}

void GraphExecutor::SetOptimizationManager(std::shared_ptr<OptimizationManager> optimization_manager) {
    optimization_manager_ = optimization_manager;
}

void GraphExecutor::SetProfiler(Profiler* profiler) {
    profiler_ = profiler;
}

bool GraphExecutor::RegisterBackend(std::shared_ptr<ExecutionBackend> backend) {
    if (!backend) return false;
    
    std::lock_guard<std::mutex> lock(backends_mutex_);
    backends_[backend->GetId()] = backend;
    return true;
}

bool GraphExecutor::UnregisterBackend(const std::string& backend_id) {
    std::lock_guard<std::mutex> lock(backends_mutex_);
    auto it = backends_.find(backend_id);
    if (it != backends_.end()) {
        backends_.erase(it);
        return true;
    }
    return false;
}

std::vector<std::string> GraphExecutor::GetAvailableBackends() const {
    std::lock_guard<std::mutex> lock(backends_mutex_);
    std::vector<std::string> backend_ids;
    for (const auto& [id, backend] : backends_) {
        backend_ids.push_back(id);
    }
    return backend_ids;
}


void GraphExecutor::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    result_cache_.clear();
}

size_t GraphExecutor::GetCacheSize() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return result_cache_.size();
}

GraphExecutor::ExecutorStats::Snapshot GraphExecutor::GetStatsSnapshot() const {
    return stats_.GetSnapshot();
}

void GraphExecutor::ResetStats() {
    // Reset stats manually since assignment is deleted
    stats_.total_executions.store(0);
    stats_.successful_executions.store(0);
    stats_.failed_executions.store(0);
    stats_.total_execution_time_ms.store(0);
    stats_.max_execution_time_ms.store(0);
    stats_.min_execution_time_ms.store(std::numeric_limits<uint64_t>::max());
    stats_.cache_hits.store(0);
    stats_.cache_misses.store(0);
}

// Private methods

ExecutionResult GraphExecutor::ExecuteGraphInternal(std::shared_ptr<Graph> graph,
                                                  const ExecutionContext& context) {
    ExecutionResult result;
    
    // Use scheduler if available
    if (scheduler_) {
        auto graph_stats = scheduler_->ExecuteGraph(graph, context);
        result.success = true;
        result.stats = graph_stats;
    } else {
        // Direct execution
        result.success = ExecuteGraphDirectly(graph, context);
    }
    
    return result;
}

bool GraphExecutor::ExecuteGraphDirectly(std::shared_ptr<Graph> graph, const ExecutionContext& context) {
    // Get topological order
    auto topo_order = graph->GetTopologicalOrder();
    
    // Execute nodes in topological order
    for (const auto& node_id : topo_order) {
        if (context.should_stop.load()) {
            return false;
        }
        
        auto node_result = ExecuteNode(graph, node_id, context);
        if (node_result.status == NodeStatus::FAILED) {
            return false;
        }
    }
    
    return true;
}

bool GraphExecutor::PreExecuteValidation(std::shared_ptr<Graph> graph, const ExecutionContext& context) {
    if (!graph) return false;
    if (!graph->IsValid()) return false;
    if (context.session_id.empty()) return false;
    
    return true;
}

bool GraphExecutor::PostExecuteValidation(const ExecutionResult& result) {
    return result.success;
}

std::vector<std::vector<std::string>> GraphExecutor::CreateExecutionStages(std::shared_ptr<Graph> graph) {
    std::vector<std::vector<std::string>> stages;
    
    // Get topological order
    auto topo_order = graph->GetTopologicalOrder();
    
    // Group nodes by execution level (simple approach)
    std::unordered_map<std::string, int> node_levels;
    std::unordered_set<std::string> processed_nodes;
    
    for (const auto& node_id : topo_order) {
        int level = 0;
        auto node = graph->GetNode(node_id);
        if (node) {
            for (const auto& input_edge : node->GetInputEdges()) {
                auto edge = graph->GetEdge(input_edge);
                if (edge) {
                    const std::string& source_node = edge->GetSourceNode();
                    if (processed_nodes.find(source_node) != processed_nodes.end()) {
                        level = std::max(level, node_levels[source_node] + 1);
                    }
                }
            }
        }
        node_levels[node_id] = level;
        processed_nodes.insert(node_id);
    }
    
    // Group nodes by level
    int max_level = 0;
    for (const auto& [node_id, level] : node_levels) {
        max_level = std::max(max_level, level);
    }
    
    stages.resize(max_level + 1);
    for (const auto& [node_id, level] : node_levels) {
        stages[level].push_back(node_id);
    }
    
    return stages;
}

std::unordered_map<std::string, BackendType> GraphExecutor::OptimizeBackendSelection(std::shared_ptr<Graph> graph) {
    std::unordered_map<std::string, BackendType> backend_mapping;
    
    // Simple backend selection for now
    auto node_ids = graph->GetNodeIds();
    for (const auto& node_id : node_ids) {
        auto node = graph->GetNode(node_id);
        if (node && node->GetType() == NodeType::MODEL_INFERENCE) {
            backend_mapping[node_id] = BackendType::CPU; // Default to CPU
        } else {
            backend_mapping[node_id] = BackendType::CPU;
        }
    }
    
    return backend_mapping;
}

std::vector<std::string> GraphExecutor::FindCriticalPath(std::shared_ptr<Graph> graph) {
    // Simple critical path calculation
    return graph->GetTopologicalOrder();
}

NodeExecutionResult GraphExecutor::ExecuteNode(std::shared_ptr<Graph> graph, const std::string& node_id,
                                             const ExecutionContext& context) {
    NodeExecutionResult result;
    auto start_time = std::chrono::steady_clock::now();
    
    auto node = graph->GetNode(node_id);
    if (!node) {
        result.status = NodeStatus::FAILED;
        result.error_message = "Node not found: " + node_id;
        return result;
    }
    
    // Prepare node inputs
    if (!PrepareNodeInputs(graph, node_id)) {
        result.status = NodeStatus::FAILED;
        result.error_message = "Failed to prepare inputs for node: " + node_id;
        return result;
    }
    
    // Execute node based on type
    switch (node->GetType()) {
        case NodeType::MODEL_INFERENCE:
            result = ExecuteModelNode(graph, node_id, context);
            break;
        case NodeType::DATA_PROCESSING:
            result = ExecuteOperatorNode(graph, node_id, context);
            break;
        default:
            result = ExecuteGenericNode(graph, node_id, context);
            break;
    }
    
    // Process node outputs
    if (result.status == NodeStatus::COMPLETED) {
        ProcessNodeOutputs(graph, node_id);
    }
    
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    
    return result;
}

NodeExecutionResult GraphExecutor::ExecuteModelNode(std::shared_ptr<Graph> graph, const std::string& node_id, const ExecutionContext& context) {
    (void)graph;
    (void)node_id;
    (void)context;
    NodeExecutionResult result;
    
    // TODO: Implement actual model inference
    // For now, simulate execution
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    result.status = NodeStatus::COMPLETED;
    return result;
}

NodeExecutionResult GraphExecutor::ExecuteOperatorNode(std::shared_ptr<Graph> graph, const std::string& node_id, const ExecutionContext& context) {
    (void)graph;
    (void)node_id;
    (void)context;
    NodeExecutionResult result;
    
    // TODO: Implement data processing
    // For now, simulate execution
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    result.status = NodeStatus::COMPLETED;
    return result;
}

NodeExecutionResult GraphExecutor::ExecuteGenericNode(std::shared_ptr<Graph> graph, const std::string& node_id, const ExecutionContext& context) {
    (void)graph;
    (void)node_id;
    (void)context;
    NodeExecutionResult result;
    
    // TODO: Implement generic node execution
    // For now, simulate execution
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    
    result.status = NodeStatus::COMPLETED;
    return result;
}

bool GraphExecutor::PrepareNodeInputs(std::shared_ptr<Graph> graph, const std::string& node_id) {
    auto node = graph->GetNode(node_id);
    if (!node) return false;
    
    std::vector<Tensor> inputs;
    
    // Collect inputs from input edges
    for (const auto& edge_id : node->GetInputEdges()) {
        auto edge = graph->GetEdge(edge_id);
        if (edge && edge->IsDataReady()) {
            std::vector<Tensor> edge_data;
            if (edge->TransferData(edge_data)) {
                for (auto& tensor : edge_data) {
                    inputs.emplace_back(std::move(tensor));
                }
            }
        }
    }
    
    node->SetInputs(std::move(inputs));
    return true;
}

bool GraphExecutor::ProcessNodeOutputs(std::shared_ptr<Graph> graph, const std::string& node_id) {
    auto node = graph->GetNode(node_id);
    if (!node) return false;
    
    // Transfer outputs to output edges
    for (const auto& edge_id : node->GetOutputEdges()) {
        auto edge = graph->GetEdge(edge_id);
        if (edge) {
            // Note: GetOutputs() returns const reference, so we need to create a copy
            // This is a limitation of the current design - ideally we'd have move semantics
            const auto& outputs_ref = node->GetOutputs();
            std::vector<Tensor> outputs_copy;
            outputs_copy.reserve(outputs_ref.size());
            for (const auto& tensor : outputs_ref) {
                // Create a copy of each tensor (this will need to be implemented properly)
                outputs_copy.emplace_back(tensor.GetDataType(), tensor.GetShape());
            }
            edge->SetData(std::move(outputs_copy));
        }
    }
    
    return true;
}

void GraphExecutor::ApplyOptimizations(std::shared_ptr<Graph> graph, const ExecutionContext& context) {
    (void)context;
    if (!optimization_manager_) return;
    
    // Collect execution metrics
    [[maybe_unused]] OptimizationMetrics metrics = CollectExecutionMetrics(graph);
    
    // Apply optimizations
    // TODO: Implement optimization application
}

OptimizationMetrics GraphExecutor::CollectExecutionMetrics(std::shared_ptr<Graph> graph) {
    (void)graph;
    OptimizationMetrics metrics;
    
    // TODO: Collect actual metrics from graph execution
    metrics.avg_latency_ms = 100.0;
    metrics.memory_usage_percent = 50.0;
    metrics.queue_depth = 5;
    
    return metrics;
}

void GraphExecutor::StartProfilingSession(const std::string& session_id) {
    if (profiler_) {
        uint64_t request_id = std::hash<std::string>{}(session_id);
        profiler_->StartSessionForRequest(request_id);
    }
}

void GraphExecutor::EndProfilingSession(const std::string& session_id) {
    if (profiler_) {
        uint64_t request_id = std::hash<std::string>{}(session_id);
        profiler_->EndSessionForRequest(request_id);
    }
}

void GraphExecutor::ProfileNodeExecution(const std::string& node_id, const std::chrono::milliseconds& duration) {
    (void)node_id;
    (void)duration;
    if (profiler_) {
        profiler_->MarkEvent(0, "node_execution_" + node_id);
    }
}

std::string GraphExecutor::GenerateCacheKey(std::shared_ptr<Graph> graph, const ExecutionContext& context) {
    // TODO: Generate proper cache key based on graph structure and context
    return graph->GetId() + "_" + context.session_id;
}

bool GraphExecutor::GetCachedResult(const std::string& cache_key, ExecutionResult& result) {
    if (!config_.enable_caching) return false;
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = result_cache_.find(cache_key);
    if (it != result_cache_.end()) {
        result = std::move(it->second);
        stats_.cache_hits.fetch_add(1);
        return true;
    }
    
    stats_.cache_misses.fetch_add(1);
    return false;
}

void GraphExecutor::CacheResult(const std::string& cache_key, ExecutionResult&& result) {
    if (!config_.enable_caching) return;
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    result_cache_[cache_key] = std::move(result);
}

void GraphExecutor::HandleExecutionError(const std::string& error, ExecutionResult& result) {
    result.success = false;
    result.error_message = error;
}

bool GraphExecutor::ShouldRetryExecution(const ExecutionResult& result) {
    return !result.success && config_.max_retry_attempts > 0;
}

void GraphExecutor::UpdateStats(const ExecutionResult& result, std::chrono::milliseconds execution_time) {
    stats_.total_executions.fetch_add(1);
    
    if (result.success) {
        stats_.successful_executions.fetch_add(1);
    } else {
        stats_.failed_executions.fetch_add(1);
    }
    
    // Update timing stats using atomic operations
    auto current_total = stats_.total_execution_time_ms.load();
    while (!stats_.total_execution_time_ms.compare_exchange_weak(current_total, current_total + execution_time.count())) {
        // Retry if compare_exchange_weak failed
    }
    
    // Update max execution time
    auto current_max = stats_.max_execution_time_ms.load();
    while (static_cast<uint64_t>(execution_time.count()) > current_max && 
           !stats_.max_execution_time_ms.compare_exchange_weak(current_max, static_cast<uint64_t>(execution_time.count()))) {
        // Retry
    }
    
    // Update min execution time
    auto current_min = stats_.min_execution_time_ms.load();
    while (static_cast<uint64_t>(execution_time.count()) < current_min && 
           !stats_.min_execution_time_ms.compare_exchange_weak(current_min, static_cast<uint64_t>(execution_time.count()))) {
        // Retry
    }
}

// GraphExecutionPipeline Implementation

GraphExecutionPipeline::GraphExecutionPipeline(std::shared_ptr<GraphExecutor> executor)
    : executor_(executor), max_parallel_graphs_(4), pipeline_timeout_(std::chrono::milliseconds(30000)),
      fail_fast_(false) {
}

std::future<std::vector<ExecutionResult>> GraphExecutionPipeline::ExecuteSequentialAsync(
    const std::vector<std::shared_ptr<Graph>>& graphs, const ExecutionContext& context) {
    
    return std::async(std::launch::async, [this, graphs, context]() {
        std::vector<ExecutionResult> results;
        
        for (const auto& graph : graphs) {
            if (context.should_stop.load()) {
                break;
            }
            
            auto result = executor_->Execute(graph, context);
            results.emplace_back(std::move(result));
            
            if (fail_fast_ && !result.success) {
                break;
            }
        }
        
        return results;
    });
}

std::future<std::vector<ExecutionResult>> GraphExecutionPipeline::ExecuteParallelAsync(
    const std::vector<std::shared_ptr<Graph>>& graphs, const ExecutionContext& context) {
    
    return std::async(std::launch::async, [this, graphs, context]() {
        std::vector<std::future<ExecutionResult>> futures;
        
        // Limit parallel execution
        int parallel_count = std::min(static_cast<int>(graphs.size()), max_parallel_graphs_);
        
        for (int i = 0; i < parallel_count; ++i) {
            futures.push_back(executor_->ExecuteAsync(graphs[i], context));
        }
        
        std::vector<ExecutionResult> results;
        for (auto& future : futures) {
            results.push_back(future.get());
        }
        
        return results;
    });
}

std::future<std::vector<ExecutionResult>> GraphExecutionPipeline::ExecuteConditionalAsync(
    const std::vector<std::shared_ptr<Graph>>& graphs,
    const std::function<int(const ExecutionContext&)>& condition_func,
    const ExecutionContext& context) {
    
    return std::async(std::launch::async, [this, graphs, condition_func, context]() {
        std::vector<ExecutionResult> results;
        
        int selected_index = condition_func(context);
        if (selected_index >= 0 && selected_index < static_cast<int>(graphs.size())) {
            auto result = executor_->Execute(graphs[selected_index], context);
            results.emplace_back(std::move(result));
        }
        
        return results;
    });
}

void GraphExecutionPipeline::SetMaxParallelGraphs(int max_parallel) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    max_parallel_graphs_ = max_parallel;
}

void GraphExecutionPipeline::SetPipelineTimeout(std::chrono::milliseconds timeout) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    pipeline_timeout_ = timeout;
}

void GraphExecutionPipeline::SetFailFast(bool fail_fast) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    fail_fast_ = fail_fast;
}

} // namespace edge_ai
