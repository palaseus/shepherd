/**
 * @file graph_scheduler.cpp
 * @brief Implementation of GraphScheduler for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 */

#include "graph/graph_scheduler.h"
#include "profiling/profiler.h"
#include <algorithm>
#include <chrono>

namespace edge_ai {

GraphScheduler::GraphScheduler(const GraphSchedulerConfig& config)
    : config_(config), initialized_(false), shutdown_requested_(false), 
      scheduling_paused_(false), active_workers_(0) {
    
    // Initialize worker load tracking
    worker_loads_.resize(config_.num_worker_threads);
    worker_busy_.resize(config_.num_worker_threads);
    for (int i = 0; i < config_.num_worker_threads; ++i) {
        worker_loads_[i] = 0;
        worker_busy_[i] = false;
    }
}

GraphScheduler::~GraphScheduler() {
    Shutdown();
}

bool GraphScheduler::Initialize() {
    if (initialized_.load()) {
        return true;
    }
    
    // Start worker threads
    for (int i = 0; i < config_.num_worker_threads; ++i) {
        worker_threads_.emplace_back(&GraphScheduler::WorkerThreadMain, this, i);
    }
    
    // Start scheduler thread
    scheduler_thread_ = std::thread(&GraphScheduler::SchedulerThreadMain, this);
    
    initialized_.store(true);
    return true;
}

void GraphScheduler::Shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    shutdown_requested_.store(true);
    
    // Notify all waiting threads
    task_queue_cv_.notify_all();
    
    // Wait for worker threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
    
    // Wait for scheduler thread
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
    
    initialized_.store(false);
}

std::future<GraphExecutionStats::Snapshot> GraphScheduler::ExecuteGraphAsync(
    std::shared_ptr<Graph> graph, const ExecutionContext& context) {
    
    return std::async(std::launch::async, [this, graph, context]() {
        return ExecuteGraph(graph, context);
    });
}

GraphExecutionStats::Snapshot GraphScheduler::ExecuteGraph(
    std::shared_ptr<Graph> graph, const ExecutionContext& context) {
    
    if (!graph || !initialized_.load()) {
        return GraphExecutionStats::Snapshot{};
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Reset graph execution state
    graph->ResetExecutionState();
    
    // Get topological order
    auto topo_order = graph->GetTopologicalOrder();
    
    // Execute nodes in topological order
    for (const auto& node_id : topo_order) {
        if (context.should_stop.load()) {
            break;
        }
        
        auto node = graph->GetNode(node_id);
        if (!node) continue;
        
        // Create execution task
        NodeExecutionTask task;
        task.node_id = node_id;
        task.graph_id = graph->GetId();
        task.session_id = context.session_id;
        task.priority = node->GetMetadata().priority;
        task.is_critical = node->GetMetadata().is_critical;
        
        // Execute node
        auto result = ExecuteNode(task);
        
        // Update node status
        node->SetExecutionResult(std::move(result));
        node->SetStatus(result.status);
        
        // Update graph stats
        graph->UpdateStats(result);
        
        // Check for failures
        if (result.status == NodeStatus::FAILED && node->GetMetadata().is_critical) {
            break;
        }
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Update stats using atomic operations
    auto current_total = stats_.total_execution_time_ms.load();
    while (!stats_.total_execution_time_ms.compare_exchange_weak(current_total, current_total + execution_time.count())) {
        // Retry if compare_exchange_weak failed
    }
    
    return graph->GetStatsSnapshot();
}

bool GraphScheduler::SubmitTask(const NodeExecutionTask& task) {
    if (!initialized_.load()) {
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(task_queue_mutex_);
        task_queue_.push_back(std::make_unique<NodeExecutionTask>(std::move(const_cast<NodeExecutionTask&>(task))));
    }
    
    task_queue_cv_.notify_one();
    stats_.total_tasks_submitted.fetch_add(1);
    
    return true;
}

bool GraphScheduler::CancelTask(const std::string& task_id) {
    [[maybe_unused]] auto task_id_ref = task_id;
    // TODO: Implement task cancellation
    return false;
}

bool GraphScheduler::CancelAllTasks() {
    std::lock_guard<std::mutex> lock(task_queue_mutex_);
    
    task_queue_.clear();
    
    return true;
}

void GraphScheduler::PauseScheduling() {
    scheduling_paused_.store(true);
}

void GraphScheduler::ResumeScheduling() {
    scheduling_paused_.store(false);
    task_queue_cv_.notify_all();
}

GraphSchedulerStats::Snapshot GraphScheduler::GetStatsSnapshot() const {
    return stats_.GetSnapshot();
}

void GraphScheduler::ResetStats() {
    // Reset stats manually since assignment is deleted
    stats_.total_tasks_submitted.store(0);
    stats_.total_tasks_completed.store(0);
    stats_.total_tasks_failed.store(0);
    stats_.total_tasks_cancelled.store(0);
    stats_.total_scheduling_time_ms.store(0);
    stats_.total_execution_time_ms.store(0);
    stats_.max_task_execution_time_ms.store(0);
    stats_.min_task_execution_time_ms.store(std::numeric_limits<uint64_t>::max());
    stats_.peak_queue_size.store(0);
    stats_.active_workers.store(0);
}

void GraphScheduler::UpdateConfig(const GraphSchedulerConfig& config) {
    config_ = config;
}

bool GraphScheduler::RegisterBackend(std::shared_ptr<ExecutionBackend> backend) {
    if (!backend) return false;
    
    std::lock_guard<std::mutex> lock(backends_mutex_);
    backends_[backend->GetId()] = backend;
    return true;
}

bool GraphScheduler::UnregisterBackend(const std::string& backend_id) {
    std::lock_guard<std::mutex> lock(backends_mutex_);
    auto it = backends_.find(backend_id);
    if (it != backends_.end()) {
        backends_.erase(it);
        return true;
    }
    return false;
}

std::vector<std::string> GraphScheduler::GetAvailableBackends() const {
    std::lock_guard<std::mutex> lock(backends_mutex_);
    std::vector<std::string> backend_ids;
    for (const auto& [id, backend] : backends_) {
        backend_ids.push_back(id);
    }
    return backend_ids;
}

void GraphScheduler::SetOptimizationManager(std::shared_ptr<OptimizationManager> optimization_manager) {
    optimization_manager_ = optimization_manager;
}

void GraphScheduler::SetProfiler(Profiler* profiler) {
    profiler_ = profiler;
}

// Private methods

void GraphScheduler::WorkerThreadMain(int worker_id) {
    active_workers_.fetch_add(1);
    
    while (!shutdown_requested_.load()) {
        NodeExecutionTask task;
        
        // Get task from queue
        {
            std::unique_lock<std::mutex> lock(task_queue_mutex_);
            task_queue_cv_.wait(lock, [this] {
                return !task_queue_.empty() || shutdown_requested_.load();
            });
            
            if (shutdown_requested_.load()) {
                break;
            }
            
            if (!task_queue_.empty()) {
                // Find highest priority task
                auto it = std::max_element(task_queue_.begin(), task_queue_.end(), 
                    [](const std::unique_ptr<NodeExecutionTask>& a, const std::unique_ptr<NodeExecutionTask>& b) {
                        return a->priority < b->priority;
                    });
                task = std::move(**it);
                task_queue_.erase(it);
            }
        }
        
        if (task.node_id.empty()) {
            continue;
        }
        
        // Execute task
        {
            std::lock_guard<std::mutex> lock(worker_busy_mutex_);
            worker_busy_[worker_id] = true;
        }
        auto result = ExecuteNode(task);
        {
            std::lock_guard<std::mutex> lock(worker_busy_mutex_);
            worker_busy_[worker_id] = false;
        }
        
        // Update stats
        auto execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - task.submit_time);
        UpdateStats(result, execution_time);
    }
    
    active_workers_.fetch_sub(1);
}

void GraphScheduler::SchedulerThreadMain() {
    while (!shutdown_requested_.load()) {
        if (!scheduling_paused_.load()) {
            // TODO: Implement dynamic scheduling logic
        }
        
        std::this_thread::sleep_for(config_.scheduling_interval);
    }
}

std::vector<NodeExecutionTask> GraphScheduler::GetReadyTasks(std::shared_ptr<Graph> graph) {
    std::vector<NodeExecutionTask> ready_tasks;
    
    auto ready_nodes = graph->GetReadyNodes();
    for (const auto& node_id : ready_nodes) {
        NodeExecutionTask task;
        task.node_id = node_id;
        task.graph_id = graph->GetId();
        ready_tasks.push_back(std::move(task));
    }
    
    return ready_tasks;
}

void GraphScheduler::ScheduleTasks(const std::vector<NodeExecutionTask>& tasks) {
    for (const auto& task : tasks) {
        SubmitTask(task);
    }
}

void GraphScheduler::UpdateNodeDependencies(std::shared_ptr<Graph> graph, const std::string& completed_node) {
    (void)graph;
    (void)completed_node;
    // TODO: Implement dependency update logic
}

NodeExecutionResult GraphScheduler::ExecuteNode(const NodeExecutionTask& task) {
    (void)task;
    NodeExecutionResult result;
    auto start_time = std::chrono::steady_clock::now();
    
    // TODO: Implement actual node execution
    // For now, simulate execution
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    result.status = NodeStatus::COMPLETED;
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);
    
    return result;
}

std::shared_ptr<ExecutionBackend> GraphScheduler::SelectBackend(const NodeExecutionTask& task) {
    (void)task;
    std::lock_guard<std::mutex> lock(backends_mutex_);
    
    if (backends_.empty()) {
        return nullptr;
    }
    
    // Simple round-robin selection for now
    static size_t backend_index = 0;
    auto it = backends_.begin();
    std::advance(it, backend_index % backends_.size());
    backend_index++;
    
    return it->second;
}

int GraphScheduler::SelectWorker(const NodeExecutionTask& task) {
    (void)task;
    // Simple round-robin worker selection
    static std::atomic<int> worker_index{0};
    return worker_index.fetch_add(1) % config_.num_worker_threads;
}

void GraphScheduler::UpdateWorkerLoads() {
    // TODO: Implement worker load tracking
}

void GraphScheduler::HandleTaskFailure(const NodeExecutionTask& task, const std::string& error) {
    (void)task;
    (void)error;
    // TODO: Implement fault tolerance logic
}

bool GraphScheduler::ShouldRetryTask(const NodeExecutionTask& task) {
    (void)task;
    // TODO: Implement retry logic
    return false;
}

void GraphScheduler::UpdateStats(const NodeExecutionResult& result, std::chrono::milliseconds execution_time) {
    stats_.total_tasks_completed.fetch_add(1);
    
    switch (result.status) {
        case NodeStatus::COMPLETED:
            stats_.total_tasks_completed.fetch_add(1);
            break;
        case NodeStatus::FAILED:
            stats_.total_tasks_failed.fetch_add(1);
            break;
        case NodeStatus::CANCELLED:
            stats_.total_tasks_cancelled.fetch_add(1);
            break;
        default:
            break;
    }
    
    // Update timing stats
    stats_.total_execution_time_ms.fetch_add(execution_time.count());
    
    // Update max execution time
    auto current_max = stats_.max_task_execution_time_ms.load();
    while (static_cast<uint64_t>(execution_time.count()) > current_max && 
           !stats_.max_task_execution_time_ms.compare_exchange_weak(current_max, static_cast<uint64_t>(execution_time.count()))) {
        // Retry
    }
    
    // Update min execution time
    auto current_min = stats_.min_task_execution_time_ms.load();
    while (static_cast<uint64_t>(execution_time.count()) < current_min && 
           !stats_.min_task_execution_time_ms.compare_exchange_weak(current_min, static_cast<uint64_t>(execution_time.count()))) {
        // Retry
    }
}

// GraphExecutionOrchestrator Implementation

GraphExecutionOrchestrator::GraphExecutionOrchestrator(std::shared_ptr<GraphScheduler> scheduler)
    : scheduler_(scheduler), execution_active_(false) {
}

std::future<GraphExecutionStats::Snapshot> GraphExecutionOrchestrator::ExecutePipelineAsync(
    const std::vector<std::shared_ptr<Graph>>& graphs, const ExecutionContext& context) {
    
    return std::async(std::launch::async, [this, graphs, context]() {
        execution_active_.store(true);
        
        GraphExecutionStats::Snapshot total_stats;
        
        for (const auto& graph : graphs) {
            if (context.should_stop.load()) {
                break;
            }
            
            (void)scheduler_->ExecuteGraph(graph, context);
            // TODO: Aggregate stats
        }
        
        execution_active_.store(false);
        return total_stats;
    });
}

std::future<GraphExecutionStats::Snapshot> GraphExecutionOrchestrator::ExecuteParallelAsync(
    const std::vector<std::shared_ptr<Graph>>& graphs, const ExecutionContext& context) {
    
    return std::async(std::launch::async, [this, graphs, context]() {
        execution_active_.store(true);
        
        std::vector<std::future<GraphExecutionStats::Snapshot>> futures;
        
        for (const auto& graph : graphs) {
            futures.push_back(scheduler_->ExecuteGraphAsync(graph, context));
        }
        
        GraphExecutionStats::Snapshot total_stats;
        
        for (auto& future : futures) {
            (void)future.get();
            // TODO: Aggregate stats
        }
        
        execution_active_.store(false);
        return total_stats;
    });
}

std::future<GraphExecutionStats::Snapshot> GraphExecutionOrchestrator::ExecuteConditionalAsync(
    const std::vector<std::shared_ptr<Graph>>& graphs,
    const std::function<int(const ExecutionContext&)>& condition_func,
    const ExecutionContext& context) {
    
    return std::async(std::launch::async, [this, graphs, condition_func, context]() {
        execution_active_.store(true);
        
        int selected_index = condition_func(context);
        if (selected_index >= 0 && selected_index < static_cast<int>(graphs.size())) {
            auto result = scheduler_->ExecuteGraph(graphs[selected_index], context);
            execution_active_.store(false);
            return result;
        }
        
        execution_active_.store(false);
        return GraphExecutionStats::Snapshot{};
    });
}

std::future<void> GraphExecutionOrchestrator::ExecuteStreamingAsync(
    std::shared_ptr<Graph> graph, const ExecutionContext& context,
    std::function<void(const std::vector<Tensor>&)> output_callback) {
    
    return std::async(std::launch::async, [this, graph, context, output_callback]() {
        execution_active_.store(true);
        
        // TODO: Implement streaming execution
        while (!context.should_stop.load() && execution_active_.load()) {
            // Execute graph iteration
            (void)scheduler_->ExecuteGraph(graph, context);
            
            // TODO: Extract outputs and call callback
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        execution_active_.store(false);
    });
}

void GraphExecutionOrchestrator::PauseExecution() {
    execution_active_.store(false);
}

void GraphExecutionOrchestrator::ResumeExecution() {
    execution_active_.store(true);
}

void GraphExecutionOrchestrator::CancelExecution() {
    execution_active_.store(false);
}

bool GraphExecutionOrchestrator::IsExecutionActive() const {
    return execution_active_.load();
}

} // namespace edge_ai
