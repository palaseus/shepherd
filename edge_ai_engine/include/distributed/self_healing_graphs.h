/**
 * @file self_healing_graphs.h
 * @brief Self-healing graphs with real-time bottleneck detection and DAG rewiring
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
#include "core/types.h"
#include "distributed/cluster_types.h"
#include "graph/graph.h"

// Forward declarations
namespace edge_ai {
    class ClusterManager;
}

namespace edge_ai {

/**
 * @brief Bottleneck detection result
 */
struct BottleneckDetectionResult {
    std::string bottleneck_id;
    std::string graph_id;
    std::string node_id;
    
    enum class BottleneckType {
        CPU_SATURATION,
        MEMORY_EXHAUSTION,
        GPU_UNDERUTILIZATION,
        NETWORK_CONGESTION,
        DISK_IO_BOTTLENECK,
        CACHE_MISS_RATE,
        THREAD_CONTENTION,
        LOCK_CONTENTION,
        QUEUE_OVERFLOW,
        LATENCY_SPIKE,
        THROUGHPUT_DEGRADATION
    } bottleneck_type;
    
    // Severity and impact
    double severity_score{0.0};  // 0.0 to 1.0
    double impact_score{0.0};    // 0.0 to 1.0
    double confidence{0.0};      // 0.0 to 1.0
    
    // Performance metrics
    double current_throughput{0.0};
    double baseline_throughput{0.0};
    double throughput_degradation{0.0};
    std::chrono::milliseconds current_latency{0};
    std::chrono::milliseconds baseline_latency{0};
    double latency_increase{0.0};
    
    // Resource utilization
    double cpu_utilization{0.0};
    double memory_utilization{0.0};
    double gpu_utilization{0.0};
    double network_utilization{0.0};
    double disk_utilization{0.0};
    
    // Detection metadata
    std::chrono::steady_clock::time_point detection_time;
    std::chrono::milliseconds detection_duration{0};
    std::vector<double> historical_metrics;
    std::string detection_algorithm;
    
    // Recommended actions
    std::vector<std::string> recommended_actions;
    std::vector<std::string> alternative_nodes;
    bool auto_remediation_enabled{true};
    
    BottleneckDetectionResult() {
        detection_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Graph healing action
 */
struct GraphHealingAction {
    std::string action_id;
    std::string graph_id;
    std::string bottleneck_id;
    
    enum class ActionType {
        NODE_MIGRATION,      // Move node to different hardware
        GRAPH_REWIRING,      // Change graph topology
        RESOURCE_ALLOCATION, // Increase resources
        LOAD_BALANCING,      // Redistribute load
        CACHE_OPTIMIZATION,  // Optimize caching
        PARALLELIZATION,     // Increase parallelism
        BATCH_SIZE_ADJUSTMENT, // Adjust batch sizes
        PRIORITY_ADJUSTMENT, // Change execution priority
        GRAPH_PARTITIONING,  // Split/merge partitions
        BACKPRESSURE_CONTROL // Apply backpressure
    } action_type;
    
    // Action parameters
    std::map<std::string, std::string> parameters;
    std::vector<std::string> affected_nodes;
    std::vector<std::string> new_connections;
    std::vector<std::string> removed_connections;
    
    // Execution details
    std::chrono::milliseconds estimated_execution_time{0};
    double estimated_improvement{0.0};
    double risk_score{0.0};
    uint32_t priority{0};
    
    // Status tracking
    bool executed{false};
    std::chrono::steady_clock::time_point execution_time;
    Status execution_status{Status::NOT_INITIALIZED};
    std::string error_message;
    
    // Results
    double actual_improvement{0.0};
    std::chrono::milliseconds actual_execution_time{0};
    std::map<std::string, double> performance_metrics;
    
    GraphHealingAction() {
        execution_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Graph performance profile
 */
struct GraphPerformanceProfile {
    std::string graph_id;
    std::string profile_id;
    
    // Performance baselines
    double baseline_throughput{0.0};
    std::chrono::milliseconds baseline_latency{0};
    double baseline_cpu_usage{0.0};
    double baseline_memory_usage{0.0};
    double baseline_gpu_usage{0.0};
    
    // Current performance
    double current_throughput{0.0};
    std::chrono::milliseconds current_latency{0};
    double current_cpu_usage{0.0};
    double current_memory_usage{0.0};
    double current_gpu_usage{0.0};
    
    // Performance trends
    std::vector<double> throughput_history;
    std::vector<std::chrono::milliseconds> latency_history;
    std::vector<double> cpu_usage_history;
    std::vector<double> memory_usage_history;
    std::vector<double> gpu_usage_history;
    
    // Node-level performance
    std::map<std::string, double> node_throughput;
    std::map<std::string, std::chrono::milliseconds> node_latency;
    std::map<std::string, double> node_cpu_usage;
    std::map<std::string, double> node_memory_usage;
    
    // Edge-level performance
    std::map<std::string, double> edge_throughput;
    std::map<std::string, std::chrono::milliseconds> edge_latency;
    std::map<std::string, double> edge_data_volume;
    
    // Timing information
    std::chrono::steady_clock::time_point last_update;
    std::chrono::milliseconds update_interval{1000};
    
    GraphPerformanceProfile() {
        last_update = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Graph topology optimization
 */
struct GraphTopologyOptimization {
    std::string optimization_id;
    std::string graph_id;
    
    enum class OptimizationType {
        NODE_PLACEMENT,      // Optimize node placement
        EDGE_OPTIMIZATION,   // Optimize edge connections
        PARALLELIZATION,     // Increase parallelism
        PIPELINING,          // Optimize pipelining
        CACHING,             // Optimize caching
        LOAD_BALANCING,      // Balance load distribution
        RESOURCE_ALLOCATION, // Optimize resource allocation
        BATCH_OPTIMIZATION   // Optimize batch processing
    } optimization_type;
    
    // Optimization parameters
    std::map<std::string, std::string> parameters;
    std::vector<std::string> nodes_to_move;
    std::vector<std::string> edges_to_add;
    std::vector<std::string> edges_to_remove;
    std::vector<std::string> nodes_to_split;
    std::vector<std::string> nodes_to_merge;
    
    // Expected improvements
    double expected_throughput_improvement{0.0};
    double expected_latency_improvement{0.0};
    double expected_resource_efficiency{0.0};
    double expected_cost_reduction{0.0};
    
    // Execution status
    bool applied{false};
    std::chrono::steady_clock::time_point application_time;
    Status application_status{Status::NOT_INITIALIZED};
    
    // Results
    double actual_throughput_improvement{0.0};
    double actual_latency_improvement{0.0};
    double actual_resource_efficiency{0.0};
    double actual_cost_reduction{0.0};
    
    GraphTopologyOptimization() {
        application_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Self-healing graphs statistics
 */
struct SelfHealingGraphsStats {
    std::atomic<uint64_t> total_bottlenecks_detected{0};
    std::atomic<uint64_t> total_healing_actions{0};
    std::atomic<uint64_t> successful_healing_actions{0};
    std::atomic<uint64_t> failed_healing_actions{0};
    
    // Performance metrics
    std::atomic<double> avg_bottleneck_detection_time_ms{0.0};
    std::atomic<double> avg_healing_action_time_ms{0.0};
    std::atomic<double> avg_performance_improvement{0.0};
    std::atomic<double> avg_latency_reduction{0.0};
    
    // Graph optimization
    std::atomic<uint64_t> total_topology_optimizations{0};
    std::atomic<uint64_t> successful_optimizations{0};
    std::atomic<uint64_t> failed_optimizations{0};
    std::atomic<double> avg_optimization_improvement{0.0};
    
    // Bottleneck types
    std::atomic<uint64_t> cpu_bottlenecks{0};
    std::atomic<uint64_t> memory_bottlenecks{0};
    std::atomic<uint64_t> gpu_bottlenecks{0};
    std::atomic<uint64_t> network_bottlenecks{0};
    std::atomic<uint64_t> disk_bottlenecks{0};
    std::atomic<uint64_t> latency_bottlenecks{0};
    
    // Healing effectiveness
    std::atomic<double> healing_success_rate{0.0};
    std::atomic<double> bottleneck_resolution_rate{0.0};
    std::atomic<double> graph_stability_score{0.0};
    std::atomic<double> auto_remediation_rate{0.0};
    
    SelfHealingGraphsStats() = default;
    
    struct Snapshot {
        uint64_t total_bottlenecks_detected;
        uint64_t total_healing_actions;
        uint64_t successful_healing_actions;
        uint64_t failed_healing_actions;
        double avg_bottleneck_detection_time_ms;
        double avg_healing_action_time_ms;
        double avg_performance_improvement;
        double avg_latency_reduction;
        uint64_t total_topology_optimizations;
        uint64_t successful_optimizations;
        uint64_t failed_optimizations;
        double avg_optimization_improvement;
        uint64_t cpu_bottlenecks;
        uint64_t memory_bottlenecks;
        uint64_t gpu_bottlenecks;
        uint64_t network_bottlenecks;
        uint64_t disk_bottlenecks;
        uint64_t latency_bottlenecks;
        double healing_success_rate;
        double bottleneck_resolution_rate;
        double graph_stability_score;
        double auto_remediation_rate;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_bottlenecks_detected = total_bottlenecks_detected.load();
        snapshot.total_healing_actions = total_healing_actions.load();
        snapshot.successful_healing_actions = successful_healing_actions.load();
        snapshot.failed_healing_actions = failed_healing_actions.load();
        snapshot.avg_bottleneck_detection_time_ms = avg_bottleneck_detection_time_ms.load();
        snapshot.avg_healing_action_time_ms = avg_healing_action_time_ms.load();
        snapshot.avg_performance_improvement = avg_performance_improvement.load();
        snapshot.avg_latency_reduction = avg_latency_reduction.load();
        snapshot.total_topology_optimizations = total_topology_optimizations.load();
        snapshot.successful_optimizations = successful_optimizations.load();
        snapshot.failed_optimizations = failed_optimizations.load();
        snapshot.avg_optimization_improvement = avg_optimization_improvement.load();
        snapshot.cpu_bottlenecks = cpu_bottlenecks.load();
        snapshot.memory_bottlenecks = memory_bottlenecks.load();
        snapshot.gpu_bottlenecks = gpu_bottlenecks.load();
        snapshot.network_bottlenecks = network_bottlenecks.load();
        snapshot.disk_bottlenecks = disk_bottlenecks.load();
        snapshot.latency_bottlenecks = latency_bottlenecks.load();
        snapshot.healing_success_rate = healing_success_rate.load();
        snapshot.bottleneck_resolution_rate = bottleneck_resolution_rate.load();
        snapshot.graph_stability_score = graph_stability_score.load();
        snapshot.auto_remediation_rate = auto_remediation_rate.load();
        return snapshot;
    }
};

/**
 * @brief Self-healing graphs system
 */
class SelfHealingGraphs {
public:
    /**
     * @brief Constructor
     * @param cluster_manager Cluster manager for node information
     */
    explicit SelfHealingGraphs(std::shared_ptr<ClusterManager> cluster_manager);
    
    /**
     * @brief Destructor
     */
    ~SelfHealingGraphs();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Bottleneck detection
    Status DetectBottlenecks();
    Status DetectNodeBottlenecks(const std::string& node_id, std::vector<BottleneckDetectionResult>& bottlenecks);
    Status DetectGraphBottlenecks(const std::string& graph_id, std::vector<BottleneckDetectionResult>& bottlenecks);
    Status RegisterBottleneckDetector(const std::string& graph_id, 
                                     std::function<bool(const GraphPerformanceProfile&)> detector);
    
    // Graph healing
    Status ExecuteHealingAction(const GraphHealingAction& action);
    Status GenerateHealingActions(const BottleneckDetectionResult& bottleneck, 
                                 std::vector<GraphHealingAction>& actions);
    Status AutoRemediateBottleneck(const BottleneckDetectionResult& bottleneck);
    Status ScheduleHealingAction(const GraphHealingAction& action, 
                                std::chrono::milliseconds delay = std::chrono::milliseconds(0));
    
    // Graph optimization
    Status OptimizeGraphTopology(const std::string& graph_id, GraphTopologyOptimization& optimization);
    Status ApplyTopologyOptimization(const GraphTopologyOptimization& optimization);
    Status EvaluateOptimizationImpact(const GraphTopologyOptimization& optimization);
    
    // Performance profiling
    Status UpdateGraphPerformanceProfile(const std::string& graph_id, const GraphPerformanceProfile& profile);
    Status GetGraphPerformanceProfile(const std::string& graph_id, GraphPerformanceProfile& profile);
    Status AnalyzePerformanceTrends(const std::string& graph_id, std::vector<double>& trends);
    
    // Graph rewiring
    Status RewireGraph(const std::string& graph_id, const std::vector<std::string>& new_connections);
    Status MigrateGraphNode(const std::string& graph_id, const std::string& node_id, 
                           const std::string& target_node_id);
    Status SplitGraphNode(const std::string& graph_id, const std::string& node_id, 
                         const std::vector<std::string>& new_nodes);
    Status MergeGraphNodes(const std::string& graph_id, const std::vector<std::string>& node_ids, 
                          const std::string& merged_node_id);
    
    // Load balancing
    Status RebalanceGraphLoad(const std::string& graph_id);
    Status OptimizeNodePlacement(const std::string& graph_id);
    Status AdjustResourceAllocation(const std::string& graph_id, const std::string& node_id, 
                                   const std::map<std::string, double>& resource_requirements);
    
    // Backpressure control
    Status ApplyBackpressure(const std::string& graph_id, const std::string& node_id, double pressure_factor);
    Status ReleaseBackpressure(const std::string& graph_id, const std::string& node_id);
    Status MonitorBackpressure(const std::string& graph_id, std::map<std::string, double>& pressure_levels);
    
    // Statistics and monitoring
    SelfHealingGraphsStats::Snapshot GetStats() const;
    void ResetStats();
    Status GenerateHealingReport();
    
    // Configuration
    void SetAutoHealingEnabled(bool enabled);
    void SetBottleneckDetectionEnabled(bool enabled);
    void SetTopologyOptimizationEnabled(bool enabled);
    void SetBackpressureControlEnabled(bool enabled);
    void SetHealingThreshold(double threshold);

private:
    // Internal detection methods
    bool DetectCPUBottleneck(const std::string& node_id, const GraphPerformanceProfile& profile);
    bool DetectMemoryBottleneck(const std::string& node_id, const GraphPerformanceProfile& profile);
    bool DetectGPUBottleneck(const std::string& node_id, const GraphPerformanceProfile& profile);
    bool DetectNetworkBottleneck(const std::string& node_id, const GraphPerformanceProfile& profile);
    bool DetectLatencyBottleneck(const std::string& node_id, const GraphPerformanceProfile& profile);
    
    // Healing action generation
    std::vector<GraphHealingAction> GenerateNodeMigrationActions(const BottleneckDetectionResult& bottleneck);
    std::vector<GraphHealingAction> GenerateResourceAllocationActions(const BottleneckDetectionResult& bottleneck);
    std::vector<GraphHealingAction> GenerateLoadBalancingActions(const BottleneckDetectionResult& bottleneck);
    std::vector<GraphHealingAction> GenerateTopologyOptimizationActions(const BottleneckDetectionResult& bottleneck);
    
    // Graph analysis algorithms
    Status AnalyzeGraphTopology(const std::string& graph_id, std::map<std::string, double>& node_metrics);
    Status CalculateGraphCriticalPath(const std::string& graph_id, std::vector<std::string>& critical_path);
    Status IdentifyGraphBottlenecks(const std::string& graph_id, std::vector<std::string>& bottleneck_nodes);
    Status OptimizeGraphLayout(const std::string& graph_id, std::map<std::string, std::string>& node_placement);
    
    // Performance optimization
    Status OptimizeNodeExecution(const std::string& graph_id, const std::string& node_id);
    Status OptimizeEdgeCommunication(const std::string& graph_id, const std::string& edge_id);
    Status OptimizeResourceUtilization(const std::string& graph_id);
    Status OptimizeBatchProcessing(const std::string& graph_id);
    
    // Threading and synchronization
    void BottleneckDetectionThread();
    void HealingExecutionThread();
    void PerformanceProfilingThread();
    void TopologyOptimizationThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> auto_healing_enabled_{true};
    std::atomic<bool> bottleneck_detection_enabled_{true};
    std::atomic<bool> topology_optimization_enabled_{true};
    std::atomic<bool> backpressure_control_enabled_{true};
    std::atomic<double> healing_threshold_{0.8};
    
    // Dependencies
    std::shared_ptr<ClusterManager> cluster_manager_;
    
    // Bottleneck detection state
    mutable std::mutex detection_mutex_;
    std::map<std::string, std::function<bool(const GraphPerformanceProfile&)>> bottleneck_detectors_;
    std::vector<BottleneckDetectionResult> detected_bottlenecks_;
    std::map<std::string, std::vector<BottleneckDetectionResult>> node_bottlenecks_;
    
    // Healing actions
    mutable std::mutex healing_mutex_;
    std::queue<GraphHealingAction> pending_healing_actions_;
    std::map<std::string, GraphHealingAction> active_healing_actions_;
    std::vector<GraphHealingAction> healing_history_;
    
    // Performance profiles
    mutable std::mutex profile_mutex_;
    std::map<std::string, GraphPerformanceProfile> graph_profiles_;
    std::map<std::string, std::vector<double>> performance_trends_;
    
    // Topology optimizations
    mutable std::mutex optimization_mutex_;
    std::queue<GraphTopologyOptimization> pending_optimizations_;
    std::map<std::string, GraphTopologyOptimization> active_optimizations_;
    std::vector<GraphTopologyOptimization> optimization_history_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    SelfHealingGraphsStats stats_;
    
    // Threading
    std::thread bottleneck_detection_thread_;
    std::thread healing_execution_thread_;
    std::thread performance_profiling_thread_;
    std::thread topology_optimization_thread_;
    
    std::condition_variable detection_cv_;
    std::condition_variable healing_cv_;
    std::condition_variable profiling_cv_;
    std::condition_variable optimization_cv_;
    
    mutable std::mutex detection_cv_mutex_;
    mutable std::mutex healing_cv_mutex_;
    mutable std::mutex profiling_cv_mutex_;
    mutable std::mutex optimization_cv_mutex_;
};

} // namespace edge_ai
