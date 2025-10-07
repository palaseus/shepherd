/**
 * @file graph_partitioner.h
 * @brief Graph partitioning and placement strategies for distributed execution
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <atomic>
#include <mutex>
#include "graph/graph_types.h"
#include "graph/graph.h"
#include "distributed/cluster_types.h"
#include "core/types.h"

namespace edge_ai {

/**
 * @brief Partitioning strategy enumeration
 */
enum class PartitioningStrategy {
    ROUND_ROBIN = 0,           // Simple round-robin distribution
    LOAD_BALANCED = 1,         // Load-based balancing
    CAPABILITY_AWARE = 2,      // Based on node capabilities
    LATENCY_OPTIMIZED = 3,     // Minimize end-to-end latency
    THROUGHPUT_OPTIMIZED = 4,  // Maximize overall throughput
    ML_BASED = 5,              // ML-driven placement
    CUSTOM = 6                 // Custom strategy
};

/**
 * @brief Graph partition representing a subgraph for execution
 */
struct GraphPartition {
    std::string partition_id;
    std::string assigned_node_id;
    std::set<std::string> node_ids;           // Nodes in this partition
    std::set<std::string> input_edges;        // Input edges from other partitions
    std::set<std::string> output_edges;       // Output edges to other partitions
    std::set<std::string> internal_edges;     // Internal edges within partition
    
    // Resource requirements
    NodeCapabilities required_capabilities;
    uint64_t estimated_memory_mb{0};
    uint32_t estimated_compute_ops{0};
    double estimated_latency_ms{0.0};
    
    // Dependencies
    std::set<std::string> depends_on_partitions;
    std::set<std::string> dependent_partitions;
    
    // Execution metadata
    std::chrono::steady_clock::time_point created_time;
    std::atomic<bool> is_executing{false};
    std::atomic<bool> is_completed{false};
    std::atomic<bool> has_failed{false};
    
    GraphPartition() = default;
    GraphPartition(const std::string& id, const std::string& node_id)
        : partition_id(id), assigned_node_id(node_id) {
        created_time = std::chrono::steady_clock::now();
    }
    
    // Disable copy, enable move
    GraphPartition(const GraphPartition&) = delete;
    GraphPartition& operator=(const GraphPartition&) = delete;
    GraphPartition(GraphPartition&&) = default;
    GraphPartition& operator=(GraphPartition&&) = default;
};

/**
 * @brief Partitioning result containing all partitions and metadata
 */
struct PartitioningResult {
    std::string graph_id;
    std::vector<std::unique_ptr<GraphPartition>> partitions;
    std::map<std::string, std::string> node_to_partition_map;
    std::map<std::string, std::string> partition_to_node_map;
    
    // Execution plan
    std::vector<std::string> execution_order;  // Topological order of partitions
    std::map<std::string, std::set<std::string>> partition_dependencies;
    
    // Performance estimates
    double estimated_total_latency_ms{0.0};
    double estimated_throughput_ops_per_sec{0.0};
    uint64_t estimated_total_memory_mb{0};
    
    // Partitioning metadata
    PartitioningStrategy strategy_used;
    std::chrono::steady_clock::time_point created_time;
    std::chrono::milliseconds partitioning_time_ms{0};
    
    PartitioningResult() = default;
    
    // Disable copy, enable move
    PartitioningResult(const PartitioningResult&) = delete;
    PartitioningResult& operator=(const PartitioningResult&) = delete;
    PartitioningResult(PartitioningResult&&) = default;
    PartitioningResult& operator=(PartitioningResult&&) = default;
};

/**
 * @brief Partitioning configuration and constraints
 */
struct PartitioningConfig {
    PartitioningStrategy strategy{PartitioningStrategy::LOAD_BALANCED};
    uint32_t max_partitions{10};
    uint32_t min_nodes_per_partition{1};
    uint32_t max_nodes_per_partition{50};
    
    // Resource constraints
    uint64_t max_memory_per_partition_mb{1024};
    uint32_t max_compute_ops_per_partition{1000000};
    double max_latency_per_partition_ms{100.0};
    
    // Load balancing parameters
    double load_balance_threshold{0.2};  // 20% load difference threshold
    bool enable_dynamic_rebalancing{true};
    uint32_t rebalancing_interval_ms{5000};
    
    // ML-based partitioning
    bool enable_ml_placement{false};
    std::string ml_model_path;
    double ml_confidence_threshold{0.8};
    
    // Custom strategy parameters
    std::map<std::string, std::string> custom_parameters;
    
    PartitioningConfig() = default;
};

/**
 * @brief Partitioning statistics and metrics
 */
struct PartitioningStats {
    std::atomic<uint32_t> total_partitionings{0};
    std::atomic<uint32_t> successful_partitionings{0};
    std::atomic<uint32_t> failed_partitionings{0};
    std::atomic<uint64_t> total_partitioning_time_ms{0};
    std::atomic<uint32_t> total_partitions_created{0};
    std::atomic<uint32_t> rebalancing_events{0};
    
    // Performance metrics
    std::atomic<double> avg_partitioning_time_ms{0.0};
    std::atomic<double> avg_partitions_per_graph{0.0};
    std::atomic<double> avg_load_balance_score{0.0};
    std::atomic<double> avg_latency_improvement_percent{0.0};
    
    PartitioningStats() = default;
    
    struct Snapshot {
        uint32_t total_partitionings;
        uint32_t successful_partitionings;
        uint32_t failed_partitionings;
        uint64_t total_partitioning_time_ms;
        uint32_t total_partitions_created;
        uint32_t rebalancing_events;
        double avg_partitioning_time_ms;
        double avg_partitions_per_graph;
        double avg_load_balance_score;
        double avg_latency_improvement_percent;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_partitionings = total_partitionings.load();
        snapshot.successful_partitionings = successful_partitionings.load();
        snapshot.failed_partitionings = failed_partitionings.load();
        snapshot.total_partitioning_time_ms = total_partitioning_time_ms.load();
        snapshot.total_partitions_created = total_partitions_created.load();
        snapshot.rebalancing_events = rebalancing_events.load();
        snapshot.avg_partitioning_time_ms = avg_partitioning_time_ms.load();
        snapshot.avg_partitions_per_graph = avg_partitions_per_graph.load();
        snapshot.avg_load_balance_score = avg_load_balance_score.load();
        snapshot.avg_latency_improvement_percent = avg_latency_improvement_percent.load();
        return snapshot;
    }
};

/**
 * @brief Graph partitioner for distributed execution
 */
class GraphPartitioner {
public:
    /**
     * @brief Constructor
     * @param config Partitioning configuration
     */
    explicit GraphPartitioner(const PartitioningConfig& config);
    
    /**
     * @brief Destructor
     */
    ~GraphPartitioner() = default;
    
    // Core partitioning methods
    std::unique_ptr<PartitioningResult> PartitionGraph(
        std::shared_ptr<Graph> graph,
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities);
    
    std::unique_ptr<PartitioningResult> RebalancePartitions(
        const PartitioningResult& current_result,
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities,
        const std::map<std::string, NodeHealth>& node_health);
    
    // Strategy-specific partitioning
    std::unique_ptr<PartitioningResult> PartitionByRoundRobin(
        std::shared_ptr<Graph> graph,
        const std::vector<std::string>& available_nodes);
    
    std::unique_ptr<PartitioningResult> PartitionByLoadBalancing(
        std::shared_ptr<Graph> graph,
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities);
    
    std::unique_ptr<PartitioningResult> PartitionByCapability(
        std::shared_ptr<Graph> graph,
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities);
    
    std::unique_ptr<PartitioningResult> PartitionByLatency(
        std::shared_ptr<Graph> graph,
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities);
    
    std::unique_ptr<PartitioningResult> PartitionByML(
        std::shared_ptr<Graph> graph,
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities);
    
    // Analysis and optimization
    double CalculateLoadBalanceScore(const PartitioningResult& result) const;
    double CalculateLatencyEstimate(const PartitioningResult& result) const;
    double CalculateThroughputEstimate(const PartitioningResult& result) const;
    
    // Configuration and statistics
    PartitioningConfig GetConfig() const;
    Status UpdateConfig(const PartitioningConfig& config);
    PartitioningStats::Snapshot GetStats() const;
    void ResetStats();
    
    // Validation
    bool ValidatePartitioning(const PartitioningResult& result) const;
    std::vector<std::string> GetPartitioningIssues(const PartitioningResult& result) const;

private:
    // Internal helper methods
    std::vector<std::string> GetExecutionOrder(std::shared_ptr<Graph> graph) const;
    std::set<std::string> GetNodeDependencies(std::shared_ptr<Graph> graph, const std::string& node_id) const;
    NodeCapabilities EstimateNodeRequirements(std::shared_ptr<Graph> graph, const std::string& node_id) const;
    uint64_t EstimateMemoryRequirements(std::shared_ptr<Graph> graph, const std::string& node_id) const;
    uint32_t EstimateComputeRequirements(std::shared_ptr<Graph> graph, const std::string& node_id) const;
    double EstimateLatency(std::shared_ptr<Graph> graph, const std::string& node_id) const;
    
    // Load balancing helpers
    std::string SelectNodeForPartition(
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities,
        const NodeCapabilities& requirements) const;
    
    void UpdateNodeLoad(const std::string& node_id, const NodeCapabilities& requirements);
    double CalculateNodeLoad(const std::string& node_id) const;
    
    // ML-based placement helpers
    std::string PredictOptimalNode(
        const std::string& node_id,
        const std::vector<std::string>& available_nodes,
        const std::map<std::string, NodeCapabilities>& node_capabilities) const;
    
    // Configuration
    PartitioningConfig config_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    PartitioningStats stats_;
    
    // Node load tracking (for load balancing)
    mutable std::mutex node_load_mutex_;
    std::map<std::string, NodeCapabilities> node_current_load_;
    std::map<std::string, uint32_t> node_task_count_;
};

} // namespace edge_ai
