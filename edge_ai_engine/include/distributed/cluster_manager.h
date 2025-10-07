/**
 * @file cluster_manager.h
 * @brief Cluster management and node orchestration
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "distributed/cluster_types.h"
#include "core/types.h"

namespace edge_ai {

/**
 * @brief Cluster manager for distributed node orchestration
 */
class ClusterManager {
public:
    using NodeEventHandler = std::function<void(const ClusterEvent&)>;
    using HealthCheckCallback = std::function<bool(const std::string&)>;
    
    /**
     * @brief Constructor
     * @param config Cluster configuration
     */
    explicit ClusterManager(const ClusterConfig& config);
    
    /**
     * @brief Destructor
     */
    ~ClusterManager();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Node management
    Status RegisterNode(const std::string& node_id, const std::string& address, 
                       uint16_t port, const NodeCapabilities& capabilities);
    Status UnregisterNode(const std::string& node_id);
    Status UpdateNodeCapabilities(const std::string& node_id, const NodeCapabilities& capabilities);
    Status UpdateNodeHealth(const std::string& node_id, const NodeHealth& health);
    
    // Node discovery and querying
    std::vector<std::string> GetActiveNodes() const;
    std::vector<std::string> GetNodesByCapability(const NodeCapabilities& required_caps) const;
    std::shared_ptr<ClusterNode> GetNode(const std::string& node_id) const;
    bool IsNodeActive(const std::string& node_id) const;
    
    // Load balancing and selection
    std::string SelectOptimalNode(const NodeCapabilities& requirements) const;
    std::vector<std::string> SelectNodesForLoadBalancing(uint32_t count) const;
    std::string SelectNodeForMigration(const std::string& from_node_id) const;
    
    // Health monitoring
    Status StartHealthMonitoring();
    Status StopHealthMonitoring();
    Status CheckNodeHealth(const std::string& node_id);
    Status MarkNodeFailed(const std::string& node_id, const std::string& reason);
    Status MarkNodeRecovered(const std::string& node_id);
    
    // Event handling
    Status RegisterEventHandler(ClusterEventType type, NodeEventHandler handler);
    Status UnregisterEventHandler(ClusterEventType type);
    void EmitEvent(const ClusterEvent& event);
    
    // Statistics and monitoring
    ClusterStats::Snapshot GetStats() const;
    void ResetStats();
    
    // Configuration
    ClusterConfig GetConfig() const;
    Status UpdateConfig(const ClusterConfig& config);
    
    // Heartbeat management
    Status SendHeartbeat(const std::string& node_id);
    Status ProcessHeartbeat(const std::string& node_id, const NodeHealth& health);
    
    // Cluster topology
    std::vector<std::string> GetClusterTopology() const;
    Status RebalanceCluster();
    bool IsClusterOverloaded() const;
    bool IsClusterUnderloaded() const;

private:
    // Internal methods
    void HealthMonitoringThread();
    void ProcessNodeTimeout(const std::string& node_id);
    void UpdateClusterStats();
    void NotifyEventHandlers(const ClusterEvent& event);
    
    // Node selection algorithms
    std::string SelectNodeByLoad(const NodeCapabilities& requirements) const;
    std::string SelectNodeByCapability(const NodeCapabilities& requirements) const;
    std::string SelectNodeByLatency(const NodeCapabilities& requirements) const;
    
    // Load balancing algorithms
    std::vector<std::string> RoundRobinSelection(uint32_t count) const;
    std::vector<std::string> LoadBasedSelection(uint32_t count) const;
    std::vector<std::string> CapabilityBasedSelection(uint32_t count, const NodeCapabilities& caps) const;
    
    // Configuration
    ClusterConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Node management
    mutable std::mutex nodes_mutex_;
    std::map<std::string, std::shared_ptr<ClusterNode>> nodes_;
    std::set<std::string> active_nodes_;
    std::set<std::string> failed_nodes_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    ClusterStats stats_;
    
    // Event handling
    mutable std::mutex event_handlers_mutex_;
    std::map<ClusterEventType, std::vector<NodeEventHandler>> event_handlers_;
    
    // Health monitoring
    std::thread health_monitoring_thread_;
    std::atomic<bool> health_monitoring_active_{false};
    std::condition_variable health_monitoring_cv_;
    mutable std::mutex health_monitoring_mutex_;
    
    // Load balancing state
    mutable std::mutex load_balancing_mutex_;
    mutable std::atomic<uint32_t> round_robin_index_{0};
    
    // Health check callback
    HealthCheckCallback health_check_callback_;
};

} // namespace edge_ai
