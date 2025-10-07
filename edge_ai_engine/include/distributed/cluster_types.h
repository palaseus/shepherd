/**
 * @file cluster_types.h
 * @brief Core data structures for distributed cluster management
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "core/types.h"
#include "backend/execution_backend.h"

namespace edge_ai {

/**
 * @brief Node capabilities and resources
 */
struct NodeCapabilities {
    // Compute resources
    std::atomic<uint32_t> cpu_cores{0};
    std::atomic<uint64_t> memory_mb{0};
    std::atomic<uint64_t> gpu_memory_mb{0};
    std::atomic<bool> has_gpu{false};
    std::atomic<bool> has_npu{false};
    
    // Network capabilities
    std::atomic<uint32_t> bandwidth_mbps{0};
    std::atomic<uint32_t> latency_ms{0};
    
    // Model support
    std::set<ModelType> supported_models;
    std::set<BackendType> supported_backends;
    
    // Performance characteristics
    std::atomic<double> compute_efficiency{1.0};  // Relative to baseline
    std::atomic<double> memory_efficiency{1.0};
    
    NodeCapabilities() = default;
    
    // Disable copy, enable move
    NodeCapabilities(const NodeCapabilities&) = delete;
    NodeCapabilities& operator=(const NodeCapabilities&) = delete;
    
    // Custom move constructor/assignment for atomic members
    NodeCapabilities(NodeCapabilities&& other) noexcept
        : cpu_cores(other.cpu_cores.load())
        , memory_mb(other.memory_mb.load())
        , gpu_memory_mb(other.gpu_memory_mb.load())
        , has_gpu(other.has_gpu.load())
        , has_npu(other.has_npu.load())
        , bandwidth_mbps(other.bandwidth_mbps.load())
        , latency_ms(other.latency_ms.load())
        , supported_models(std::move(other.supported_models))
        , supported_backends(std::move(other.supported_backends))
        , compute_efficiency(other.compute_efficiency.load())
        , memory_efficiency(other.memory_efficiency.load()) {}
    
    NodeCapabilities& operator=(NodeCapabilities&& other) noexcept {
        if (this != &other) {
            cpu_cores.store(other.cpu_cores.load());
            memory_mb.store(other.memory_mb.load());
            gpu_memory_mb.store(other.gpu_memory_mb.load());
            has_gpu.store(other.has_gpu.load());
            has_npu.store(other.has_npu.load());
            bandwidth_mbps.store(other.bandwidth_mbps.load());
            latency_ms.store(other.latency_ms.load());
            supported_models = std::move(other.supported_models);
            supported_backends = std::move(other.supported_backends);
            compute_efficiency.store(other.compute_efficiency.load());
            memory_efficiency.store(other.memory_efficiency.load());
        }
        return *this;
    }
    
    // Snapshot for thread-safe access
    struct Snapshot {
        uint32_t cpu_cores;
        uint64_t memory_mb;
        uint64_t gpu_memory_mb;
        bool has_gpu;
        bool has_npu;
        uint32_t bandwidth_mbps;
        uint32_t latency_ms;
        std::set<ModelType> supported_models;
        std::set<BackendType> supported_backends;
        double compute_efficiency;
        double memory_efficiency;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.cpu_cores = cpu_cores.load();
        snapshot.memory_mb = memory_mb.load();
        snapshot.gpu_memory_mb = gpu_memory_mb.load();
        snapshot.has_gpu = has_gpu.load();
        snapshot.has_npu = has_npu.load();
        snapshot.bandwidth_mbps = bandwidth_mbps.load();
        snapshot.latency_ms = latency_ms.load();
        snapshot.supported_models = supported_models;
        snapshot.supported_backends = supported_backends;
        snapshot.compute_efficiency = compute_efficiency.load();
        snapshot.memory_efficiency = memory_efficiency.load();
        return snapshot;
    }
};

/**
 * @brief Cluster node status enumeration
 */
enum class ClusterNodeStatus {
    UNKNOWN = 0,
    REGISTERING = 1,
    ACTIVE = 2,
    BUSY = 3,
    OVERLOADED = 4,
    FAILED = 5,
    MAINTENANCE = 6,
    SHUTTING_DOWN = 7
};

/**
 * @brief Node health and status information
 */
struct NodeHealth {
    std::atomic<ClusterNodeStatus> status{static_cast<ClusterNodeStatus>(0)};
    std::atomic<uint64_t> last_heartbeat_ms{0};
    std::atomic<double> cpu_usage_percent{0.0};
    std::atomic<double> memory_usage_percent{0.0};
    std::atomic<double> gpu_usage_percent{0.0};
    std::atomic<uint32_t> active_tasks{0};
    std::atomic<uint32_t> failed_tasks{0};
    std::atomic<double> avg_latency_ms{0.0};
    std::atomic<double> throughput_ops_per_sec{0.0};
    
    // Health metrics
    std::atomic<uint32_t> consecutive_failures{0};
    std::atomic<uint64_t> total_uptime_ms{0};
    std::atomic<uint64_t> total_downtime_ms{0};
    
    NodeHealth() = default;
    
    // Disable copy, enable move
    NodeHealth(const NodeHealth&) = delete;
    NodeHealth& operator=(const NodeHealth&) = delete;
    
    // Custom move constructor/assignment for atomic members
    NodeHealth(NodeHealth&& other) noexcept
        : status(other.status.load())
        , last_heartbeat_ms(other.last_heartbeat_ms.load())
        , cpu_usage_percent(other.cpu_usage_percent.load())
        , memory_usage_percent(other.memory_usage_percent.load())
        , gpu_usage_percent(other.gpu_usage_percent.load())
        , active_tasks(other.active_tasks.load())
        , failed_tasks(other.failed_tasks.load())
        , avg_latency_ms(other.avg_latency_ms.load())
        , throughput_ops_per_sec(other.throughput_ops_per_sec.load())
        , consecutive_failures(other.consecutive_failures.load())
        , total_uptime_ms(other.total_uptime_ms.load())
        , total_downtime_ms(other.total_downtime_ms.load()) {}
    
    NodeHealth& operator=(NodeHealth&& other) noexcept {
        if (this != &other) {
            status.store(other.status.load());
            last_heartbeat_ms.store(other.last_heartbeat_ms.load());
            cpu_usage_percent.store(other.cpu_usage_percent.load());
            memory_usage_percent.store(other.memory_usage_percent.load());
            gpu_usage_percent.store(other.gpu_usage_percent.load());
            active_tasks.store(other.active_tasks.load());
            failed_tasks.store(other.failed_tasks.load());
            avg_latency_ms.store(other.avg_latency_ms.load());
            throughput_ops_per_sec.store(other.throughput_ops_per_sec.load());
            consecutive_failures.store(other.consecutive_failures.load());
            total_uptime_ms.store(other.total_uptime_ms.load());
            total_downtime_ms.store(other.total_downtime_ms.load());
        }
        return *this;
    }
    
    struct Snapshot {
        ClusterNodeStatus status;
        uint64_t last_heartbeat_ms;
        double cpu_usage_percent;
        double memory_usage_percent;
        double gpu_usage_percent;
        uint32_t active_tasks;
        uint32_t failed_tasks;
        double avg_latency_ms;
        double throughput_ops_per_sec;
        uint32_t consecutive_failures;
        uint64_t total_uptime_ms;
        uint64_t total_downtime_ms;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.status = status.load();
        snapshot.last_heartbeat_ms = last_heartbeat_ms.load();
        snapshot.cpu_usage_percent = cpu_usage_percent.load();
        snapshot.memory_usage_percent = memory_usage_percent.load();
        snapshot.gpu_usage_percent = gpu_usage_percent.load();
        snapshot.active_tasks = active_tasks.load();
        snapshot.failed_tasks = failed_tasks.load();
        snapshot.avg_latency_ms = avg_latency_ms.load();
        snapshot.throughput_ops_per_sec = throughput_ops_per_sec.load();
        snapshot.consecutive_failures = consecutive_failures.load();
        snapshot.total_uptime_ms = total_uptime_ms.load();
        snapshot.total_downtime_ms = total_downtime_ms.load();
        return snapshot;
    }
};

/**
 * @brief Cluster node representation
 */
struct ClusterNode {
    std::string node_id;
    std::string address;
    uint16_t port;
    NodeCapabilities capabilities;
    NodeHealth health;
    std::chrono::steady_clock::time_point registration_time;
    std::chrono::steady_clock::time_point last_seen;
    
    // Node metadata
    std::string hostname;
    std::string os_version;
    std::string hardware_info;
    std::map<std::string, std::string> custom_attributes;
    
    ClusterNode() = default;
    ClusterNode(const std::string& id, const std::string& addr, uint16_t p)
        : node_id(id), address(addr), port(p) {
        registration_time = std::chrono::steady_clock::now();
        last_seen = registration_time;
    }
    
    // Disable copy, enable move
    ClusterNode(const ClusterNode&) = delete;
    ClusterNode& operator=(const ClusterNode&) = delete;
    ClusterNode(ClusterNode&&) = default;
    ClusterNode& operator=(ClusterNode&&) = default;
};

/**
 * @brief Cluster topology and configuration
 */
struct ClusterConfig {
    std::string cluster_id;
    std::string coordinator_node_id;
    uint32_t heartbeat_interval_ms{1000};
    uint32_t node_timeout_ms{5000};
    uint32_t max_nodes{100};
    bool enable_auto_discovery{true};
    bool enable_load_balancing{true};
    bool enable_fault_tolerance{true};
    
    // Network settings
    uint32_t max_bandwidth_mbps{1000};
    uint32_t max_latency_ms{100};
    
    // Performance thresholds
    double max_cpu_usage_percent{80.0};
    double max_memory_usage_percent{85.0};
    double max_gpu_usage_percent{90.0};
    
    ClusterConfig() = default;
};

/**
 * @brief Cluster statistics and monitoring
 */
struct ClusterStats {
    std::atomic<uint32_t> total_nodes{0};
    std::atomic<uint32_t> active_nodes{0};
    std::atomic<uint32_t> failed_nodes{0};
    std::atomic<uint32_t> total_tasks{0};
    std::atomic<uint32_t> active_tasks{0};
    std::atomic<uint32_t> completed_tasks{0};
    std::atomic<uint32_t> failed_tasks{0};
    std::atomic<uint64_t> total_bandwidth_used_mbps{0};
    std::atomic<double> avg_cluster_latency_ms{0.0};
    std::atomic<double> cluster_throughput_ops_per_sec{0.0};
    
    // Load balancing metrics
    std::atomic<uint32_t> rebalancing_events{0};
    std::atomic<uint32_t> migration_events{0};
    std::atomic<uint64_t> total_migration_time_ms{0};
    
    ClusterStats() = default;
    
    struct Snapshot {
        uint32_t total_nodes;
        uint32_t active_nodes;
        uint32_t failed_nodes;
        uint32_t total_tasks;
        uint32_t active_tasks;
        uint32_t completed_tasks;
        uint32_t failed_tasks;
        uint64_t total_bandwidth_used_mbps;
        double avg_cluster_latency_ms;
        double cluster_throughput_ops_per_sec;
        uint32_t rebalancing_events;
        uint32_t migration_events;
        uint64_t total_migration_time_ms;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_nodes = total_nodes.load();
        snapshot.active_nodes = active_nodes.load();
        snapshot.failed_nodes = failed_nodes.load();
        snapshot.total_tasks = total_tasks.load();
        snapshot.active_tasks = active_tasks.load();
        snapshot.completed_tasks = completed_tasks.load();
        snapshot.failed_tasks = failed_tasks.load();
        snapshot.total_bandwidth_used_mbps = total_bandwidth_used_mbps.load();
        snapshot.avg_cluster_latency_ms = avg_cluster_latency_ms.load();
        snapshot.cluster_throughput_ops_per_sec = cluster_throughput_ops_per_sec.load();
        snapshot.rebalancing_events = rebalancing_events.load();
        snapshot.migration_events = migration_events.load();
        snapshot.total_migration_time_ms = total_migration_time_ms.load();
        return snapshot;
    }
};


/**
 * @brief Cluster event types
 */
enum class ClusterEventType {
    NODE_REGISTERED = 0,
    NODE_UNREGISTERED = 1,
    NODE_FAILED = 2,
    NODE_RECOVERED = 3,
    NODE_OVERLOADED = 4,
    NODE_UNDERLOADED = 5,
    REBALANCING_STARTED = 6,
    REBALANCING_COMPLETED = 7,
    MIGRATION_STARTED = 8,
    MIGRATION_COMPLETED = 9,
    CLUSTER_OVERLOADED = 10,
    CLUSTER_UNDERLOADED = 11
};

/**
 * @brief Cluster event for monitoring and notifications
 */
struct ClusterEvent {
    ClusterEventType type;
    std::string node_id;
    std::string message;
    std::chrono::steady_clock::time_point timestamp;
    std::map<std::string, std::string> metadata;
    
    ClusterEvent() = default;
    ClusterEvent(ClusterEventType t, const std::string& nid, const std::string& msg)
        : type(t), node_id(nid), message(msg), timestamp(std::chrono::steady_clock::now()) {}
};

} // namespace edge_ai
