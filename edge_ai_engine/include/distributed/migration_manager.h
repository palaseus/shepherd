/**
 * @file migration_manager.h
 * @brief Real-time graph migration and state transfer management
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
#include <future>
#include "graph/graph_types.h"
#include "distributed/cluster_types.h"
#include "distributed/graph_partitioner.h"
#include "core/types.h"

namespace edge_ai {

/**
 * @brief Migration strategy enumeration
 */
enum class MigrationStrategy {
    IMMEDIATE = 0,          // Stop execution, migrate, restart
    GRACEFUL = 1,           // Wait for current execution to complete
    HOT_STANDBY = 2,        // Run on both nodes, switch when ready
    STREAMING = 3,          // Migrate while streaming data
    CUSTOM = 4              // Custom migration strategy
};

/**
 * @brief Migration state enumeration
 */
enum class MigrationState {
    PENDING = 0,
    PREPARING = 1,
    TRANSFERRING = 2,
    VALIDATING = 3,
    SWITCHING = 4,
    COMPLETED = 5,
    FAILED = 6,
    CANCELLED = 7
};

/**
 * @brief Migration request for moving partitions between nodes
 */
struct MigrationRequest {
    std::string migration_id;
    std::string partition_id;
    std::string source_node_id;
    std::string target_node_id;
    MigrationStrategy strategy{MigrationStrategy::GRACEFUL};
    
    // Migration metadata
    std::chrono::steady_clock::time_point request_time;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point completion_time;
    std::atomic<MigrationState> state{MigrationState::PENDING};
    
    // Migration parameters
    uint32_t timeout_ms{30000};
    bool preserve_state{true};
    bool validate_after_migration{true};
    bool rollback_on_failure{true};
    
    // State information
    std::map<std::string, std::vector<uint8_t>> partition_state;
    
    // Custom attributes for internal use
    std::map<std::string, std::string> custom_attributes;
    std::map<std::string, std::vector<uint8_t>> execution_context;
    std::map<std::string, std::vector<uint8_t>> stream_buffers;
    
    // Progress tracking
    std::atomic<double> progress_percent{0.0};
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint64_t> total_bytes{0};
    
    // Error handling
    std::string error_message;
    std::string failure_reason;
    std::atomic<uint32_t> retry_count{0};
    std::atomic<uint32_t> max_retries{3};
    
    MigrationRequest() = default;
    MigrationRequest(const std::string& id, const std::string& part_id,
                    const std::string& source, const std::string& target)
        : migration_id(id), partition_id(part_id), source_node_id(source), target_node_id(target) {
        request_time = std::chrono::steady_clock::now();
    }
    
    // Disable copy, enable move
    MigrationRequest(const MigrationRequest&) = delete;
    MigrationRequest& operator=(const MigrationRequest&) = delete;
    MigrationRequest(MigrationRequest&&) = default;
    MigrationRequest& operator=(MigrationRequest&&) = default;
};

/**
 * @brief Migration result containing outcome and metrics
 */
struct MigrationResult {
    std::string migration_id;
    std::string partition_id;
    Status migration_status{Status::NOT_INITIALIZED};
    MigrationState final_state{MigrationState::FAILED};
    
    // Timing information
    std::chrono::milliseconds total_time_ms{0};
    std::chrono::milliseconds preparation_time_ms{0};
    std::chrono::milliseconds transfer_time_ms{0};
    std::chrono::milliseconds validation_time_ms{0};
    std::chrono::milliseconds switching_time_ms{0};
    
    // Performance metrics
    uint64_t bytes_transferred{0};
    double transfer_rate_mbps{0.0};
    uint32_t retry_count{0};
    double downtime_ms{0.0};
    
    // Validation results
    bool state_consistency_valid{false};
    bool performance_acceptable{false};
    std::vector<std::string> validation_errors;
    
    // Error information
    std::string error_message;
    std::string stack_trace;
    
    MigrationResult() = default;
};

/**
 * @brief Migration manager configuration
 */
struct MigrationManagerConfig {
    // Migration parameters
    MigrationStrategy default_strategy{MigrationStrategy::GRACEFUL};
    uint32_t default_timeout_ms{30000};
    uint32_t max_concurrent_migrations{5};
    bool enable_parallel_migrations{true};
    
    // State preservation
    bool enable_state_checkpointing{true};
    uint32_t checkpoint_interval_ms{1000};
    bool enable_incremental_checkpoints{true};
    
    // Validation
    bool enable_pre_migration_validation{true};
    bool enable_post_migration_validation{true};
    bool enable_performance_validation{true};
    double performance_degradation_threshold{0.1};  // 10% max degradation
    
    // Fault tolerance
    bool enable_rollback_on_failure{true};
    uint32_t max_retries{3};
    uint32_t retry_delay_ms{2000};
    
    // Performance optimization
    bool enable_data_compression{true};
    bool enable_parallel_transfer{true};
    uint32_t transfer_chunk_size_kb{64};
    
    MigrationManagerConfig() = default;
};

/**
 * @brief Migration manager statistics
 */
struct MigrationManagerStats {
    std::atomic<uint32_t> total_migrations_requested{0};
    std::atomic<uint32_t> total_migrations_completed{0};
    std::atomic<uint32_t> total_migrations_failed{0};
    std::atomic<uint32_t> total_migrations_cancelled{0};
    std::atomic<uint32_t> total_retries{0};
    
    // Performance metrics
    std::atomic<double> avg_migration_time_ms{0.0};
    std::atomic<double> avg_transfer_rate_mbps{0.0};
    std::atomic<double> avg_downtime_ms{0.0};
    std::atomic<uint64_t> total_bytes_transferred{0};
    
    // Strategy usage
    std::atomic<uint32_t> immediate_migrations{0};
    std::atomic<uint32_t> graceful_migrations{0};
    std::atomic<uint32_t> hot_standby_migrations{0};
    std::atomic<uint32_t> streaming_migrations{0};
    
    // Validation metrics
    std::atomic<uint32_t> validation_failures{0};
    std::atomic<uint32_t> rollback_events{0};
    std::atomic<uint32_t> performance_degradations{0};
    
    MigrationManagerStats() = default;
    
    struct Snapshot {
        uint32_t total_migrations_requested;
        uint32_t total_migrations_completed;
        uint32_t total_migrations_failed;
        uint32_t total_migrations_cancelled;
        uint32_t total_retries;
        double avg_migration_time_ms;
        double avg_transfer_rate_mbps;
        double avg_downtime_ms;
        uint64_t total_bytes_transferred;
        uint32_t immediate_migrations;
        uint32_t graceful_migrations;
        uint32_t hot_standby_migrations;
        uint32_t streaming_migrations;
        uint32_t validation_failures;
        uint32_t rollback_events;
        uint32_t performance_degradations;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_migrations_requested = total_migrations_requested.load();
        snapshot.total_migrations_completed = total_migrations_completed.load();
        snapshot.total_migrations_failed = total_migrations_failed.load();
        snapshot.total_migrations_cancelled = total_migrations_cancelled.load();
        snapshot.total_retries = total_retries.load();
        snapshot.avg_migration_time_ms = avg_migration_time_ms.load();
        snapshot.avg_transfer_rate_mbps = avg_transfer_rate_mbps.load();
        snapshot.avg_downtime_ms = avg_downtime_ms.load();
        snapshot.total_bytes_transferred = total_bytes_transferred.load();
        snapshot.immediate_migrations = immediate_migrations.load();
        snapshot.graceful_migrations = graceful_migrations.load();
        snapshot.hot_standby_migrations = hot_standby_migrations.load();
        snapshot.streaming_migrations = streaming_migrations.load();
        snapshot.validation_failures = validation_failures.load();
        snapshot.rollback_events = rollback_events.load();
        snapshot.performance_degradations = performance_degradations.load();
        return snapshot;
    }
};

/**
 * @brief Migration manager for real-time graph migration
 */
class MigrationManager {
public:
    using MigrationProgressCallback = std::function<void(const std::string&, double)>;
    using MigrationCompletionCallback = std::function<void(const MigrationResult&)>;
    
    /**
     * @brief Constructor
     * @param config Migration manager configuration
     */
    explicit MigrationManager(const MigrationManagerConfig& config);
    
    /**
     * @brief Destructor
     */
    ~MigrationManager();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Migration operations
    std::future<MigrationResult> MigratePartition(
        const std::string& partition_id,
        const std::string& source_node_id,
        const std::string& target_node_id,
        MigrationStrategy strategy = MigrationStrategy::GRACEFUL);
    
    Status CancelMigration(const std::string& migration_id);
    Status CancelAllMigrations();
    
    // State management
    Status CheckpointPartitionState(const std::string& partition_id, 
                                   const std::string& node_id);
    Status RestorePartitionState(const std::string& partition_id,
                                const std::string& node_id,
                                const std::map<std::string, std::vector<uint8_t>>& state);
    
    // Validation
    Status ValidateMigrationFeasibility(const std::string& partition_id,
                                       const std::string& source_node_id,
                                       const std::string& target_node_id);
    Status ValidatePostMigration(const std::string& partition_id,
                                const std::string& node_id);
    
    // Monitoring and statistics
    MigrationManagerStats::Snapshot GetStats() const;
    void ResetStats();
    std::vector<std::string> GetActiveMigrations() const;
    std::vector<std::string> GetFailedMigrations() const;
    double GetMigrationProgress(const std::string& migration_id) const;
    
    // Event handling
    Status RegisterProgressCallback(MigrationProgressCallback callback);
    Status RegisterCompletionCallback(MigrationCompletionCallback callback);
    
    // Configuration
    MigrationManagerConfig GetConfig() const;
    Status UpdateConfig(const MigrationManagerConfig& config);

private:
    // Internal migration methods
    void MigrationThreadMain();
    MigrationResult ExecuteMigration(std::unique_ptr<MigrationRequest> request);
    
    // Strategy-specific implementations
    MigrationResult ExecuteImmediateMigration(std::unique_ptr<MigrationRequest> request);
    MigrationResult ExecuteGracefulMigration(std::unique_ptr<MigrationRequest> request);
    MigrationResult ExecuteHotStandbyMigration(std::unique_ptr<MigrationRequest> request);
    MigrationResult ExecuteStreamingMigration(std::unique_ptr<MigrationRequest> request);
    
    // State management
    Status SerializePartitionState(const std::string& partition_id,
                                  const std::string& node_id,
                                  std::map<std::string, std::vector<uint8_t>>& state);
    Status DeserializePartitionState(const std::map<std::string, std::vector<uint8_t>>& state,
                                    const std::string& partition_id,
                                    const std::string& node_id);
    
    // Data transfer
    Status TransferPartitionData(const MigrationRequest& request);
    Status CompressStateData(const std::map<std::string, std::vector<uint8_t>>& state,
                            std::vector<uint8_t>& compressed_data);
    Status DecompressStateData(const std::vector<uint8_t>& compressed_data,
                              std::map<std::string, std::vector<uint8_t>>& state);
    
    // Validation
    bool ValidateNodeCapabilities(const std::string& node_id, 
                                 const std::string& partition_id);
    bool ValidateNetworkConnectivity(const std::string& source_node_id,
                                    const std::string& target_node_id);
    bool ValidateStateConsistency(const std::string& partition_id,
                                 const std::string& node_id);
    bool ValidatePerformance(const std::string& partition_id,
                            const std::string& node_id);
    
    // Error handling and recovery
    Status RollbackMigration(const MigrationRequest& request);
    Status RetryMigration(std::unique_ptr<MigrationRequest> request);
    void HandleMigrationFailure(const MigrationRequest& request, const std::string& error);
    
    // Configuration
    MigrationManagerConfig config_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Migration management
    mutable std::mutex migrations_mutex_;
    std::map<std::string, std::unique_ptr<MigrationRequest>> active_migrations_;
    std::map<std::string, MigrationResult> completed_migrations_;
    std::condition_variable migrations_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    MigrationManagerStats stats_;
    
    // Threading
    std::thread migration_thread_;
    
    // Event callbacks
    mutable std::mutex callbacks_mutex_;
    std::vector<MigrationProgressCallback> progress_callbacks_;
    std::vector<MigrationCompletionCallback> completion_callbacks_;
    
    // State checkpointing
    mutable std::mutex checkpoints_mutex_;
    std::map<std::string, std::map<std::string, std::vector<uint8_t>>> partition_checkpoints_;
};

} // namespace edge_ai
