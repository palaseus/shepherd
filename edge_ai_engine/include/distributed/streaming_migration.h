/**
 * @file streaming_migration.h
 * @brief Zero-loss streaming migration system with stateful checkpointed contexts
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
 * @brief Migration state enumeration
 */
enum class MigrationState {
    PENDING = 0,
    PREPARING = 1,
    CHECKPOINTING = 2,
    TRANSFERRING = 3,
    RESUMING = 4,
    COMPLETED = 5,
    FAILED = 6,
    CANCELLED = 7
};

/**
 * @brief Stream data packet
 */
struct StreamPacket {
    uint64_t sequence_id;
    uint64_t timestamp_ns;
    std::vector<uint8_t> data;
    std::string source_node_id;
    std::string target_node_id;
    bool is_ordered{true};
    bool is_critical{false};
    
    // Metadata
    std::map<std::string, std::string> metadata;
    uint32_t priority{0};
    std::chrono::milliseconds ttl_ms{5000};
    
    StreamPacket() {
        timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

/**
 * @brief Migration checkpoint
 */
struct MigrationCheckpoint {
    std::string checkpoint_id;
    std::string context_id;
    std::string node_id;
    
    // State data
    std::vector<uint8_t> input_state;
    std::vector<uint8_t> output_state;
    std::vector<uint8_t> intermediate_state;
    std::vector<uint8_t> graph_state;
    
    // Sequence tracking
    uint64_t last_processed_sequence{0};
    uint64_t last_committed_sequence{0};
    std::vector<uint64_t> pending_sequences;
    
    // Timing information
    std::chrono::steady_clock::time_point checkpoint_time;
    std::chrono::milliseconds processing_latency{0};
    
    // Validation data
    std::vector<uint8_t> checksum;
    uint32_t version{1};
    bool is_valid{false};
    
    MigrationCheckpoint() {
        checkpoint_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Migration request
 */
struct MigrationRequest {
    std::string migration_id;
    std::string source_node_id;
    std::string target_node_id;
    std::string graph_id;
    std::vector<std::string> context_ids;
    
    // Migration parameters
    MigrationState state{MigrationState::PENDING};
    std::chrono::milliseconds timeout_ms{30000};
    bool allow_partial_migration{false};
    bool preserve_ordering{true};
    bool enable_compression{true};
    
    // Priority and scheduling
    uint32_t priority{0};
    std::chrono::steady_clock::time_point scheduled_time;
    std::chrono::milliseconds max_downtime_ms{100};
    
    // Progress tracking
    double progress_percent{0.0};
    std::string current_operation;
    std::vector<std::string> completed_operations;
    
    // Error handling
    std::string error_message;
    uint32_t retry_count{0};
    uint32_t max_retries{3};
    
    MigrationRequest() {
        scheduled_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Migration result
 */
struct MigrationResult {
    std::string migration_id;
    std::string source_node_id;
    std::string target_node_id;
    
    // Result status
    Status migration_status{Status::NOT_INITIALIZED};
    MigrationState final_state{MigrationState::PENDING};
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    
    // Performance metrics
    std::chrono::milliseconds total_migration_time{0};
    std::chrono::milliseconds downtime{0};
    std::chrono::milliseconds data_transfer_time{0};
    std::chrono::milliseconds checkpoint_time{0};
    
    // Data integrity
    size_t total_data_transferred{0};
    size_t packets_migrated{0};
    size_t packets_lost{0};
    size_t packets_duplicated{0};
    bool data_integrity_verified{false};
    
    // Quality metrics
    double migration_efficiency{0.0};
    double data_compression_ratio{0.0};
    double ordering_preservation_rate{0.0};
    
    MigrationResult() {
        start_time = std::chrono::steady_clock::now();
        end_time = start_time;
    }
};

/**
 * @brief Stream buffer for ordered delivery
 */
struct StreamBuffer {
    std::string buffer_id;
    std::string node_id;
    std::string graph_id;
    
    // Buffer management
    std::deque<StreamPacket> packets;
    std::map<uint64_t, StreamPacket> out_of_order_packets;
    uint64_t expected_sequence{0};
    uint64_t last_delivered_sequence{0};
    
    // Buffer limits
    size_t max_buffer_size{1024 * 1024};  // 1MB
    size_t max_packet_count{1000};
    std::chrono::milliseconds max_packet_age{10000};  // 10 seconds
    
    // Statistics
    std::atomic<uint64_t> total_packets_received{0};
    std::atomic<uint64_t> total_packets_delivered{0};
    std::atomic<uint64_t> total_packets_dropped{0};
    std::atomic<uint64_t> total_duplicates{0};
    
    // Timing
    std::chrono::steady_clock::time_point last_activity;
    std::chrono::milliseconds avg_delivery_latency{0};
    
    StreamBuffer() {
        last_activity = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Streaming migration statistics
 */
struct StreamingMigrationStats {
    std::atomic<uint64_t> total_migrations{0};
    std::atomic<uint64_t> successful_migrations{0};
    std::atomic<uint64_t> failed_migrations{0};
    std::atomic<uint64_t> cancelled_migrations{0};
    
    // Performance metrics
    std::atomic<double> avg_migration_time_ms{0.0};
    std::atomic<double> avg_downtime_ms{0.0};
    std::atomic<double> avg_data_transfer_rate_mbps{0.0};
    std::atomic<double> avg_checkpoint_time_ms{0.0};
    
    // Data integrity metrics
    std::atomic<uint64_t> total_packets_migrated{0};
    std::atomic<uint64_t> total_packets_lost{0};
    std::atomic<uint64_t> total_packets_duplicated{0};
    std::atomic<double> data_integrity_rate{0.0};
    std::atomic<double> ordering_preservation_rate{0.0};
    
    // Quality metrics
    std::atomic<double> migration_success_rate{0.0};
    std::atomic<double> zero_loss_migration_rate{0.0};
    std::atomic<double> avg_compression_ratio{0.0};
    std::atomic<double> avg_migration_efficiency{0.0};
    
    // Buffer statistics
    std::atomic<uint64_t> total_buffers_created{0};
    std::atomic<uint64_t> total_buffers_destroyed{0};
    std::atomic<uint64_t> total_packets_buffered{0};
    std::atomic<double> avg_buffer_utilization{0.0};
    
    StreamingMigrationStats() = default;
    
    struct Snapshot {
        uint64_t total_migrations;
        uint64_t successful_migrations;
        uint64_t failed_migrations;
        uint64_t cancelled_migrations;
        double avg_migration_time_ms;
        double avg_downtime_ms;
        double avg_data_transfer_rate_mbps;
        double avg_checkpoint_time_ms;
        uint64_t total_packets_migrated;
        uint64_t total_packets_lost;
        uint64_t total_packets_duplicated;
        double data_integrity_rate;
        double ordering_preservation_rate;
        double migration_success_rate;
        double zero_loss_migration_rate;
        double avg_compression_ratio;
        double avg_migration_efficiency;
        uint64_t total_buffers_created;
        uint64_t total_buffers_destroyed;
        uint64_t total_packets_buffered;
        double avg_buffer_utilization;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_migrations = total_migrations.load();
        snapshot.successful_migrations = successful_migrations.load();
        snapshot.failed_migrations = failed_migrations.load();
        snapshot.cancelled_migrations = cancelled_migrations.load();
        snapshot.avg_migration_time_ms = avg_migration_time_ms.load();
        snapshot.avg_downtime_ms = avg_downtime_ms.load();
        snapshot.avg_data_transfer_rate_mbps = avg_data_transfer_rate_mbps.load();
        snapshot.avg_checkpoint_time_ms = avg_checkpoint_time_ms.load();
        snapshot.total_packets_migrated = total_packets_migrated.load();
        snapshot.total_packets_lost = total_packets_lost.load();
        snapshot.total_packets_duplicated = total_packets_duplicated.load();
        snapshot.data_integrity_rate = data_integrity_rate.load();
        snapshot.ordering_preservation_rate = ordering_preservation_rate.load();
        snapshot.migration_success_rate = migration_success_rate.load();
        snapshot.zero_loss_migration_rate = zero_loss_migration_rate.load();
        snapshot.avg_compression_ratio = avg_compression_ratio.load();
        snapshot.avg_migration_efficiency = avg_migration_efficiency.load();
        snapshot.total_buffers_created = total_buffers_created.load();
        snapshot.total_buffers_destroyed = total_buffers_destroyed.load();
        snapshot.total_packets_buffered = total_packets_buffered.load();
        snapshot.avg_buffer_utilization = avg_buffer_utilization.load();
        return snapshot;
    }
};

/**
 * @brief Zero-loss streaming migration system
 */
class StreamingMigration {
public:
    /**
     * @brief Constructor
     * @param cluster_manager Cluster manager for node information
     */
    explicit StreamingMigration(std::shared_ptr<ClusterManager> cluster_manager);
    
    /**
     * @brief Destructor
     */
    ~StreamingMigration();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Migration management
    Status RequestMigration(const MigrationRequest& request);
    Status ExecuteMigration(const std::string& migration_id);
    Status CancelMigration(const std::string& migration_id);
    Status GetMigrationStatus(const std::string& migration_id, MigrationResult& result);
    
    // Stream buffer management
    Status CreateStreamBuffer(const std::string& buffer_id, const std::string& node_id, 
                             const std::string& graph_id, StreamBuffer& buffer);
    Status DestroyStreamBuffer(const std::string& buffer_id);
    Status AddPacketToBuffer(const std::string& buffer_id, const StreamPacket& packet);
    Status GetNextPacket(const std::string& buffer_id, StreamPacket& packet);
    
    // Checkpoint management
    Status CreateCheckpoint(const std::string& context_id, const std::string& node_id, 
                           MigrationCheckpoint& checkpoint);
    Status RestoreFromCheckpoint(const std::string& context_id, const std::string& node_id, 
                                const MigrationCheckpoint& checkpoint);
    Status ValidateCheckpoint(const MigrationCheckpoint& checkpoint);
    
    // Data transfer
    Status TransferStreamData(const std::string& source_node_id, const std::string& target_node_id, 
                             const std::vector<StreamPacket>& packets);
    Status CompressStreamData(const std::vector<StreamPacket>& packets, std::vector<uint8_t>& compressed);
    Status DecompressStreamData(const std::vector<uint8_t>& compressed, std::vector<StreamPacket>& packets);
    
    // Ordering and sequencing
    Status EnsureOrderedDelivery(const std::string& buffer_id);
    Status HandleOutOfOrderPackets(const std::string& buffer_id);
    Status DetectDuplicatePackets(const std::string& buffer_id);
    
    // Data integrity
    Status CalculateChecksum(const std::vector<uint8_t>& data, std::vector<uint8_t>& checksum);
    Status VerifyDataIntegrity(const std::vector<uint8_t>& data, const std::vector<uint8_t>& checksum);
    Status DetectDataCorruption(const std::vector<StreamPacket>& packets);
    
    // Performance optimization
    Status OptimizeMigrationPath(const std::string& source_node_id, const std::string& target_node_id, 
                                std::vector<std::string>& optimal_path);
    Status EstimateMigrationTime(const MigrationRequest& request, std::chrono::milliseconds& estimated_time);
    Status OptimizeBufferSizes();
    
    // Statistics and monitoring
    StreamingMigrationStats::Snapshot GetStats() const;
    void ResetStats();
    Status GenerateMigrationReport();
    
    // Configuration
    void SetCompressionEnabled(bool enabled);
    void SetOrderingPreservationEnabled(bool enabled);
    void SetMaxBufferSize(size_t max_size);
    void SetCheckpointInterval(std::chrono::milliseconds interval);
    void SetMaxDowntime(std::chrono::milliseconds max_downtime);

private:
    // Internal migration methods
    Status PrepareMigration(const MigrationRequest& request);
    Status ExecuteCheckpointing(const std::string& migration_id);
    Status ExecuteDataTransfer(const std::string& migration_id);
    Status ExecuteResumption(const std::string& migration_id);
    Status CleanupMigration(const std::string& migration_id);
    
    // Stream processing algorithms
    Status ProcessIncomingPackets(const std::string& buffer_id);
    Status ReorderPackets(const std::string& buffer_id);
    Status RemoveExpiredPackets(const std::string& buffer_id);
    Status OptimizeBufferUtilization(const std::string& buffer_id);
    
    // Compression algorithms
    Status CompressPacket(const StreamPacket& packet, std::vector<uint8_t>& compressed);
    Status DecompressPacket(const std::vector<uint8_t>& compressed, StreamPacket& packet);
    double CalculateCompressionRatio(const std::vector<uint8_t>& original, const std::vector<uint8_t>& compressed);
    
    // Network optimization
    Status SelectOptimalTransferPath(const std::string& source_node_id, const std::string& target_node_id);
    Status OptimizeTransferBatchSize(const std::vector<StreamPacket>& packets);
    Status ImplementRetryLogic(const std::string& migration_id, uint32_t retry_count);
    
    // Threading and synchronization
    void MigrationExecutionThread();
    void StreamProcessingThread();
    void BufferMaintenanceThread();
    void CheckpointThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> compression_enabled_{true};
    std::atomic<bool> ordering_preservation_enabled_{true};
    std::atomic<size_t> max_buffer_size_{1024 * 1024};
    std::atomic<std::chrono::milliseconds> checkpoint_interval_{std::chrono::milliseconds(1000)};
    std::atomic<std::chrono::milliseconds> max_downtime_{std::chrono::milliseconds(100)};
    
    // Dependencies
    std::shared_ptr<ClusterManager> cluster_manager_;
    
    // Migration state
    mutable std::mutex migration_mutex_;
    std::map<std::string, MigrationRequest> pending_migrations_;
    std::map<std::string, MigrationRequest> active_migrations_;
    std::map<std::string, MigrationResult> completed_migrations_;
    
    // Stream buffers
    mutable std::mutex buffer_mutex_;
    std::map<std::string, StreamBuffer> stream_buffers_;
    std::map<std::string, std::vector<StreamPacket>> buffer_queues_;
    
    // Checkpoints
    mutable std::mutex checkpoint_mutex_;
    std::map<std::string, MigrationCheckpoint> checkpoints_;
    std::map<std::string, std::vector<uint8_t>> checkpoint_data_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    StreamingMigrationStats stats_;
    
    // Threading
    std::thread migration_execution_thread_;
    std::thread stream_processing_thread_;
    std::thread buffer_maintenance_thread_;
    std::thread checkpoint_thread_;
    
    std::condition_variable migration_cv_;
    std::condition_variable stream_cv_;
    std::condition_variable buffer_cv_;
    std::condition_variable checkpoint_cv_;
    
    mutable std::mutex migration_cv_mutex_;
    mutable std::mutex stream_cv_mutex_;
    mutable std::mutex buffer_cv_mutex_;
    mutable std::mutex checkpoint_cv_mutex_;
};

} // namespace edge_ai
