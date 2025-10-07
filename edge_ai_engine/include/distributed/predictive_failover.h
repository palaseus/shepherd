/**
 * @file predictive_failover.h
 * @brief Predictive failover system with online ML models for node failure prediction
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
#include "core/types.h"
#include "distributed/cluster_types.h"
#include "ml_policy/ml_based_policy.h"

// Forward declarations
namespace edge_ai {
    class ClusterManager;
}

namespace edge_ai {

/**
 * @brief Node health prediction model
 */
struct NodeHealthPrediction {
    std::string node_id;
    std::chrono::steady_clock::time_point prediction_time;
    
    // Failure probability predictions
    double cpu_failure_probability{0.0};
    double memory_failure_probability{0.0};
    double gpu_failure_probability{0.0};
    double network_failure_probability{0.0};
    double thermal_failure_probability{0.0};
    double overall_failure_probability{0.0};
    
    // Time to failure estimates (in milliseconds)
    std::chrono::milliseconds estimated_time_to_failure{0};
    std::chrono::milliseconds confidence_interval_ms{0};
    
    // Performance degradation predictions
    double predicted_cpu_degradation{0.0};
    double predicted_memory_degradation{0.0};
    double predicted_throughput_degradation{0.0};
    double predicted_latency_increase{0.0};
    
    // Model confidence
    double model_confidence{0.0};
    std::string model_version;
    std::vector<double> feature_importance;
    
    NodeHealthPrediction() {
        prediction_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Failover decision
 */
struct FailoverDecision {
    std::string decision_id;
    std::string source_node_id;
    std::string target_node_id;
    
    enum class FailoverType {
        PREVENTIVE,      // Proactive migration before failure
        REACTIVE,        // Emergency migration after failure
        LOAD_BALANCING,  // Performance-based migration
        MAINTENANCE      // Scheduled maintenance migration
    } failover_type;
    
    // Migration details
    std::vector<std::string> tasks_to_migrate;
    std::vector<std::string> graphs_to_migrate;
    std::chrono::milliseconds estimated_migration_time{0};
    double migration_priority{0.0};
    
    // Risk assessment
    double failure_risk{0.0};
    double migration_risk{0.0};
    double data_loss_risk{0.0};
    
    // Performance impact
    double expected_performance_impact{0.0};
    double expected_latency_impact{0.0};
    double expected_throughput_impact{0.0};
    
    // Decision metadata
    std::string reasoning;
    double confidence_score{0.0};
    std::chrono::steady_clock::time_point decision_time;
    bool executed{false};
    Status execution_status{Status::NOT_INITIALIZED};
    
    FailoverDecision() {
        decision_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Streaming context for zero-loss migration
 */
struct StreamingContext {
    std::string context_id;
    std::string node_id;
    std::string graph_id;
    
    // Stream state
    std::vector<uint8_t> input_buffer;
    std::vector<uint8_t> output_buffer;
    std::vector<uint8_t> intermediate_state;
    
    // Sequence tracking
    uint64_t last_processed_sequence{0};
    uint64_t last_committed_sequence{0};
    std::deque<uint64_t> pending_sequences;
    
    // Timing information
    std::chrono::steady_clock::time_point last_activity;
    std::chrono::milliseconds max_idle_time{5000};
    
    // Checkpoint information
    std::vector<uint8_t> checkpoint_data;
    std::chrono::steady_clock::time_point last_checkpoint;
    std::chrono::milliseconds checkpoint_interval{1000};
    
    // Migration state
    bool migration_in_progress{false};
    std::string target_node_id;
    std::chrono::steady_clock::time_point migration_start_time;
    
    StreamingContext() {
        last_activity = std::chrono::steady_clock::now();
        last_checkpoint = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Anomaly detection result
 */
struct AnomalyDetectionResult {
    std::string node_id;
    std::string graph_id;
    std::string task_id;
    
    enum class AnomalyType {
        PERFORMANCE_DEGRADATION,
        MEMORY_LEAK,
        CPU_SPIKE,
        NETWORK_LATENCY,
        THROUGHPUT_DROP,
        ERROR_RATE_INCREASE,
        RESOURCE_STARVATION
    } anomaly_type;
    
    double severity_score{0.0};  // 0.0 to 1.0
    double confidence{0.0};
    std::string description;
    std::chrono::steady_clock::time_point detection_time;
    
    // Historical context
    std::vector<double> historical_values;
    double baseline_value{0.0};
    double current_value{0.0};
    double deviation_percentage{0.0};
    
    // Recommended actions
    std::vector<std::string> recommended_actions;
    bool auto_remediation_enabled{true};
    
    AnomalyDetectionResult() {
        detection_time = std::chrono::steady_clock::now();
    }
};

/**
 * @brief Predictive failover statistics
 */
struct PredictiveFailoverStats {
    std::atomic<uint64_t> total_predictions{0};
    std::atomic<uint64_t> accurate_predictions{0};
    std::atomic<uint64_t> false_positives{0};
    std::atomic<uint64_t> false_negatives{0};
    
    // Failover metrics
    std::atomic<uint64_t> total_failovers{0};
    std::atomic<uint64_t> preventive_failovers{0};
    std::atomic<uint64_t> reactive_failovers{0};
    std::atomic<uint64_t> successful_failovers{0};
    std::atomic<uint64_t> failed_failovers{0};
    
    // Migration metrics
    std::atomic<uint64_t> total_migrations{0};
    std::atomic<uint64_t> zero_loss_migrations{0};
    std::atomic<uint64_t> data_loss_migrations{0};
    std::atomic<double> avg_migration_time_ms{0.0};
    std::atomic<double> avg_downtime_ms{0.0};
    
    // Anomaly detection
    std::atomic<uint64_t> total_anomalies_detected{0};
    std::atomic<uint64_t> auto_remediated_anomalies{0};
    std::atomic<double> avg_detection_latency_ms{0.0};
    
    // Performance metrics
    std::atomic<double> prediction_accuracy{0.0};
    std::atomic<double> failover_success_rate{0.0};
    std::atomic<double> zero_loss_migration_rate{0.0};
    std::atomic<double> avg_recovery_time_ms{0.0};
    
    PredictiveFailoverStats() = default;
    
    struct Snapshot {
        uint64_t total_predictions;
        uint64_t accurate_predictions;
        uint64_t false_positives;
        uint64_t false_negatives;
        uint64_t total_failovers;
        uint64_t preventive_failovers;
        uint64_t reactive_failovers;
        uint64_t successful_failovers;
        uint64_t failed_failovers;
        uint64_t total_migrations;
        uint64_t zero_loss_migrations;
        uint64_t data_loss_migrations;
        double avg_migration_time_ms;
        double avg_downtime_ms;
        uint64_t total_anomalies_detected;
        uint64_t auto_remediated_anomalies;
        double avg_detection_latency_ms;
        double prediction_accuracy;
        double failover_success_rate;
        double zero_loss_migration_rate;
        double avg_recovery_time_ms;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.total_predictions = total_predictions.load();
        snapshot.accurate_predictions = accurate_predictions.load();
        snapshot.false_positives = false_positives.load();
        snapshot.false_negatives = false_negatives.load();
        snapshot.total_failovers = total_failovers.load();
        snapshot.preventive_failovers = preventive_failovers.load();
        snapshot.reactive_failovers = reactive_failovers.load();
        snapshot.successful_failovers = successful_failovers.load();
        snapshot.failed_failovers = failed_failovers.load();
        snapshot.total_migrations = total_migrations.load();
        snapshot.zero_loss_migrations = zero_loss_migrations.load();
        snapshot.data_loss_migrations = data_loss_migrations.load();
        snapshot.avg_migration_time_ms = avg_migration_time_ms.load();
        snapshot.avg_downtime_ms = avg_downtime_ms.load();
        snapshot.total_anomalies_detected = total_anomalies_detected.load();
        snapshot.auto_remediated_anomalies = auto_remediated_anomalies.load();
        snapshot.avg_detection_latency_ms = avg_detection_latency_ms.load();
        snapshot.prediction_accuracy = prediction_accuracy.load();
        snapshot.failover_success_rate = failover_success_rate.load();
        snapshot.zero_loss_migration_rate = zero_loss_migration_rate.load();
        snapshot.avg_recovery_time_ms = avg_recovery_time_ms.load();
        return snapshot;
    }
};

/**
 * @brief Predictive failover system
 */
class PredictiveFailover {
public:
    /**
     * @brief Constructor
     * @param cluster_manager Cluster manager for node information
     * @param ml_policy ML policy for failure prediction
     */
    PredictiveFailover(std::shared_ptr<ClusterManager> cluster_manager,
                      std::shared_ptr<MLBasedPolicy> ml_policy);
    
    /**
     * @brief Destructor
     */
    ~PredictiveFailover();
    
    // Lifecycle management
    Status Initialize();
    Status Shutdown();
    bool IsInitialized() const;
    
    // Health prediction
    Status PredictNodeHealth(const std::string& node_id, NodeHealthPrediction& prediction);
    Status PredictClusterHealth(std::vector<NodeHealthPrediction>& predictions);
    Status UpdateHealthModel(const std::string& node_id, const std::vector<double>& features, bool failure_occurred);
    
    // Failover management
    Status EvaluateFailoverNeeds();
    Status ExecuteFailover(const FailoverDecision& decision);
    Status SchedulePreventiveMigration(const std::string& node_id, const std::vector<std::string>& tasks);
    Status ExecuteEmergencyFailover(const std::string& failed_node_id);
    
    // Streaming context management
    Status CreateStreamingContext(const std::string& context_id, const std::string& node_id, 
                                 const std::string& graph_id, StreamingContext& context);
    Status UpdateStreamingContext(const std::string& context_id, const std::vector<uint8_t>& data);
    Status CheckpointStreamingContext(const std::string& context_id);
    Status MigrateStreamingContext(const std::string& context_id, const std::string& target_node_id);
    
    // Anomaly detection
    Status DetectAnomalies();
    Status RegisterAnomalyDetector(const std::string& node_id, 
                                  std::function<bool(const std::vector<double>&)> detector);
    Status HandleAnomaly(const AnomalyDetectionResult& anomaly);
    Status AutoRemediateAnomaly(const AnomalyDetectionResult& anomaly);
    
    // Zero-loss migration
    Status PrepareZeroLossMigration(const std::string& source_node_id, const std::string& target_node_id);
    Status ExecuteZeroLossMigration(const std::string& migration_id);
    Status ValidateMigrationIntegrity(const std::string& migration_id);
    
    // Statistics and monitoring
    PredictiveFailoverStats::Snapshot GetStats() const;
    void ResetStats();
    Status GenerateFailoverReport();
    
    // Configuration
    void SetPredictionThreshold(double threshold);
    void SetFailoverEnabled(bool enabled);
    void SetAnomalyDetectionEnabled(bool enabled);
    void SetAutoRemediationEnabled(bool enabled);
    void SetCheckpointInterval(std::chrono::milliseconds interval);

private:
    // Internal prediction methods
    double CalculateFailureProbability(const std::string& node_id, const std::vector<double>& features);
    std::chrono::milliseconds EstimateTimeToFailure(const std::string& node_id, double failure_probability);
    std::vector<std::string> SelectMigrationTargets(const std::string& source_node_id, 
                                                   const std::vector<std::string>& tasks);
    
    // ML model management
    Status TrainFailurePredictionModel();
    Status UpdateModelWeights(const std::vector<double>& features, bool actual_failure);
    double ValidatePredictionAccuracy();
    
    // Streaming migration algorithms
    Status CreateMigrationCheckpoint(const StreamingContext& context);
    Status TransferStreamingData(const StreamingContext& context, const std::string& target_node_id);
    Status ResumeStreamingExecution(const std::string& context_id, const std::string& target_node_id);
    
    // Anomaly detection algorithms
    bool DetectPerformanceAnomaly(const std::string& node_id, const std::vector<double>& metrics);
    bool DetectMemoryAnomaly(const std::string& node_id, const std::vector<double>& metrics);
    bool DetectNetworkAnomaly(const std::string& node_id, const std::vector<double>& metrics);
    std::vector<std::string> GenerateRemediationActions(const AnomalyDetectionResult& anomaly);
    
    // Threading and synchronization
    void HealthPredictionThread();
    void AnomalyDetectionThread();
    void FailoverExecutionThread();
    void StreamingMigrationThread();
    
    // Member variables
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    std::atomic<bool> failover_enabled_{true};
    std::atomic<bool> anomaly_detection_enabled_{true};
    std::atomic<bool> auto_remediation_enabled_{true};
    std::atomic<double> prediction_threshold_{0.7};
    std::atomic<std::chrono::milliseconds> checkpoint_interval_{std::chrono::milliseconds(1000)};
    
    // Dependencies
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<MLBasedPolicy> ml_policy_;
    
    // Health prediction state
    mutable std::mutex prediction_mutex_;
    std::map<std::string, NodeHealthPrediction> node_predictions_;
    std::map<std::string, std::vector<double>> historical_features_;
    std::map<std::string, std::vector<bool>> historical_failures_;
    
    // Failover state
    mutable std::mutex failover_mutex_;
    std::queue<FailoverDecision> pending_failovers_;
    std::map<std::string, FailoverDecision> active_failovers_;
    std::vector<FailoverDecision> failover_history_;
    
    // Streaming contexts
    mutable std::mutex streaming_mutex_;
    std::map<std::string, StreamingContext> streaming_contexts_;
    std::map<std::string, std::vector<uint8_t>> migration_checkpoints_;
    
    // Anomaly detection
    mutable std::mutex anomaly_mutex_;
    std::map<std::string, std::function<bool(const std::vector<double>&)>> anomaly_detectors_;
    std::vector<AnomalyDetectionResult> detected_anomalies_;
    std::map<std::string, std::vector<double>> node_metrics_history_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    PredictiveFailoverStats stats_;
    
    // Threading
    std::thread health_prediction_thread_;
    std::thread anomaly_detection_thread_;
    std::thread failover_execution_thread_;
    std::thread streaming_migration_thread_;
    
    std::condition_variable prediction_cv_;
    std::condition_variable anomaly_cv_;
    std::condition_variable failover_cv_;
    std::condition_variable streaming_cv_;
    
    mutable std::mutex prediction_cv_mutex_;
    mutable std::mutex anomaly_cv_mutex_;
    mutable std::mutex failover_cv_mutex_;
    mutable std::mutex streaming_cv_mutex_;
};

} // namespace edge_ai
