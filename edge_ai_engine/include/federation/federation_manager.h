/**
 * @file federation_manager.h
 * @brief Cross-cluster intelligence and federated learning manager
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
#include "ml_policy/ml_based_policy.h"
#include "governance/governance_manager.h"

// Forward declarations
namespace edge_ai {
    class ClusterManager;
    class MLBasedPolicy;
    class GovernanceManager;
}

namespace edge_ai {

/**
 * @brief Federation protocol types
 */
enum class FederationProtocol {
    FEDERATED_AVERAGING = 0,
    FEDERATED_DROPOUT = 1,
    FEDERATED_META_LEARNING = 2,
    FEDERATED_REINFORCEMENT_LEARNING = 3,
    FEDERATED_TRANSFER_LEARNING = 4,
    CROSS_CLUSTER_ENSEMBLE = 5
};

/**
 * @brief Federation participant information
 */
struct FederationParticipant {
    std::string cluster_id;
    std::string node_id;
    std::string endpoint;
    std::string protocol_version;
    
    // Capabilities
    std::vector<std::string> supported_models;
    std::vector<std::string> supported_algorithms;
    double compute_capacity;
    double memory_capacity;
    double bandwidth_capacity;
    
    // Status
    bool is_active;
    std::chrono::steady_clock::time_point last_heartbeat;
    std::chrono::steady_clock::time_point joined_at;
    uint32_t participation_score;
    double trust_level;
    
    // Performance metrics
    double avg_contribution_quality;
    double avg_response_time_ms;
    uint64_t total_contributions;
    uint64_t successful_contributions;
};

/**
 * @brief Federation round configuration
 */
struct FederationRound {
    std::string round_id;
    std::string federation_id;
    FederationProtocol protocol;
    
    // Round parameters
    std::vector<std::string> participant_clusters;
    std::chrono::milliseconds round_timeout;
    uint32_t min_participants;
    uint32_t max_participants;
    
    // Learning parameters
    double learning_rate;
    uint32_t local_epochs;
    uint32_t batch_size;
    std::string model_architecture;
    
    // Aggregation parameters
    std::string aggregation_strategy;
    double aggregation_weight;
    bool differential_privacy_enabled;
    double privacy_budget;
    
    // Status
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    bool is_active;
    bool is_completed;
    std::string status_message;
};

/**
 * @brief Federation contribution
 */
struct FederationContribution {
    std::string contribution_id;
    std::string round_id;
    std::string participant_cluster_id;
    std::string participant_node_id;
    
    // Model updates
    std::vector<double> model_weights;
    std::vector<double> model_gradients;
    std::map<std::string, double> model_metrics;
    
    // Metadata
    std::chrono::steady_clock::time_point submission_time;
    std::chrono::milliseconds computation_time;
    double contribution_quality;
    double privacy_budget_used;
    
    // Validation
    bool is_validated;
    std::string validation_status;
    std::vector<std::string> validation_errors;
};

/**
 * @brief Federation aggregation result
 */
struct FederationAggregation {
    std::string aggregation_id;
    std::string round_id;
    
    // Aggregated model
    std::vector<double> aggregated_weights;
    std::map<std::string, double> aggregated_metrics;
    
    // Aggregation metadata
    std::vector<std::string> contributing_clusters;
    uint32_t num_contributions;
    double aggregation_quality;
    std::chrono::milliseconds aggregation_time;
    
    // Performance metrics
    double model_improvement;
    double convergence_rate;
    double diversity_score;
    
    // Privacy metrics
    double privacy_budget_consumed;
    double privacy_loss;
    bool privacy_guarantees_met;
};

/**
 * @brief Cross-cluster knowledge sharing
 */
struct KnowledgeSharing {
    std::string sharing_id;
    std::string source_cluster_id;
    std::string target_cluster_id;
    
    // Knowledge content
    std::string knowledge_type; // "model", "policy", "insights", "best_practices"
    std::map<std::string, std::string> knowledge_data;
    std::vector<double> knowledge_embeddings;
    
    // Sharing metadata
    std::chrono::steady_clock::time_point sharing_time;
    std::chrono::milliseconds transfer_time;
    double knowledge_quality;
    double relevance_score;
    
    // Privacy and security
    bool is_encrypted;
    std::string encryption_method;
    double privacy_level;
    bool requires_consent;
    
    // Feedback
    bool is_acknowledged;
    double usefulness_rating;
    std::string feedback_message;
};

/**
 * @brief Federation statistics
 */
struct FederationStats {
    // Participation metrics
    std::atomic<uint64_t> total_rounds{0};
    std::atomic<uint64_t> completed_rounds{0};
    std::atomic<uint64_t> failed_rounds{0};
    std::atomic<uint64_t> total_participants{0};
    std::atomic<uint64_t> active_participants{0};
    
    // Learning metrics
    std::atomic<double> avg_model_improvement{0.0};
    std::atomic<double> avg_convergence_rate{0.0};
    std::atomic<double> avg_contribution_quality{0.0};
    std::atomic<double> federation_effectiveness{0.0};
    
    // Communication metrics
    std::atomic<uint64_t> total_contributions{0};
    std::atomic<uint64_t> successful_contributions{0};
    std::atomic<double> avg_communication_time_ms{0.0};
    std::atomic<double> avg_aggregation_time_ms{0.0};
    
    // Knowledge sharing
    std::atomic<uint64_t> knowledge_shares_sent{0};
    std::atomic<uint64_t> knowledge_shares_received{0};
    std::atomic<double> avg_knowledge_quality{0.0};
    std::atomic<double> knowledge_utilization_rate{0.0};
    
    // Privacy and security
    std::atomic<double> avg_privacy_budget_used{0.0};
    std::atomic<uint64_t> privacy_violations{0};
    std::atomic<double> trust_score{0.0};
    
    /**
     * @brief Get a snapshot of current statistics
     */
    struct Snapshot {
        uint64_t total_rounds;
        uint64_t completed_rounds;
        uint64_t failed_rounds;
        uint64_t total_participants;
        uint64_t active_participants;
        double avg_model_improvement;
        double avg_convergence_rate;
        double avg_contribution_quality;
        double federation_effectiveness;
        uint64_t total_contributions;
        uint64_t successful_contributions;
        double avg_communication_time_ms;
        double avg_aggregation_time_ms;
        uint64_t knowledge_shares_sent;
        uint64_t knowledge_shares_received;
        double avg_knowledge_quality;
        double knowledge_utilization_rate;
        double avg_privacy_budget_used;
        uint64_t privacy_violations;
        double trust_score;
    };
    
    Snapshot GetSnapshot() const {
        return {
            total_rounds.load(),
            completed_rounds.load(),
            failed_rounds.load(),
            total_participants.load(),
            active_participants.load(),
            avg_model_improvement.load(),
            avg_convergence_rate.load(),
            avg_contribution_quality.load(),
            federation_effectiveness.load(),
            total_contributions.load(),
            successful_contributions.load(),
            avg_communication_time_ms.load(),
            avg_aggregation_time_ms.load(),
            knowledge_shares_sent.load(),
            knowledge_shares_received.load(),
            avg_knowledge_quality.load(),
            knowledge_utilization_rate.load(),
            avg_privacy_budget_used.load(),
            privacy_violations.load(),
            trust_score.load()
        };
    }
};

/**
 * @class FederationManager
 * @brief Cross-cluster intelligence and federated learning manager
 */
class FederationManager {
public:
    explicit FederationManager(std::shared_ptr<ClusterManager> cluster_manager,
                             std::shared_ptr<MLBasedPolicy> ml_policy,
                             std::shared_ptr<GovernanceManager> governance_manager);
    virtual ~FederationManager();
    
    /**
     * @brief Initialize the federation manager
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the federation manager
     */
    Status Shutdown();
    
    /**
     * @brief Check if the federation manager is initialized
     */
    bool IsInitialized() const;
    
    // Federation Management
    
    /**
     * @brief Create a new federation
     */
    Status CreateFederation(const std::string& federation_id, 
                           const std::vector<std::string>& participant_clusters,
                           FederationProtocol protocol);
    
    /**
     * @brief Join an existing federation
     */
    Status JoinFederation(const std::string& federation_id, const std::string& cluster_id);
    
    /**
     * @brief Leave a federation
     */
    Status LeaveFederation(const std::string& federation_id, const std::string& cluster_id);
    
    /**
     * @brief Get federation participants
     */
    std::vector<FederationParticipant> GetFederationParticipants(const std::string& federation_id) const;
    
    /**
     * @brief Update participant information
     */
    Status UpdateParticipantInfo(const std::string& cluster_id, const FederationParticipant& participant);
    
    // Federation Rounds
    
    /**
     * @brief Start a federation round
     */
    Status StartFederationRound(const FederationRound& round);
    
    /**
     * @brief Submit contribution to federation round
     */
    Status SubmitContribution(const FederationContribution& contribution);
    
    /**
     * @brief Aggregate contributions from federation round
     */
    Status AggregateContributions(const std::string& round_id, FederationAggregation& aggregation);
    
    /**
     * @brief Complete federation round
     */
    Status CompleteFederationRound(const std::string& round_id);
    
    /**
     * @brief Get federation round status
     */
    Status GetFederationRoundStatus(const std::string& round_id, FederationRound& round) const;
    
    /**
     * @brief Get federation round history
     */
    std::vector<FederationRound> GetFederationRoundHistory(const std::string& federation_id) const;
    
    // Knowledge Sharing
    
    /**
     * @brief Share knowledge with other clusters
     */
    Status ShareKnowledge(const KnowledgeSharing& knowledge);
    
    /**
     * @brief Receive knowledge from other clusters
     */
    Status ReceiveKnowledge(const std::string& source_cluster_id, const KnowledgeSharing& knowledge);
    
    /**
     * @brief Request knowledge from other clusters
     */
    Status RequestKnowledge(const std::string& target_cluster_id, const std::string& knowledge_type);
    
    /**
     * @brief Get shared knowledge history
     */
    std::vector<KnowledgeSharing> GetKnowledgeHistory(const std::string& cluster_id) const;
    
    /**
     * @brief Rate knowledge usefulness
     */
    Status RateKnowledgeUsefulness(const std::string& sharing_id, double rating, const std::string& feedback);
    
    // Meta-Learning and Transfer
    
    /**
     * @brief Perform federated meta-learning
     */
    Status PerformFederatedMetaLearning(const std::vector<std::string>& cluster_ids,
                                      std::map<std::string, double>& meta_learned_params);
    
    /**
     * @brief Transfer learned policies across clusters
     */
    Status TransferLearnedPolicies(const std::string& source_cluster_id,
                                 const std::string& target_cluster_id,
                                 const std::vector<std::string>& policy_ids);
    
    /**
     * @brief Adapt policies to local cluster context
     */
    Status AdaptPoliciesToLocalContext(const std::string& cluster_id,
                                     const std::vector<std::string>& policy_ids);
    
    // Cross-Cluster Coordination
    
    /**
     * @brief Coordinate global load balancing
     */
    Status CoordinateGlobalLoadBalancing(const std::vector<std::string>& cluster_ids,
                                       std::map<std::string, double>& load_distribution);
    
    /**
     * @brief Coordinate cross-cluster task migration
     */
    Status CoordinateTaskMigration(const std::string& source_cluster_id,
                                 const std::string& target_cluster_id,
                                 const std::vector<std::string>& task_ids);
    
    /**
     * @brief Synchronize cross-cluster configurations
     */
    Status SynchronizeCrossClusterConfigurations(const std::vector<std::string>& cluster_ids);
    
    // Privacy and Security
    
    /**
     * @brief Enable differential privacy for federation
     */
    Status EnableDifferentialPrivacy(const std::string& federation_id, double privacy_budget);
    
    /**
     * @brief Validate privacy guarantees
     */
    Status ValidatePrivacyGuarantees(const std::string& round_id, bool& privacy_guarantees_met);
    
    /**
     * @brief Update trust scores for participants
     */
    Status UpdateTrustScores(const std::map<std::string, double>& trust_scores);
    
    // Analytics and Reporting
    
    /**
     * @brief Generate federation effectiveness report
     */
    Status GenerateFederationReport(const std::string& federation_id,
                                  std::map<std::string, double>& effectiveness_metrics);
    
    /**
     * @brief Generate cross-cluster collaboration report
     */
    Status GenerateCollaborationReport(std::map<std::string, double>& collaboration_metrics);
    
    /**
     * @brief Get federation statistics
     */
    FederationStats::Snapshot GetStats() const;
    
    /**
     * @brief Reset federation statistics
     */
    void ResetStats();
    
    /**
     * @brief Generate federation insights
     */
    Status GenerateFederationInsights(std::vector<std::string>& insights);

private:
    // Core components
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<MLBasedPolicy> ml_policy_;
    std::shared_ptr<GovernanceManager> governance_manager_;
    
    // State management
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Federation storage
    mutable std::mutex federations_mutex_;
    std::map<std::string, std::vector<FederationParticipant>> federations_;
    std::map<std::string, FederationParticipant> participants_;
    
    // Round storage
    mutable std::mutex rounds_mutex_;
    std::map<std::string, FederationRound> active_rounds_;
    std::map<std::string, std::vector<FederationRound>> round_history_;
    std::map<std::string, std::vector<FederationContribution>> round_contributions_;
    
    // Knowledge sharing storage
    mutable std::mutex knowledge_mutex_;
    std::map<std::string, std::vector<KnowledgeSharing>> knowledge_history_;
    std::map<std::string, std::vector<KnowledgeSharing>> received_knowledge_;
    
    // Background threads
    std::thread federation_thread_;
    std::thread knowledge_thread_;
    std::thread coordination_thread_;
    
    // Condition variables
    std::mutex federation_cv_mutex_;
    std::condition_variable federation_cv_;
    std::mutex knowledge_cv_mutex_;
    std::condition_variable knowledge_cv_;
    std::mutex coordination_cv_mutex_;
    std::condition_variable coordination_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    FederationStats stats_;
    
    // Private methods
    
    /**
     * @brief Background federation thread
     */
    void FederationThread();
    
    /**
     * @brief Background knowledge sharing thread
     */
    void KnowledgeThread();
    
    /**
     * @brief Background coordination thread
     */
    void CoordinationThread();
    
    /**
     * @brief Validate federation contribution
     */
    bool ValidateContribution(const FederationContribution& contribution) const;
    
    /**
     * @brief Aggregate model weights using federated averaging
     */
    Status AggregateWeightsFederatedAveraging(const std::vector<FederationContribution>& contributions,
                                            std::vector<double>& aggregated_weights);
    
    /**
     * @brief Calculate contribution quality score
     */
    double CalculateContributionQuality(const FederationContribution& contribution) const;
    
    /**
     * @brief Update federation statistics
     */
    void UpdateStats(const FederationAggregation& aggregation);
    
    /**
     * @brief Calculate federation effectiveness
     */
    double CalculateFederationEffectiveness(const std::string& federation_id) const;
    
    /**
     * @brief Synchronize with remote clusters
     */
    Status SynchronizeWithRemoteClusters(const std::vector<std::string>& cluster_ids);
    
    /**
     * @brief Handle federation round timeout
     */
    Status HandleFederationRoundTimeout(const std::string& round_id);
    
    /**
     * @brief Process knowledge sharing requests
     */
    Status ProcessKnowledgeSharingRequests();
    
    /**
     * @brief Validate knowledge sharing permissions
     */
    bool ValidateKnowledgeSharingPermissions(const std::string& source_cluster_id,
                                           const std::string& target_cluster_id) const;
    
    /**
     * @brief Encrypt knowledge for sharing
     */
    Status EncryptKnowledge(KnowledgeSharing& knowledge);
    
    /**
     * @brief Decrypt received knowledge
     */
    Status DecryptKnowledge(KnowledgeSharing& knowledge);
    
    /**
     * @brief Calculate knowledge relevance score
     */
    double CalculateKnowledgeRelevance(const KnowledgeSharing& knowledge,
                                     const std::string& target_cluster_id) const;
    
    /**
     * @brief Update participant trust scores
     */
    Status UpdateParticipantTrustScores();
    
    /**
     * @brief Detect malicious participants
     */
    std::vector<std::string> DetectMaliciousParticipants(const std::string& federation_id) const;
    
    /**
     * @brief Apply differential privacy to contributions
     */
    Status ApplyDifferentialPrivacy(std::vector<FederationContribution>& contributions,
                                  double privacy_budget);
};

} // namespace edge_ai
