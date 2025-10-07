/**
 * @file security_manager.h
 * @brief Security and trust layer for autonomous edge AI governance
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
// TODO: Optional cryptography support - uncomment when crypto++ is available
// #include <cryptopp/cryptlib.h>
// #include <cryptopp/aes.h>
// #include <cryptopp/modes.h>
// #include <cryptopp/rsa.h>
// #include <cryptopp/sha.h>
// #include <cryptopp/hmac.h>
#include "core/types.h"
#include "distributed/cluster_types.h"
#include "governance/governance_manager.h"
#include "federation/federation_manager.h"
#include "analytics/telemetry_analytics.h"

// Forward declarations
namespace edge_ai {
    class ClusterManager;
    class GovernanceManager;
    class FederationManager;
    class TelemetryAnalytics;
}

namespace edge_ai {

/**
 * @brief Security threat types
 */
enum class SecurityThreatType {
    MALICIOUS_NODE = 0,
    DATA_POISONING = 1,
    MODEL_POISONING = 2,
    PRIVACY_BREACH = 3,
    UNAUTHORIZED_ACCESS = 4,
    DENIAL_OF_SERVICE = 5,
    MAN_IN_THE_MIDDLE = 6,
    REPLAY_ATTACK = 7,
    BYZANTINE_FAULT = 8,
    ADVERSARIAL_ATTACK = 9
};

/**
 * @brief Security threat severity levels
 */
enum class SecurityThreatSeverity {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
    CRITICAL = 3,
    EMERGENCY = 4
};

/**
 * @brief Security threat detection result
 */
struct SecurityThreatDetection {
    std::string threat_id;
    std::string cluster_id;
    std::string node_id;
    std::string source_id;
    
    // Threat characteristics
    SecurityThreatType threat_type;
    SecurityThreatSeverity severity;
    std::string threat_description;
    std::string attack_vector;
    
    // Detection metadata
    std::chrono::steady_clock::time_point detection_time;
    std::chrono::milliseconds detection_duration;
    double confidence_score;
    bool is_confirmed;
    
    // Impact assessment
    std::vector<std::string> affected_services;
    std::vector<std::string> affected_tenants;
    double impact_score;
    std::string impact_description;
    
    // Evidence and indicators
    std::map<std::string, std::string> evidence_indicators;
    std::vector<std::string> behavioral_anomalies;
    std::vector<std::string> network_anomalies;
    std::vector<std::string> performance_anomalies;
    
    // Response actions
    std::vector<std::string> immediate_actions;
    std::vector<std::string> mitigation_strategies;
    std::vector<std::string> recovery_actions;
    double response_urgency;
    
    // Investigation
    std::string investigation_status;
    std::vector<std::string> investigation_notes;
    std::string root_cause_analysis;
    std::vector<std::string> similar_threats;
};

/**
 * @brief Privacy-preserving protocol configuration
 */
struct PrivacyPreservingConfig {
    std::string config_id;
    std::string protocol_name;
    
    // Differential privacy parameters
    double epsilon; // Privacy budget
    double delta;   // Failure probability
    bool adaptive_epsilon_enabled;
    double max_epsilon_per_query;
    
    // Homomorphic encryption parameters
    bool homomorphic_encryption_enabled;
    std::string encryption_scheme;
    uint32_t key_size;
    std::string key_generation_method;
    
    // Secure multi-party computation
    bool smpc_enabled;
    uint32_t min_participants;
    uint32_t max_participants;
    std::string smpc_protocol;
    
    // Federated learning privacy
    bool federated_privacy_enabled;
    double noise_scale;
    double clipping_threshold;
    bool secure_aggregation_enabled;
    
    // Data anonymization
    bool data_anonymization_enabled;
    std::vector<std::string> anonymization_techniques;
    double k_anonymity_threshold;
    double l_diversity_threshold;
    
    // Privacy monitoring
    bool privacy_monitoring_enabled;
    std::vector<std::string> monitored_privacy_metrics;
    double privacy_violation_threshold;
    std::chrono::milliseconds privacy_audit_interval;
};

/**
 * @brief Trust score calculation
 */
struct TrustScore {
    std::string entity_id;
    std::string entity_type; // "node", "cluster", "tenant", "user"
    
    // Trust components
    double behavioral_trust;
    double performance_trust;
    double security_trust;
    double reliability_trust;
    double collaboration_trust;
    
    // Overall trust score
    double overall_trust_score;
    double trust_confidence;
    std::string trust_level; // "untrusted", "low", "medium", "high", "trusted"
    
    // Trust history
    std::vector<double> trust_history;
    double trust_trend;
    std::chrono::steady_clock::time_point last_updated;
    
    // Trust factors
    std::map<std::string, double> trust_factors;
    std::vector<std::string> trust_indicators;
    std::vector<std::string> trust_violations;
    
    // Recommendations
    std::vector<std::string> trust_improvement_recommendations;
    double trust_recovery_time;
    std::string trust_status;
};

/**
 * @brief Secure communication channel
 */
struct SecureChannel {
    std::string channel_id;
    std::string source_id;
    std::string destination_id;
    
    // Encryption parameters
    std::string encryption_algorithm;
    std::string key_exchange_method;
    uint32_t key_size;
    std::chrono::milliseconds key_rotation_interval;
    
    // Authentication
    bool mutual_authentication_enabled;
    std::string authentication_method;
    std::string certificate_authority;
    std::chrono::steady_clock::time_point certificate_expiry;
    
    // Integrity protection
    bool integrity_protection_enabled;
    std::string integrity_algorithm;
    bool replay_protection_enabled;
    uint32_t sequence_number_window;
    
    // Channel status
    bool is_active;
    bool is_secure;
    std::chrono::steady_clock::time_point established_time;
    std::chrono::steady_clock::time_point last_activity;
    
    // Performance metrics
    double latency_ms;
    double throughput_mbps;
    uint64_t messages_sent;
    uint64_t messages_received;
    uint64_t encryption_errors;
    uint64_t decryption_errors;
    
    // Security metrics
    double security_score;
    std::vector<std::string> security_violations;
    std::vector<std::string> authentication_failures;
    bool is_compromised;
};

/**
 * @brief Security policy definition
 */
struct SecurityPolicy {
    std::string policy_id;
    std::string name;
    std::string description;
    
    // Policy scope
    std::vector<std::string> target_entities;
    std::vector<std::string> target_services;
    std::vector<std::string> target_tenants;
    
    // Policy rules
    std::vector<std::string> access_rules;
    std::vector<std::string> encryption_rules;
    std::vector<std::string> authentication_rules;
    std::vector<std::string> authorization_rules;
    
    // Policy enforcement
    bool auto_enforcement_enabled;
    std::vector<std::string> enforcement_actions;
    std::chrono::milliseconds enforcement_timeout;
    double policy_priority;
    
    // Policy metadata
    std::string policy_version;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_updated;
    std::string created_by;
    bool is_active;
    
    // Compliance requirements
    std::vector<std::string> compliance_standards;
    std::vector<std::string> audit_requirements;
    std::chrono::milliseconds audit_frequency;
    std::string compliance_officer;
};

/**
 * @brief Security statistics
 */
struct SecurityStats {
    // Threat detection
    std::atomic<uint64_t> total_threats_detected{0};
    std::atomic<uint64_t> confirmed_threats{0};
    std::atomic<uint64_t> false_positives{0};
    std::atomic<uint64_t> threats_blocked{0};
    std::atomic<double> threat_detection_accuracy{0.0};
    
    // Security incidents
    std::atomic<uint64_t> security_incidents{0};
    std::atomic<uint64_t> incidents_resolved{0};
    std::atomic<uint64_t> incidents_escalated{0};
    std::atomic<double> avg_incident_resolution_time_ms{0.0};
    
    // Trust management
    std::atomic<uint64_t> trust_assessments{0};
    std::atomic<uint64_t> trust_violations{0};
    std::atomic<double> avg_trust_score{0.0};
    std::atomic<double> trust_improvement_rate{0.0};
    
    // Privacy protection
    std::atomic<uint64_t> privacy_checks{0};
    std::atomic<uint64_t> privacy_violations{0};
    std::atomic<double> privacy_compliance_rate{0.0};
    std::atomic<double> privacy_budget_consumed{0.0};
    
    // Encryption and authentication
    std::atomic<uint64_t> encryption_operations{0};
    std::atomic<uint64_t> authentication_attempts{0};
    std::atomic<uint64_t> authentication_failures{0};
    std::atomic<double> authentication_success_rate{0.0};
    
    // Performance impact
    std::atomic<double> security_overhead_percent{0.0};
    std::atomic<double> avg_security_processing_time_ms{0.0};
    std::atomic<double> security_throughput{0.0};
    
    /**
     * @brief Get a snapshot of current statistics
     */
    struct Snapshot {
        uint64_t total_threats_detected;
        uint64_t confirmed_threats;
        uint64_t false_positives;
        uint64_t threats_blocked;
        double threat_detection_accuracy;
        uint64_t security_incidents;
        uint64_t incidents_resolved;
        uint64_t incidents_escalated;
        double avg_incident_resolution_time_ms;
        uint64_t trust_assessments;
        uint64_t trust_violations;
        double avg_trust_score;
        double trust_improvement_rate;
        uint64_t privacy_checks;
        uint64_t privacy_violations;
        double privacy_compliance_rate;
        double privacy_budget_consumed;
        uint64_t encryption_operations;
        uint64_t authentication_attempts;
        uint64_t authentication_failures;
        double authentication_success_rate;
        double security_overhead_percent;
        double avg_security_processing_time_ms;
        double security_throughput;
    };
    
    Snapshot GetSnapshot() const {
        return {
            total_threats_detected.load(),
            confirmed_threats.load(),
            false_positives.load(),
            threats_blocked.load(),
            threat_detection_accuracy.load(),
            security_incidents.load(),
            incidents_resolved.load(),
            incidents_escalated.load(),
            avg_incident_resolution_time_ms.load(),
            trust_assessments.load(),
            trust_violations.load(),
            avg_trust_score.load(),
            trust_improvement_rate.load(),
            privacy_checks.load(),
            privacy_violations.load(),
            privacy_compliance_rate.load(),
            privacy_budget_consumed.load(),
            encryption_operations.load(),
            authentication_attempts.load(),
            authentication_failures.load(),
            authentication_success_rate.load(),
            security_overhead_percent.load(),
            avg_security_processing_time_ms.load(),
            security_throughput.load()
        };
    }
};

/**
 * @class SecurityManager
 * @brief Security and trust layer for autonomous edge AI governance
 */
class SecurityManager {
public:
    explicit SecurityManager(std::shared_ptr<ClusterManager> cluster_manager,
                           std::shared_ptr<GovernanceManager> governance_manager,
                           std::shared_ptr<FederationManager> federation_manager,
                           std::shared_ptr<TelemetryAnalytics> telemetry_analytics);
    virtual ~SecurityManager();
    
    /**
     * @brief Initialize the security manager
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the security manager
     */
    Status Shutdown();
    
    /**
     * @brief Check if the security manager is initialized
     */
    bool IsInitialized() const;
    
    // Threat Detection and Response
    
    /**
     * @brief Detect security threats
     */
    Status DetectSecurityThreats(const std::string& cluster_id,
                               std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Detect specific threat type
     */
    Status DetectThreatType(SecurityThreatType threat_type,
                          const std::string& cluster_id,
                          std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Respond to security threat
     */
    Status RespondToSecurityThreat(const std::string& threat_id,
                                 const std::vector<std::string>& response_actions);
    
    /**
     * @brief Escalate security incident
     */
    Status EscalateSecurityIncident(const std::string& threat_id,
                                  const std::string& escalation_reason);
    
    /**
     * @brief Get security threat history
     */
    std::vector<SecurityThreatDetection> GetSecurityThreatHistory(
        const std::string& cluster_id,
        std::chrono::hours lookback_hours = std::chrono::hours(24)) const;
    
    // Privacy-Preserving Protocols
    
    /**
     * @brief Configure privacy-preserving protocols
     */
    Status ConfigurePrivacyPreservingProtocols(const PrivacyPreservingConfig& config);
    
    /**
     * @brief Apply differential privacy
     */
    Status ApplyDifferentialPrivacy(const std::vector<double>& data,
                                  double epsilon,
                                  std::vector<double>& privatized_data);
    
    /**
     * @brief Perform secure aggregation
     */
    Status PerformSecureAggregation(const std::vector<std::vector<double>>& local_data,
                                  const std::vector<std::string>& participant_ids,
                                  std::vector<double>& aggregated_data);
    
    /**
     * @brief Encrypt sensitive data
     */
    Status EncryptSensitiveData(const std::vector<uint8_t>& plaintext,
                              const std::string& key_id,
                              std::vector<uint8_t>& ciphertext);
    
    /**
     * @brief Decrypt sensitive data
     */
    Status DecryptSensitiveData(const std::vector<uint8_t>& ciphertext,
                              const std::string& key_id,
                              std::vector<uint8_t>& plaintext);
    
    /**
     * @brief Validate privacy guarantees
     */
    Status ValidatePrivacyGuarantees(const std::string& operation_id,
                                   bool& privacy_guarantees_met);
    
    // Trust Management
    
    /**
     * @brief Calculate trust score for entity
     */
    Status CalculateTrustScore(const std::string& entity_id,
                             const std::string& entity_type,
                             TrustScore& trust_score);
    
    /**
     * @brief Update trust score
     */
    Status UpdateTrustScore(const std::string& entity_id,
                          const std::map<std::string, double>& trust_factors);
    
    /**
     * @brief Get trust score history
     */
    std::vector<TrustScore> GetTrustScoreHistory(const std::string& entity_id,
                                               std::chrono::hours lookback_hours = std::chrono::hours(168)) const;
    
    /**
     * @brief Revoke trust for entity
     */
    Status RevokeTrust(const std::string& entity_id, const std::string& revocation_reason);
    
    /**
     * @brief Restore trust for entity
     */
    Status RestoreTrust(const std::string& entity_id, const std::string& restoration_reason);
    
    // Secure Communication
    
    /**
     * @brief Establish secure channel
     */
    Status EstablishSecureChannel(const std::string& source_id,
                                const std::string& destination_id,
                                SecureChannel& channel);
    
    /**
     * @brief Terminate secure channel
     */
    Status TerminateSecureChannel(const std::string& channel_id);
    
    /**
     * @brief Send secure message
     */
    Status SendSecureMessage(const std::string& channel_id,
                           const std::vector<uint8_t>& message,
                           std::vector<uint8_t>& encrypted_message);
    
    /**
     * @brief Receive secure message
     */
    Status ReceiveSecureMessage(const std::string& channel_id,
                              const std::vector<uint8_t>& encrypted_message,
                              std::vector<uint8_t>& decrypted_message);
    
    /**
     * @brief Get secure channel status
     */
    Status GetSecureChannelStatus(const std::string& channel_id, SecureChannel& channel) const;
    
    // Security Policy Management
    
    /**
     * @brief Create security policy
     */
    Status CreateSecurityPolicy(const SecurityPolicy& policy);
    
    /**
     * @brief Update security policy
     */
    Status UpdateSecurityPolicy(const std::string& policy_id, const SecurityPolicy& policy);
    
    /**
     * @brief Delete security policy
     */
    Status DeleteSecurityPolicy(const std::string& policy_id);
    
    /**
     * @brief Enforce security policy
     */
    Status EnforceSecurityPolicy(const std::string& policy_id,
                               const std::string& target_entity,
                               bool& policy_violated);
    
    /**
     * @brief Get security policies
     */
    std::vector<SecurityPolicy> GetSecurityPolicies() const;
    
    // Authentication and Authorization
    
    /**
     * @brief Authenticate entity
     */
    Status AuthenticateEntity(const std::string& entity_id,
                            const std::string& credentials,
                            bool& authentication_successful);
    
    /**
     * @brief Authorize operation
     */
    Status AuthorizeOperation(const std::string& entity_id,
                            const std::string& operation,
                            const std::string& resource,
                            bool& authorization_granted);
    
    /**
     * @brief Generate authentication token
     */
    Status GenerateAuthenticationToken(const std::string& entity_id,
                                     const std::string& token_type,
                                     std::string& token);
    
    /**
     * @brief Validate authentication token
     */
    Status ValidateAuthenticationToken(const std::string& token,
                                     std::string& entity_id,
                                     bool& token_valid);
    
    // Security Analytics and Reporting
    
    /**
     * @brief Generate security report
     */
    Status GenerateSecurityReport(const std::string& cluster_id,
                                std::map<std::string, double>& security_metrics);
    
    /**
     * @brief Generate threat intelligence report
     */
    Status GenerateThreatIntelligenceReport(const std::string& cluster_id,
                                          std::map<std::string, double>& threat_metrics);
    
    /**
     * @brief Generate compliance report
     */
    Status GenerateComplianceReport(const std::string& cluster_id,
                                  std::map<std::string, double>& compliance_metrics);
    
    /**
     * @brief Get security statistics
     */
    SecurityStats::Snapshot GetStats() const;
    
    /**
     * @brief Reset security statistics
     */
    void ResetStats();
    
    /**
     * @brief Generate security insights
     */
    Status GenerateSecurityInsights(std::vector<std::string>& insights);

private:
    // Core components
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<GovernanceManager> governance_manager_;
    std::shared_ptr<FederationManager> federation_manager_;
    std::shared_ptr<TelemetryAnalytics> telemetry_analytics_;
    
    // State management
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Security storage
    mutable std::mutex threats_mutex_;
    std::deque<SecurityThreatDetection> threat_history_;
    std::map<std::string, SecurityThreatDetection> active_threats_;
    
    // Trust storage
    mutable std::mutex trust_mutex_;
    std::map<std::string, TrustScore> trust_scores_;
    std::map<std::string, std::deque<TrustScore>> trust_history_;
    
    // Secure channels storage
    mutable std::mutex channels_mutex_;
    std::map<std::string, SecureChannel> secure_channels_;
    
    // Security policies storage
    mutable std::mutex policies_mutex_;
    std::map<std::string, SecurityPolicy> security_policies_;
    
    // Privacy configuration storage
    mutable std::mutex privacy_mutex_;
    std::map<std::string, PrivacyPreservingConfig> privacy_configs_;
    
    // Cryptographic components
    std::map<std::string, std::vector<uint8_t>> encryption_keys_;
    std::map<std::string, std::vector<uint8_t>> decryption_keys_;
    std::map<std::string, std::string> authentication_tokens_;
    
    // Background threads
    std::thread threat_detection_thread_;
    std::thread trust_assessment_thread_;
    std::thread security_monitoring_thread_;
    std::thread privacy_protection_thread_;
    
    // Condition variables
    std::mutex threat_detection_cv_mutex_;
    std::condition_variable threat_detection_cv_;
    std::mutex trust_assessment_cv_mutex_;
    std::condition_variable trust_assessment_cv_;
    std::mutex security_monitoring_cv_mutex_;
    std::condition_variable security_monitoring_cv_;
    std::mutex privacy_protection_cv_mutex_;
    std::condition_variable privacy_protection_cv_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    SecurityStats stats_;
    
    // Private methods
    
    /**
     * @brief Background threat detection thread
     */
    void ThreatDetectionThread();
    
    /**
     * @brief Background trust assessment thread
     */
    void TrustAssessmentThread();
    
    /**
     * @brief Background security monitoring thread
     */
    void SecurityMonitoringThread();
    
    /**
     * @brief Background privacy protection thread
     */
    void PrivacyProtectionThread();
    
    /**
     * @brief Detect malicious nodes
     */
    Status DetectMaliciousNodes(const std::string& cluster_id,
                              std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Detect data poisoning attacks
     */
    Status DetectDataPoisoningAttacks(const std::string& cluster_id,
                                    std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Detect model poisoning attacks
     */
    Status DetectModelPoisoningAttacks(const std::string& cluster_id,
                                     std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Detect privacy breaches
     */
    Status DetectPrivacyBreaches(const std::string& cluster_id,
                               std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Detect unauthorized access
     */
    Status DetectUnauthorizedAccess(const std::string& cluster_id,
                                  std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Detect denial of service attacks
     */
    Status DetectDenialOfServiceAttacks(const std::string& cluster_id,
                                      std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Detect byzantine faults
     */
    Status DetectByzantineFaults(const std::string& cluster_id,
                               std::vector<SecurityThreatDetection>& threats);
    
    /**
     * @brief Calculate behavioral trust score
     */
    double CalculateBehavioralTrustScore(const std::string& entity_id) const;
    
    /**
     * @brief Calculate performance trust score
     */
    double CalculatePerformanceTrustScore(const std::string& entity_id) const;
    
    /**
     * @brief Calculate security trust score
     */
    double CalculateSecurityTrustScore(const std::string& entity_id) const;
    
    /**
     * @brief Calculate reliability trust score
     */
    double CalculateReliabilityTrustScore(const std::string& entity_id) const;
    
    /**
     * @brief Calculate collaboration trust score
     */
    double CalculateCollaborationTrustScore(const std::string& entity_id) const;
    
    /**
     * @brief Generate encryption key
     */
    Status GenerateEncryptionKey(const std::string& key_id, uint32_t key_size);
    
    /**
     * @brief Rotate encryption keys
     */
    Status RotateEncryptionKeys();
    
    /**
     * @brief Validate security policy
     */
    bool ValidateSecurityPolicy(const SecurityPolicy& policy) const;
    
    /**
     * @brief Update security statistics
     */
    void UpdateStats(const SecurityThreatDetection& threat);
    
    /**
     * @brief Update trust statistics
     */
    void UpdateTrustStats(const TrustScore& trust_score);
    
    /**
     * @brief Calculate security overhead
     */
    double CalculateSecurityOverhead() const;
    
    /**
     * @brief Cleanup expired data
     */
    Status CleanupExpiredData();
    
    /**
     * @brief Backup security state
     */
    Status BackupSecurityState(const std::string& backup_id);
    
    /**
     * @brief Restore security state
     */
    Status RestoreSecurityState(const std::string& backup_id);
    
    /**
     * @brief Initialize cryptographic components
     */
    Status InitializeCryptographicComponents();
    
    /**
     * @brief Perform secure key exchange
     */
    Status PerformSecureKeyExchange(const std::string& channel_id,
                                  const std::string& peer_id);
    
    /**
     * @brief Validate message integrity
     */
    Status ValidateMessageIntegrity(const std::vector<uint8_t>& message,
                                  const std::vector<uint8_t>& signature,
                                  bool& integrity_valid);
    
    /**
     * @brief Generate message signature
     */
    Status GenerateMessageSignature(const std::vector<uint8_t>& message,
                                  std::vector<uint8_t>& signature);
    
    /**
     * @brief Perform secure multi-party computation
     */
    Status PerformSecureMultiPartyComputation(const std::vector<std::vector<double>>& inputs,
                                            const std::vector<std::string>& participants,
                                            std::vector<double>& result);
    
    /**
     * @brief Apply homomorphic encryption
     */
    Status ApplyHomomorphicEncryption(const std::vector<double>& data,
                                    std::vector<std::vector<uint8_t>>& encrypted_data);
    
    /**
     * @brief Perform homomorphic computation
     */
    Status PerformHomomorphicComputation(const std::vector<std::vector<uint8_t>>& encrypted_data,
                                       const std::string& computation_type,
                                       std::vector<uint8_t>& encrypted_result);
    
    /**
     * @brief Decrypt homomorphic result
     */
    Status DecryptHomomorphicResult(const std::vector<uint8_t>& encrypted_result,
                                  std::vector<double>& decrypted_result);
};

} // namespace edge_ai
