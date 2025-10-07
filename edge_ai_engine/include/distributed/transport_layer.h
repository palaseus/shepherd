/**
 * @file transport_layer.h
 * @brief Pluggable transport layer for distributed communication
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
#include "core/types.h"

namespace edge_ai {

/**
 * @brief Transport protocol enumeration
 */
enum class TransportProtocol {
    MOCK = 0,           // Mock transport for testing
    TCP = 1,            // TCP sockets
    UDP = 2,            // UDP sockets
    GRPC = 3,           // gRPC
    QUIC = 4,           // QUIC protocol
    RDMA = 5,           // RDMA (InfiniBand)
    CUSTOM = 6          // Custom transport
};

/**
 * @brief Message types for distributed communication
 */
enum class MessageType {
    HEARTBEAT = 0,
    TASK_REQUEST = 1,
    TASK_RESPONSE = 2,
    DATA_TRANSFER = 3,
    MIGRATION_REQUEST = 4,
    MIGRATION_RESPONSE = 5,
    NODE_REGISTRATION = 6,
    NODE_UNREGISTRATION = 7,
    CLUSTER_UPDATE = 8,
    ERROR_MESSAGE = 9,
    CUSTOM = 10
};

/**
 * @brief Network message structure
 */
struct NetworkMessage {
    std::string message_id;
    std::string source_node_id;
    std::string target_node_id;
    MessageType type{MessageType::CUSTOM};
    std::vector<uint8_t> payload;
    std::map<std::string, std::string> headers;
    
    // Message metadata
    std::chrono::steady_clock::time_point timestamp;
    uint32_t sequence_number{0};
    bool requires_ack{false};
    uint32_t timeout_ms{5000};
    
    NetworkMessage() = default;
    NetworkMessage(const std::string& id, const std::string& source, 
                  const std::string& target, MessageType t)
        : message_id(id), source_node_id(source), target_node_id(target), type(t) {
        timestamp = std::chrono::steady_clock::now();
    }
    
    // Disable copy, enable move
    NetworkMessage(const NetworkMessage&) = delete;
    NetworkMessage& operator=(const NetworkMessage&) = delete;
    NetworkMessage(NetworkMessage&&) = default;
    NetworkMessage& operator=(NetworkMessage&&) = default;
};

/**
 * @brief Transport layer configuration
 */
struct TransportConfig {
    TransportProtocol protocol{TransportProtocol::MOCK};
    std::string local_address{"127.0.0.1"};
    uint16_t local_port{8080};
    uint32_t max_message_size_mb{10};
    uint32_t connection_timeout_ms{5000};
    uint32_t send_timeout_ms{10000};
    uint32_t receive_timeout_ms{10000};
    
    // Protocol-specific settings
    bool enable_compression{true};
    bool enable_encryption{false};
    uint32_t compression_level{6};
    std::string encryption_key;
    
    // Reliability settings
    bool enable_retry{true};
    uint32_t max_retries{3};
    uint32_t retry_delay_ms{1000};
    bool enable_acknowledgments{true};
    
    // Performance settings
    uint32_t send_buffer_size_kb{64};
    uint32_t receive_buffer_size_kb{64};
    uint32_t max_concurrent_connections{100};
    bool enable_nagle_algorithm{false};
    
    TransportConfig() = default;
};

/**
 * @brief Transport layer statistics
 */
struct TransportStats {
    std::atomic<uint64_t> messages_sent{0};
    std::atomic<uint64_t> messages_received{0};
    std::atomic<uint64_t> messages_failed{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    
    // Performance metrics
    std::atomic<double> avg_send_latency_ms{0.0};
    std::atomic<double> avg_receive_latency_ms{0.0};
    std::atomic<double> throughput_mbps{0.0};
    std::atomic<uint32_t> active_connections{0};
    
    // Error metrics
    std::atomic<uint32_t> connection_failures{0};
    std::atomic<uint32_t> timeout_errors{0};
    std::atomic<uint32_t> retry_events{0};
    
    TransportStats() = default;
    
    struct Snapshot {
        uint64_t messages_sent;
        uint64_t messages_received;
        uint64_t messages_failed;
        uint64_t bytes_sent;
        uint64_t bytes_received;
        double avg_send_latency_ms;
        double avg_receive_latency_ms;
        double throughput_mbps;
        uint32_t active_connections;
        uint32_t connection_failures;
        uint32_t timeout_errors;
        uint32_t retry_events;
    };
    
    Snapshot GetSnapshot() const {
        Snapshot snapshot;
        snapshot.messages_sent = messages_sent.load();
        snapshot.messages_received = messages_received.load();
        snapshot.messages_failed = messages_failed.load();
        snapshot.bytes_sent = bytes_sent.load();
        snapshot.bytes_received = bytes_received.load();
        snapshot.avg_send_latency_ms = avg_send_latency_ms.load();
        snapshot.avg_receive_latency_ms = avg_receive_latency_ms.load();
        snapshot.throughput_mbps = throughput_mbps.load();
        snapshot.active_connections = active_connections.load();
        snapshot.connection_failures = connection_failures.load();
        snapshot.timeout_errors = timeout_errors.load();
        snapshot.retry_events = retry_events.load();
        return snapshot;
    }
};

/**
 * @brief Abstract transport layer interface
 */
class TransportLayer {
public:
    using MessageHandler = std::function<void(const NetworkMessage&)>;
    using ConnectionHandler = std::function<void(const std::string&, bool)>;
    
    /**
     * @brief Constructor
     * @param config Transport configuration
     */
    explicit TransportLayer(const TransportConfig& config) : config_(config) {}
    
    /**
     * @brief Destructor
     */
    virtual ~TransportLayer() = default;
    
    // Lifecycle management
    virtual Status Initialize() = 0;
    virtual Status Shutdown() = 0;
    virtual bool IsInitialized() const = 0;
    
    // Message handling
    virtual Status SendMessage(const NetworkMessage& message) = 0;
    virtual std::future<NetworkMessage> SendMessageAsync(const NetworkMessage& message) = 0;
    virtual Status BroadcastMessage(const NetworkMessage& message, 
                                   const std::vector<std::string>& target_nodes) = 0;
    
    // Connection management
    virtual Status ConnectToNode(const std::string& node_id, 
                                const std::string& address, uint16_t port) = 0;
    virtual Status DisconnectFromNode(const std::string& node_id) = 0;
    virtual Status DisconnectAll() = 0;
    virtual bool IsConnected(const std::string& node_id) const = 0;
    
    // Event handling
    virtual Status RegisterMessageHandler(MessageType type, MessageHandler handler) = 0;
    virtual Status RegisterConnectionHandler(ConnectionHandler handler) = 0;
    virtual Status UnregisterMessageHandler(MessageType type) = 0;
    
    // Statistics and monitoring
    virtual TransportStats::Snapshot GetStats() const = 0;
    virtual void ResetStats() = 0;
    
    // Configuration
    virtual TransportConfig GetConfig() const = 0;
    virtual Status UpdateConfig(const TransportConfig& config) = 0;

protected:
    TransportConfig config_;
};

/**
 * @brief Mock transport layer for testing and development
 */
class MockTransportLayer : public TransportLayer {
public:
    /**
     * @brief Constructor
     * @param config Transport configuration
     */
    explicit MockTransportLayer(const TransportConfig& config);
    
    /**
     * @brief Destructor
     */
    ~MockTransportLayer() override;
    
    // Lifecycle management
    Status Initialize() override;
    Status Shutdown() override;
    bool IsInitialized() const override;
    
    // Message handling
    Status SendMessage(const NetworkMessage& message) override;
    std::future<NetworkMessage> SendMessageAsync(const NetworkMessage& message) override;
    Status BroadcastMessage(const NetworkMessage& message, 
                           const std::vector<std::string>& target_nodes) override;
    
    // Connection management
    Status ConnectToNode(const std::string& node_id, 
                        const std::string& address, uint16_t port) override;
    Status DisconnectFromNode(const std::string& node_id) override;
    Status DisconnectAll() override;
    bool IsConnected(const std::string& node_id) const override;
    
    // Event handling
    Status RegisterMessageHandler(MessageType type, MessageHandler handler) override;
    Status RegisterConnectionHandler(ConnectionHandler handler) override;
    Status UnregisterMessageHandler(MessageType type) override;
    
    // Statistics and monitoring
    TransportStats::Snapshot GetStats() const override;
    void ResetStats() override;
    
    // Configuration
    TransportConfig GetConfig() const override;
    Status UpdateConfig(const TransportConfig& config) override;
    
    // Mock-specific methods
    Status SimulateNetworkDelay(uint32_t delay_ms);
    Status SimulateNetworkFailure(const std::string& node_id, bool fail);
    Status SimulateMessageLoss(double loss_rate);

private:
    // Internal methods
    void MessageProcessingThread();
    void SimulateNetworkLatency(const NetworkMessage& message);
    bool ShouldDropMessage(const NetworkMessage& message);
    
    // Configuration
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_requested_{false};
    
    // Message handling
    mutable std::mutex message_handlers_mutex_;
    std::map<MessageType, MessageHandler> message_handlers_;
    ConnectionHandler connection_handler_;
    
    // Connection management
    mutable std::mutex connections_mutex_;
    std::set<std::string> connected_nodes_;
    std::map<std::string, std::string> node_addresses_;
    std::map<std::string, uint16_t> node_ports_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    TransportStats stats_;
    
    // Threading
    std::thread message_processing_thread_;
    std::condition_variable message_processing_cv_;
    mutable std::mutex message_processing_mutex_;
    
    // Mock simulation
    std::atomic<uint32_t> simulated_delay_ms_{0};
    std::atomic<double> message_loss_rate_{0.0};
    mutable std::mutex network_failures_mutex_;
    std::set<std::string> failed_nodes_;
};

/**
 * @brief Transport layer factory
 */
class TransportLayerFactory {
public:
    /**
     * @brief Create transport layer instance
     * @param config Transport configuration
     * @return Unique pointer to transport layer
     */
    static std::unique_ptr<TransportLayer> CreateTransportLayer(const TransportConfig& config);
    
    /**
     * @brief Get supported protocols
     * @return Vector of supported transport protocols
     */
    static std::vector<TransportProtocol> GetSupportedProtocols();
    
    /**
     * @brief Check if protocol is supported
     * @param protocol Transport protocol
     * @return True if protocol is supported
     */
    static bool IsProtocolSupported(TransportProtocol protocol);
};

} // namespace edge_ai
