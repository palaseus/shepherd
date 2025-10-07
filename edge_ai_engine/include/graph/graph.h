/**
 * @file graph.h
 * @brief Graph class for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Graph class that represents a directed acyclic graph (DAG)
 * of nodes and edges for multi-model, multi-device execution.
 */

#pragma once

#include "graph_types.h"
#include "core/types.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <atomic>
#include <queue>
#include <functional>

namespace edge_ai {

/**
 * @class GraphNode
 * @brief Represents a node in the execution graph
 */
class GraphNode {
public:
    GraphNode(const std::string& id, NodeType type, const NodeMetadata& metadata);
    ~GraphNode() = default;
    
    // Getters
    const std::string& GetId() const { return id_; }
    NodeType GetType() const { return type_; }
    const NodeMetadata& GetMetadata() const { return metadata_; }
    NodeStatus GetStatus() const { return status_.load(); }
    const std::vector<std::string>& GetInputEdges() const { return input_edges_; }
    const std::vector<std::string>& GetOutputEdges() const { return output_edges_; }
    const std::vector<Tensor>& GetInputs() const { return inputs_; }
    const std::vector<Tensor>& GetOutputs() const { return outputs_; }
    const NodeExecutionResult& GetExecutionResult() const { return execution_result_; }
    
    // Setters
    void SetStatus(NodeStatus status) { status_.store(status); }
    void SetInputs(std::vector<Tensor>&& inputs) { inputs_ = std::move(inputs); }
    void SetOutputs(std::vector<Tensor>&& outputs) { outputs_ = std::move(outputs); }
    void SetExecutionResult(NodeExecutionResult&& result) { execution_result_ = std::move(result); }
    
    // Move-only operations for Tensor vectors
    void AddInput(Tensor&& input) { inputs_.emplace_back(std::move(input)); }
    void AddOutput(Tensor&& output) { outputs_.emplace_back(std::move(output)); }
    
    // Edge management
    void AddInputEdge(const std::string& edge_id);
    void AddOutputEdge(const std::string& edge_id);
    void RemoveInputEdge(const std::string& edge_id);
    void RemoveOutputEdge(const std::string& edge_id);
    
    // Execution helpers
    bool IsReady() const;
    bool HasAllInputs() const;
    void ClearInputs();
    void ClearOutputs();
    
    // Validation
    bool Validate() const;
    
private:
    std::string id_;
    NodeType type_;
    NodeMetadata metadata_;
    std::atomic<NodeStatus> status_;
    std::vector<std::string> input_edges_;
    std::vector<std::string> output_edges_;
    std::vector<Tensor> inputs_;
    std::vector<Tensor> outputs_;
    NodeExecutionResult execution_result_;
    mutable std::mutex mutex_;
};

/**
 * @class GraphEdge
 * @brief Represents an edge in the execution graph
 */
class GraphEdge {
public:
    GraphEdge(const std::string& id, const std::string& source_node, 
              const std::string& target_node, EdgeType type, const EdgeMetadata& metadata);
    ~GraphEdge() = default;
    
    // Getters
    const std::string& GetId() const { return id_; }
    const std::string& GetSourceNode() const { return source_node_; }
    const std::string& GetTargetNode() const { return target_node_; }
    EdgeType GetType() const { return type_; }
    const EdgeMetadata& GetMetadata() const { return metadata_; }
    const std::vector<Tensor>& GetData() const { return data_; }
    bool IsDataReady() const { return data_ready_.load(); }
    
    // Setters
    void SetData(std::vector<Tensor>&& data);
    void SetDataReady(bool ready) { data_ready_.store(ready); }
    
    // Data management
    void ClearData();
    bool TransferData(std::vector<Tensor>& target_data);
    
    // Validation
    bool Validate() const;
    
private:
    std::string id_;
    std::string source_node_;
    std::string target_node_;
    EdgeType type_;
    EdgeMetadata metadata_;
    std::vector<Tensor> data_;
    std::atomic<bool> data_ready_;
    mutable std::mutex mutex_;
};

/**
 * @class Graph
 * @brief Represents a directed acyclic graph (DAG) for execution
 */
class Graph {
public:
    Graph(const std::string& id, const GraphConfig& config = GraphConfig{});
    ~Graph() = default;
    
    // Graph management
    bool AddNode(std::unique_ptr<GraphNode> node);
    bool AddEdge(std::unique_ptr<GraphEdge> edge);
    bool RemoveNode(const std::string& node_id);
    bool RemoveEdge(const std::string& edge_id);
    
    // Getters
    const std::string& GetId() const { return id_; }
    const GraphConfig& GetConfig() const { return config_; }
    GraphNode* GetNode(const std::string& node_id);
    GraphEdge* GetEdge(const std::string& edge_id);
    const std::vector<std::string>& GetNodeIds() const { return node_ids_; }
    const std::vector<std::string>& GetEdgeIds() const { return edge_ids_; }
    const GraphExecutionStats& GetStats() const { return stats_; }
    
    // Graph analysis
    bool IsValid() const;
    bool IsAcyclic() const;
    std::vector<std::string> GetTopologicalOrder() const;
    std::vector<std::string> GetReadyNodes() const;
    std::vector<std::string> GetNodesByType(NodeType type) const;
    std::vector<std::string> GetSourceNodes() const;
    std::vector<std::string> GetSinkNodes() const;
    
    // Execution state
    bool IsExecutionComplete() const;
    bool HasFailedNodes() const;
    void ResetExecutionState();
    
    // Statistics
    GraphExecutionStats::Snapshot GetStatsSnapshot() const;
    void UpdateStats(const NodeExecutionResult& result);
    
    // Validation
    bool ValidateGraph() const;
    std::vector<std::string> GetValidationErrors() const;
    
    // Serialization
    std::string ToJson() const;
    bool FromJson(const std::string& json);
    
private:
    std::string id_;
    GraphConfig config_;
    std::unordered_map<std::string, std::unique_ptr<GraphNode>> nodes_;
    std::unordered_map<std::string, std::unique_ptr<GraphEdge>> edges_;
    std::vector<std::string> node_ids_;
    std::vector<std::string> edge_ids_;
    GraphExecutionStats stats_;
    mutable std::mutex mutex_;
    
    // Helper methods
    bool HasCycles() const;
    void DFS(const std::string& node_id, std::unordered_set<std::string>& visited,
             std::unordered_set<std::string>& rec_stack, bool& has_cycle) const;
    std::vector<std::string> TopologicalSort() const;
};

/**
 * @class GraphBuilder
 * @brief Builder class for constructing graphs
 */
class GraphBuilder {
public:
    GraphBuilder(const std::string& graph_id, const GraphConfig& config = GraphConfig{});
    ~GraphBuilder() = default;
    
    // Node operations
    GraphBuilder& AddNode(const std::string& node_id, NodeType type, const NodeMetadata& metadata);
    GraphBuilder& AddModelNode(const std::string& node_id, const std::string& model_path, 
                               ModelType model_type, const NodeMetadata& metadata = NodeMetadata{});
    GraphBuilder& AddProcessingNode(const std::string& node_id, const NodeMetadata& metadata = NodeMetadata{});
    GraphBuilder& AddCustomNode(const std::string& node_id, const NodeMetadata& metadata = NodeMetadata{});
    
    // Edge operations
    GraphBuilder& AddEdge(const std::string& edge_id, const std::string& source_node,
                          const std::string& target_node, EdgeType type = EdgeType::DATA_FLOW,
                          const EdgeMetadata& metadata = EdgeMetadata{});
    GraphBuilder& ConnectNodes(const std::string& source_node, const std::string& target_node,
                               const std::string& edge_name = "");
    
    // Pipeline operations
    GraphBuilder& CreatePipeline(const std::vector<std::string>& node_ids);
    GraphBuilder& CreateParallel(const std::vector<std::string>& node_ids);
    
    // Build
    std::unique_ptr<Graph> Build();
    bool IsValid() const;
    std::vector<std::string> GetValidationErrors() const;
    
private:
    std::string graph_id_;
    GraphConfig config_;
    std::unordered_map<std::string, std::unique_ptr<GraphNode>> nodes_;
    std::unordered_map<std::string, std::unique_ptr<GraphEdge>> edges_;
    std::vector<std::string> node_ids_;
    std::vector<std::string> edge_ids_;
    
    std::string GenerateEdgeId(const std::string& source, const std::string& target) const;
};

} // namespace edge_ai
