/**
 * @file graph.cpp
 * @brief Implementation of Graph classes for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 */

#include "graph/graph.h"
#include "profiling/profiler.h"
#include <algorithm>
#include <queue>
#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace edge_ai {

// GraphNode Implementation

GraphNode::GraphNode(const std::string& id, NodeType type, const NodeMetadata& metadata)
    : id_(id), type_(type), metadata_(metadata), status_(NodeStatus::PENDING) {
}

void GraphNode::AddInputEdge(const std::string& edge_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(input_edges_.begin(), input_edges_.end(), edge_id) == input_edges_.end()) {
        input_edges_.push_back(edge_id);
    }
}

void GraphNode::AddOutputEdge(const std::string& edge_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (std::find(output_edges_.begin(), output_edges_.end(), edge_id) == output_edges_.end()) {
        output_edges_.push_back(edge_id);
    }
}

void GraphNode::RemoveInputEdge(const std::string& edge_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(input_edges_.begin(), input_edges_.end(), edge_id);
    if (it != input_edges_.end()) {
        input_edges_.erase(it);
    }
}

void GraphNode::RemoveOutputEdge(const std::string& edge_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find(output_edges_.begin(), output_edges_.end(), edge_id);
    if (it != output_edges_.end()) {
        output_edges_.erase(it);
    }
}

bool GraphNode::IsReady() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_.load() == NodeStatus::READY && HasAllInputs();
}

bool GraphNode::HasAllInputs() const {
    return !input_edges_.empty() && inputs_.size() == input_edges_.size();
}

void GraphNode::ClearInputs() {
    std::lock_guard<std::mutex> lock(mutex_);
    inputs_.clear();
}

void GraphNode::ClearOutputs() {
    std::lock_guard<std::mutex> lock(mutex_);
    outputs_.clear();
}

bool GraphNode::Validate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Basic validation
    if (id_.empty()) return false;
    if (metadata_.name.empty()) return false;
    
    // Type-specific validation
    if (type_ == NodeType::MODEL_INFERENCE) {
        if (metadata_.model_path.empty()) return false;
        if (metadata_.input_shapes.empty() || metadata_.output_shapes.empty()) return false;
    }
    
    return true;
}

// GraphEdge Implementation

GraphEdge::GraphEdge(const std::string& id, const std::string& source_node,
                     const std::string& target_node, EdgeType type, const EdgeMetadata& metadata)
    : id_(id), source_node_(source_node), target_node_(target_node), 
      type_(type), metadata_(metadata), data_ready_(false) {
}

void GraphEdge::SetData(std::vector<Tensor>&& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_ = std::move(data);
    data_ready_.store(true);
}

void GraphEdge::ClearData() {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.clear();
    data_ready_.store(false);
}

bool GraphEdge::TransferData(std::vector<Tensor>& target_data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (data_ready_.load() && !data_.empty()) {
        target_data = std::move(data_);
        data_ready_.store(false);
        return true;
    }
    return false;
}

bool GraphEdge::Validate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Basic validation
    if (id_.empty()) return false;
    if (source_node_.empty() || target_node_.empty()) return false;
    if (source_node_ == target_node_) return false; // No self-loops
    
    return true;
}

// Graph Implementation

Graph::Graph(const std::string& id, const GraphConfig& config)
    : id_(id), config_(config) {
}

bool Graph::AddNode(std::unique_ptr<GraphNode> node) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!node || node->GetId().empty()) return false;
    if (nodes_.find(node->GetId()) != nodes_.end()) return false;
    
    nodes_[node->GetId()] = std::move(node);
    node_ids_.push_back(node->GetId());
    return true;
}

bool Graph::AddEdge(std::unique_ptr<GraphEdge> edge) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!edge || edge->GetId().empty()) return false;
    if (edges_.find(edge->GetId()) != edges_.end()) return false;
    
    // Validate that source and target nodes exist
    if (nodes_.find(edge->GetSourceNode()) == nodes_.end() ||
        nodes_.find(edge->GetTargetNode()) == nodes_.end()) {
        return false;
    }
    
    edges_[edge->GetId()] = std::move(edge);
    edge_ids_.push_back(edge->GetId());
    
    // Update node connections
    nodes_[edge->GetSourceNode()]->AddOutputEdge(edge->GetId());
    nodes_[edge->GetTargetNode()]->AddInputEdge(edge->GetId());
    
    return true;
}

bool Graph::RemoveNode(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto node_it = nodes_.find(node_id);
    if (node_it == nodes_.end()) return false;
    
    // Remove all connected edges
    std::vector<std::string> edges_to_remove;
    for (const auto& edge_id : node_it->second->GetInputEdges()) {
        edges_to_remove.push_back(edge_id);
    }
    for (const auto& edge_id : node_it->second->GetOutputEdges()) {
        edges_to_remove.push_back(edge_id);
    }
    
    for (const auto& edge_id : edges_to_remove) {
        RemoveEdge(edge_id);
    }
    
    // Remove node
    nodes_.erase(node_it);
    auto id_it = std::find(node_ids_.begin(), node_ids_.end(), node_id);
    if (id_it != node_ids_.end()) {
        node_ids_.erase(id_it);
    }
    
    return true;
}

bool Graph::RemoveEdge(const std::string& edge_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto edge_it = edges_.find(edge_id);
    if (edge_it == edges_.end()) return false;
    
    // Update node connections
    auto source_node = nodes_.find(edge_it->second->GetSourceNode());
    auto target_node = nodes_.find(edge_it->second->GetTargetNode());
    
    if (source_node != nodes_.end()) {
        source_node->second->RemoveOutputEdge(edge_id);
    }
    if (target_node != nodes_.end()) {
        target_node->second->RemoveInputEdge(edge_id);
    }
    
    // Remove edge
    edges_.erase(edge_it);
    auto id_it = std::find(edge_ids_.begin(), edge_ids_.end(), edge_id);
    if (id_it != edge_ids_.end()) {
        edge_ids_.erase(id_it);
    }
    
    return true;
}

GraphNode* Graph::GetNode(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = nodes_.find(node_id);
    return (it != nodes_.end()) ? it->second.get() : nullptr;
}

GraphEdge* Graph::GetEdge(const std::string& edge_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = edges_.find(edge_id);
    return (it != edges_.end()) ? it->second.get() : nullptr;
}

bool Graph::IsValid() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check if graph is acyclic
    if (!IsAcyclic()) return false;
    
    // Validate all nodes
    for (const auto& [id, node] : nodes_) {
        if (!node->Validate()) return false;
    }
    
    // Validate all edges
    for (const auto& [id, edge] : edges_) {
        if (!edge->Validate()) return false;
    }
    
    return true;
}

bool Graph::IsAcyclic() const {
    return !HasCycles();
}

bool Graph::HasCycles() const {
    std::unordered_set<std::string> visited;
    std::unordered_set<std::string> rec_stack;
    bool has_cycle = false;
    
    for (const auto& [id, node] : nodes_) {
        if (visited.find(id) == visited.end()) {
            DFS(id, visited, rec_stack, has_cycle);
            if (has_cycle) break;
        }
    }
    
    return has_cycle;
}

void Graph::DFS(const std::string& node_id, std::unordered_set<std::string>& visited,
                std::unordered_set<std::string>& rec_stack, bool& has_cycle) const {
    visited.insert(node_id);
    rec_stack.insert(node_id);
    
    auto node_it = nodes_.find(node_id);
    if (node_it != nodes_.end()) {
        for (const auto& edge_id : node_it->second->GetOutputEdges()) {
            auto edge_it = edges_.find(edge_id);
            if (edge_it != edges_.end()) {
                const std::string& target_node = edge_it->second->GetTargetNode();
                
                if (visited.find(target_node) == visited.end()) {
                    DFS(target_node, visited, rec_stack, has_cycle);
                } else if (rec_stack.find(target_node) != rec_stack.end()) {
                    has_cycle = true;
                    return;
                }
            }
        }
    }
    
    rec_stack.erase(node_id);
}

std::vector<std::string> Graph::GetTopologicalOrder() const {
    return TopologicalSort();
}

std::vector<std::string> Graph::TopologicalSort() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> result;
    std::unordered_map<std::string, int> in_degree;
    
    // Calculate in-degrees
    for (const auto& [id, node] : nodes_) {
        in_degree[id] = 0;
    }
    
    for (const auto& [id, edge] : edges_) {
        in_degree[edge->GetTargetNode()]++;
    }
    
    // Find nodes with no incoming edges
    std::queue<std::string> queue;
    for (const auto& [id, degree] : in_degree) {
        if (degree == 0) {
            queue.push(id);
        }
    }
    
    // Process nodes
    while (!queue.empty()) {
        std::string current = queue.front();
        queue.pop();
        result.push_back(current);
        
        auto node_it = nodes_.find(current);
        if (node_it != nodes_.end()) {
            for (const auto& edge_id : node_it->second->GetOutputEdges()) {
                auto edge_it = edges_.find(edge_id);
                if (edge_it != edges_.end()) {
                    const std::string& target = edge_it->second->GetTargetNode();
                    in_degree[target]--;
                    if (in_degree[target] == 0) {
                        queue.push(target);
                    }
                }
            }
        }
    }
    
    return result;
}

std::vector<std::string> Graph::GetReadyNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ready_nodes;
    
    for (const auto& [id, node] : nodes_) {
        if (node->IsReady()) {
            ready_nodes.push_back(id);
        }
    }
    
    return ready_nodes;
}

std::vector<std::string> Graph::GetNodesByType(NodeType type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> nodes_of_type;
    
    for (const auto& [id, node] : nodes_) {
        if (node->GetType() == type) {
            nodes_of_type.push_back(id);
        }
    }
    
    return nodes_of_type;
}

std::vector<std::string> Graph::GetSourceNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> source_nodes;
    
    for (const auto& [id, node] : nodes_) {
        if (node->GetInputEdges().empty()) {
            source_nodes.push_back(id);
        }
    }
    
    return source_nodes;
}

std::vector<std::string> Graph::GetSinkNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> sink_nodes;
    
    for (const auto& [id, node] : nodes_) {
        if (node->GetOutputEdges().empty()) {
            sink_nodes.push_back(id);
        }
    }
    
    return sink_nodes;
}

bool Graph::IsExecutionComplete() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& [id, node] : nodes_) {
        NodeStatus status = node->GetStatus();
        if (status != NodeStatus::COMPLETED && status != NodeStatus::FAILED && 
            status != NodeStatus::CANCELLED && status != NodeStatus::SKIPPED) {
            return false;
        }
    }
    
    return true;
}

bool Graph::HasFailedNodes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& [id, node] : nodes_) {
        if (node->GetStatus() == NodeStatus::FAILED) {
            return true;
        }
    }
    
    return false;
}

void Graph::ResetExecutionState() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& [id, node] : nodes_) {
        node->SetStatus(NodeStatus::PENDING);
        node->ClearInputs();
        node->ClearOutputs();
    }
    
    for (const auto& [id, edge] : edges_) {
        edge->ClearData();
    }
    
    // Reset stats
    stats_ = GraphExecutionStats{};
}

GraphExecutionStats::Snapshot Graph::GetStatsSnapshot() const {
    return stats_.GetSnapshot();
}

void Graph::UpdateStats(const NodeExecutionResult& result) {
    stats_.total_nodes_executed.fetch_add(1);
    
    switch (result.status) {
        case NodeStatus::COMPLETED:
            stats_.successful_nodes.fetch_add(1);
            break;
        case NodeStatus::FAILED:
            stats_.failed_nodes.fetch_add(1);
            break;
        case NodeStatus::CANCELLED:
            stats_.cancelled_nodes.fetch_add(1);
            break;
        default:
            break;
    }
    
    // Update timing stats
    auto exec_time = result.execution_time;
    stats_.total_execution_time_ms.fetch_add(exec_time.count());
    
    // Update max execution time
    auto current_max = stats_.max_node_execution_time_ms.load();
    while (static_cast<uint64_t>(exec_time.count()) > current_max && 
           !stats_.max_node_execution_time_ms.compare_exchange_weak(current_max, static_cast<uint64_t>(exec_time.count()))) {
        // Retry
    }
    
    // Update min execution time
    auto current_min = stats_.min_node_execution_time_ms.load();
    while (static_cast<uint64_t>(exec_time.count()) < current_min && 
            !stats_.min_node_execution_time_ms.compare_exchange_weak(current_min, static_cast<uint64_t>(exec_time.count()))) {
        // Retry
    }
}

bool Graph::ValidateGraph() const {
    return IsValid();
}

std::vector<std::string> Graph::GetValidationErrors() const {
    std::vector<std::string> errors;
    
    if (!IsAcyclic()) {
        errors.push_back("Graph contains cycles");
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Validate nodes
    for (const auto& [id, node] : nodes_) {
        if (!node->Validate()) {
            errors.push_back("Invalid node: " + id);
        }
    }
    
    // Validate edges
    for (const auto& [id, edge] : edges_) {
        if (!edge->Validate()) {
            errors.push_back("Invalid edge: " + id);
        }
    }
    
    return errors;
}

std::string Graph::ToJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    json graph_json;
    graph_json["id"] = id_;
    graph_json["config"] = {
        {"default_execution_mode", static_cast<int>(config_.default_execution_mode)},
        {"max_parallel_nodes", config_.max_parallel_nodes},
        {"global_timeout", config_.global_timeout.count()},
        {"enable_profiling", config_.enable_profiling},
        {"enable_optimization", config_.enable_optimization},
        {"enable_distributed_execution", config_.enable_distributed_execution},
        {"optimization_policy", config_.optimization_policy},
        {"max_memory_usage", config_.max_memory_usage},
        {"fail_fast", config_.fail_fast}
    };
    
    // Serialize nodes
    json nodes_json = json::array();
    for (const auto& [id, node] : nodes_) {
        json node_json;
        node_json["id"] = node->GetId();
        node_json["type"] = static_cast<int>(node->GetType());
        node_json["metadata"] = {
            {"name", node->GetMetadata().name},
            {"description", node->GetMetadata().description},
            {"model_path", node->GetMetadata().model_path},
            {"model_type", static_cast<int>(node->GetMetadata().model_type)},
            {"timeout_ms", node->GetMetadata().timeout_ms.count()},
            {"priority", node->GetMetadata().priority},
            {"is_critical", node->GetMetadata().is_critical}
        };
        nodes_json.push_back(node_json);
    }
    graph_json["nodes"] = nodes_json;
    
    // Serialize edges
    json edges_json = json::array();
    for (const auto& [id, edge] : edges_) {
        json edge_json;
        edge_json["id"] = edge->GetId();
        edge_json["source_node"] = edge->GetSourceNode();
        edge_json["target_node"] = edge->GetTargetNode();
        edge_json["type"] = static_cast<int>(edge->GetType());
        edge_json["metadata"] = {
            {"name", edge->GetMetadata().name},
            {"description", edge->GetMetadata().description},
            {"data_type", static_cast<int>(edge->GetMetadata().data_type)},
            {"transfer_timeout_ms", edge->GetMetadata().transfer_timeout_ms.count()},
            {"buffer_size", edge->GetMetadata().buffer_size},
            {"is_buffered", edge->GetMetadata().is_buffered}
        };
        edges_json.push_back(edge_json);
    }
    graph_json["edges"] = edges_json;
    
    return graph_json.dump(2);
}

bool Graph::FromJson(const std::string& json_str) {
    try {
        json graph_json = json::parse(json_str);
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Clear existing graph
        nodes_.clear();
        edges_.clear();
        node_ids_.clear();
        edge_ids_.clear();
        
        // Parse graph ID
        if (graph_json.contains("id")) {
            id_ = graph_json["id"];
        }
        
        // Parse config
        if (graph_json.contains("config")) {
            auto config_json = graph_json["config"];
            if (config_json.contains("default_execution_mode")) {
                config_.default_execution_mode = static_cast<ExecutionMode>(config_json["default_execution_mode"]);
            }
            if (config_json.contains("max_parallel_nodes")) {
                config_.max_parallel_nodes = config_json["max_parallel_nodes"];
            }
            if (config_json.contains("global_timeout")) {
                config_.global_timeout = std::chrono::milliseconds(config_json["global_timeout"]);
            }
            if (config_json.contains("enable_profiling")) {
                config_.enable_profiling = config_json["enable_profiling"];
            }
            if (config_json.contains("enable_optimization")) {
                config_.enable_optimization = config_json["enable_optimization"];
            }
            if (config_json.contains("enable_distributed_execution")) {
                config_.enable_distributed_execution = config_json["enable_distributed_execution"];
            }
            if (config_json.contains("optimization_policy")) {
                config_.optimization_policy = config_json["optimization_policy"];
            }
            if (config_json.contains("max_memory_usage")) {
                config_.max_memory_usage = config_json["max_memory_usage"];
            }
            if (config_json.contains("fail_fast")) {
                config_.fail_fast = config_json["fail_fast"];
            }
        }
        
        // Parse nodes
        if (graph_json.contains("nodes")) {
            for (const auto& node_json : graph_json["nodes"]) {
                std::string node_id = node_json["id"];
                NodeType type = static_cast<NodeType>(node_json["type"]);
                
                NodeMetadata metadata;
                if (node_json.contains("metadata")) {
                    auto meta_json = node_json["metadata"];
                    if (meta_json.contains("name")) metadata.name = meta_json["name"];
                    if (meta_json.contains("description")) metadata.description = meta_json["description"];
                    if (meta_json.contains("model_path")) metadata.model_path = meta_json["model_path"];
                    if (meta_json.contains("model_type")) metadata.model_type = static_cast<ModelType>(meta_json["model_type"]);
                    if (meta_json.contains("timeout_ms")) metadata.timeout_ms = std::chrono::milliseconds(meta_json["timeout_ms"]);
                    if (meta_json.contains("priority")) metadata.priority = meta_json["priority"];
                    if (meta_json.contains("is_critical")) metadata.is_critical = meta_json["is_critical"];
                }
                
                auto node = std::make_unique<GraphNode>(node_id, type, metadata);
                nodes_[node_id] = std::move(node);
                node_ids_.push_back(node_id);
            }
        }
        
        // Parse edges
        if (graph_json.contains("edges")) {
            for (const auto& edge_json : graph_json["edges"]) {
                std::string edge_id = edge_json["id"];
                std::string source_node = edge_json["source_node"];
                std::string target_node = edge_json["target_node"];
                EdgeType type = static_cast<EdgeType>(edge_json["type"]);
                
                EdgeMetadata metadata;
                if (edge_json.contains("metadata")) {
                    auto meta_json = edge_json["metadata"];
                    if (meta_json.contains("name")) metadata.name = meta_json["name"];
                    if (meta_json.contains("description")) metadata.description = meta_json["description"];
                    if (meta_json.contains("data_type")) metadata.data_type = static_cast<DataType>(meta_json["data_type"]);
                    if (meta_json.contains("transfer_timeout_ms")) metadata.transfer_timeout_ms = std::chrono::milliseconds(meta_json["transfer_timeout_ms"]);
                    if (meta_json.contains("buffer_size")) metadata.buffer_size = meta_json["buffer_size"];
                    if (meta_json.contains("is_buffered")) metadata.is_buffered = meta_json["is_buffered"];
                }
                
                auto edge = std::make_unique<GraphEdge>(edge_id, source_node, target_node, type, metadata);
                edges_[edge_id] = std::move(edge);
                edge_ids_.push_back(edge_id);
                
                // Update node connections
                nodes_[source_node]->AddOutputEdge(edge_id);
                nodes_[target_node]->AddInputEdge(edge_id);
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// GraphBuilder Implementation

GraphBuilder::GraphBuilder(const std::string& graph_id, const GraphConfig& config)
    : graph_id_(graph_id), config_(config) {
}

GraphBuilder& GraphBuilder::AddNode(const std::string& node_id, NodeType type, const NodeMetadata& metadata) {
    auto node = std::make_unique<GraphNode>(node_id, type, metadata);
    nodes_[node_id] = std::move(node);
    node_ids_.push_back(node_id);
    return *this;
}

GraphBuilder& GraphBuilder::AddModelNode(const std::string& node_id, const std::string& model_path, 
                                         ModelType model_type, const NodeMetadata& metadata) {
    NodeMetadata model_metadata = metadata;
    model_metadata.model_path = model_path;
    model_metadata.model_type = model_type;
    model_metadata.name = metadata.name.empty() ? node_id : metadata.name;
    
    return AddNode(node_id, NodeType::MODEL_INFERENCE, model_metadata);
}

GraphBuilder& GraphBuilder::AddProcessingNode(const std::string& node_id, const NodeMetadata& metadata) {
    NodeMetadata proc_metadata = metadata;
    proc_metadata.name = metadata.name.empty() ? node_id : metadata.name;
    
    return AddNode(node_id, NodeType::DATA_PROCESSING, proc_metadata);
}

GraphBuilder& GraphBuilder::AddCustomNode(const std::string& node_id, const NodeMetadata& metadata) {
    NodeMetadata custom_metadata = metadata;
    custom_metadata.name = metadata.name.empty() ? node_id : metadata.name;
    
    return AddNode(node_id, NodeType::CUSTOM_OPERATOR, custom_metadata);
}

GraphBuilder& GraphBuilder::AddEdge(const std::string& edge_id, const std::string& source_node,
                                    const std::string& target_node, EdgeType type, const EdgeMetadata& metadata) {
    auto edge = std::make_unique<GraphEdge>(edge_id, source_node, target_node, type, metadata);
    edges_[edge_id] = std::move(edge);
    edge_ids_.push_back(edge_id);
    return *this;
}

GraphBuilder& GraphBuilder::ConnectNodes(const std::string& source_node, const std::string& target_node,
                                         const std::string& edge_name) {
    std::string edge_id = edge_name.empty() ? GenerateEdgeId(source_node, target_node) : edge_name;
    return AddEdge(edge_id, source_node, target_node);
}

GraphBuilder& GraphBuilder::CreatePipeline(const std::vector<std::string>& node_ids) {
    for (size_t i = 0; i < node_ids.size() - 1; ++i) {
        ConnectNodes(node_ids[i], node_ids[i + 1]);
    }
    return *this;
}

GraphBuilder& GraphBuilder::CreateParallel(const std::vector<std::string>& node_ids) {
    (void)node_ids;
    // For parallel execution, we don't add edges between the nodes
    // The scheduler will handle parallel execution
    return *this;
}

std::unique_ptr<Graph> GraphBuilder::Build() {
    if (!IsValid()) {
        return nullptr;
    }
    
    auto graph = std::make_unique<Graph>(graph_id_, config_);
    
    // Add all nodes
    for (auto& [id, node] : nodes_) {
        graph->AddNode(std::move(node));
    }
    nodes_.clear();
    
    // Add all edges
    for (auto& [id, edge] : edges_) {
        graph->AddEdge(std::move(edge));
    }
    edges_.clear();
    
    return graph;
}

bool GraphBuilder::IsValid() const {
    // Check if all nodes exist
    for (const auto& [id, node] : nodes_) {
        if (!node->Validate()) return false;
    }
    
    // Check if all edges reference existing nodes
    for (const auto& [id, edge] : edges_) {
        if (!edge->Validate()) return false;
        if (nodes_.find(edge->GetSourceNode()) == nodes_.end() ||
            nodes_.find(edge->GetTargetNode()) == nodes_.end()) {
            return false;
        }
    }
    
    return true;
}

std::vector<std::string> GraphBuilder::GetValidationErrors() const {
    std::vector<std::string> errors;
    
    // Validate nodes
    for (const auto& [id, node] : nodes_) {
        if (!node->Validate()) {
            errors.push_back("Invalid node: " + id);
        }
    }
    
    // Validate edges
    for (const auto& [id, edge] : edges_) {
        if (!edge->Validate()) {
            errors.push_back("Invalid edge: " + id);
        } else {
            if (nodes_.find(edge->GetSourceNode()) == nodes_.end()) {
                errors.push_back("Edge " + id + " references non-existent source node: " + edge->GetSourceNode());
            }
            if (nodes_.find(edge->GetTargetNode()) == nodes_.end()) {
                errors.push_back("Edge " + id + " references non-existent target node: " + edge->GetTargetNode());
            }
        }
    }
    
    return errors;
}

std::string GraphBuilder::GenerateEdgeId(const std::string& source, const std::string& target) const {
    return source + "_to_" + target;
}

} // namespace edge_ai
