/**
 * @file graph_compiler.h
 * @brief Graph compiler for Edge AI Inference Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the GraphCompiler class that takes declarative graph
 * specifications (JSON/YAML) and compiles them into optimized execution plans.
 */

#pragma once

#include "graph_types.h"
#include "graph.h"
#include "core/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>

namespace edge_ai {

/**
 * @struct CompilationResult
 * @brief Result of graph compilation
 */
struct CompilationResult {
    bool success;
    std::string error_message;
    std::unique_ptr<Graph> compiled_graph;
    std::vector<std::string> warnings;
    std::chrono::milliseconds compilation_time;
    
    CompilationResult() : success(false), compilation_time(std::chrono::milliseconds(0)) {}
};

/**
 * @struct OptimizationHints
 * @brief Hints for graph optimization during compilation
 */
struct OptimizationHints {
    bool enable_fusion;                         // Enable operator fusion
    bool enable_parallelization;                // Enable automatic parallelization
    bool enable_memory_optimization;            // Enable memory optimization
    bool enable_backend_optimization;           // Enable backend selection optimization
    std::unordered_map<std::string, BackendType> preferred_backends; // Node-specific backend preferences
    std::unordered_map<std::string, int> node_priorities; // Node execution priorities
    size_t max_memory_usage;                    // Maximum memory usage limit
    int max_parallel_nodes;                     // Maximum parallel nodes
    
    OptimizationHints() : enable_fusion(true), enable_parallelization(true),
                         enable_memory_optimization(true), enable_backend_optimization(true),
                         max_memory_usage(1024 * 1024 * 1024), max_parallel_nodes(4) {}
};

/**
 * @class GraphCompiler
 * @brief Compiles declarative graph specifications into optimized execution graphs
 */
class GraphCompiler {
public:
    GraphCompiler();
    ~GraphCompiler() = default;
    
    // Compilation methods
    CompilationResult CompileFromJson(const std::string& json_spec, 
                                     const OptimizationHints& hints = OptimizationHints{});
    CompilationResult CompileFromYaml(const std::string& yaml_spec,
                                     const OptimizationHints& hints = OptimizationHints{});
    CompilationResult CompileFromFile(const std::string& file_path,
                                     const OptimizationHints& hints = OptimizationHints{});
    
    // Graph optimization
    std::unique_ptr<Graph> OptimizeGraph(std::unique_ptr<Graph> graph,
                                        const OptimizationHints& hints);
    
    // Validation
    bool ValidateSpecification(const std::string& json_spec);
    std::vector<std::string> GetValidationErrors(const std::string& json_spec);
    
    // Utility methods
    std::string GetSupportedFormats() const;
    bool IsFormatSupported(const std::string& format) const;
    
private:
    // JSON parsing
    CompilationResult ParseJsonSpecification(const std::string& json_spec,
                                           const OptimizationHints& hints);
    bool ParseJsonNodes(const nlohmann::json& nodes_json, GraphBuilder& builder);
    bool ParseJsonEdges(const nlohmann::json& edges_json, GraphBuilder& builder);
    bool ParseJsonConfig(const nlohmann::json& config_json, GraphConfig& config);
    
    // YAML parsing
    CompilationResult ParseYamlSpecification(const std::string& yaml_spec,
                                           const OptimizationHints& hints);
    
    // Optimization passes
    void ApplyFusionOptimization(Graph& graph);
    void ApplyParallelizationOptimization(Graph& graph, const OptimizationHints& hints);
    void ApplyMemoryOptimization(Graph& graph, const OptimizationHints& hints);
    void ApplyBackendOptimization(Graph& graph, const OptimizationHints& hints);
    
    // Helper methods
    NodeType ParseNodeType(const std::string& type_str);
    EdgeType ParseEdgeType(const std::string& type_str);
    ExecutionMode ParseExecutionMode(const std::string& mode_str);
    ModelType ParseModelType(const std::string& type_str);
    DataType ParseDataType(const std::string& type_str);
    BackendType ParseBackendType(const std::string& type_str);
    
    std::string GetFileExtension(const std::string& file_path);
    bool IsJsonFile(const std::string& file_path);
    bool IsYamlFile(const std::string& file_path);
    
    // Validation helpers
    bool ValidateNodeSpecification(const nlohmann::json& node_json);
    bool ValidateEdgeSpecification(const nlohmann::json& edge_json);
    bool ValidateConfigSpecification(const nlohmann::json& config_json);
    
    // Supported formats
    std::vector<std::string> supported_formats_;
};

/**
 * @class GraphSpecificationValidator
 * @brief Validates graph specifications before compilation
 */
class GraphSpecificationValidator {
public:
    GraphSpecificationValidator();
    ~GraphSpecificationValidator() = default;
    
    // Validation methods
    bool ValidateJsonSpecification(const std::string& json_spec);
    bool ValidateYamlSpecification(const std::string& yaml_spec);
    std::vector<std::string> GetValidationErrors(const std::string& spec, const std::string& format);
    
    // Schema validation
    bool ValidateAgainstSchema(const std::string& spec, const std::string& format);
    bool ValidateJsonSchema(const nlohmann::json& spec);
    bool ValidateYamlSchema(const std::string& yaml_spec);
    
private:
    
    // Schema definitions
    nlohmann::json GetJsonSchema();
    std::string GetYamlSchema();
};

} // namespace edge_ai
