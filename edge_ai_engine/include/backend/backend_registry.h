/**
 * @file backend_registry.h
 * @brief Backend registry for managing execution backends
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the BackendRegistry class which manages registration,
 * selection, and lifecycle of execution backends.
 */

#pragma once

#include "execution_backend.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <string>

namespace edge_ai {

// Forward declarations
class Model;
class InferenceRequest;

/**
 * @struct BackendSelectionCriteria
 * @brief Criteria for backend selection
 */
struct BackendSelectionCriteria {
    ModelType preferred_model_type{ModelType::UNKNOWN};
    DataType preferred_data_type{DataType::UNKNOWN};
    size_t batch_size{1};
    bool prefer_batching{false};
    bool prefer_low_latency{false};
    bool prefer_high_throughput{false};
    
    BackendSelectionCriteria() = default;
};

/**
 * @class BackendRegistry
 * @brief Registry for managing execution backends
 * 
 * The BackendRegistry class manages the registration, selection, and
 * lifecycle of execution backends. It provides intelligent backend
 * selection based on request characteristics and backend capabilities.
 */
class BackendRegistry {
public:
    /**
     * @brief Constructor
     */
    BackendRegistry();
    
    /**
     * @brief Destructor
     */
    ~BackendRegistry();
    
    // Disable copy constructor and assignment operator
    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;
    
    /**
     * @brief Register a backend
     * @param backend Backend to register
     * @return Status indicating success or failure
     */
    Status RegisterBackend(std::shared_ptr<ExecutionBackend> backend);
    
    /**
     * @brief Unregister a backend
     * @param backend_type Type of backend to unregister
     * @return Status indicating success or failure
     */
    Status UnregisterBackend(BackendType backend_type);
    
    /**
     * @brief Get a backend by type
     * @param backend_type Type of backend
     * @return Backend if found, nullptr otherwise
     */
    std::shared_ptr<ExecutionBackend> GetBackend(BackendType backend_type) const;
    
    /**
     * @brief Select the best backend for a request
     * @param model Model to execute
     * @param request Inference request
     * @param criteria Selection criteria
     * @return Best backend for the request
     */
    std::shared_ptr<ExecutionBackend> SelectBackend(const Model& model,
                                                   const InferenceRequest& request,
                                                   const BackendSelectionCriteria& criteria = BackendSelectionCriteria{}) const;
    
    /**
     * @brief Get all registered backends
     * @return Vector of registered backends
     */
    std::vector<std::shared_ptr<ExecutionBackend>> GetAllBackends() const;
    
    /**
     * @brief Get available backend types
     * @return Vector of available backend types
     */
    std::vector<BackendType> GetAvailableBackendTypes() const;
    
    /**
     * @brief Check if a backend type is available
     * @param backend_type Backend type to check
     * @return True if available, false otherwise
     */
    bool IsBackendAvailable(BackendType backend_type) const;
    
    /**
     * @brief Initialize all registered backends
     * @return Status indicating success or failure
     */
    Status InitializeAllBackends();
    
    /**
     * @brief Shutdown all registered backends
     * @return Status indicating success or failure
     */
    Status ShutdownAllBackends();
    
    /**
     * @brief Get registry statistics
     * @return Registry statistics
     */
    struct RegistryStats {
        size_t total_backends{0};
        size_t initialized_backends{0};
        std::unordered_map<BackendType, size_t> backend_type_counts;
        std::unordered_map<BackendType, uint64_t> backend_selection_counts;
    };
    
    RegistryStats GetStats() const;

private:
    mutable std::mutex registry_mutex_;
    std::unordered_map<BackendType, std::shared_ptr<ExecutionBackend>> backends_;
    std::unordered_map<BackendType, uint64_t> selection_counts_;
    
    /**
     * @brief Calculate backend score for selection
     * @param backend Backend to score
     * @param model Model to execute
     * @param request Inference request
     * @param criteria Selection criteria
     * @return Backend score (higher is better)
     */
    double CalculateBackendScore(std::shared_ptr<ExecutionBackend> backend,
                                const Model& model,
                                const InferenceRequest& request,
                                const BackendSelectionCriteria& criteria) const;
    
    /**
     * @brief Check if backend supports request requirements
     * @param backend Backend to check
     * @param model Model to execute
     * @param request Inference request
     * @return True if supported, false otherwise
     */
    bool BackendSupportsRequest(std::shared_ptr<ExecutionBackend> backend,
                               const Model& model,
                               const InferenceRequest& request) const;
};

} // namespace edge_ai
