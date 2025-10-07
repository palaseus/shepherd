/**
 * @file backend_registry.cpp
 * @brief Backend registry implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "backend/backend_registry.h"
#include "core/model.h"
#include <algorithm>
#include <numeric>

namespace edge_ai {

BackendRegistry::BackendRegistry() {
}

BackendRegistry::~BackendRegistry() {
    ShutdownAllBackends();
}

Status BackendRegistry::RegisterBackend(std::shared_ptr<ExecutionBackend> backend) {
    if (!backend) {
        return Status::INVALID_ARGUMENT;
    }
    
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    BackendType backend_type = backend->GetBackendType();
    backends_[backend_type] = backend;
    selection_counts_[backend_type] = 0;
    
    return Status::SUCCESS;
}

Status BackendRegistry::UnregisterBackend(BackendType backend_type) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = backends_.find(backend_type);
    if (it != backends_.end()) {
        // Shutdown the backend before removing
        it->second->Shutdown();
        backends_.erase(it);
        selection_counts_.erase(backend_type);
        return Status::SUCCESS;
    }
    
    return Status::FAILURE;
}

std::shared_ptr<ExecutionBackend> BackendRegistry::GetBackend(BackendType backend_type) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    auto it = backends_.find(backend_type);
    if (it != backends_.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::shared_ptr<ExecutionBackend> BackendRegistry::SelectBackend(const Model& model,
                                                               const InferenceRequest& request,
                                                               const BackendSelectionCriteria& criteria) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    if (backends_.empty()) {
        return nullptr;
    }
    
    std::shared_ptr<ExecutionBackend> best_backend = nullptr;
    double best_score = -1.0;
    
    for (const auto& [backend_type, backend] : backends_) {
        // Check if backend supports the request
        if (!BackendSupportsRequest(backend, model, request)) {
            continue;
        }
        
        // Calculate score for this backend
        double score = CalculateBackendScore(backend, model, request, criteria);
        
        if (score > best_score) {
            best_score = score;
            best_backend = backend;
        }
    }
    
    // Update selection count
    if (best_backend) {
        const_cast<BackendRegistry*>(this)->selection_counts_[best_backend->GetBackendType()]++;
    }
    
    return best_backend;
}

std::vector<std::shared_ptr<ExecutionBackend>> BackendRegistry::GetAllBackends() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<std::shared_ptr<ExecutionBackend>> result;
    result.reserve(backends_.size());
    
    for (const auto& [backend_type, backend] : backends_) {
        result.push_back(backend);
    }
    
    return result;
}

std::vector<BackendType> BackendRegistry::GetAvailableBackendTypes() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    std::vector<BackendType> result;
    result.reserve(backends_.size());
    
    for (const auto& [backend_type, backend] : backends_) {
        result.push_back(backend_type);
    }
    
    return result;
}

bool BackendRegistry::IsBackendAvailable(BackendType backend_type) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return backends_.find(backend_type) != backends_.end();
}

Status BackendRegistry::InitializeAllBackends() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    Status overall_status = Status::SUCCESS;
    
    for (const auto& [backend_type, backend] : backends_) {
        Status status = backend->Initialize();
        if (status != Status::SUCCESS) {
            overall_status = status;
        }
    }
    
    return overall_status;
}

Status BackendRegistry::ShutdownAllBackends() {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    Status overall_status = Status::SUCCESS;
    
    for (const auto& [backend_type, backend] : backends_) {
        Status status = backend->Shutdown();
        if (status != Status::SUCCESS) {
            overall_status = status;
        }
    }
    
    return overall_status;
}

BackendRegistry::RegistryStats BackendRegistry::GetStats() const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    
    RegistryStats stats;
    stats.total_backends = backends_.size();
    
    for (const auto& [backend_type, backend] : backends_) {
        stats.backend_type_counts[backend_type]++;
        
        auto it = selection_counts_.find(backend_type);
        if (it != selection_counts_.end()) {
            stats.backend_selection_counts[backend_type] = it->second;
        }
    }
    
    return stats;
}

double BackendRegistry::CalculateBackendScore(std::shared_ptr<ExecutionBackend> backend,
                                            const Model& model,
                                            const InferenceRequest& request,
                                            const BackendSelectionCriteria& criteria) const {
    double score = 0.0;
    
    // Base score
    score += 1.0;
    
    // Model type preference
    if (criteria.preferred_model_type != ModelType::UNKNOWN) {
        if (model.GetType() == criteria.preferred_model_type) {
            score += 2.0;
        }
    }
    
    // Data type preference
    if (criteria.preferred_data_type != DataType::UNKNOWN) {
        for (const auto& input : request.inputs) {
            if (input.GetDataType() == criteria.preferred_data_type) {
                score += 1.0;
                break;
            }
        }
    }
    
    // Batching preference
    if (criteria.prefer_batching && criteria.batch_size > 1) {
        auto capabilities = backend->GetCapabilities();
        if (capabilities.supports_batching && capabilities.max_batch_size >= criteria.batch_size) {
            score += 3.0;
        }
    }
    
    // Latency vs throughput preference
    if (criteria.prefer_low_latency) {
        // Prefer CPU for low latency (in this mock implementation)
        if (backend->GetBackendType() == BackendType::CPU) {
            score += 2.0;
        }
    } else if (criteria.prefer_high_throughput) {
        // Prefer GPU for high throughput
        if (backend->GetBackendType() == BackendType::GPU) {
            score += 2.0;
        }
    }
    
    // Backend type preference (GPU generally preferred for larger workloads)
    if (request.inputs.size() > 4) {
        if (backend->GetBackendType() == BackendType::GPU) {
            score += 1.0;
        }
    }
    
    return score;
}

bool BackendRegistry::BackendSupportsRequest(std::shared_ptr<ExecutionBackend> backend,
                                           const Model& model,
                                           const InferenceRequest& request) const {
    // Check model type support
    if (!backend->SupportsModelType(model.GetType())) {
        return false;
    }
    
    // Check data type support
    for (const auto& input : request.inputs) {
        if (!backend->SupportsDataType(input.GetDataType())) {
            return false;
        }
    }
    
    return true;
}

} // namespace edge_ai
