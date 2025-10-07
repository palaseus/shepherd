/**
 * @file execution_backend.cpp
 * @brief Execution backend interface implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "backend/execution_backend.h"
#include <algorithm>

namespace edge_ai {

ExecutionBackend::ExecutionBackend(BackendType backend_type, std::shared_ptr<Device> device)
    : backend_type_(backend_type)
    , device_(device)
    , initialized_(false) {
}

BackendStats::Snapshot ExecutionBackend::GetStats() const {
    return stats_.GetSnapshot();
}

void ExecutionBackend::UpdateStats(std::chrono::microseconds execution_time, bool success, size_t memory_used) {
    stats_.total_executions.fetch_add(1);
    
    if (success) {
        stats_.successful_executions.fetch_add(1);
    } else {
        stats_.failed_executions.fetch_add(1);
    }
    
    // Update timing statistics
    auto current_total = stats_.total_execution_time.load();
    stats_.total_execution_time.store(current_total + execution_time);
    
    // Update average execution time
    uint64_t total_executions = stats_.total_executions.load();
    if (total_executions > 0) {
        auto total_time = stats_.total_execution_time.load();
        stats_.average_execution_time.store(std::chrono::microseconds(total_time.count() / total_executions));
    }
    
    // Update min/max execution time
    auto current_min = stats_.min_execution_time.load();
    if (execution_time < current_min) {
        stats_.min_execution_time.store(execution_time);
    }
    
    auto current_max = stats_.max_execution_time.load();
    if (execution_time > current_max) {
        stats_.max_execution_time.store(execution_time);
    }
    
    // Update memory statistics
    if (memory_used > 0) {
        stats_.total_memory_allocated.fetch_add(memory_used);
        
        auto current_peak = stats_.peak_memory_usage.load();
        if (memory_used > current_peak) {
            stats_.peak_memory_usage.store(memory_used);
        }
    }
}

} // namespace edge_ai
