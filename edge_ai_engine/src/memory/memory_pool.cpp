/**
 * @file memory_pool.cpp
 * @brief Memory pool implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "memory/memory_pool.h"
#include <stdexcept>

namespace edge_ai {

MemoryPool::MemoryPool(size_t pool_size, std::shared_ptr<Device> device)
    : pool_size_(pool_size)
    , device_(device) {
    InitializePool();
}

MemoryPool::~MemoryPool() {
    CleanupPool();
}

void* MemoryPool::Allocate(size_t size, [[maybe_unused]] size_t alignment) {
    try {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        
        if (size == 0 || size > pool_size_) {
            return nullptr;
        }
        
        // Simple allocation strategy - just allocate from the pool
        void* ptr = static_cast<char*>(pool_memory_) + allocated_blocks_.size() * size;
        allocated_blocks_[ptr] = size;
        
        return ptr;
    } catch (const std::exception& e) {
        return nullptr;
    }
}

Status MemoryPool::Deallocate(void* ptr) {
    try {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        
        auto it = allocated_blocks_.find(ptr);
        if (it != allocated_blocks_.end()) {
            allocated_blocks_.erase(it);
            return Status::SUCCESS;
        }
        
        return Status::FAILURE;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

MemoryPoolStats MemoryPool::GetPoolStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    MemoryPoolStats stats;
    stats.pool_size = pool_size_;
    stats.allocated_size = allocated_blocks_.size() * 1024; // Placeholder
    stats.free_size = pool_size_ - stats.allocated_size;
    stats.total_allocations = allocated_blocks_.size();
    stats.utilization_ratio = static_cast<double>(stats.allocated_size) / pool_size_;
    
    return stats;
}

bool MemoryPool::IsFull() const {
    return GetPoolStats().free_size == 0;
}

bool MemoryPool::IsEmpty() const {
    return allocated_blocks_.empty();
}

double MemoryPool::GetUtilizationRatio() const {
    return GetPoolStats().utilization_ratio;
}

Status MemoryPool::InitializePool() {
    try {
        pool_memory_ = std::aligned_alloc(64, pool_size_);
        if (!pool_memory_) {
            return Status::OUT_OF_MEMORY;
        }
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

void MemoryPool::CleanupPool() {
    if (pool_memory_) {
        std::free(pool_memory_);
        pool_memory_ = nullptr;
    }
    allocated_blocks_.clear();
}

} // namespace edge_ai
