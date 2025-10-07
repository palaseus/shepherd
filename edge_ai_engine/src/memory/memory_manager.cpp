/**
 * @file memory_manager.cpp
 * @brief Memory management and allocation system implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "memory/memory_manager.h"
#include "profiling/profiler.h"
#include <stdexcept>
#include <cstdlib>

namespace edge_ai {

MemoryManager::MemoryManager(const MemoryConfig& config)
    : config_(config)
    , initialized_(false)
    , memory_pooling_enabled_(true)
    , memory_limit_(config.max_memory_usage) {
}

MemoryManager::~MemoryManager() {
    Cleanup();
}

Status MemoryManager::Initialize() {
    try {
        if (initialized_) {
            return Status::ALREADY_INITIALIZED;
        }
        
        // Initialize memory pools
        Status status = InitializeMemoryPools();
        if (status != Status::SUCCESS) {
            return status;
        }
        
        // Initialize memory allocators
        status = InitializeMemoryAllocators();
        if (status != Status::SUCCESS) {
            return status;
        }
        
        initialized_ = true;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status MemoryManager::Shutdown() {
    try {
        if (!initialized_) {
            return Status::SUCCESS;
        }
        
        Cleanup();
        initialized_ = false;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

void* MemoryManager::AllocateMemory(size_t size, size_t alignment, std::shared_ptr<Device> device) {
    try {
        if (!initialized_) {
            return nullptr;
        }
        
        if (size == 0) {
            return nullptr;
        }
        
        // Profile memory allocation
        PROFILER_SCOPED_EVENT(0, "mem_alloc");
        
        // Check memory limit
        if (IsMemoryLimitExceeded()) {
            return nullptr;
        }
        
        // Try to find suitable memory pool first
        if (memory_pooling_enabled_) {
            MemoryPool* pool = FindSuitableMemoryPool(size, device);
            if (pool) {
                void* ptr = pool->Allocate(size, alignment);
                if (ptr) {
                    UpdateMemoryStats(size, 0);
                    UpdateDeviceMemoryUsage(device, size, 0);
                    return ptr;
                }
            }
        }
        
        // Fallback to direct allocation
        void* ptr = std::aligned_alloc(alignment, size);
        if (ptr) {
            UpdateMemoryStats(size, 0);
            UpdateDeviceMemoryUsage(device, size, 0);
        }
        
        return ptr;
    } catch (const std::exception& e) {
        return nullptr;
    }
}

Status MemoryManager::DeallocateMemory(void* ptr, [[maybe_unused]] std::shared_ptr<Device> device) {
    try {
        if (!ptr) {
            return Status::INVALID_ARGUMENT;
        }
        
        // Profile memory deallocation
        PROFILER_SCOPED_EVENT(0, "mem_free");
        
        // Try to deallocate from memory pools first
        if (memory_pooling_enabled_) {
            for (auto& pool : memory_pools_) {
                if (pool->Deallocate(ptr) == Status::SUCCESS) {
                    return Status::SUCCESS;
                }
            }
        }
        
        // Fallback to direct deallocation
        std::free(ptr);
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status MemoryManager::AllocateTensorMemory(Tensor& tensor, std::shared_ptr<Device> device) {
    try {
        if (!tensor.IsValid()) {
            return Status::INVALID_ARGUMENT;
        }
        
        size_t size = tensor.GetSize();
        void* ptr = AllocateMemory(size, 64, device);
        if (!ptr) {
            return Status::OUT_OF_MEMORY;
        }
        
        return tensor.SetData(ptr, size);
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status MemoryManager::DeallocateTensorMemory(Tensor& tensor, std::shared_ptr<Device> device) {
    try {
        void* ptr = tensor.GetData();
        if (ptr) {
            return DeallocateMemory(ptr, device);
        }
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status MemoryManager::AllocateMemoryPool(size_t pool_size, std::shared_ptr<Device> device) {
    try {
        if (!initialized_) {
            return Status::NOT_INITIALIZED;
        }
        
        auto pool = std::make_unique<MemoryPool>(pool_size, device);
        memory_pools_.push_back(std::move(pool));
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

MemoryStats::Snapshot MemoryManager::GetMemoryStats() const {
    return stats_.GetSnapshot();
}

DeviceMemoryUsage MemoryManager::GetDeviceMemoryUsage(std::shared_ptr<Device> device) const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(device_memory_mutex_));
    auto it = device_memory_usage_.find(device);
    if (it != device_memory_usage_.end()) {
        return it->second;
    }
    return DeviceMemoryUsage{};
}

Status MemoryManager::SetMemoryConfig(const MemoryConfig& config) {
    try {
        config_ = config;
        memory_limit_ = config.max_memory_usage;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

MemoryConfig MemoryManager::GetMemoryConfig() const {
    return config_;
}

Status MemoryManager::SetMemoryPooling(bool enable) {
    memory_pooling_enabled_ = enable;
    return Status::SUCCESS;
}

bool MemoryManager::IsMemoryPoolingEnabled() const {
    return memory_pooling_enabled_;
}

Status MemoryManager::DefragmentMemory() {
    // Placeholder implementation
    return Status::SUCCESS;
}

Status MemoryManager::ClearAllMemory() {
    try {
        // Clear all memory pools
        for ([[maybe_unused]] auto& pool : memory_pools_) {
            // This would clear the pool
        }
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

double MemoryManager::GetMemoryFragmentationRatio() const {
    // Placeholder implementation
    return 0.0;
}

Status MemoryManager::SetMemoryLimit(size_t limit) {
    memory_limit_ = limit;
    return Status::SUCCESS;
}

size_t MemoryManager::GetMemoryLimit() const {
    return memory_limit_;
}

bool MemoryManager::IsMemoryLimitExceeded() const {
    return stats_.current_usage.load() > memory_limit_;
}

// Private methods
Status MemoryManager::InitializeMemoryPools() {
    try {
        // Create default memory pools
        for (int i = 0; i < config_.num_memory_pools; ++i) {
            size_t pool_size = config_.memory_pool_size / config_.num_memory_pools;
            auto pool = std::make_unique<MemoryPool>(pool_size, nullptr);
            memory_pools_.push_back(std::move(pool));
        }
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status MemoryManager::InitializeMemoryAllocators() {
    try {
        // Create default memory allocators
        // Create a basic memory allocator implementation
        class BasicMemoryAllocator : public MemoryAllocator {
        public:
            BasicMemoryAllocator(std::shared_ptr<Device> device) : MemoryAllocator(device) {}
            
            void* Allocate(size_t size, size_t alignment = 64) override {
                return std::aligned_alloc(alignment, size);
            }
            
            Status Deallocate(void* ptr) override {
                if (ptr) {
                    std::free(ptr);
                    return Status::SUCCESS;
                }
                return Status::INVALID_ARGUMENT;
            }
            
            MemoryAllocatorStats GetAllocatorStats() const override {
                return MemoryAllocatorStats{};
            }
        };
        
        auto allocator = std::make_unique<BasicMemoryAllocator>(nullptr);
        allocators_.push_back(std::move(allocator));
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

void MemoryManager::UpdateMemoryStats(size_t allocated_size, size_t deallocated_size) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (allocated_size > 0) {
        stats_.total_allocations.fetch_add(1);
        stats_.current_usage.fetch_add(allocated_size);
        stats_.total_allocated.fetch_add(allocated_size);
        stats_.peak_usage = std::max(stats_.peak_usage.load(), 
                                           stats_.current_usage.load());
    }
    
    if (deallocated_size > 0) {
        stats_.total_deallocations.fetch_add(1);
        stats_.current_usage.fetch_sub(deallocated_size);
    }
}

void MemoryManager::UpdateDeviceMemoryUsage(std::shared_ptr<Device> device, size_t allocated_size, size_t deallocated_size) {
    std::lock_guard<std::mutex> lock(device_memory_mutex_);
    
    auto& usage = device_memory_usage_[device];
    if (allocated_size > 0) {
        usage.allocated_memory += allocated_size;
        usage.total_memory += allocated_size;
    }
    
    if (deallocated_size > 0) {
        usage.allocated_memory -= deallocated_size;
    }
    
    usage.free_memory = usage.total_memory - usage.allocated_memory;
    usage.utilization_ratio = static_cast<double>(usage.allocated_memory) / usage.total_memory;
}

MemoryPool* MemoryManager::FindSuitableMemoryPool(size_t size, [[maybe_unused]] std::shared_ptr<Device> device) {
    for (auto& pool : memory_pools_) {
        if (!pool->IsFull() && pool->GetPoolStats().free_size >= size) {
            return pool.get();
        }
    }
    return nullptr;
}

void MemoryManager::Cleanup() {
    memory_pools_.clear();
    allocators_.clear();
    
    std::lock_guard<std::mutex> lock(device_memory_mutex_);
    device_memory_usage_.clear();
}

// MemoryPool implementation
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

// MemoryAllocator implementation
MemoryAllocator::MemoryAllocator(std::shared_ptr<Device> device)
    : device_(device) {
}

} // namespace edge_ai
