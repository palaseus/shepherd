/**
 * @file memory_manager.h
 * @brief Memory management and allocation system
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryManager class which handles memory allocation,
 * deallocation, and optimization for the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>

namespace edge_ai {

// Forward declarations
class MemoryPool;
class MemoryAllocator;
class Device;

/**
 * @class MemoryManager
 * @brief Memory management and allocation system
 * 
 * The MemoryManager class provides efficient memory allocation and management
 * for tensors and other data structures used in the Edge AI Engine.
 */
class MemoryManager {
public:
    /**
     * @brief Constructor
     * @param config Memory manager configuration
     */
    explicit MemoryManager(const MemoryConfig& config = MemoryConfig{});
    
    /**
     * @brief Destructor
     */
    ~MemoryManager();
    
    // Disable copy constructor and assignment operator
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    /**
     * @brief Initialize the memory manager
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the memory manager
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Allocate memory for a tensor
     * @param size Size of memory to allocate
     * @param alignment Memory alignment requirement
     * @param device Device to allocate memory on
     * @return Pointer to allocated memory, or nullptr if failed
     */
    void* AllocateMemory(size_t size, size_t alignment = 64, std::shared_ptr<Device> device = nullptr);
    
    /**
     * @brief Deallocate memory
     * @param ptr Pointer to memory to deallocate
     * @param device Device where memory was allocated
     * @return Status indicating success or failure
     */
    Status DeallocateMemory(void* ptr, std::shared_ptr<Device> device = nullptr);
    
    /**
     * @brief Allocate memory for a tensor with specific properties
     * @param tensor Tensor to allocate memory for
     * @param device Device to allocate memory on
     * @return Status indicating success or failure
     */
    Status AllocateTensorMemory(Tensor& tensor, std::shared_ptr<Device> device = nullptr);
    
    /**
     * @brief Deallocate memory for a tensor
     * @param tensor Tensor to deallocate memory for
     * @param device Device where memory was allocated
     * @return Status indicating success or failure
     */
    Status DeallocateTensorMemory(Tensor& tensor, std::shared_ptr<Device> device = nullptr);
    
    /**
     * @brief Allocate memory pool
     * @param pool_size Size of memory pool
     * @param device Device to allocate pool on
     * @return Status indicating success or failure
     */
    Status AllocateMemoryPool(size_t pool_size, std::shared_ptr<Device> device = nullptr);
    
            /**
             * @brief Get memory statistics
             * @return Memory statistics
             */
            MemoryStats::Snapshot GetMemoryStats() const;
    
    /**
     * @brief Get memory usage for a device
     * @param device Device to get memory usage for
     * @return Memory usage information
     */
    DeviceMemoryUsage GetDeviceMemoryUsage(std::shared_ptr<Device> device) const;
    
    /**
     * @brief Set memory manager configuration
     * @param config Memory manager configuration
     * @return Status indicating success or failure
     */
    Status SetMemoryConfig(const MemoryConfig& config);
    
    /**
     * @brief Get current memory manager configuration
     * @return Current memory manager configuration
     */
    MemoryConfig GetMemoryConfig() const;
    
    /**
     * @brief Enable or disable memory pooling
     * @param enable Enable memory pooling
     * @return Status indicating success or failure
     */
    Status SetMemoryPooling(bool enable);
    
    /**
     * @brief Check if memory pooling is enabled
     * @return True if memory pooling is enabled
     */
    bool IsMemoryPoolingEnabled() const;
    
    /**
     * @brief Defragment memory
     * @return Status indicating success or failure
     */
    Status DefragmentMemory();
    
    /**
     * @brief Clear all allocated memory
     * @return Status indicating success or failure
     */
    Status ClearAllMemory();
    
    /**
     * @brief Get memory fragmentation ratio
     * @return Memory fragmentation ratio (0.0 to 1.0)
     */
    double GetMemoryFragmentationRatio() const;
    
    /**
     * @brief Set memory limit
     * @param limit Memory limit in bytes
     * @return Status indicating success or failure
     */
    Status SetMemoryLimit(size_t limit);
    
    /**
     * @brief Get current memory limit
     * @return Current memory limit in bytes
     */
    size_t GetMemoryLimit() const;
    
    /**
     * @brief Check if memory limit is exceeded
     * @return True if memory limit is exceeded
     */
    bool IsMemoryLimitExceeded() const;

private:
    // Configuration
    MemoryConfig config_;
    
    // State
    bool initialized_;
    bool memory_pooling_enabled_;
    size_t memory_limit_;
    
    // Memory pools
    std::vector<std::unique_ptr<MemoryPool>> memory_pools_;
    std::mutex pools_mutex_;
    
    // Memory allocators
    std::vector<std::unique_ptr<MemoryAllocator>> allocators_;
    std::mutex allocators_mutex_;
    
    // Device memory tracking
    std::unordered_map<std::shared_ptr<Device>, DeviceMemoryUsage> device_memory_usage_;
    std::mutex device_memory_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    MemoryStats stats_;
    
    /**
     * @brief Initialize memory pools
     * @return Status indicating success or failure
     */
    Status InitializeMemoryPools();
    
    /**
     * @brief Initialize memory allocators
     * @return Status indicating success or failure
     */
    Status InitializeMemoryAllocators();
    
    /**
     * @brief Update memory statistics
     * @param allocated_size Size of memory allocated
     * @param deallocated_size Size of memory deallocated
     */
    void UpdateMemoryStats(size_t allocated_size, size_t deallocated_size);
    
    /**
     * @brief Update device memory usage
     * @param device Device to update
     * @param allocated_size Size of memory allocated
     * @param deallocated_size Size of memory deallocated
     */
    void UpdateDeviceMemoryUsage(std::shared_ptr<Device> device, size_t allocated_size, size_t deallocated_size);
    
    /**
     * @brief Find suitable memory pool
     * @param size Size of memory needed
     * @param device Device to allocate on
     * @return Suitable memory pool, or nullptr if none found
     */
    MemoryPool* FindSuitableMemoryPool(size_t size, std::shared_ptr<Device> device);
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};

/**
 * @class MemoryPool
 * @brief Memory pool for efficient allocation
 */
class MemoryPool {
public:
    /**
     * @brief Constructor
     * @param pool_size Size of memory pool
     * @param device Device to allocate pool on
     */
    MemoryPool(size_t pool_size, std::shared_ptr<Device> device);
    
    /**
     * @brief Destructor
     */
    ~MemoryPool();
    
    // Disable copy constructor and assignment operator
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    /**
     * @brief Allocate memory from pool
     * @param size Size of memory to allocate
     * @param alignment Memory alignment requirement
     * @return Pointer to allocated memory, or nullptr if failed
     */
    void* Allocate(size_t size, size_t alignment = 64);
    
    /**
     * @brief Deallocate memory back to pool
     * @param ptr Pointer to memory to deallocate
     * @return Status indicating success or failure
     */
    Status Deallocate(void* ptr);
    
    /**
     * @brief Get pool statistics
     * @return Pool statistics
     */
    MemoryPoolStats GetPoolStats() const;
    
    /**
     * @brief Check if pool is full
     * @return True if pool is full
     */
    bool IsFull() const;
    
    /**
     * @brief Check if pool is empty
     * @return True if pool is empty
     */
    bool IsEmpty() const;
    
    /**
     * @brief Get pool utilization ratio
     * @return Pool utilization ratio (0.0 to 1.0)
     */
    double GetUtilizationRatio() const;

private:
    size_t pool_size_;
    void* pool_memory_;
    std::shared_ptr<Device> device_;
    
    // Allocation tracking
    std::unordered_map<void*, size_t> allocated_blocks_;
    std::mutex allocation_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    MemoryPoolStats stats_;
    
    /**
     * @brief Initialize memory pool
     * @return Status indicating success or failure
     */
    Status InitializePool();
    
    /**
     * @brief Cleanup memory pool
     */
    void CleanupPool();
};

/**
 * @class MemoryAllocator
 * @brief Memory allocator interface
 */
class MemoryAllocator {
public:
    /**
     * @brief Constructor
     * @param device Device to allocate memory on
     */
    explicit MemoryAllocator(std::shared_ptr<Device> device);
    
    /**
     * @brief Destructor
     */
    virtual ~MemoryAllocator() = default;
    
    // Disable copy constructor and assignment operator
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;
    
    /**
     * @brief Allocate memory
     * @param size Size of memory to allocate
     * @param alignment Memory alignment requirement
     * @return Pointer to allocated memory, or nullptr if failed
     */
    virtual void* Allocate(size_t size, size_t alignment = 64) = 0;
    
    /**
     * @brief Deallocate memory
     * @param ptr Pointer to memory to deallocate
     * @return Status indicating success or failure
     */
    virtual Status Deallocate(void* ptr) = 0;
    
    /**
     * @brief Get allocator statistics
     * @return Allocator statistics
     */
    virtual MemoryAllocatorStats GetAllocatorStats() const = 0;

protected:
    std::shared_ptr<Device> device_;
};


} // namespace edge_ai
