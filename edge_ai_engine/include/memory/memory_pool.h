/**
 * @file memory_pool.h
 * @brief Memory pool interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryPool class for memory management in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace edge_ai {

// Forward declarations
class Device;

/**
 * @class MemoryPool
 * @brief Memory pool for efficient allocation
 * 
 * The MemoryPool class provides efficient memory allocation and deallocation
 * for the Edge AI Engine.
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

} // namespace edge_ai
