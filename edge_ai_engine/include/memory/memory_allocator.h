/**
 * @file memory_allocator.h
 * @brief Memory allocator interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryAllocator class for memory allocation in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>

namespace edge_ai {

// Forward declarations
class Device;

/**
 * @class MemoryAllocator
 * @brief Memory allocator interface
 * 
 * The MemoryAllocator class provides a common interface for memory allocation
 * on different devices in the Edge AI Engine.
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
