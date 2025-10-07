/**
 * @file memory_config.h
 * @brief Memory configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MemoryConfig class for memory configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"

namespace edge_ai {

/**
 * @struct MemoryConfig
 * @brief Configuration for memory manager
 */
struct MemoryConfig {
    // Memory pool configuration
    bool enable_memory_pooling = true;
    size_t memory_pool_size = 512 * 1024 * 1024; // 512MB
    int num_memory_pools = 4;
    
    // Memory allocation configuration
    size_t default_alignment = 64;
    bool enable_memory_alignment = true;
    bool enable_memory_reuse = true;
    
    // Memory limit configuration
    size_t max_memory_usage = 1024 * 1024 * 1024; // 1GB
    bool enable_memory_limit = true;
    
    // Performance configuration
    bool enable_memory_defragmentation = true;
    double fragmentation_threshold = 0.3; // 30%
    
    MemoryConfig() = default;
};

} // namespace edge_ai
