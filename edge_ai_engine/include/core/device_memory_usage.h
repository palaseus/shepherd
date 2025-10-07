/**
 * @file device_memory_usage.h
 * @brief Device memory usage interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the DeviceMemoryUsage class for device memory usage in the Edge AI Engine.
 */

#pragma once

#include "types.h"

namespace edge_ai {

/**
 * @struct DeviceMemoryUsage
 * @brief Memory usage information for a device
 */
struct DeviceMemoryUsage {
    size_t total_memory = 0;
    size_t allocated_memory = 0;
    size_t free_memory = 0;
    size_t peak_memory_usage = 0;
    double utilization_ratio = 0.0;
    
    DeviceMemoryUsage() = default;
};

} // namespace edge_ai
