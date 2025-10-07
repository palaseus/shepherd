/**
 * @file cpu_accelerator.h
 * @brief CPU accelerator interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the CPUAccelerator class for CPU acceleration in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <string>
#include <vector>

namespace edge_ai {

/**
 * @class CPUAccelerator
 * @brief CPU accelerator utility class
 * 
 * The CPUAccelerator class provides CPU-based acceleration for AI inference
 * in the Edge AI Engine.
 */
class CPUAccelerator {
public:
    /**
     * @brief Constructor
     */
    CPUAccelerator();
    
    /**
     * @brief Destructor
     */
    ~CPUAccelerator();
    
    // Disable copy constructor and assignment operator
    CPUAccelerator(const CPUAccelerator&) = delete;
    CPUAccelerator& operator=(const CPUAccelerator&) = delete;
    
    /**
     * @brief Initialize the CPU accelerator
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the CPU accelerator
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Execute a kernel on the CPU
     * @param kernel_name Name of the kernel to execute
     * @param args Arguments for the kernel
     * @return Status indicating success or failure
     */
    Status ExecuteKernel(const std::string& kernel_name, const std::vector<void*>& args);
};

} // namespace edge_ai
