/**
 * @file gpu_accelerator.h
 * @brief GPU accelerator interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the GPUAccelerator class for GPU acceleration in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <string>
#include <vector>

namespace edge_ai {

/**
 * @class GPUAccelerator
 * @brief GPU accelerator utility class
 * 
 * The GPUAccelerator class provides GPU-based acceleration for AI inference
 * in the Edge AI Engine.
 */
class GPUAccelerator {
public:
    /**
     * @brief Constructor
     */
    GPUAccelerator();
    
    /**
     * @brief Destructor
     */
    ~GPUAccelerator();
    
    // Disable copy constructor and assignment operator
    GPUAccelerator(const GPUAccelerator&) = delete;
    GPUAccelerator& operator=(const GPUAccelerator&) = delete;
    
    /**
     * @brief Initialize the GPU accelerator
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the GPU accelerator
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Execute a kernel on the GPU
     * @param kernel_name Name of the kernel to execute
     * @param args Arguments for the kernel
     * @return Status indicating success or failure
     */
    Status ExecuteKernel(const std::string& kernel_name, const std::vector<void*>& args);
};

} // namespace edge_ai
