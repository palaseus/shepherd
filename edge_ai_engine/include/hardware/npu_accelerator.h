/**
 * @file npu_accelerator.h
 * @brief NPU accelerator interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the NPUAccelerator class for NPU acceleration in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <string>
#include <vector>

namespace edge_ai {

/**
 * @class NPUAccelerator
 * @brief NPU accelerator utility class
 * 
 * The NPUAccelerator class provides NPU-based acceleration for AI inference
 * in the Edge AI Engine.
 */
class NPUAccelerator {
public:
    /**
     * @brief Constructor
     */
    NPUAccelerator();
    
    /**
     * @brief Destructor
     */
    ~NPUAccelerator();
    
    // Disable copy constructor and assignment operator
    NPUAccelerator(const NPUAccelerator&) = delete;
    NPUAccelerator& operator=(const NPUAccelerator&) = delete;
    
    /**
     * @brief Initialize the NPU accelerator
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the NPU accelerator
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Execute a kernel on the NPU
     * @param kernel_name Name of the kernel to execute
     * @param args Arguments for the kernel
     * @return Status indicating success or failure
     */
    Status ExecuteKernel(const std::string& kernel_name, const std::vector<void*>& args);
};

} // namespace edge_ai
