/**
 * @file gpu_device.h
 * @brief GPU device interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the GPUDevice class for GPU devices in the Edge AI Engine.
 */

#pragma once

#include "device.h"
#include <string>
#include <vector>

namespace edge_ai {

/**
 * @class GPUDevice
 * @brief GPU device implementation
 * 
 * The GPUDevice class provides GPU-based acceleration for AI inference.
 */
class GPUDevice : public Device {
public:
    /**
     * @brief Constructor
     * @param device_id Device ID
     */
    explicit GPUDevice(int device_id);
    
    /**
     * @brief Destructor
     */
    ~GPUDevice() override;
    
    /**
     * @brief Initialize the GPU device
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the GPU device
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Allocate memory on the device
     * @param size Size of memory to allocate
     * @param ptr Pointer to store allocated memory address
     * @return Status indicating success or failure
     */
    Status AllocateMemory(size_t size, void** ptr);
    
    /**
     * @brief Deallocate memory on the device
     * @param ptr Pointer to memory to deallocate
     * @return Status indicating success or failure
     */
    Status DeallocateMemory(void* ptr);
    
    /**
     * @brief Execute a kernel on the device
     * @param kernel_name Name of the kernel to execute
     * @param args Arguments for the kernel
     * @return Status indicating success or failure
     */
    Status ExecuteKernel(const std::string& kernel_name, const std::vector<void*>& args);
};

} // namespace edge_ai
