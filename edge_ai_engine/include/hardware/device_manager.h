/**
 * @file device_manager.h
 * @brief Hardware device management interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the DeviceManager class and device interfaces for
 * managing hardware acceleration in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>
#include <vector>
#include <string>
#include <mutex>

namespace edge_ai {

// Forward declarations
class Device;
class CPUDevice;
class GPUDevice;

/**
 * @class DeviceManager
 * @brief Manages hardware devices for acceleration
 * 
 * The DeviceManager class handles discovery, initialization, and management
 * of hardware devices for AI inference acceleration.
 */
class DeviceManager {
public:
    /**
     * @brief Constructor
     */
    DeviceManager();
    
    /**
     * @brief Destructor
     */
    ~DeviceManager();
    
    // Disable copy constructor and assignment operator
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;
    
    /**
     * @brief Initialize the device manager
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the device manager
     * @return Status indicating success or failure
     */
    Status Shutdown();
    
    /**
     * @brief Get all available devices
     * @return Vector of available devices
     */
    std::vector<std::shared_ptr<Device>> GetAvailableDevices() const;
    
    /**
     * @brief Get a specific device
     * @param type Device type
     * @param device_id Device ID
     * @return Shared pointer to device, or nullptr if not found
     */
    std::shared_ptr<Device> GetDevice(DeviceType type, int device_id) const;
    
    /**
     * @brief Get the best available device
     * @param preferred_type Preferred device type
     * @return Shared pointer to best device, or nullptr if none available
     */
    std::shared_ptr<Device> GetBestDevice(DeviceType preferred_type) const;
    
    /**
     * @brief Add a device
     * @param device Device to add
     * @return Status indicating success or failure
     */
    Status AddDevice(std::shared_ptr<Device> device);
    
    /**
     * @brief Remove a device
     * @param device Device to remove
     * @return Status indicating success or failure
     */
    Status RemoveDevice(std::shared_ptr<Device> device);
    
    /**
     * @brief Get device information
     * @param device Device to get info for
     * @return Device information
     */
    DeviceInfo GetDeviceInfo(std::shared_ptr<Device> device) const;

private:
    bool initialized_;
    std::vector<std::shared_ptr<Device>> devices_;
    mutable std::mutex devices_mutex_;
    
    /**
     * @brief Initialize devices
     * @return Status indicating success or failure
     */
    Status InitializeDevices();
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};

/**
 * @class Device
 * @brief Base class for hardware devices
 * 
 * The Device class provides a common interface for all hardware devices
 * used in the Edge AI Engine.
 */
class Device {
public:
    /**
     * @brief Constructor
     * @param type Device type
     * @param device_id Device ID
     */
    Device(DeviceType type, int device_id);
    
    /**
     * @brief Destructor
     */
    virtual ~Device();
    
    // Disable copy constructor and assignment operator
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    
    /**
     * @brief Get device type
     * @return Device type
     */
    DeviceType GetDeviceType() const;
    
    /**
     * @brief Get device ID
     * @return Device ID
     */
    int GetDeviceId() const;
    
    /**
     * @brief Get device name
     * @return Device name
     */
    std::string GetDeviceName() const;
    
    /**
     * @brief Get device vendor
     * @return Device vendor
     */
    std::string GetVendor() const;
    
    /**
     * @brief Get device version
     * @return Device version
     */
    std::string GetVersion() const;
    
    /**
     * @brief Get device memory size
     * @return Memory size in bytes
     */
    size_t GetMemorySize() const;
    
    /**
     * @brief Get number of compute units
     * @return Number of compute units
     */
    int GetComputeUnits() const;
    
    /**
     * @brief Check if device is available
     * @return True if device is available
     */
    bool IsAvailable() const;

protected:
    DeviceType device_type_;
    int device_id_;
    std::string device_name_;
    std::string vendor_;
    std::string version_;
    size_t memory_size_;
    int compute_units_;
    bool available_;
};

/**
 * @class CPUDevice
 * @brief CPU device implementation
 * 
 * The CPUDevice class provides CPU-based acceleration for AI inference.
 */
class CPUDevice : public Device {
public:
    /**
     * @brief Constructor
     * @param device_id Device ID
     */
    explicit CPUDevice(int device_id);
    
    /**
     * @brief Destructor
     */
    ~CPUDevice() override;
    
    /**
     * @brief Initialize the CPU device
     * @return Status indicating success or failure
     */
    Status Initialize();
    
    /**
     * @brief Shutdown the CPU device
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
