/**
 * @file device.h
 * @brief Device interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Device class for representing hardware devices in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <memory>
#include <string>
#include <vector>

namespace edge_ai {

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

} // namespace edge_ai
