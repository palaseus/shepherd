/**
 * @file device_info.h
 * @brief Device information interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the DeviceInfo class for device information in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <string>

namespace edge_ai {

/**
 * @struct DeviceInfo
 * @brief Information about a device
 */
struct DeviceInfo {
    DeviceType type;
    std::string name;
    std::string vendor;
    std::string version;
    size_t memory_size;
    int compute_units;
    bool available;
    
    DeviceInfo() : type(DeviceType::CPU), memory_size(0), compute_units(0), available(false) {}
};

} // namespace edge_ai
