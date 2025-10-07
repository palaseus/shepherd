/**
 * @file device.cpp
 * @brief Device implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "core/device.h"

namespace edge_ai {

Device::Device(DeviceType type, int device_id)
    : device_type_(type)
    , device_id_(device_id)
    , memory_size_(0)
    , compute_units_(0)
    , available_(false) {
}

Device::~Device() = default;

DeviceType Device::GetDeviceType() const {
    return device_type_;
}

int Device::GetDeviceId() const {
    return device_id_;
}

std::string Device::GetDeviceName() const {
    return device_name_;
}

std::string Device::GetVendor() const {
    return vendor_;
}

std::string Device::GetVersion() const {
    return version_;
}

size_t Device::GetMemorySize() const {
    return memory_size_;
}

int Device::GetComputeUnits() const {
    return compute_units_;
}

bool Device::IsAvailable() const {
    return available_;
}

} // namespace edge_ai
