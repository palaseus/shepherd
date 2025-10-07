/**
 * @file device_manager.cpp
 * @brief Hardware device management implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "hardware/device_manager.h"
#include <stdexcept>
#include <thread>

namespace edge_ai {

DeviceManager::DeviceManager()
    : initialized_(false) {
}

DeviceManager::~DeviceManager() {
    Cleanup();
}

Status DeviceManager::Initialize() {
    try {
        if (initialized_) {
            return Status::ALREADY_INITIALIZED;
        }
        
        // Initialize devices
        Status status = InitializeDevices();
        if (status != Status::SUCCESS) {
            return status;
        }
        
        initialized_ = true;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status DeviceManager::Shutdown() {
    try {
        if (!initialized_) {
            return Status::SUCCESS;
        }
        
        Cleanup();
        initialized_ = false;
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

std::vector<std::shared_ptr<Device>> DeviceManager::GetAvailableDevices() const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    return devices_;
}

std::shared_ptr<Device> DeviceManager::GetDevice(DeviceType type, int device_id) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    
    for (const auto& device : devices_) {
        if (device->GetDeviceType() == type && device->GetDeviceId() == device_id) {
            return device;
        }
    }
    
    return nullptr;
}

std::shared_ptr<Device> DeviceManager::GetBestDevice(DeviceType preferred_type) const {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    
    // First try to find preferred type
    for (const auto& device : devices_) {
        if (device->GetDeviceType() == preferred_type && device->IsAvailable()) {
            return device;
        }
    }
    
    // Fallback to any available device
    for (const auto& device : devices_) {
        if (device->IsAvailable()) {
            return device;
        }
    }
    
    return nullptr;
}

Status DeviceManager::AddDevice(std::shared_ptr<Device> device) {
    try {
        if (!device) {
            return Status::INVALID_ARGUMENT;
        }
        
        std::lock_guard<std::mutex> lock(devices_mutex_);
        devices_.push_back(device);
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status DeviceManager::RemoveDevice(std::shared_ptr<Device> device) {
    try {
        if (!device) {
            return Status::INVALID_ARGUMENT;
        }
        
        std::lock_guard<std::mutex> lock(devices_mutex_);
        auto it = std::find(devices_.begin(), devices_.end(), device);
        if (it != devices_.end()) {
            devices_.erase(it);
            return Status::SUCCESS;
        }
        
        return Status::FAILURE;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

DeviceInfo DeviceManager::GetDeviceInfo(std::shared_ptr<Device> device) const {
    DeviceInfo info;
    
    try {
        if (device) {
            info.type = device->GetDeviceType();
            info.name = device->GetDeviceName();
            info.vendor = device->GetVendor();
            info.version = device->GetVersion();
            info.memory_size = device->GetMemorySize();
            info.compute_units = device->GetComputeUnits();
            info.available = device->IsAvailable();
        }
    } catch (const std::exception& e) {
        // Return default info on error
    }
    
    return info;
}

// Private methods
Status DeviceManager::InitializeDevices() {
    try {
        // Create CPU device
        auto cpu_device = std::make_shared<CPUDevice>(0);
        devices_.push_back(cpu_device);
        
        // Try to create GPU device if available
        auto gpu_device = std::make_shared<GPUDevice>(0);
        if (gpu_device->IsAvailable()) {
            devices_.push_back(gpu_device);
        }
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

void DeviceManager::Cleanup() {
    std::lock_guard<std::mutex> lock(devices_mutex_);
    devices_.clear();
}

// Device base class implementation
Device::Device(DeviceType type, int device_id)
    : device_type_(type)
    , device_id_(device_id)
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

// CPUDevice implementation is in cpu_device.cpp

// GPUDevice implementation
GPUDevice::GPUDevice(int device_id)
    : Device(DeviceType::GPU, device_id) {
    Initialize();
}

GPUDevice::~GPUDevice() = default;

Status GPUDevice::Initialize() {
    try {
        device_name_ = "GPU";
        vendor_ = "NVIDIA";
        version_ = "1.0";
        memory_size_ = 4ULL * 1024 * 1024 * 1024; // 4GB placeholder
        compute_units_ = 1024; // Placeholder
        available_ = false; // Assume not available by default
        
        // In practice, this would check for actual GPU availability
        // For now, we'll assume it's not available
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status GPUDevice::Shutdown() {
    available_ = false;
    return Status::SUCCESS;
}

Status GPUDevice::AllocateMemory(size_t size, void** ptr) {
    try {
        if (!ptr || size == 0) {
            return Status::INVALID_ARGUMENT;
        }
        
        if (!available_) {
            return Status::HARDWARE_NOT_AVAILABLE;
        }
        
        // Placeholder implementation
        *ptr = std::aligned_alloc(64, size);
        if (!*ptr) {
            return Status::OUT_OF_MEMORY;
        }
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status GPUDevice::DeallocateMemory(void* ptr) {
    try {
        if (ptr) {
            std::free(ptr);
        }
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

Status GPUDevice::ExecuteKernel([[maybe_unused]] const std::string& kernel_name, [[maybe_unused]] const std::vector<void*>& args) {
    try {
        if (!available_) {
            return Status::HARDWARE_NOT_AVAILABLE;
        }
        
        // Placeholder implementation
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
}

} // namespace edge_ai
