/**
 * @file hardware_utilization_stats.h
 * @brief Hardware utilization statistics interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the HardwareUtilizationStats class for hardware utilization statistics in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <string>
#include <chrono>

namespace edge_ai {

/**
 * @struct HardwareUtilizationStats
 * @brief Statistics for hardware utilization
 */
struct HardwareUtilizationStats {
    std::string device_name;
    double current_utilization = 0.0;
    double average_utilization = 0.0;
    double peak_utilization = 0.0;
    double min_utilization = 1.0;
    uint64_t measurement_count = 0;
    std::chrono::steady_clock::time_point last_measurement;
    
    HardwareUtilizationStats() = default;
    explicit HardwareUtilizationStats(const std::string& name) : device_name(name) {}
};

} // namespace edge_ai
