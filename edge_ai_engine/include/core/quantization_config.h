/**
 * @file quantization_config.h
 * @brief Quantization configuration interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the QuantizationConfig class for quantization configuration in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <string>

namespace edge_ai {

/**
 * @struct QuantizationConfig
 * @brief Configuration for model quantization
 */
struct QuantizationConfig {
    // Quantization type
    DataType target_type = DataType::INT8;
    bool per_channel_quantization = false;
    bool symmetric_quantization = true;
    
    // Calibration configuration
    bool enable_calibration = true;
    size_t calibration_samples = 100;
    std::string calibration_dataset_path;
    
    // Accuracy preservation
    float max_accuracy_loss = 0.01f; // 1%
    bool enable_accuracy_validation = true;
    
    // Hardware-specific optimization
    bool optimize_for_hardware = true;
    std::string target_hardware;
    
    QuantizationConfig() = default;
};

} // namespace edge_ai
