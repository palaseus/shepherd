/**
 * @file quantizer.h
 * @brief Model quantization interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Quantizer class for model quantization in the Edge AI Engine.
 */

#pragma once

#include "../core/types.h"
#include <memory>

namespace edge_ai {

// Forward declarations
class Model;

/**
 * @class Quantizer
 * @brief Model quantization utility class
 * 
 * The Quantizer class provides functionality for quantizing models to reduce
 * their size and improve inference performance.
 */
class Quantizer {
public:
    /**
     * @brief Constructor
     */
    Quantizer();
    
    /**
     * @brief Destructor
     */
    ~Quantizer();
    
    // Disable copy constructor and assignment operator
    Quantizer(const Quantizer&) = delete;
    Quantizer& operator=(const Quantizer&) = delete;
    
    /**
     * @brief Quantize a model
     * @param input_model Input model to quantize
     * @param output_model Quantized model (will be populated)
     * @param config Quantization configuration
     * @return Status indicating success or failure
     */
    Status QuantizeModel(std::shared_ptr<Model> input_model,
                        std::shared_ptr<Model>& output_model,
                        const QuantizationConfig& config);
};

} // namespace edge_ai
