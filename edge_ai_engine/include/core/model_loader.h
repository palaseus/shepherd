/**
 * @file model_loader.h
 * @brief Model loading and management interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the ModelLoader class which handles loading and
 * managing models in various formats (ONNX, TensorFlow Lite, PyTorch Mobile).
 */

#pragma once

#include "types.h"
#include <memory>
#include <string>
#include <vector>

namespace edge_ai {

// Forward declarations
class Model;
class Device;

/**
 * @class ModelLoader
 * @brief Handles loading and management of AI models
 * 
 * The ModelLoader class is responsible for loading models from various formats,
 * validating model compatibility, and providing access to model metadata.
 */
class ModelLoader {
public:
    /**
     * @brief Constructor
     * @param device Target device for model loading
     */
    explicit ModelLoader(std::shared_ptr<Device> device);
    
    /**
     * @brief Destructor
     */
    ~ModelLoader();
    
    // Disable copy constructor and assignment operator
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    
    /**
     * @brief Load a model from file
     * @param model_path Path to the model file
     * @param model_type Type of model to load
     * @return Status indicating success or failure
     */
    Status LoadModel(const std::string& model_path, ModelType model_type);
    
    /**
     * @brief Load a model from memory
     * @param model_data Model data in memory
     * @param model_size Size of model data
     * @param model_type Type of model to load
     * @return Status indicating success or failure
     */
    Status LoadModel(const void* model_data, size_t model_size, ModelType model_type);
    
    /**
     * @brief Unload the current model
     * @return Status indicating success or failure
     */
    Status UnloadModel();
    
    /**
     * @brief Check if a model is loaded
     * @return True if model is loaded, false otherwise
     */
    bool IsModelLoaded() const;
    
    /**
     * @brief Get the loaded model
     * @return Shared pointer to the model, or nullptr if not loaded
     */
    std::shared_ptr<Model> GetModel() const;
    
    /**
     * @brief Get model information
     * @return Model information structure
     */
    ModelInfo GetModelInfo() const;
    
    /**
     * @brief Validate model compatibility
     * @param model_path Path to model file
     * @param model_type Type of model
     * @return Status indicating compatibility
     */
    Status ValidateModel(const std::string& model_path, ModelType model_type) const;
    
    /**
     * @brief Get supported model types
     * @return Vector of supported model types
     */
    std::vector<ModelType> GetSupportedModelTypes() const;
    
    /**
     * @brief Get model file size
     * @param model_path Path to model file
     * @return Size of model file in bytes, or 0 if error
     */
    size_t GetModelFileSize(const std::string& model_path) const;
    
    /**
     * @brief Set device for model loading
     * @param device Target device
     * @return Status indicating success or failure
     */
    Status SetDevice(std::shared_ptr<Device> device);
    
    /**
     * @brief Get current device
     * @return Shared pointer to current device
     */
    std::shared_ptr<Device> GetDevice() const;

private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<Model> model_;
    bool model_loaded_;
    
    /**
     * @brief Detect model type from file extension
     * @param model_path Path to model file
     * @return Detected model type or UNKNOWN if not recognized
     */
    ModelType DetectModelType(const std::string& model_path) const;
    
    /**
     * @brief Load ONNX model
     * @param model_path Path to ONNX model file
     * @return Status indicating success or failure
     */
    Status LoadONNXModel(const std::string& model_path);
    
    /**
     * @brief Load ONNX model from memory
     * @param model_data Model data in memory
     * @param model_size Size of model data
     * @return Status indicating success or failure
     */
    Status LoadONNXModel(const void* model_data, size_t model_size);
    
    /**
     * @brief Load TensorFlow Lite model
     * @param model_path Path to TFLite model file
     * @return Status indicating success or failure
     */
    Status LoadTensorFlowLiteModel(const std::string& model_path);
    
    /**
     * @brief Load TensorFlow Lite model from memory
     * @param model_data Model data in memory
     * @param model_size Size of model data
     * @return Status indicating success or failure
     */
    Status LoadTensorFlowLiteModel(const void* model_data, size_t model_size);
    
    /**
     * @brief Load PyTorch Mobile model
     * @param model_path Path to PyTorch Mobile model file
     * @return Status indicating success or failure
     */
    Status LoadPyTorchMobileModel(const std::string& model_path);
    
    /**
     * @brief Load PyTorch Mobile model from memory
     * @param model_data Model data in memory
     * @param model_size Size of model data
     * @return Status indicating success or failure
     */
    Status LoadPyTorchMobileModel(const void* model_data, size_t model_size);
    
    /**
     * @brief Validate model file exists and is readable
     * @param model_path Path to model file
     * @return Status indicating success or failure
     */
    Status ValidateModelFile(const std::string& model_path) const;
    
    /**
     * @brief Extract model metadata
     * @param model_path Path to model file
     * @param model_type Type of model
     * @return Model information structure
     */
    ModelInfo ExtractModelInfo(const std::string& model_path, ModelType model_type) const;
    
    /**
     * @brief Cleanup resources
     */
    void Cleanup();
};

} // namespace edge_ai
