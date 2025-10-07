/**
 * @file model.h
 * @brief Model interface
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the Model class for representing AI models in the Edge AI Engine.
 */

#pragma once

#include "types.h"
#include <memory>
#include <string>
#include <vector>

namespace edge_ai {

/**
 * @class Model
 * @brief Represents an AI model
 * 
 * The Model class represents an AI model that can be loaded and executed
 * by the Edge AI Engine.
 */
class Model {
public:
    /**
     * @brief Constructor
     */
    Model();
    
    /**
     * @brief Destructor
     */
    virtual ~Model();
    
    // Disable copy constructor and assignment operator
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    
    /**
     * @brief Get model name
     * @return Model name
     */
    std::string GetName() const;
    
    /**
     * @brief Get model type
     * @return Model type
     */
    ModelType GetType() const;
    
    /**
     * @brief Get model version
     * @return Model version
     */
    std::string GetVersion() const;
    
    /**
     * @brief Get model size
     * @return Model size in bytes
     */
    size_t GetSize() const;
    
    /**
     * @brief Check if model is optimized
     * @return True if model is optimized
     */
    bool IsOptimized() const;
    
    /**
     * @brief Get input shapes
     * @return Vector of input shapes
     */
    std::vector<TensorShape> GetInputShapes() const;
    
    /**
     * @brief Get output shapes
     * @return Vector of output shapes
     */
    std::vector<TensorShape> GetOutputShapes() const;
    
    /**
     * @brief Get input types
     * @return Vector of input types
     */
    std::vector<DataType> GetInputTypes() const;
    
    /**
     * @brief Get output types
     * @return Vector of output types
     */
    std::vector<DataType> GetOutputTypes() const;
    
    /**
     * @brief Set model name
     * @param name Model name
     */
    void SetName(const std::string& name);
    
    /**
     * @brief Set model type
     * @param type Model type
     */
    void SetType(ModelType type);
    
    /**
     * @brief Set model version
     * @param version Model version
     */
    void SetVersion(const std::string& version);
    
    /**
     * @brief Set model size
     * @param size Model size in bytes
     */
    void SetSize(size_t size);
    
    /**
     * @brief Set optimization status
     * @param optimized True if model is optimized
     */
    void SetOptimized(bool optimized);
    
    /**
     * @brief Set input shapes
     * @param shapes Vector of input shapes
     */
    void SetInputShapes(const std::vector<TensorShape>& shapes);
    
    /**
     * @brief Set output shapes
     * @param shapes Vector of output shapes
     */
    void SetOutputShapes(const std::vector<TensorShape>& shapes);
    
    /**
     * @brief Set input types
     * @param types Vector of input types
     */
    void SetInputTypes(const std::vector<DataType>& types);
    
    /**
     * @brief Set output types
     * @param types Vector of output types
     */
    void SetOutputTypes(const std::vector<DataType>& types);

protected:
    std::string name_;
    ModelType type_;
    std::string version_;
    size_t size_;
    bool optimized_;
    std::vector<TensorShape> input_shapes_;
    std::vector<TensorShape> output_shapes_;
    std::vector<DataType> input_types_;
    std::vector<DataType> output_types_;
};

} // namespace edge_ai
