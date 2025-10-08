/**
 * @file model.cpp
 * @brief Model implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include "core/model.h"

namespace edge_ai {

Model::Model()
    : type_(ModelType::UNKNOWN)
    , size_(0)
    , optimized_(false)
    , quantization_type_(DataType::UNKNOWN)
    , quantized_(false)
    , pruned_(false)
    , pruning_ratio_(0.0f)
    , graph_optimized_(false) {
}

Model::~Model() = default;

std::string Model::GetName() const {
    return name_;
}

ModelType Model::GetType() const {
    return type_;
}

std::string Model::GetVersion() const {
    return version_;
}

size_t Model::GetSize() const {
    return size_;
}

bool Model::IsOptimized() const {
    return optimized_;
}

std::vector<TensorShape> Model::GetInputShapes() const {
    return input_shapes_;
}

std::vector<TensorShape> Model::GetOutputShapes() const {
    return output_shapes_;
}

std::vector<DataType> Model::GetInputTypes() const {
    return input_types_;
}

std::vector<DataType> Model::GetOutputTypes() const {
    return output_types_;
}

void Model::SetName(const std::string& name) {
    name_ = name;
}

void Model::SetType(ModelType type) {
    type_ = type;
}

void Model::SetVersion(const std::string& version) {
    version_ = version;
}

void Model::SetSize(size_t size) {
    size_ = size;
}

void Model::SetOptimized(bool optimized) {
    optimized_ = optimized;
}

void Model::SetInputShapes(const std::vector<TensorShape>& shapes) {
    input_shapes_ = shapes;
}

void Model::SetOutputShapes(const std::vector<TensorShape>& shapes) {
    output_shapes_ = shapes;
}

void Model::SetInputTypes(const std::vector<DataType>& types) {
    input_types_ = types;
}

void Model::SetOutputTypes(const std::vector<DataType>& types) {
    output_types_ = types;
}

void Model::SetQuantizationType(DataType type) {
    quantization_type_ = type;
}

void Model::SetQuantized(bool quantized) {
    quantized_ = quantized;
}

void Model::SetPruned(bool pruned) {
    pruned_ = pruned;
}

void Model::SetPruningRatio(float ratio) {
    pruning_ratio_ = ratio;
}

void Model::SetGraphOptimized(bool optimized) {
    graph_optimized_ = optimized;
}

bool Model::IsValid() const {
    return !name_.empty() && 
           type_ != ModelType::UNKNOWN && 
           size_ > 0 && 
           !input_shapes_.empty() && 
           !output_shapes_.empty();
}

} // namespace edge_ai
