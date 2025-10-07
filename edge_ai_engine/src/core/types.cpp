/**
 * @file types.cpp
 * @brief Implementation of core data types and structures
 * @author AI Co-Developer
 * @date 2024
 */

#include "core/types.h"
#include <algorithm>
#include <numeric>

namespace edge_ai {

// TensorShape implementation
size_t TensorShape::GetTotalElements() const {
    if (dimensions.empty()) {
        return 0;
    }
    return std::accumulate(dimensions.begin(), dimensions.end(), 1ULL, std::multiplies<size_t>());
}

bool TensorShape::IsValid() const {
    return !dimensions.empty() && 
           std::all_of(dimensions.begin(), dimensions.end(), [](int64_t dim) { return dim > 0; });
}

std::string TensorShape::ToString() const {
    std::string result = "[";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(dimensions[i]);
    }
    result += "]";
    return result;
}

// Tensor implementation
Tensor::Tensor(DataType type, const TensorShape& shape, void* data)
    : data_type_(type), shape_(shape), data_(data), size_(0), owns_data_(data == nullptr) {
    if (owns_data_) {
        AllocateMemory();
    } else {
        size_ = GetSize();
    }
}

Tensor::~Tensor() {
    if (owns_data_) {
        DeallocateMemory();
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_type_(other.data_type_)
    , shape_(std::move(other.shape_))
    , data_(other.data_)
    , size_(other.size_)
    , owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_) {
            DeallocateMemory();
        }
        
        data_type_ = other.data_type_;
        shape_ = std::move(other.shape_);
        data_ = other.data_;
        size_ = other.size_;
        owns_data_ = other.owns_data_;
        
        other.data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
    }
    return *this;
}

Status Tensor::SetData(void* data, size_t size) {
    if (data == nullptr || size == 0) {
        return Status::INVALID_ARGUMENT;
    }
    
    if (owns_data_) {
        DeallocateMemory();
    }
    
    data_ = data;
    size_ = size;
    owns_data_ = false;
    
    return Status::SUCCESS;
}

Status Tensor::Reshape(const TensorShape& new_shape) {
    if (!new_shape.IsValid()) {
        return Status::INVALID_ARGUMENT;
    }
    
    size_t new_size = new_shape.GetTotalElements() * GetDataTypeSize();
    if (new_size != size_) {
        return Status::INVALID_ARGUMENT;
    }
    
    shape_ = new_shape;
    return Status::SUCCESS;
}

bool Tensor::IsValid() const {
    return data_type_ != DataType::UNKNOWN && 
           shape_.IsValid() && 
           data_ != nullptr && 
           size_ > 0;
}

std::string Tensor::ToString() const {
    std::string result = "Tensor(";
    result += "type=" + std::to_string(static_cast<int>(data_type_));
    result += ", shape=" + shape_.ToString();
    result += ", size=" + std::to_string(size_);
    result += ")";
    return result;
}

void Tensor::AllocateMemory() {
    size_ = shape_.GetTotalElements() * GetDataTypeSize();
    if (size_ > 0) {
        data_ = std::aligned_alloc(64, size_);
        if (data_ == nullptr) {
            size_ = 0;
        }
    }
}

void Tensor::DeallocateMemory() {
    if (data_ != nullptr) {
        std::free(data_);
        data_ = nullptr;
    }
    size_ = 0;
}

size_t Tensor::GetDataTypeSize() const {
    switch (data_type_) {
        case DataType::FLOAT32: return sizeof(float);
        case DataType::FLOAT16: return sizeof(uint16_t);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::INT16: return sizeof(int16_t);
        case DataType::INT8: return sizeof(int8_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::BOOL: return sizeof(bool);
        default: return 0;
    }
}

} // namespace edge_ai
