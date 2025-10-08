/**
 * @file cuda_backend.h
 * @brief CUDA GPU execution backend for high-performance inference
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the CUDABackend class which provides real GPU acceleration
 * using CUDA for high-performance AI model inference.
 */

#pragma once

#include "execution_backend.h"
#include <memory>
#include <vector>
#include <mutex>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#else
// Mock CUDA types when CUDA is not available
typedef int cudaError_t;
typedef int cublasStatus_t;
typedef int cudnnStatus_t;
typedef int cudaDataType;
typedef struct { int major, minor; } cudaDeviceProp;
typedef void* cudaStream_t;
typedef void* cublasHandle_t;
typedef void* cudnnHandle_t;
#endif

namespace edge_ai {

/**
 * @class CUDABackend
 * @brief CUDA GPU execution backend
 * 
 * The CUDABackend class provides real GPU acceleration using CUDA for:
 * - High-performance matrix operations
 * - Parallel inference execution
 * - Memory management on GPU
 * - Batch processing optimization
 */
class CUDABackend : public ExecutionBackend {
public:
    /**
     * @brief Constructor
     * @param device GPU device to use
     * @param device_id CUDA device ID
     */
    explicit CUDABackend(std::shared_ptr<Device> device, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~CUDABackend() override;
    
    /**
     * @brief Initialize the CUDA backend
     * @return Status indicating success or failure
     */
    Status Initialize() override;
    
    /**
     * @brief Shutdown the CUDA backend
     * @return Status indicating success or failure
     */
    Status Shutdown() override;
    
    /**
     * @brief Execute inference on a single request
     * @param model Model to execute
     * @param request Inference request
     * @param result Inference result
     * @return Status indicating success or failure
     */
    Status Execute(const Model& model, 
                  const InferenceRequest& request, 
                  InferenceResult& result) override;
    
    /**
     * @brief Execute inference on a batch of requests
     * @param model Model to execute
     * @param requests Batch of inference requests
     * @param results Batch of inference results
     * @return Status indicating success or failure
     */
    Status ExecuteBatch(const Model& model,
                       const std::vector<InferenceRequest>& requests,
                       std::vector<InferenceResult>& results) override;
    
    /**
     * @brief Get CUDA backend capabilities
     * @return CUDA backend capabilities
     */
    BackendCapabilities GetCapabilities() const override;
    
    /**
     * @brief Check if CUDA backend supports a specific model type
     * @param model_type Model type to check
     * @return True if supported, false otherwise
     */
    bool SupportsModelType(ModelType model_type) const override;
    
    /**
     * @brief Check if CUDA backend supports a specific data type
     * @param data_type Data type to check
     * @return True if supported, false otherwise
     */
    bool SupportsDataType(DataType data_type) const override;
    
    /**
     * @brief Get CUDA backend name
     * @return CUDA backend name
     */
    std::string GetName() const override;
    
    /**
     * @brief Get CUDA backend version
     * @return CUDA backend version
     */
    std::string GetVersion() const override;
    
    /**
     * @brief Get CUDA backend unique identifier
     * @return CUDA backend unique identifier
     */
    std::string GetId() const override;
    
    /**
     * @brief Get CUDA device properties
     * @return CUDA device properties
     */
    cudaDeviceProp GetDeviceProperties() const;
    
    /**
     * @brief Get available GPU memory
     * @return Available GPU memory in bytes
     */
    size_t GetAvailableMemory() const;
    
    /**
     * @brief Get total GPU memory
     * @return Total GPU memory in bytes
     */
    size_t GetTotalMemory() const;

private:
    int device_id_;
    cudaDeviceProp device_props_;
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
    cudaStream_t stream_;
    std::mutex execution_mutex_;
    
    // Memory management
    std::vector<void*> allocated_ptrs_;
    size_t total_allocated_;
    size_t max_memory_;
    
    /**
     * @brief Initialize CUDA runtime
     * @return Status indicating success or failure
     */
    Status InitializeCUDA();
    
    /**
     * @brief Initialize cuBLAS
     * @return Status indicating success or failure
     */
    Status InitializeCUBLAS();
    
    /**
     * @brief Initialize cuDNN
     * @return Status indicating success or failure
     */
    Status InitializeCUDNN();
    
    /**
     * @brief Cleanup CUDA resources
     */
    void CleanupCUDA();
    
    /**
     * @brief Allocate GPU memory
     * @param size Size in bytes
     * @return Pointer to allocated memory or nullptr on failure
     */
    void* AllocateGPUMemory(size_t size);
    
    /**
     * @brief Free GPU memory
     * @param ptr Pointer to memory to free
     */
    void FreeGPUMemory(void* ptr);
    
    /**
     * @brief Copy data from host to device
     * @param dst Device pointer
     * @param src Host pointer
     * @param size Size in bytes
     * @return Status indicating success or failure
     */
    Status CopyHostToDevice(void* dst, const void* src, size_t size);
    
    /**
     * @brief Copy data from device to host
     * @param dst Host pointer
     * @param src Device pointer
     * @param size Size in bytes
     * @return Status indicating success or failure
     */
    Status CopyDeviceToHost(void* dst, const void* src, size_t size);
    
    /**
     * @brief Execute ONNX model on GPU
     * @param model Model to execute
     * @param request Inference request
     * @param result Inference result
     * @return Status indicating success or failure
     */
    Status ExecuteONNXModel(const Model& model, 
                           const InferenceRequest& request, 
                           InferenceResult& result);
    
    /**
     * @brief Execute TensorFlow Lite model on GPU
     * @param model Model to execute
     * @param request Inference request
     * @param result Inference result
     * @return Status indicating success or failure
     */
    Status ExecuteTFLiteModel(const Model& model, 
                             const InferenceRequest& request, 
                             InferenceResult& result);
    
    /**
     * @brief Execute PyTorch Mobile model on GPU
     * @param model Model to execute
     * @param request Inference request
     * @param result Inference result
     * @return Status indicating success or failure
     */
    Status ExecutePyTorchMobileModel(const Model& model, 
                                    const InferenceRequest& request, 
                                    InferenceResult& result);
    
    /**
     * @brief Create output tensors on GPU
     * @param model Model information
     * @param result Inference result
     * @return Status indicating success or failure
     */
    Status CreateGPUOutputs(const Model& model, InferenceResult& result);
    
    /**
     * @brief Convert data type to CUDA type
     * @param data_type Edge AI data type
     * @return CUDA data type
     */
    cudaDataType ConvertDataType(DataType data_type) const;
    
    /**
     * @brief Check CUDA error and convert to Status
     * @param cuda_error CUDA error code
     * @return Status indicating success or failure
     */
    Status CheckCUDAError(cudaError_t cuda_error) const;
    
    /**
     * @brief Check cuBLAS error and convert to Status
     * @param cublas_error cuBLAS error code
     * @return Status indicating success or failure
     */
    Status CheckCUBLASError(cublasStatus_t cublas_error) const;
    
    /**
     * @brief Check cuDNN error and convert to Status
     * @param cudnn_error cuDNN error code
     * @return Status indicating success or failure
     */
    Status CheckCUDNNError(cudnnStatus_t cudnn_error) const;
};

} // namespace edge_ai
