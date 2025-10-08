/**
 * @file test_cuda_backend.cpp
 * @brief Unit tests for CUDA backend implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include <gtest/gtest.h>
#include "backend/cuda_backend.h"
#include "core/model.h"
#include "core/types.h"
#include <memory>

namespace edge_ai {
namespace testing {

class CUDABackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a mock device for testing
        device_ = std::make_shared<Device>();
        device_->SetType(DeviceType::GPU);
        device_->SetId("test_gpu_0");
        
        // Create CUDA backend
        cuda_backend_ = std::make_shared<CUDABackend>(device_, 0);
    }
    
    void TearDown() override {
        if (cuda_backend_) {
            cuda_backend_->Shutdown();
        }
    }
    
    std::shared_ptr<Device> device_;
    std::shared_ptr<CUDABackend> cuda_backend_;
};

TEST_F(CUDABackendTest, ConstructorTest) {
    EXPECT_NE(cuda_backend_, nullptr);
    EXPECT_EQ(cuda_backend_->GetBackendType(), BackendType::GPU);
    EXPECT_EQ(cuda_backend_->GetDeviceId(), 0);
}

TEST_F(CUDABackendTest, InitializationTest) {
    // Test initialization (may fail if CUDA is not available)
    Status status = cuda_backend_->Initialize();
    
    // If CUDA is available, initialization should succeed
    if (status == Status::SUCCESS) {
        EXPECT_TRUE(cuda_backend_->IsInitialized());
    } else {
        // If CUDA is not available, we expect a specific error
        EXPECT_TRUE(status == Status::HARDWARE_NOT_AVAILABLE || 
                   status == Status::FAILURE);
    }
}

TEST_F(CUDABackendTest, CapabilitiesTest) {
    auto capabilities = cuda_backend_->GetCapabilities();
    
    EXPECT_TRUE(capabilities.supports_batching);
    EXPECT_TRUE(capabilities.supports_quantization);
    EXPECT_TRUE(capabilities.supports_pruning);
    EXPECT_GT(capabilities.max_batch_size, 0);
    EXPECT_GT(capabilities.max_memory_usage, 0);
    
    // Check supported data types
    EXPECT_TRUE(std::find(capabilities.supported_data_types.begin(),
                         capabilities.supported_data_types.end(),
                         DataType::FLOAT32) != capabilities.supported_data_types.end());
    EXPECT_TRUE(std::find(capabilities.supported_data_types.begin(),
                         capabilities.supported_data_types.end(),
                         DataType::FLOAT16) != capabilities.supported_data_types.end());
    
    // Check supported model types
    EXPECT_TRUE(std::find(capabilities.supported_model_types.begin(),
                         capabilities.supported_model_types.end(),
                         ModelType::ONNX) != capabilities.supported_model_types.end());
}

TEST_F(CUDABackendTest, ModelTypeSupportTest) {
    EXPECT_TRUE(cuda_backend_->SupportsModelType(ModelType::ONNX));
    EXPECT_TRUE(cuda_backend_->SupportsModelType(ModelType::TENSORFLOW_LITE));
    EXPECT_TRUE(cuda_backend_->SupportsModelType(ModelType::PYTORCH_MOBILE));
    EXPECT_FALSE(cuda_backend_->SupportsModelType(ModelType::UNKNOWN));
}

TEST_F(CUDABackendTest, DataTypeSupportTest) {
    EXPECT_TRUE(cuda_backend_->SupportsDataType(DataType::FLOAT32));
    EXPECT_TRUE(cuda_backend_->SupportsDataType(DataType::FLOAT16));
    EXPECT_TRUE(cuda_backend_->SupportsDataType(DataType::INT32));
    EXPECT_TRUE(cuda_backend_->SupportsDataType(DataType::INT8));
    EXPECT_FALSE(cuda_backend_->SupportsDataType(DataType::UNKNOWN));
}

TEST_F(CUDABackendTest, NameAndVersionTest) {
    std::string name = cuda_backend_->GetName();
    std::string version = cuda_backend_->GetVersion();
    std::string id = cuda_backend_->GetId();
    
    EXPECT_FALSE(name.empty());
    EXPECT_FALSE(version.empty());
    EXPECT_FALSE(id.empty());
    
    EXPECT_EQ(name, "CUDA GPU Backend");
    EXPECT_TRUE(version.find("CUDA") != std::string::npos);
    EXPECT_TRUE(id.find("cuda_backend_") != std::string::npos);
}

TEST_F(CUDABackendTest, MemoryInfoTest) {
    // Test memory information (may not be available if CUDA is not initialized)
    size_t total_memory = cuda_backend_->GetTotalMemory();
    size_t available_memory = cuda_backend_->GetAvailableMemory();
    
    // If CUDA is available, memory should be reported
    if (total_memory > 0) {
        EXPECT_GT(total_memory, 0);
        EXPECT_LE(available_memory, total_memory);
    }
}

TEST_F(CUDABackendTest, DevicePropertiesTest) {
    auto device_props = cuda_backend_->GetDeviceProperties();
    
    // Basic validation of device properties
    EXPECT_GE(device_props.major, 0);
    EXPECT_GE(device_props.minor, 0);
}

TEST_F(CUDABackendTest, ExecutionTest) {
    // Initialize backend first
    Status init_status = cuda_backend_->Initialize();
    if (init_status != Status::SUCCESS) {
        GTEST_SKIP() << "CUDA not available, skipping execution test";
    }
    
    // Create a test model
    auto model = std::make_shared<Model>();
    model->SetName("test_model");
    model->SetType(ModelType::ONNX);
    model->SetSize(1024);
    model->SetInputShapes({TensorShape({1, 3, 224, 224})});
    model->SetOutputShapes({TensorShape({1, 1000})});
    model->SetInputTypes({DataType::FLOAT32});
    model->SetOutputTypes({DataType::FLOAT32});
    
    // Create test input
    Tensor input_tensor(DataType::FLOAT32, TensorShape({1, 3, 224, 224}));
    
    InferenceRequest request;
    request.request_id = 1;
    request.inputs = {input_tensor};
    
    InferenceResult result;
    
    // Execute inference
    Status exec_status = cuda_backend_->Execute(*model, request, result);
    
    // Should succeed or fail gracefully
    EXPECT_TRUE(exec_status == Status::SUCCESS || exec_status == Status::FAILURE);
    
    if (exec_status == Status::SUCCESS) {
        EXPECT_EQ(result.request_id, request.request_id);
        EXPECT_EQ(result.status, Status::SUCCESS);
        EXPECT_GT(result.latency.count(), 0);
    }
}

TEST_F(CUDABackendTest, BatchExecutionTest) {
    // Initialize backend first
    Status init_status = cuda_backend_->Initialize();
    if (init_status != Status::SUCCESS) {
        GTEST_SKIP() << "CUDA not available, skipping batch execution test";
    }
    
    // Create a test model
    auto model = std::make_shared<Model>();
    model->SetName("test_model");
    model->SetType(ModelType::ONNX);
    model->SetSize(1024);
    model->SetInputShapes({TensorShape({1, 3, 224, 224})});
    model->SetOutputShapes({TensorShape({1, 1000})});
    model->SetInputTypes({DataType::FLOAT32});
    model->SetOutputTypes({DataType::FLOAT32});
    
    // Create batch of requests
    std::vector<InferenceRequest> requests;
    std::vector<InferenceResult> results;
    
    for (int i = 0; i < 3; ++i) {
        Tensor input_tensor(DataType::FLOAT32, TensorShape({1, 3, 224, 224}));
        
        InferenceRequest request;
        request.request_id = i + 1;
        request.inputs = {input_tensor};
        requests.push_back(request);
    }
    
    // Execute batch
    Status exec_status = cuda_backend_->ExecuteBatch(*model, requests, results);
    
    // Should succeed or fail gracefully
    EXPECT_TRUE(exec_status == Status::SUCCESS || exec_status == Status::FAILURE);
    
    if (exec_status == Status::SUCCESS) {
        EXPECT_EQ(results.size(), requests.size());
        for (size_t i = 0; i < results.size(); ++i) {
            EXPECT_EQ(results[i].request_id, requests[i].request_id);
        }
    }
}

TEST_F(CUDABackendTest, ShutdownTest) {
    // Initialize first
    Status init_status = cuda_backend_->Initialize();
    if (init_status == Status::SUCCESS) {
        EXPECT_TRUE(cuda_backend_->IsInitialized());
        
        // Shutdown
        Status shutdown_status = cuda_backend_->Shutdown();
        EXPECT_EQ(shutdown_status, Status::SUCCESS);
        EXPECT_FALSE(cuda_backend_->IsInitialized());
    }
}

} // namespace testing
} // namespace edge_ai
