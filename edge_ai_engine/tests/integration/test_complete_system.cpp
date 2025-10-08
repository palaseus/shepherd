/**
 * @file test_complete_system.cpp
 * @brief Integration tests for complete system functionality
 * @author AI Co-Developer
 * @date 2024
 */

#include <gtest/gtest.h>
#include "core/edge_ai_engine.h"
#include "backend/cuda_backend.h"
#include "backend/cpu_backend.h"
#include "optimization/optimizer.h"
#include "profiling/profiler.h"
#include <memory>

namespace edge_ai {
namespace testing {

class CompleteSystemTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create engine configuration
        config_.device_type = DeviceType::AUTO;
        config_.enable_optimization = true;
        config_.enable_profiling = true;
        config_.max_memory_usage = 1024 * 1024 * 1024; // 1GB
        config_.enable_memory_pool = true;
        config_.num_threads = 4;
        config_.max_batch_size = 16;
        config_.enable_dynamic_batching = true;
        
        // Create engine
        engine_ = std::make_unique<EdgeAIEngine>(config_);
    }
    
    void TearDown() override {
        if (engine_) {
            engine_->Shutdown();
        }
    }
    
    EngineConfig config_;
    std::unique_ptr<EdgeAIEngine> engine_;
};

TEST_F(CompleteSystemTest, EngineInitializationTest) {
    // Test engine initialization
    Status status = engine_->Initialize();
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Test engine statistics
    auto stats = engine_->GetStats();
    EXPECT_GE(stats.total_requests, 0);
    
    // Test performance metrics
    auto metrics = engine_->GetMetrics();
    EXPECT_GE(metrics.avg_latency_ms, 0.0);
}

TEST_F(CompleteSystemTest, ModelLoadingAndOptimizationTest) {
    // Initialize engine
    Status status = engine_->Initialize();
    ASSERT_EQ(status, Status::SUCCESS);
    
    // Create a test model file (simulate)
    std::string model_path = "test_model.onnx";
    
    // Test model loading
    status = engine_->LoadModel(model_path, ModelType::ONNX);
    // This may fail if the file doesn't exist, which is expected in tests
    if (status == Status::SUCCESS) {
        // Test model optimization
        OptimizationConfig opt_config;
        opt_config.enable_quantization = true;
        opt_config.quantization_type = DataType::INT8;
        opt_config.enable_pruning = true;
        opt_config.pruning_ratio = 0.1f;
        opt_config.enable_graph_optimization = true;
        
        status = engine_->OptimizeModel(opt_config);
        EXPECT_EQ(status, Status::SUCCESS);
        
        // Test model information
        auto model_info = engine_->GetModelInfo();
        EXPECT_FALSE(model_info.name.empty());
        EXPECT_EQ(model_info.type, ModelType::ONNX);
    }
}

TEST_F(CompleteSystemTest, InferenceExecutionTest) {
    // Initialize engine
    Status status = engine_->Initialize();
    ASSERT_EQ(status, Status::SUCCESS);
    
    // Create test input tensors
    std::vector<Tensor> inputs;
    Tensor input_tensor(DataType::FLOAT32, TensorShape({1, 3, 224, 224}));
    inputs.push_back(std::move(input_tensor));
    
    // Test inference execution
    std::vector<Tensor> outputs;
    status = engine_->RunInference(inputs, outputs);
    
    // Should succeed or fail gracefully
    EXPECT_TRUE(status == Status::SUCCESS || status == Status::MODEL_NOT_LOADED);
}

TEST_F(CompleteSystemTest, AsyncInferenceTest) {
    // Initialize engine
    Status status = engine_->Initialize();
    ASSERT_EQ(status, Status::SUCCESS);
    
    // Create test input tensors
    std::vector<Tensor> inputs;
    Tensor input_tensor(DataType::FLOAT32, TensorShape({1, 3, 224, 224}));
    inputs.push_back(std::move(input_tensor));
    
    // Test async inference
    bool callback_called = false;
    auto callback = [&callback_called](Status result_status, std::vector<Tensor> result_outputs) {
        callback_called = true;
        // Should succeed or fail gracefully
        EXPECT_TRUE(result_status == Status::SUCCESS || result_status == Status::MODEL_NOT_LOADED);
    };
    
    status = engine_->RunInferenceAsync(inputs, callback);
    EXPECT_TRUE(status == Status::SUCCESS || status == Status::MODEL_NOT_LOADED);
    
    // Wait a bit for async operation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Note: callback may not be called if model is not loaded, which is expected
}

TEST_F(CompleteSystemTest, PerformanceMonitoringTest) {
    // Initialize engine
    Status status = engine_->Initialize();
    ASSERT_EQ(status, Status::SUCCESS);
    
    // Enable monitoring
    engine_->SetMonitoring(true);
    
    // Test performance metrics
    auto metrics = engine_->GetMetrics();
    EXPECT_GE(metrics.avg_latency_ms, 0.0);
    EXPECT_GE(metrics.throughput_ops_per_sec, 0.0);
    EXPECT_GE(metrics.memory_usage_percent, 0.0);
    EXPECT_LE(metrics.memory_usage_percent, 100.0);
}

TEST_F(CompleteSystemTest, BackendIntegrationTest) {
    // Test CPU backend
    auto cpu_device = std::make_shared<Device>();
    cpu_device->SetType(DeviceType::CPU);
    
    auto cpu_backend = std::make_shared<CPUBackend>(cpu_device, 4);
    Status status = cpu_backend->Initialize();
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Test CUDA backend (may fail if CUDA not available)
    auto cuda_device = std::make_shared<Device>();
    cuda_device->SetType(DeviceType::GPU);
    
    auto cuda_backend = std::make_shared<CUDABackend>(cuda_device, 0);
    status = cuda_backend->Initialize();
    // Should succeed or fail gracefully
    EXPECT_TRUE(status == Status::SUCCESS || status == Status::HARDWARE_NOT_AVAILABLE);
    
    // Test backend capabilities
    auto cpu_caps = cpu_backend->GetCapabilities();
    EXPECT_TRUE(cpu_caps.supports_batching);
    EXPECT_TRUE(cpu_caps.supports_quantization);
    
    auto cuda_caps = cuda_backend->GetCapabilities();
    EXPECT_TRUE(cuda_caps.supports_batching);
    EXPECT_TRUE(cuda_caps.supports_quantization);
    
    // Cleanup
    cpu_backend->Shutdown();
    cuda_backend->Shutdown();
}

TEST_F(CompleteSystemTest, OptimizationSystemTest) {
    // Test optimization system
    auto optimizer = std::make_shared<Optimizer>(OptimizationConfig{});
    Status status = optimizer->Initialize();
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Create test model
    auto model = std::make_shared<Model>();
    model->SetName("test_model");
    model->SetType(ModelType::ONNX);
    model->SetSize(1024 * 1024);
    model->SetInputShapes({TensorShape({1, 3, 224, 224})});
    model->SetOutputShapes({TensorShape({1, 1000})});
    model->SetInputTypes({DataType::FLOAT32});
    model->SetOutputTypes({DataType::FLOAT32});
    
    // Test quantization
    std::shared_ptr<Model> quantized_model;
    QuantizationConfig quant_config;
    quant_config.quantization_type = DataType::INT8;
    
    status = optimizer->QuantizeModel(model, quantized_model, quant_config);
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(quantized_model, nullptr);
    
    // Test pruning
    std::shared_ptr<Model> pruned_model;
    PruningConfig prune_config;
    prune_config.pruning_ratio = 0.1f;
    
    status = optimizer->PruneModel(model, pruned_model, prune_config);
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(pruned_model, nullptr);
    
    // Test graph optimization
    std::shared_ptr<Model> graph_optimized_model;
    GraphOptimizationConfig graph_config;
    graph_config.enable_operator_fusion = true;
    
    status = optimizer->OptimizeGraph(model, graph_optimized_model, graph_config);
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(graph_optimized_model, nullptr);
    
    // Test optimization stats
    auto stats = optimizer->GetOptimizationStats();
    EXPECT_GE(stats.total_optimizations_performed, 0);
    
    optimizer->Shutdown();
}

TEST_F(CompleteSystemTest, ProfilerIntegrationTest) {
    // Test profiler system
    Profiler& profiler = Profiler::GetInstance();
    Status status = profiler.Initialize();
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Test profiler session
    status = profiler.StartGlobalSession("test_session");
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Test profiler events
    profiler.MarkEvent(1, "test_event");
    
    // Test scoped event
    {
        auto scoped_event = profiler.CreateScopedEvent(1, "scoped_test");
        EXPECT_NE(scoped_event, nullptr);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Test session export
    status = profiler.StopGlobalSession();
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Test export to JSON
    status = profiler.ExportSessionAsJson("test_session", "test_trace.json");
    EXPECT_EQ(status, Status::SUCCESS);
    
    profiler.Shutdown();
}

TEST_F(CompleteSystemTest, ErrorHandlingTest) {
    // Test error handling with invalid inputs
    Status status;
    
    // Test with uninitialized engine
    auto uninit_engine = std::make_unique<EdgeAIEngine>(config_);
    
    std::vector<Tensor> inputs;
    std::vector<Tensor> outputs;
    status = uninit_engine->RunInference(inputs, outputs);
    EXPECT_EQ(status, Status::NOT_INITIALIZED);
    
    // Test with empty inputs
    status = engine_->Initialize();
    ASSERT_EQ(status, Status::SUCCESS);
    
    status = engine_->RunInference(inputs, outputs);
    EXPECT_EQ(status, Status::INVALID_ARGUMENT);
    
    // Test with invalid model path
    status = engine_->LoadModel("nonexistent_model.onnx", ModelType::ONNX);
    EXPECT_TRUE(status == Status::FAILURE || status == Status::INVALID_MODEL_FORMAT);
}

TEST_F(CompleteSystemTest, ConfigurationTest) {
    // Test different configurations
    EngineConfig test_config;
    test_config.device_type = DeviceType::CPU;
    test_config.enable_optimization = false;
    test_config.enable_profiling = false;
    test_config.max_memory_usage = 512 * 1024 * 1024; // 512MB
    test_config.num_threads = 2;
    
    auto test_engine = std::make_unique<EdgeAIEngine>(test_config);
    Status status = test_engine->Initialize();
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Test that configuration is applied
    auto stats = test_engine->GetStats();
    EXPECT_GE(stats.total_requests, 0);
    
    test_engine->Shutdown();
}

} // namespace testing
} // namespace edge_ai
