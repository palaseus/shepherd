/**
 * @file basic_inference.cpp
 * @brief Basic inference example for Edge AI Engine
 * @author AI Co-Developer
 * @date 2024
 * 
 * This example demonstrates basic usage of the Edge AI Engine for model inference.
 */

#include "core/edge_ai_engine.h"
#include "core/types.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

using namespace edge_ai;

int main(int argc, char* argv[]) {
    try {
        std::cout << "Edge AI Engine - Basic Inference Example" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Check command line arguments
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <model_path> [model_type]" << std::endl;
            std::cerr << "  model_path: Path to the model file" << std::endl;
            std::cerr << "  model_type: Model type (onnx, tflite, pt) [default: onnx]" << std::endl;
            return 1;
        }
        
        std::string model_path = argv[1];
        std::string model_type_str = (argc > 2) ? argv[2] : "onnx";
        
        // Convert model type string to enum
        ModelType model_type = ModelType::ONNX;
        if (model_type_str == "tflite") {
            model_type = ModelType::TENSORFLOW_LITE;
        } else if (model_type_str == "pt" || model_type_str == "pth") {
            model_type = ModelType::PYTORCH_MOBILE;
        }
        
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Model type: " << model_type_str << std::endl;
        
        // Create engine configuration
        EngineConfig config;
        config.device_type = DeviceType::CPU;
        config.max_memory_usage = 512 * 1024 * 1024; // 512MB
        config.enable_memory_pool = true;
        config.num_threads = 4;
        config.enable_optimization = true;
        config.enable_profiling = true;
        config.max_batch_size = 16;
        config.enable_dynamic_batching = true;
        
        std::cout << "Engine configuration:" << std::endl;
        std::cout << "  Device type: CPU" << std::endl;
        std::cout << "  Max memory: 512MB" << std::endl;
        std::cout << "  Threads: 4" << std::endl;
        std::cout << "  Optimization: enabled" << std::endl;
        std::cout << "  Profiling: enabled" << std::endl;
        std::cout << "  Max batch size: 16" << std::endl;
        std::cout << "  Dynamic batching: enabled" << std::endl;
        
        // Create and initialize engine
        std::cout << "\nInitializing Edge AI Engine..." << std::endl;
        EdgeAIEngine engine(config);
        
        Status status = engine.Initialize();
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to initialize engine: " << static_cast<int>(status) << std::endl;
            return 1;
        }
        std::cout << "Engine initialized successfully" << std::endl;
        
        // Load model
        std::cout << "\nLoading model..." << std::endl;
        status = engine.LoadModel(model_path, model_type);
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to load model: " << static_cast<int>(status) << std::endl;
            return 1;
        }
        std::cout << "Model loaded successfully" << std::endl;
        
        // Get model information
        ModelInfo model_info = engine.GetModelInfo();
        std::cout << "\nModel information:" << std::endl;
        std::cout << "  Name: " << model_info.name << std::endl;
        std::cout << "  Type: " << static_cast<int>(model_info.type) << std::endl;
        std::cout << "  Version: " << model_info.version << std::endl;
        std::cout << "  Size: " << model_info.model_size << " bytes" << std::endl;
        std::cout << "  Optimized: " << (model_info.is_optimized ? "Yes" : "No") << std::endl;
        std::cout << "  Input shapes: " << model_info.input_shapes.size() << std::endl;
        std::cout << "  Output shapes: " << model_info.output_shapes.size() << std::endl;
        
        // Optimize model
        std::cout << "\nOptimizing model..." << std::endl;
        OptimizationConfig opt_config;
        opt_config.enable_quantization = true;
        opt_config.quantization_type = DataType::INT8;
        opt_config.enable_pruning = false;
        opt_config.enable_graph_optimization = true;
        opt_config.enable_hardware_acceleration = true;
        
        status = engine.OptimizeModel(opt_config);
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to optimize model: " << static_cast<int>(status) << std::endl;
            return 1;
        }
        std::cout << "Model optimized successfully" << std::endl;
        
        // Enable monitoring
        engine.SetMonitoring(true);
        
        // Create sample input data
        std::cout << "\nCreating sample input data..." << std::endl;
        std::vector<Tensor> inputs;
        
        // Create a sample input tensor (assuming image input)
        TensorShape input_shape({1, 3, 224, 224}); // Batch=1, Channels=3, Height=224, Width=224
        Tensor input_tensor(DataType::FLOAT32, input_shape);
        
        // Fill with random data
        float* data = static_cast<float*>(input_tensor.GetData());
        size_t size = input_tensor.GetSize() / sizeof(float);
        for (size_t i = 0; i < size; ++i) {
            data[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        inputs.push_back(std::move(input_tensor));
        std::cout << "Sample input created: " << input_shape.ToString() << std::endl;
        
        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        std::vector<Tensor> outputs;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        status = engine.RunInference(inputs, outputs);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to run inference: " << static_cast<int>(status) << std::endl;
            return 1;
        }
        
        auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Inference completed successfully" << std::endl;
        std::cout << "Inference time: " << inference_time.count() << " microseconds" << std::endl;
        std::cout << "Number of outputs: " << outputs.size() << std::endl;
        
        // Display output information
        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto& output = outputs[i];
            std::cout << "Output " << i << ":" << std::endl;
            std::cout << "  Shape: " << output.GetShape().ToString() << std::endl;
            std::cout << "  Data type: " << static_cast<int>(output.GetDataType()) << std::endl;
            std::cout << "  Size: " << output.GetSize() << " bytes" << std::endl;
        }
        
        // Get engine statistics
        std::cout << "\nEngine statistics:" << std::endl;
        EngineStats stats = engine.GetStats();
        std::cout << "  Model loaded: " << (stats.model_loaded ? "Yes" : "No") << std::endl;
        std::cout << "  Model optimized: " << (stats.model_optimized ? "Yes" : "No") << std::endl;
        std::cout << "  Model size: " << stats.model_size << " bytes" << std::endl;
        std::cout << "  Total inferences: " << stats.total_inferences << std::endl;
        std::cout << "  Successful inferences: " << stats.successful_inferences << std::endl;
        std::cout << "  Failed inferences: " << stats.failed_inferences << std::endl;
        std::cout << "  Current memory usage: " << stats.current_memory_usage << " bytes" << std::endl;
        std::cout << "  Peak memory usage: " << stats.peak_memory_usage << " bytes" << std::endl;
        
        // Get performance metrics
        std::cout << "\nPerformance metrics:" << std::endl;
        PerformanceMetrics metrics = engine.GetMetrics();
        std::cout << "  Min latency: " << metrics.min_latency.count() << " microseconds" << std::endl;
        std::cout << "  Max latency: " << metrics.max_latency.count() << " microseconds" << std::endl;
        std::cout << "  Average latency: " << metrics.average_latency.count() << " microseconds" << std::endl;
        std::cout << "  P95 latency: " << metrics.p95_latency.count() << " microseconds" << std::endl;
        std::cout << "  P99 latency: " << metrics.p99_latency.count() << " microseconds" << std::endl;
        std::cout << "  Inferences per second: " << metrics.inferences_per_second << std::endl;
        std::cout << "  Memory utilization: " << (metrics.memory_utilization * 100) << "%" << std::endl;
        std::cout << "  Error rate: " << (metrics.error_rate * 100) << "%" << std::endl;
        
        // Shutdown engine
        std::cout << "\nShutting down engine..." << std::endl;
        status = engine.Shutdown();
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to shutdown engine: " << static_cast<int>(status) << std::endl;
            return 1;
        }
        std::cout << "Engine shutdown successfully" << std::endl;
        
        std::cout << "\nExample completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
}
