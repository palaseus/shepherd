/**
 * @file mock_gpu_backend.h
 * @brief Mock GPU execution backend for testing and simulation
 * @author AI Co-Developer
 * @date 2024
 * 
 * This file contains the MockGPUBackend class which simulates GPU execution
 * with higher throughput and different latency characteristics than CPU.
 */

#pragma once

#include "execution_backend.h"
#include <thread>
#include <vector>
#include <random>

namespace edge_ai {

/**
 * @class MockGPUBackend
 * @brief Mock GPU execution backend
 * 
 * The MockGPUBackend class simulates GPU execution with:
 * - Higher throughput than CPU
 * - Different latency characteristics
 * - Simulated memory usage patterns
 * - Batch processing optimization
 */
class MockGPUBackend : public ExecutionBackend {
public:
    /**
     * @brief Constructor
     * @param device GPU device to use
     * @param throughput_multiplier Throughput multiplier vs CPU (default 2.0)
     */
    explicit MockGPUBackend(std::shared_ptr<Device> device, double throughput_multiplier = 2.0);
    
    /**
     * @brief Destructor
     */
    ~MockGPUBackend() override = default;
    
    /**
     * @brief Initialize the mock GPU backend
     * @return Status indicating success or failure
     */
    Status Initialize() override;
    
    /**
     * @brief Shutdown the mock GPU backend
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
     * @brief Get mock GPU backend capabilities
     * @return Mock GPU backend capabilities
     */
    BackendCapabilities GetCapabilities() const override;
    
    /**
     * @brief Check if mock GPU backend supports a specific model type
     * @param model_type Model type to check
     * @return True if supported, false otherwise
     */
    bool SupportsModelType(ModelType model_type) const override;
    
    /**
     * @brief Check if mock GPU backend supports a specific data type
     * @param data_type Data type to check
     * @return True if supported, false otherwise
     */
    bool SupportsDataType(DataType data_type) const override;
    
    /**
     * @brief Get mock GPU backend name
     * @return Mock GPU backend name
     */
    std::string GetName() const override;
    
    /**
     * @brief Get mock GPU backend version
     * @return Mock GPU backend version
     */
    std::string GetVersion() const override;
    
    /**
     * @brief Get mock GPU backend unique identifier
     * @return Mock GPU backend unique identifier
     */
    std::string GetId() const override;
    
    /**
     * @brief Set throughput multiplier
     * @param multiplier Throughput multiplier vs CPU
     */
    void SetThroughputMultiplier(double multiplier);
    
    /**
     * @brief Get throughput multiplier
     * @return Throughput multiplier
     */
    double GetThroughputMultiplier() const;

private:
    double throughput_multiplier_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> latency_distribution_;
    
    /**
     * @brief Simulate GPU inference execution
     * @param model Model to execute
     * @param request Inference request
     * @param result Inference result
     * @return Status indicating success or failure
     */
    Status SimulateExecution(const Model& model, 
                           const InferenceRequest& request, 
                           InferenceResult& result);
    
    /**
     * @brief Simulate batch GPU inference execution
     * @param model Model to execute
     * @param requests Batch of inference requests
     * @param results Batch of inference results
     * @return Status indicating success or failure
     */
    Status SimulateBatchExecution(const Model& model,
                                const std::vector<InferenceRequest>& requests,
                                std::vector<InferenceResult>& results);
    
    /**
     * @brief Create dummy output tensors
     * @param model Model information
     * @param result Inference result
     */
    void CreateDummyOutputs(const Model& model, InferenceResult& result);
    
    /**
     * @brief Calculate simulated execution time
     * @param model Model information
     * @param batch_size Batch size
     * @return Simulated execution time
     */
    std::chrono::microseconds CalculateExecutionTime(const Model& model, size_t batch_size = 1);
};

} // namespace edge_ai
