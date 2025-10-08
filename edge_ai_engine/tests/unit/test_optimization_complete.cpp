/**
 * @file test_optimization_complete.cpp
 * @brief Unit tests for complete optimization system implementation
 * @author AI Co-Developer
 * @date 2024
 */

#include <gtest/gtest.h>
#include "optimization/optimizer.h"
#include "core/model.h"
#include "core/types.h"
#include <memory>

namespace edge_ai {
namespace testing {

class OptimizationCompleteTest : public ::testing::Test {
protected:
    void SetUp() override {
        optimizer_ = std::make_shared<Optimizer>(OptimizationConfig{});
        
        // Create a test model
        test_model_ = std::make_shared<Model>();
        test_model_->SetName("test_model");
        test_model_->SetType(ModelType::ONNX);
        test_model_->SetVersion("1.0");
        test_model_->SetSize(1024 * 1024); // 1MB
        test_model_->SetInputShapes({TensorShape({1, 3, 224, 224})});
        test_model_->SetOutputShapes({TensorShape({1, 1000})});
        test_model_->SetInputTypes({DataType::FLOAT32});
        test_model_->SetOutputTypes({DataType::FLOAT32});
        
        // Initialize optimizer
        Status status = optimizer_->Initialize();
        ASSERT_EQ(status, Status::SUCCESS);
    }
    
    void TearDown() override {
        if (optimizer_) {
            optimizer_->Shutdown();
        }
    }
    
    std::shared_ptr<Optimizer> optimizer_;
    std::shared_ptr<Model> test_model_;
};

TEST_F(OptimizationCompleteTest, QuantizationINT8Test) {
    // Test INT8 quantization
    QuantizationConfig config;
    config.quantization_bits = 8;
    config.enable_dynamic_quantization = false;
    
    std::shared_ptr<Model> quantized_model;
    Status status = optimizer_->QuantizeModel(test_model_, quantized_model, config);
    
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(quantized_model, nullptr);
    EXPECT_NE(quantized_model.get(), test_model_.get()); // Should be a copy
    
    // Check quantization effects
    EXPECT_TRUE(quantized_model->IsValid());
    EXPECT_LT(quantized_model->GetSize(), test_model_->GetSize()); // Should be smaller
    EXPECT_EQ(quantized_model->GetSize(), test_model_->GetSize() / 4); // INT8 is 4x smaller
    
    // Check data types
    auto input_types = quantized_model->GetInputTypes();
    auto output_types = quantized_model->GetOutputTypes();
    
    EXPECT_EQ(input_types[0], DataType::INT8);
    EXPECT_EQ(output_types[0], DataType::INT8);
}

TEST_F(OptimizationCompleteTest, QuantizationFP16Test) {
    // Test FP16 quantization
    QuantizationConfig config;
    config.quantization_bits = 16;
    config.enable_dynamic_quantization = false;
    
    std::shared_ptr<Model> quantized_model;
    Status status = optimizer_->QuantizeModel(test_model_, quantized_model, config);
    
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(quantized_model, nullptr);
    
    // Check quantization effects
    EXPECT_LT(quantized_model->GetSize(), test_model_->GetSize()); // Should be smaller
    EXPECT_EQ(quantized_model->GetSize(), test_model_->GetSize() / 2); // FP16 is 2x smaller
    
    // Check data types
    auto input_types = quantized_model->GetInputTypes();
    auto output_types = quantized_model->GetOutputTypes();
    
    EXPECT_EQ(input_types[0], DataType::FLOAT16);
    EXPECT_EQ(output_types[0], DataType::FLOAT16);
}

TEST_F(OptimizationCompleteTest, PruningTest) {
    // Test pruning
    PruningConfig config;
    config.pruning_ratio = 0.1f; // 10% pruning
    config.enable_structured_pruning = true;
    
    std::shared_ptr<Model> pruned_model;
    Status status = optimizer_->PruneModel(test_model_, pruned_model, config);
    
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(pruned_model, nullptr);
    EXPECT_NE(pruned_model.get(), test_model_.get()); // Should be a copy
    
    // Check pruning effects
    EXPECT_TRUE(pruned_model->IsValid());
    EXPECT_LT(pruned_model->GetSize(), test_model_->GetSize()); // Should be smaller
    
    // Check that size reduction matches pruning ratio
    size_t expected_size = static_cast<size_t>(test_model_->GetSize() * (1.0f - config.pruning_ratio));
    EXPECT_EQ(pruned_model->GetSize(), expected_size);
    
    // Check model name includes pruning info
    std::string name = pruned_model->GetName();
    EXPECT_TRUE(name.find("pruned_10pct") != std::string::npos);
}

TEST_F(OptimizationCompleteTest, GraphOptimizationTest) {
    // Test graph optimization
    GraphOptimizationConfig config;
    config.enable_operator_fusion = true;
    config.enable_constant_folding = true;
    
    std::shared_ptr<Model> optimized_model;
    Status status = optimizer_->OptimizeGraph(test_model_, optimized_model, config);
    
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(optimized_model, nullptr);
    EXPECT_NE(optimized_model.get(), test_model_.get()); // Should be a copy
    
    // Check optimization effects
    EXPECT_TRUE(optimized_model->IsValid());
    EXPECT_LT(optimized_model->GetSize(), test_model_->GetSize()); // Should be smaller
    
    // Check model name includes optimization info
    std::string name = optimized_model->GetName();
    EXPECT_TRUE(name.find("graph_optimized") != std::string::npos);
}

TEST_F(OptimizationCompleteTest, FullOptimizationPipelineTest) {
    // Test complete optimization pipeline
    OptimizationConfig config;
    config.enable_graph_optimization = true;
    config.enable_operator_fusion = true;
    config.enable_constant_folding = true;
    config.enable_pruning = true;
    config.pruning_ratio = 0.1f;
    config.enable_quantization = true;
    config.quantization_type = DataType::INT8;
    
    std::shared_ptr<Model> optimized_model;
    Status status = optimizer_->OptimizeModel(test_model_, optimized_model, config);
    
    EXPECT_EQ(status, Status::SUCCESS);
    EXPECT_NE(optimized_model, nullptr);
    EXPECT_NE(optimized_model.get(), test_model_.get()); // Should be a copy
    
    // Check that model is marked as optimized
    EXPECT_TRUE(optimized_model->IsOptimized());
    
    // Check that size is significantly reduced
    EXPECT_LT(optimized_model->GetSize(), test_model_->GetSize());
    
    // The final size should be much smaller due to quantization (4x) and pruning (0.9x)
    size_t expected_size = static_cast<size_t>(test_model_->GetSize() * 0.9f * 0.95f * 0.98f / 4);
    EXPECT_NEAR(optimized_model->GetSize(), expected_size, expected_size * 0.1f); // 10% tolerance
}

TEST_F(OptimizationCompleteTest, OptimizationStatsTest) {
    // Test optimization statistics
    auto initial_stats = optimizer_->GetOptimizationStats();
    
    // Perform some optimizations
    OptimizationConfig config;
    config.enable_quantization = true;
    config.quantization_type = DataType::INT8;
    
    std::shared_ptr<Model> optimized_model;
    Status status = optimizer_->OptimizeModel(test_model_, optimized_model, config);
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Check that stats were updated
    auto final_stats = optimizer_->GetOptimizationStats();
    EXPECT_GT(final_stats.total_optimization_time, initial_stats.total_optimization_time);
    EXPECT_GT(final_stats.total_optimization_time, initial_stats.total_optimization_time);
}

TEST_F(OptimizationCompleteTest, InvalidInputTest) {
    // Test with invalid inputs
    std::shared_ptr<Model> output_model;
    
    // Test with null input model
    Status status = optimizer_->QuantizeModel(nullptr, output_model, QuantizationConfig{});
    EXPECT_EQ(status, Status::INVALID_ARGUMENT);
    
    // Test with uninitialized optimizer
    auto uninit_optimizer = std::make_shared<Optimizer>(OptimizationConfig{});
    status = uninit_optimizer->QuantizeModel(test_model_, output_model, QuantizationConfig{});
    EXPECT_EQ(status, Status::NOT_INITIALIZED);
}

TEST_F(OptimizationCompleteTest, ModelCopyTest) {
    // Test that optimization creates proper copies
    QuantizationConfig config;
    config.quantization_bits = 8;
    
    std::shared_ptr<Model> quantized_model;
    Status status = optimizer_->QuantizeModel(test_model_, quantized_model, config);
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Original model should be unchanged
    EXPECT_EQ(test_model_->GetSize(), 1024 * 1024);
    EXPECT_EQ(test_model_->GetInputTypes()[0], DataType::FLOAT32);
    EXPECT_EQ(test_model_->GetOutputTypes()[0], DataType::FLOAT32);
    
    // Quantized model should be different
    EXPECT_LT(quantized_model->GetSize(), test_model_->GetSize());
    EXPECT_EQ(quantized_model->GetInputTypes()[0], DataType::INT8);
    EXPECT_EQ(quantized_model->GetOutputTypes()[0], DataType::INT8);
}

TEST_F(OptimizationCompleteTest, MultipleOptimizationsTest) {
    // Test multiple optimization passes
    std::shared_ptr<Model> current_model = test_model_;
    
    // First: Graph optimization
    GraphOptimizationConfig graph_config;
    graph_config.enable_operator_fusion = true;
    
    std::shared_ptr<Model> graph_optimized;
    Status status = optimizer_->OptimizeGraph(current_model, graph_optimized, graph_config);
    EXPECT_EQ(status, Status::SUCCESS);
    current_model = graph_optimized;
    
    // Second: Pruning
    PruningConfig pruning_config;
    pruning_config.pruning_ratio = 0.2f;
    
    std::shared_ptr<Model> pruned;
    status = optimizer_->PruneModel(current_model, pruned, pruning_config);
    EXPECT_EQ(status, Status::SUCCESS);
    current_model = pruned;
    
    // Third: Quantization
    QuantizationConfig quant_config;
    quant_config.quantization_bits = 8;
    
    std::shared_ptr<Model> quantized;
    status = optimizer_->QuantizeModel(current_model, quantized, quant_config);
    EXPECT_EQ(status, Status::SUCCESS);
    
    // Final model should be much smaller than original
    EXPECT_LT(quantized->GetSize(), test_model_->GetSize());
    EXPECT_TRUE(quantized->IsOptimized());
}

} // namespace testing
} // namespace edge_ai
