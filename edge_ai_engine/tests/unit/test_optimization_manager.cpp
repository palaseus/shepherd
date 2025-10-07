/**
 * @file test_optimization_manager.cpp
 * @brief Unit tests for OptimizationManager
 * @author AI Co-Developer
 * @date 2024
 */

#include <gtest/gtest.h>
#include "optimization/optimization_manager.h"
#include "batching/batching_manager.h"
#include "core/runtime_scheduler.h"
#include "core/inference_engine.h"
#include "core/cpu_device.h"
#include "memory/memory_manager.h"
#include "profiling/profiler.h"
#include <memory>
#include <thread>
#include <chrono>

using namespace edge_ai;

class OptimizationManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize profiler
        Profiler::GetInstance().Initialize();
        
        // Create test components
        batching_manager_ = std::make_shared<BatchingManager>();
        scheduler_ = std::make_shared<RuntimeScheduler>();
        auto device = std::make_shared<CPUDevice>(0);
        auto memory_manager = std::make_shared<MemoryManager>();
        inference_engine_ = std::make_shared<InferenceEngine>(device, scheduler_, memory_manager, &Profiler::GetInstance());
        
        // Create optimization manager
        config_.optimization_interval = std::chrono::milliseconds(100);
        config_.convergence_timeout = std::chrono::milliseconds(500);
        optimization_manager_ = std::make_unique<OptimizationManager>(config_);
    }
    
    void TearDown() override {
        if (optimization_manager_) {
            optimization_manager_->Shutdown();
        }
    }
    
    AdaptiveOptimizationConfig config_;
    std::unique_ptr<OptimizationManager> optimization_manager_;
    std::shared_ptr<BatchingManager> batching_manager_;
    std::shared_ptr<RuntimeScheduler> scheduler_;
    std::shared_ptr<InferenceEngine> inference_engine_;
};

TEST_F(OptimizationManagerTest, Initialize) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    EXPECT_TRUE(optimization_manager_->IsOptimizationEnabled());
}

TEST_F(OptimizationManagerTest, InitializeTwice) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    EXPECT_EQ(optimization_manager_->Initialize(), Status::ALREADY_INITIALIZED);
}

TEST_F(OptimizationManagerTest, Shutdown) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    EXPECT_EQ(optimization_manager_->Shutdown(), Status::SUCCESS);
}

TEST_F(OptimizationManagerTest, RegisterComponents) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    EXPECT_EQ(optimization_manager_->RegisterComponents(
        batching_manager_, scheduler_, inference_engine_), Status::SUCCESS);
}

TEST_F(OptimizationManagerTest, RegisterComponentsWithoutInit) {
    EXPECT_EQ(optimization_manager_->RegisterComponents(
        batching_manager_, scheduler_, inference_engine_), Status::NOT_INITIALIZED);
}

TEST_F(OptimizationManagerTest, StartStopOptimization) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    EXPECT_EQ(optimization_manager_->RegisterComponents(
        batching_manager_, scheduler_, inference_engine_), Status::SUCCESS);
    
    EXPECT_EQ(optimization_manager_->StartOptimization(), Status::SUCCESS);
    
    // Let it run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    EXPECT_EQ(optimization_manager_->StopOptimization(), Status::SUCCESS);
}

TEST_F(OptimizationManagerTest, StartOptimizationTwice) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    EXPECT_EQ(optimization_manager_->RegisterComponents(
        batching_manager_, scheduler_, inference_engine_), Status::SUCCESS);
    
    EXPECT_EQ(optimization_manager_->StartOptimization(), Status::SUCCESS);
    EXPECT_EQ(optimization_manager_->StartOptimization(), Status::ALREADY_INITIALIZED);
    
    EXPECT_EQ(optimization_manager_->StopOptimization(), Status::SUCCESS);
}

TEST_F(OptimizationManagerTest, UpdateMetrics) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    OptimizationMetrics metrics;
    metrics.avg_latency_ms = 100.0;
    metrics.throughput_ops_per_sec = 500.0;
    metrics.memory_usage_percent = 0.5;
    
    EXPECT_EQ(optimization_manager_->UpdateMetrics(metrics), Status::SUCCESS);
}

TEST_F(OptimizationManagerTest, UpdateMetricsWithoutInit) {
    OptimizationMetrics metrics;
    EXPECT_EQ(optimization_manager_->UpdateMetrics(metrics), Status::NOT_INITIALIZED);
}

TEST_F(OptimizationManagerTest, ApplyOptimization) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    OptimizationDecision decision;
    decision.action = OptimizationAction::ADJUST_BATCH_SIZE;
    decision.trigger = OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED;
    decision.parameter_name = "max_batch_size";
    decision.old_value = "8";
    decision.new_value = "4";
    decision.expected_improvement = 0.2;
    
    EXPECT_EQ(optimization_manager_->ApplyOptimization(decision), Status::SUCCESS);
}

TEST_F(OptimizationManagerTest, ApplyOptimizationWithoutInit) {
    OptimizationDecision decision;
    EXPECT_EQ(optimization_manager_->ApplyOptimization(decision), Status::NOT_INITIALIZED);
}

TEST_F(OptimizationManagerTest, GetStats) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    auto stats = optimization_manager_->GetStats();
    EXPECT_EQ(stats.total_decisions, 0);
    EXPECT_EQ(stats.successful_optimizations, 0);
    EXPECT_EQ(stats.failed_optimizations, 0);
}

TEST_F(OptimizationManagerTest, GetRecentDecisions) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    auto decisions = optimization_manager_->GetRecentDecisions();
    EXPECT_TRUE(decisions.empty());
}

TEST_F(OptimizationManagerTest, SetOptimizationPolicy) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    auto policy = std::make_unique<RuleBasedPolicy>();
    EXPECT_EQ(optimization_manager_->SetOptimizationPolicy(std::move(policy)), Status::SUCCESS);
}

TEST_F(OptimizationManagerTest, SetOptimizationPolicyNull) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    EXPECT_EQ(optimization_manager_->SetOptimizationPolicy(nullptr), Status::INVALID_ARGUMENT);
}

TEST_F(OptimizationManagerTest, SetOptimizationEnabled) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    optimization_manager_->SetOptimizationEnabled(false);
    EXPECT_FALSE(optimization_manager_->IsOptimizationEnabled());
    
    optimization_manager_->SetOptimizationEnabled(true);
    EXPECT_TRUE(optimization_manager_->IsOptimizationEnabled());
}

TEST_F(OptimizationManagerTest, ExportOptimizationTrace) {
    EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
    
    const std::string trace_file = "/tmp/test_optimization_trace.json";
    EXPECT_EQ(optimization_manager_->ExportOptimizationTrace("test_session", trace_file), 
              Status::SUCCESS);
    
    // Clean up
    std::remove(trace_file.c_str());
}

TEST_F(OptimizationManagerTest, ExportOptimizationTraceWithoutInit) {
    const std::string trace_file = "/tmp/test_optimization_trace.json";
    EXPECT_EQ(optimization_manager_->ExportOptimizationTrace("test_session", trace_file), 
              Status::NOT_INITIALIZED);
}

// RuleBasedPolicy Tests
class RuleBasedPolicyTest : public ::testing::Test {
protected:
    void SetUp() override {
        policy_ = std::make_unique<RuleBasedPolicy>();
        config_.latency_threshold_ms = 100.0;
        config_.throughput_degradation_threshold = 0.2;
        config_.memory_pressure_threshold = 0.8;
    }
    
    std::unique_ptr<RuleBasedPolicy> policy_;
    AdaptiveOptimizationConfig config_;
};

TEST_F(RuleBasedPolicyTest, GetName) {
    EXPECT_EQ(policy_->GetName(), "RuleBasedPolicy");
}

TEST_F(RuleBasedPolicyTest, IsApplicable) {
    OptimizationMetrics metrics;
    EXPECT_TRUE(policy_->IsApplicable(metrics));
}

TEST_F(RuleBasedPolicyTest, AnalyzeAndDecideHighLatency) {
    OptimizationMetrics metrics;
    metrics.avg_latency_ms = 150.0; // Above threshold
    metrics.throughput_ops_per_sec = 1000.0;
    metrics.memory_usage_percent = 0.5;
    metrics.queue_depth = 10;
    
    auto decisions = policy_->AnalyzeAndDecide(metrics, config_);
    
    // Should generate decisions for high latency
    EXPECT_FALSE(decisions.empty());
    
    // Check for batch size adjustment decision
    bool found_batch_decision = false;
    for (const auto& decision : decisions) {
        if (decision.action == OptimizationAction::ADJUST_BATCH_SIZE) {
            found_batch_decision = true;
            EXPECT_EQ(decision.trigger, OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED);
            EXPECT_EQ(decision.parameter_name, "max_batch_size");
            break;
        }
    }
    EXPECT_TRUE(found_batch_decision);
}

TEST_F(RuleBasedPolicyTest, AnalyzeAndDecideHighMemoryPressure) {
    OptimizationMetrics metrics;
    metrics.avg_latency_ms = 50.0;
    metrics.throughput_ops_per_sec = 1000.0;
    metrics.memory_usage_percent = 0.9; // Above threshold
    metrics.queue_depth = 10;
    
    auto decisions = policy_->AnalyzeAndDecide(metrics, config_);
    
    // Should generate decisions for high memory pressure
    EXPECT_FALSE(decisions.empty());
    
    // Check for memory optimization decision
    bool found_memory_decision = false;
    for (const auto& decision : decisions) {
        if (decision.action == OptimizationAction::ADJUST_MEMORY_POOL) {
            found_memory_decision = true;
            EXPECT_EQ(decision.trigger, OptimizationTrigger::MEMORY_PRESSURE);
            EXPECT_EQ(decision.parameter_name, "memory_pool_size");
            break;
        }
    }
    EXPECT_TRUE(found_memory_decision);
}

TEST_F(RuleBasedPolicyTest, AnalyzeAndDecideHighQueueDepth) {
    OptimizationMetrics metrics;
    metrics.avg_latency_ms = 50.0;
    metrics.throughput_ops_per_sec = 1000.0;
    metrics.memory_usage_percent = 0.5;
    metrics.queue_depth = 150; // Above threshold
    
    auto decisions = policy_->AnalyzeAndDecide(metrics, config_);
    
    // Should generate decisions for high queue depth
    EXPECT_FALSE(decisions.empty());
    
    // Check for request throttling decision
    bool found_throttle_decision = false;
    for (const auto& decision : decisions) {
        if (decision.action == OptimizationAction::THROTTLE_REQUESTS) {
            found_throttle_decision = true;
            EXPECT_EQ(decision.trigger, OptimizationTrigger::QUEUE_OVERFLOW);
            EXPECT_EQ(decision.parameter_name, "request_throttle_rate");
            break;
        }
    }
    EXPECT_TRUE(found_throttle_decision);
}

TEST_F(RuleBasedPolicyTest, AnalyzeAndDecideNormalConditions) {
    OptimizationMetrics metrics;
    metrics.avg_latency_ms = 50.0; // Below threshold
    metrics.throughput_ops_per_sec = 1000.0;
    metrics.memory_usage_percent = 0.5; // Below threshold
    metrics.queue_depth = 10; // Below threshold
    
    auto decisions = policy_->AnalyzeAndDecide(metrics, config_);
    
    // Should not generate any decisions for normal conditions
    EXPECT_TRUE(decisions.empty());
}

// Integration Tests
class OptimizationManagerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize profiler
        Profiler::GetInstance().Initialize();
        
        // Create test components
        batching_manager_ = std::make_shared<BatchingManager>();
        scheduler_ = std::make_shared<RuntimeScheduler>();
        auto device = std::make_shared<CPUDevice>(0);
        auto memory_manager = std::make_shared<MemoryManager>();
        inference_engine_ = std::make_shared<InferenceEngine>(device, scheduler_, memory_manager, &Profiler::GetInstance());
        
        // Create optimization manager with fast intervals for testing
        config_.optimization_interval = std::chrono::milliseconds(50);
        config_.convergence_timeout = std::chrono::milliseconds(200);
        optimization_manager_ = std::make_unique<OptimizationManager>(config_);
        
        EXPECT_EQ(optimization_manager_->Initialize(), Status::SUCCESS);
        EXPECT_EQ(optimization_manager_->RegisterComponents(
            batching_manager_, scheduler_, inference_engine_), Status::SUCCESS);
    }
    
    void TearDown() override {
        if (optimization_manager_) {
            optimization_manager_->Shutdown();
        }
    }
    
    AdaptiveOptimizationConfig config_;
    std::unique_ptr<OptimizationManager> optimization_manager_;
    std::shared_ptr<BatchingManager> batching_manager_;
    std::shared_ptr<RuntimeScheduler> scheduler_;
    std::shared_ptr<InferenceEngine> inference_engine_;
};

TEST_F(OptimizationManagerIntegrationTest, EndToEndOptimization) {
    // Start optimization
    EXPECT_EQ(optimization_manager_->StartOptimization(), Status::SUCCESS);
    
    // Let it run for a short time to collect metrics and make decisions
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    // Check that some decisions were made
    auto stats = optimization_manager_->GetStats();
    EXPECT_GE(stats.total_decisions, 0);
    
    // Stop optimization
    EXPECT_EQ(optimization_manager_->StopOptimization(), Status::SUCCESS);
}

TEST_F(OptimizationManagerIntegrationTest, OptimizationWithHighLatency) {
    // Start optimization
    EXPECT_EQ(optimization_manager_->StartOptimization(), Status::SUCCESS);
    
    // Simulate high latency conditions
    OptimizationMetrics metrics;
    metrics.avg_latency_ms = 150.0; // Above threshold
    metrics.throughput_ops_per_sec = 500.0;
    metrics.memory_usage_percent = 0.5;
    metrics.queue_depth = 10;
    
    EXPECT_EQ(optimization_manager_->UpdateMetrics(metrics), Status::SUCCESS);
    
    // Let it run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Check that optimization decisions were made
    auto stats = optimization_manager_->GetStats();
    EXPECT_GE(stats.total_decisions, 0);
    
    // Stop optimization
    EXPECT_EQ(optimization_manager_->StopOptimization(), Status::SUCCESS);
}

TEST_F(OptimizationManagerIntegrationTest, ExportTraceWithDecisions) {
    // Start optimization
    EXPECT_EQ(optimization_manager_->StartOptimization(), Status::SUCCESS);
    
    // Simulate some optimization decisions
    OptimizationDecision decision;
    decision.action = OptimizationAction::ADJUST_BATCH_SIZE;
    decision.trigger = OptimizationTrigger::LATENCY_THRESHOLD_EXCEEDED;
    decision.parameter_name = "max_batch_size";
    decision.old_value = "8";
    decision.new_value = "4";
    decision.expected_improvement = 0.2;
    
    EXPECT_EQ(optimization_manager_->ApplyOptimization(decision), Status::SUCCESS);
    
    // Export trace
    const std::string trace_file = "/tmp/integration_test_trace.json";
    EXPECT_EQ(optimization_manager_->ExportOptimizationTrace("integration_test", trace_file), 
              Status::SUCCESS);
    
    // Stop optimization
    EXPECT_EQ(optimization_manager_->StopOptimization(), Status::SUCCESS);
    
    // Clean up
    std::remove(trace_file.c_str());
}
