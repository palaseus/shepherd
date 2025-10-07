/**
 * @file test_reactive_scheduler.cpp
 * @brief Unit tests for reactive cluster scheduler
 */

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>
#include "distributed/reactive_scheduler.h"
#include "distributed/cluster_manager.h"
#include "optimization/optimization_manager.h"
#include "graph/graph.h"

namespace edge_ai {

class ReactiveSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create cluster config
        ClusterConfig cluster_config;
        cluster_config.max_nodes = 10;
        cluster_config.enable_fault_tolerance = true;
        cluster_config.heartbeat_interval_ms = 1000;
        
        // Create dependencies
        cluster_manager_ = std::make_shared<ClusterManager>(cluster_config);
        optimization_manager_ = std::make_shared<OptimizationManager>();
        
        // Initialize cluster manager
        auto status = cluster_manager_->Initialize();
        ASSERT_EQ(status, Status::SUCCESS);
        
        // Initialize optimization manager
        status = optimization_manager_->Initialize();
        ASSERT_EQ(status, Status::SUCCESS);
        
        // Create reactive scheduler
        reactive_scheduler_ = std::make_unique<ReactiveScheduler>(optimization_manager_, cluster_manager_);
        
        // Initialize reactive scheduler
        status = reactive_scheduler_->Initialize();
        ASSERT_EQ(status, Status::SUCCESS);
        
        // Register test nodes
        RegisterTestNodes();
    }
    
    void TearDown() override {
        if (reactive_scheduler_) {
            reactive_scheduler_->Shutdown();
        }
        if (cluster_manager_) {
            cluster_manager_->Shutdown();
        }
        if (optimization_manager_) {
            optimization_manager_->Shutdown();
        }
    }
    
    void RegisterTestNodes() {
        // Register CPU node
        NodeCapabilities cpu_capabilities;
        cpu_capabilities.cpu_cores.store(8);
        cpu_capabilities.memory_mb.store(16384);
        cpu_capabilities.has_gpu.store(false);
        cpu_capabilities.compute_efficiency.store(1.0);
        cpu_capabilities.memory_efficiency.store(1.0);
        
        auto status = cluster_manager_->RegisterNode("cpu_node_1", "192.168.1.10", 8080, cpu_capabilities);
        ASSERT_EQ(status, Status::SUCCESS);
        
        // Register GPU node
        NodeCapabilities gpu_capabilities;
        gpu_capabilities.cpu_cores.store(4);
        gpu_capabilities.memory_mb.store(32768);
        gpu_capabilities.has_gpu.store(true);
        gpu_capabilities.gpu_memory_mb.store(8192);
        gpu_capabilities.compute_efficiency.store(2.0);
        gpu_capabilities.memory_efficiency.store(1.5);
        
        status = cluster_manager_->RegisterNode("gpu_node_1", "192.168.1.11", 8080, gpu_capabilities);
        ASSERT_EQ(status, Status::SUCCESS);
        
        // Register NPU node
        NodeCapabilities npu_capabilities;
        npu_capabilities.cpu_cores.store(2);
        npu_capabilities.memory_mb.store(8192);
        npu_capabilities.has_npu.store(true);
        npu_capabilities.compute_efficiency.store(3.0);
        npu_capabilities.memory_efficiency.store(2.0);
        
        status = cluster_manager_->RegisterNode("npu_node_1", "192.168.1.12", 8080, npu_capabilities);
        ASSERT_EQ(status, Status::SUCCESS);
    }
    
    std::shared_ptr<ClusterManager> cluster_manager_;
    std::shared_ptr<OptimizationManager> optimization_manager_;
    std::unique_ptr<ReactiveScheduler> reactive_scheduler_;
};

TEST_F(ReactiveSchedulerTest, Initialize) {
    EXPECT_TRUE(reactive_scheduler_->IsInitialized());
}

TEST_F(ReactiveSchedulerTest, BasicFunctionality) {
    // Test basic functionality
    auto stats = reactive_scheduler_->GetStats();
    EXPECT_GE(stats.total_decisions, 0);
}

// Additional tests can be added here as the system matures

} // namespace edge_ai
