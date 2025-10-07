#include <gtest/gtest.h>
#include <autonomous/autonomous_optimizer.h>
#include <core/types.h>

using namespace edge_ai;

class AutonomousOptimizerSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
    config_.feedback_collection_interval = std::chrono::milliseconds(10);
    config_.optimization_trigger_interval = std::chrono::milliseconds(100);
    config_.action_execution_interval = std::chrono::milliseconds(50);
    config_.session_evaluation_window = std::chrono::seconds(1);
        
        config_.performance_regression_threshold = 0.1;
        config_.improvement_threshold = 0.05;
        config_.confidence_threshold = 0.8;
        config_.risk_tolerance_threshold = 0.2;
        
        config_.enable_automatic_rollback = true;
        config_.max_rollback_time = std::chrono::seconds(30);
        config_.max_concurrent_optimizations = 5;
        config_.require_human_approval = false;
        
        config_.enable_online_learning = true;
        config_.learning_rate = 0.01;
        config_.experience_buffer_size = 1000;
        config_.model_update_interval = std::chrono::seconds(60);
        
        config_.global_performance_targets["latency"] = 50.0;
        config_.global_performance_targets["throughput"] = 1000.0;
        config_.optimization_weights["latency"] = 0.4;
        config_.optimization_weights["throughput"] = 0.6;
        config_.priority_metrics = {"latency", "throughput", "accuracy"};
        
        config_.enable_dag_optimization = true;
        config_.enable_architecture_evolution = true;
        config_.enable_resource_optimization = true;
        config_.enable_cross_cluster_optimization = true;
    }

    void TearDown() override {
        if (optimizer_.IsInitialized()) {
            optimizer_.Shutdown();
        }
    }

    AutonomousOptimizationConfig config_;
    AutonomousOptimizer optimizer_;
};

TEST_F(AutonomousOptimizerSimpleTest, Initialize) {
    EXPECT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(optimizer_.IsInitialized());
}

TEST_F(AutonomousOptimizerSimpleTest, InitializeAlreadyInitialized) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    EXPECT_EQ(optimizer_.Initialize(config_), Status::ALREADY_INITIALIZED);
}

TEST_F(AutonomousOptimizerSimpleTest, StartOptimizationSession) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    std::string session_id;
    std::string session_name = "test_session";
    std::string cluster_id = "test_cluster";
    std::vector<std::string> target_dags = {"dag1", "dag2"};
    
    EXPECT_EQ(optimizer_.StartOptimizationSession(session_name, cluster_id, target_dags, session_id), Status::SUCCESS);
    EXPECT_FALSE(session_id.empty());
}

TEST_F(AutonomousOptimizerSimpleTest, StartOptimizationSessionNotInitialized) {
    std::string session_id;
    std::string session_name = "test_session";
    std::string cluster_id = "test_cluster";
    std::vector<std::string> target_dags = {"dag1"};
    
    EXPECT_EQ(optimizer_.StartOptimizationSession(session_name, cluster_id, target_dags, session_id), Status::NOT_INITIALIZED);
}

TEST_F(AutonomousOptimizerSimpleTest, StopOptimizationSession) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    std::string session_id;
    std::string session_name = "test_session";
    std::string cluster_id = "test_cluster";
    std::vector<std::string> target_dags = {"dag1"};
    
    ASSERT_EQ(optimizer_.StartOptimizationSession(session_name, cluster_id, target_dags, session_id), Status::SUCCESS);
    EXPECT_EQ(optimizer_.StopOptimizationSession(session_id), Status::SUCCESS);
}

TEST_F(AutonomousOptimizerSimpleTest, StopOptimizationSessionNotInitialized) {
    EXPECT_EQ(optimizer_.StopOptimizationSession("nonexistent_session"), Status::NOT_INITIALIZED);
}

TEST_F(AutonomousOptimizerSimpleTest, ContinuousOptimizationConfig) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    ContinuousOptimizationConfig cont_config;
    cont_config.feedback_collection_interval = std::chrono::milliseconds(10);
    cont_config.optimization_interval = std::chrono::milliseconds(50);
    cont_config.learning_update_interval = std::chrono::milliseconds(20);
    cont_config.performance_improvement_threshold = 0.1;
    cont_config.learning_rate = 0.01;
    cont_config.max_optimization_overhead_percent = 10.0;
    cont_config.exploration_rate = 0.05;
    cont_config.batch_size = 1;
    
    // Test configuration initialization without starting
    EXPECT_EQ(optimizer_.InitializeContinuousOptimization(cont_config), Status::SUCCESS);
}

TEST_F(AutonomousOptimizerSimpleTest, PerformanceFeedback) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    PerformanceFeedback feedback;
    feedback.feedback_id = "test_feedback";
    feedback.timestamp = std::chrono::steady_clock::now();
    feedback.latency_ms = 50.0;
    feedback.throughput_rps = 1000.0;
    feedback.memory_usage_mb = 200.0;
    feedback.cpu_usage_percent = 70.0;
    feedback.power_consumption_watts = 150.0;
    feedback.accuracy = 0.95;
    feedback.workload_type = "test_workload";
    feedback.cluster_state = "healthy";
    feedback.network_condition = "stable";
    
    EXPECT_EQ(optimizer_.CollectPerformanceFeedback(feedback), Status::SUCCESS);
}

TEST_F(AutonomousOptimizerSimpleTest, OptimizationInsights) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    std::vector<OptimizationInsight> insights;
    EXPECT_EQ(optimizer_.GenerateOptimizationInsights(insights), Status::SUCCESS);
    EXPECT_GE(insights.size(), 0);
}

TEST_F(AutonomousOptimizerSimpleTest, GetStats) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    auto stats = optimizer_.GetStats();
    EXPECT_GE(stats.total_sessions, 0);
    EXPECT_GE(stats.total_actions_planned, 0);
    EXPECT_GE(stats.total_actions_executed, 0);
    EXPECT_GE(stats.total_feedback_events, 0);
    EXPECT_GE(stats.total_insights_generated, 0);
    EXPECT_GE(stats.avg_optimization_time_ms, 0.0);
    EXPECT_GE(stats.avg_rollback_time_ms, 0.0);
    EXPECT_GE(stats.success_rate, 0.0);
    EXPECT_LE(stats.success_rate, 1.0);
    EXPECT_GE(stats.avg_improvement_percentage, 0.0);
    EXPECT_GE(stats.avg_confidence_score, 0.0);
    EXPECT_LE(stats.avg_confidence_score, 1.0);
    EXPECT_GE(stats.avg_risk_score, 0.0);
    EXPECT_LE(stats.avg_risk_score, 1.0);
    EXPECT_GE(stats.avg_cpu_overhead_percent, 0.0);
    EXPECT_LE(stats.avg_cpu_overhead_percent, 100.0);
    EXPECT_GE(stats.avg_memory_overhead_mb, 0.0);
    EXPECT_GE(stats.avg_network_overhead_mbps, 0.0);
}

TEST_F(AutonomousOptimizerSimpleTest, Shutdown) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(optimizer_.IsInitialized());
    
    EXPECT_EQ(optimizer_.Shutdown(), Status::SUCCESS);
    EXPECT_FALSE(optimizer_.IsInitialized());
}

TEST_F(AutonomousOptimizerSimpleTest, ShutdownNotInitialized) {
    EXPECT_EQ(optimizer_.Shutdown(), Status::NOT_INITIALIZED);
}

TEST_F(AutonomousOptimizerSimpleTest, MultipleOptimizationSessions) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    // Create multiple optimization sessions
    std::string session1, session2, session3;
    EXPECT_EQ(optimizer_.StartOptimizationSession("session1", "cluster_1", {"dag_1"}, session1), Status::SUCCESS);
    EXPECT_EQ(optimizer_.StartOptimizationSession("session2", "cluster_2", {"dag_2"}, session2), Status::SUCCESS);
    EXPECT_EQ(optimizer_.StartOptimizationSession("session3", "cluster_3", {"dag_3"}, session3), Status::SUCCESS);
    
    EXPECT_FALSE(session1.empty());
    EXPECT_FALSE(session2.empty());
    EXPECT_FALSE(session3.empty());
    EXPECT_NE(session1, session2);
    EXPECT_NE(session2, session3);
    EXPECT_NE(session1, session3);
    
    // Stop all sessions
    EXPECT_EQ(optimizer_.StopOptimizationSession(session1), Status::SUCCESS);
    EXPECT_EQ(optimizer_.StopOptimizationSession(session2), Status::SUCCESS);
    EXPECT_EQ(optimizer_.StopOptimizationSession(session3), Status::SUCCESS);
}

TEST_F(AutonomousOptimizerSimpleTest, StatsValidation) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    auto stats = optimizer_.GetStats();
    
    // Validate all stats fields are properly initialized
    EXPECT_GE(stats.total_sessions, 0);
    EXPECT_GE(stats.total_actions_planned, 0);
    EXPECT_GE(stats.total_actions_executed, 0);
    EXPECT_GE(stats.total_feedback_events, 0);
    EXPECT_GE(stats.total_insights_generated, 0);
    EXPECT_GE(stats.avg_optimization_time_ms, 0.0);
    EXPECT_GE(stats.avg_rollback_time_ms, 0.0);
    EXPECT_GE(stats.success_rate, 0.0);
    EXPECT_LE(stats.success_rate, 1.0);
    EXPECT_GE(stats.avg_improvement_percentage, 0.0);
    EXPECT_GE(stats.avg_confidence_score, 0.0);
    EXPECT_LE(stats.avg_confidence_score, 1.0);
    EXPECT_GE(stats.avg_risk_score, 0.0);
    EXPECT_LE(stats.avg_risk_score, 1.0);
    EXPECT_GE(stats.avg_cpu_overhead_percent, 0.0);
    EXPECT_LE(stats.avg_cpu_overhead_percent, 100.0);
    EXPECT_GE(stats.avg_memory_overhead_mb, 0.0);
    EXPECT_GE(stats.avg_network_overhead_mbps, 0.0);
}

TEST_F(AutonomousOptimizerSimpleTest, MultiplePerformanceFeedback) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    // Send multiple performance feedback events
    for (int i = 0; i < 5; ++i) {
        PerformanceFeedback feedback;
        feedback.timestamp = std::chrono::steady_clock::now();
        feedback.latency_ms = 50.0 + i * 10.0;
        feedback.throughput_rps = 1000.0 - i * 100.0;
        feedback.memory_usage_mb = 200.0 + i * 50.0;
        feedback.cpu_usage_percent = 0.7 + i * 0.05;
        feedback.power_consumption_watts = 150.0 + i * 10.0;
        feedback.accuracy = 0.95 - i * 0.01;
        
        EXPECT_EQ(optimizer_.CollectPerformanceFeedback(feedback), Status::SUCCESS);
    }
    
    auto stats = optimizer_.GetStats();
    EXPECT_GE(stats.total_feedback_events, 5);
}