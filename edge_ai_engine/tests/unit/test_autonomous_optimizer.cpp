#include <gtest/gtest.h>
#include <autonomous/autonomous_optimizer.h>
#include <core/types.h>

using namespace edge_ai;

class AutonomousOptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.optimization_enabled = true;
        config_.learning_enabled = true;
        config_.safety_mode = "balanced";
        config_.optimization_interval_ms = 1000;
        config_.learning_rate = 0.01;
        config_.safety_threshold = 0.1;
        config_.max_optimization_actions = 10;
        config_.enable_automatic_rollback = true;
        config_.enable_performance_monitoring = true;
        config_.enable_anomaly_detection = true;
        config_.enable_continuous_learning = true;
        config_.enable_adaptive_optimization = true;
        config_.enable_ml_optimization = true;
        config_.enable_evolutionary_optimization = true;
        config_.enable_neural_architecture_search = true;
        config_.enable_autonomous_optimization = true;
        config_.enable_cross_cluster_optimization = true;
        config_.enable_predictive_optimization = true;
        config_.enable_self_healing = true;
        config_.enable_autonomous_governance = true;
        config_.enable_federation = true;
        config_.enable_evolution = true;
        config_.enable_telemetry_analytics = true;
        config_.enable_security = true;
        config_.enable_autonomous_dag_generation = true;
        config_.enable_synthetic_testing = true;
    }

    void TearDown() override {
        if (optimizer_.IsInitialized()) {
            optimizer_.Shutdown();
        }
    }

    AutonomousOptimizationConfig config_;
    AutonomousOptimizer optimizer_;
};

TEST_F(AutonomousOptimizerTest, Initialize) {
    EXPECT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(optimizer_.IsInitialized());
}

TEST_F(AutonomousOptimizerTest, InitializeInvalidConfig) {
    AutonomousOptimizationConfig invalid_config;
    invalid_config.optimization_interval_ms = 0; // Invalid
    EXPECT_EQ(optimizer_.Initialize(invalid_config), Status::INVALID_ARGUMENT);
    EXPECT_FALSE(optimizer_.IsInitialized());
}

TEST_F(AutonomousOptimizerTest, StartOptimization) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    std::string session_id;
    EXPECT_EQ(optimizer_.StartOptimization(session_id), Status::SUCCESS);
    EXPECT_FALSE(session_id.empty());
}

TEST_F(AutonomousOptimizerTest, StartOptimizationNotInitialized) {
    std::string session_id;
    EXPECT_EQ(optimizer_.StartOptimization(session_id), Status::NOT_INITIALIZED);
}

TEST_F(AutonomousOptimizerTest, StopOptimization) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    std::string session_id;
    ASSERT_EQ(optimizer_.StartOptimization(session_id), Status::SUCCESS);
    
    EXPECT_EQ(optimizer_.StopOptimization(session_id), Status::SUCCESS);
}

TEST_F(AutonomousOptimizerTest, StopOptimizationNotInitialized) {
    EXPECT_EQ(optimizer_.StopOptimization("nonexistent_session"), Status::NOT_INITIALIZED);
}

TEST_F(AutonomousOptimizerTest, ContinuousOptimization) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    ContinuousOptimizationConfig cont_config;
    cont_config.feedback_collection_interval_ms = 100;
    cont_config.optimization_interval_ms = 500;
    cont_config.learning_update_interval_ms = 200;
    cont_config.monitoring_interval_ms = 50;
    cont_config.feedback_threshold = 0.1;
    cont_config.learning_rate = 0.01;
    cont_config.safety_constraints_enabled = true;
    cont_config.adaptation_rate = 0.05;
    
    EXPECT_EQ(optimizer_.InitializeContinuousOptimization(cont_config), Status::SUCCESS);
    EXPECT_EQ(optimizer_.StartContinuousOptimization(), Status::SUCCESS);
    EXPECT_TRUE(optimizer_.IsContinuousOptimizationActive());
    
    // Let it run briefly
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    EXPECT_EQ(optimizer_.StopContinuousOptimization(), Status::SUCCESS);
    EXPECT_FALSE(optimizer_.IsContinuousOptimizationActive());
}

TEST_F(AutonomousOptimizerTest, PerformanceFeedback) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    PerformanceFeedback feedback;
    feedback.timestamp = std::chrono::steady_clock::now();
    feedback.latency_ms = 50.0;
    feedback.throughput_rps = 1000.0;
    feedback.memory_usage_mb = 200.0;
    feedback.cpu_utilization = 0.7;
    feedback.gpu_utilization = 0.8;
    feedback.power_consumption_w = 150.0;
    feedback.accuracy = 0.95;
    feedback.reliability = 0.99;
    feedback.scalability = 0.85;
    feedback.efficiency = 0.9;
    feedback.quality = 0.92;
    feedback.innovation = 0.88;
    feedback.adaptability = 0.87;
    feedback.autonomy = 0.91;
    feedback.intelligence = 0.89;
    feedback.evolution = 0.86;
    feedback.governance = 0.93;
    feedback.federation = 0.84;
    feedback.analytics = 0.9;
    feedback.security = 0.96;
    feedback.testing = 0.88;
    feedback.optimization = 0.92;
    
    EXPECT_EQ(optimizer_.CollectPerformanceFeedback(feedback), Status::SUCCESS);
}

TEST_F(AutonomousOptimizerTest, OptimizationInsights) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    std::vector<OptimizationInsight> insights;
    EXPECT_EQ(optimizer_.GenerateOptimizationInsights(insights), Status::SUCCESS);
    EXPECT_GE(insights.size(), 0);
}

TEST_F(AutonomousOptimizerTest, GetStats) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    
    auto stats = optimizer_.GetStats();
    EXPECT_GE(stats.total_optimization_sessions, 0);
    EXPECT_GE(stats.total_optimization_actions, 0);
    EXPECT_GE(stats.total_learning_updates, 0);
    EXPECT_GE(stats.total_feedback_collected, 0);
    EXPECT_GE(stats.total_insights_generated, 0);
    EXPECT_GE(stats.total_continuous_optimization_time_ms, 0);
    EXPECT_GE(stats.average_optimization_time_ms, 0.0);
    EXPECT_GE(stats.average_learning_time_ms, 0.0);
    EXPECT_GE(stats.average_feedback_processing_time_ms, 0.0);
    EXPECT_GE(stats.average_insight_generation_time_ms, 0.0);
    EXPECT_GE(stats.success_rate, 0.0);
    EXPECT_LE(stats.success_rate, 1.0);
    EXPECT_GE(stats.optimization_improvement, 0.0);
    EXPECT_GE(stats.learning_effectiveness, 0.0);
    EXPECT_LE(stats.learning_effectiveness, 1.0);
    EXPECT_GE(stats.adaptation_rate, 0.0);
    EXPECT_LE(stats.adaptation_rate, 1.0);
    EXPECT_GE(stats.autonomy_level, 0.0);
    EXPECT_LE(stats.autonomy_level, 1.0);
    EXPECT_GE(stats.intelligence_level, 0.0);
    EXPECT_LE(stats.intelligence_level, 1.0);
    EXPECT_GE(stats.evolution_level, 0.0);
    EXPECT_LE(stats.evolution_level, 1.0);
    EXPECT_GE(stats.governance_level, 0.0);
    EXPECT_LE(stats.governance_level, 1.0);
    EXPECT_GE(stats.federation_level, 0.0);
    EXPECT_LE(stats.federation_level, 1.0);
    EXPECT_GE(stats.analytics_level, 0.0);
    EXPECT_LE(stats.analytics_level, 1.0);
    EXPECT_GE(stats.security_level, 0.0);
    EXPECT_LE(stats.security_level, 1.0);
    EXPECT_GE(stats.testing_level, 0.0);
    EXPECT_LE(stats.testing_level, 1.0);
    EXPECT_GE(stats.optimization_level, 0.0);
    EXPECT_LE(stats.optimization_level, 1.0);
}

TEST_F(AutonomousOptimizerTest, Shutdown) {
    ASSERT_EQ(optimizer_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(optimizer_.IsInitialized());
    
    EXPECT_EQ(optimizer_.Shutdown(), Status::SUCCESS);
    EXPECT_FALSE(optimizer_.IsInitialized());
}

TEST_F(AutonomousOptimizerTest, ShutdownNotInitialized) {
    EXPECT_EQ(optimizer_.Shutdown(), Status::NOT_INITIALIZED);
}
