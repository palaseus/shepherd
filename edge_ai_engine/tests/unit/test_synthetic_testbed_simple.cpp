#include <gtest/gtest.h>
#include <autonomous/synthetic_testbed.h>
#include <core/types.h>

using namespace edge_ai;

class SyntheticTestbedSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No config needed - Initialize() takes no parameters
    }

    void TearDown() override {
        if (testbed_.IsInitialized()) {
            testbed_.Shutdown();
        }
    }

    SyntheticTestbed testbed_;
};

TEST_F(SyntheticTestbedSimpleTest, Initialize) {
    EXPECT_EQ(testbed_.Initialize(), Status::SUCCESS);
    EXPECT_TRUE(testbed_.IsInitialized());
}

TEST_F(SyntheticTestbedSimpleTest, InitializeAlreadyInitialized) {
    ASSERT_EQ(testbed_.Initialize(), Status::SUCCESS);
    EXPECT_EQ(testbed_.Initialize(), Status::ALREADY_INITIALIZED);
}

TEST_F(SyntheticTestbedSimpleTest, GetStats) {
    ASSERT_EQ(testbed_.Initialize(), Status::SUCCESS);
    
    auto stats = testbed_.GetStats();
    EXPECT_GE(stats.total_tests_executed, 0);
    EXPECT_GE(stats.successful_tests, 0);
    EXPECT_GE(stats.failed_tests, 0);
    EXPECT_GE(stats.total_scenarios_created, 0);
    EXPECT_GE(stats.avg_test_duration_ms, 0.0);
    EXPECT_GE(stats.avg_throughput_achieved, 0.0);
    EXPECT_GE(stats.avg_latency_achieved, 0.0);
    EXPECT_GE(stats.avg_accuracy_achieved, 0.0);
    EXPECT_LE(stats.avg_accuracy_achieved, 1.0);
    EXPECT_GE(stats.avg_cpu_utilization, 0.0);
    EXPECT_LE(stats.avg_cpu_utilization, 100.0);
    EXPECT_GE(stats.avg_memory_utilization, 0.0);
    EXPECT_GE(stats.avg_gpu_utilization, 0.0);
    EXPECT_LE(stats.avg_gpu_utilization, 100.0);
    EXPECT_GE(stats.avg_network_utilization, 0.0);
    EXPECT_LE(stats.avg_network_utilization, 100.0);
    EXPECT_GE(stats.total_failures_simulated, 0);
    EXPECT_GE(stats.total_recoveries_simulated, 0);
    EXPECT_GE(stats.avg_recovery_time_ms, 0.0);
    EXPECT_GE(stats.avg_availability, 0.0);
    EXPECT_LE(stats.avg_availability, 1.0);
    EXPECT_GE(stats.total_cross_cluster_requests, 0);
    EXPECT_GE(stats.avg_cross_cluster_latency_ms, 0.0);
    EXPECT_GE(stats.avg_cross_cluster_bandwidth_utilization, 0.0);
    EXPECT_LE(stats.avg_cross_cluster_bandwidth_utilization, 100.0);
}

TEST_F(SyntheticTestbedSimpleTest, Shutdown) {
    ASSERT_EQ(testbed_.Initialize(), Status::SUCCESS);
    EXPECT_TRUE(testbed_.IsInitialized());
    
    EXPECT_EQ(testbed_.Shutdown(), Status::SUCCESS);
    EXPECT_FALSE(testbed_.IsInitialized());
}

TEST_F(SyntheticTestbedSimpleTest, ShutdownNotInitialized) {
    EXPECT_EQ(testbed_.Shutdown(), Status::NOT_INITIALIZED);
}

TEST_F(SyntheticTestbedSimpleTest, StatsValidation) {
    ASSERT_EQ(testbed_.Initialize(), Status::SUCCESS);
    
    auto stats = testbed_.GetStats();
    
    // Validate all stats fields are properly initialized
    EXPECT_GE(stats.total_tests_executed, 0);
    EXPECT_GE(stats.successful_tests, 0);
    EXPECT_GE(stats.failed_tests, 0);
    EXPECT_GE(stats.total_scenarios_created, 0);
    EXPECT_GE(stats.avg_test_duration_ms, 0.0);
    EXPECT_GE(stats.avg_throughput_achieved, 0.0);
    EXPECT_GE(stats.avg_latency_achieved, 0.0);
    EXPECT_GE(stats.avg_accuracy_achieved, 0.0);
    EXPECT_LE(stats.avg_accuracy_achieved, 1.0);
    EXPECT_GE(stats.avg_cpu_utilization, 0.0);
    EXPECT_LE(stats.avg_cpu_utilization, 100.0);
    EXPECT_GE(stats.avg_memory_utilization, 0.0);
    EXPECT_GE(stats.avg_gpu_utilization, 0.0);
    EXPECT_LE(stats.avg_gpu_utilization, 100.0);
    EXPECT_GE(stats.avg_network_utilization, 0.0);
    EXPECT_LE(stats.avg_network_utilization, 100.0);
    EXPECT_GE(stats.total_failures_simulated, 0);
    EXPECT_GE(stats.total_recoveries_simulated, 0);
    EXPECT_GE(stats.avg_recovery_time_ms, 0.0);
    EXPECT_GE(stats.avg_availability, 0.0);
    EXPECT_LE(stats.avg_availability, 1.0);
    EXPECT_GE(stats.total_cross_cluster_requests, 0);
    EXPECT_GE(stats.avg_cross_cluster_latency_ms, 0.0);
    EXPECT_GE(stats.avg_cross_cluster_bandwidth_utilization, 0.0);
    EXPECT_LE(stats.avg_cross_cluster_bandwidth_utilization, 100.0);
}

TEST_F(SyntheticTestbedSimpleTest, MultipleInitialization) {
    // Test multiple initialization attempts
    EXPECT_EQ(testbed_.Initialize(), Status::SUCCESS);
    EXPECT_TRUE(testbed_.IsInitialized());
    
    EXPECT_EQ(testbed_.Initialize(), Status::ALREADY_INITIALIZED);
    EXPECT_TRUE(testbed_.IsInitialized());
    
    EXPECT_EQ(testbed_.Shutdown(), Status::SUCCESS);
    EXPECT_FALSE(testbed_.IsInitialized());
    
    EXPECT_EQ(testbed_.Initialize(), Status::SUCCESS);
    EXPECT_TRUE(testbed_.IsInitialized());
}

TEST_F(SyntheticTestbedSimpleTest, StatsConsistency) {
    ASSERT_EQ(testbed_.Initialize(), Status::SUCCESS);
    
    // Get stats multiple times and ensure consistency
    auto stats1 = testbed_.GetStats();
    auto stats2 = testbed_.GetStats();
    
    // Stats should be consistent between calls
    EXPECT_EQ(stats1.total_tests_executed, stats2.total_tests_executed);
    EXPECT_EQ(stats1.successful_tests, stats2.successful_tests);
    EXPECT_EQ(stats1.failed_tests, stats2.failed_tests);
    EXPECT_EQ(stats1.total_scenarios_created, stats2.total_scenarios_created);
}