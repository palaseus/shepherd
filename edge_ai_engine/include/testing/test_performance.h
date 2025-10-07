#ifndef EDGE_AI_ENGINE_TEST_PERFORMANCE_H
#define EDGE_AI_ENGINE_TEST_PERFORMANCE_H

#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <core/types.h>
#include <testing/test_framework.h>
#include <testing/test_common.h>

namespace edge_ai {
namespace testing {

// Using common PerformanceThresholds from test_common.h

// Performance configuration
struct PerformanceConfiguration {
    bool run_parallel = true;
    uint32_t max_parallel_tests = 4;
    uint32_t timeout_seconds = 300;
    bool stop_on_first_failure = false;
    uint32_t max_retries = 3;
    std::chrono::milliseconds retry_delay = std::chrono::milliseconds(1000);
    bool enable_coverage = false;
    bool enable_profiling = false;
    bool enable_telemetry = false;
    uint32_t performance_test_iterations = 10;
    uint32_t warmup_iterations = 3;
    uint32_t measurement_iterations = 7;
    uint32_t cooldown_iterations = 2;
    PerformanceThresholds thresholds;
    std::map<std::string, std::string> environment_variables;
    std::vector<std::string> include_directories;
    std::vector<std::string> library_directories;
    std::vector<std::string> libraries;
    std::vector<std::string> temp_directories;
};

// Performance test result
struct PerformanceTestResult {
    std::string test_name;
    std::string module_name;
    bool passed = false;
    bool flaky = false;
    std::chrono::milliseconds duration{0};
    double code_coverage_percent = 0.0;
    double stability_score = 0.0;
    double cpu_usage_percent = 0.0;
    double memory_usage_mb = 0.0;
    double network_usage_mbps = 0.0;
    double power_consumption_watts = 0.0;
    std::vector<uint64_t> execution_times;
    double avg_execution_time_us = 0.0;
    double min_execution_time_us = 0.0;
    double max_execution_time_us = 0.0;
    double std_dev_execution_time_us = 0.0;
    double throughput_ops_per_sec = 0.0;
    std::string error_message;
    std::map<std::string, std::string> metadata;
    std::vector<std::string> tags;
};

// Using common PerformanceStatistics from test_common.h

// TestPerformance class
class TestPerformance {
public:
    TestPerformance();
    ~TestPerformance();

    // Configuration
    Status SetConfiguration(const PerformanceConfiguration& config);
    Status SetTestEnvironment(const TestEnvironment& env);

    // Performance test execution
    Status RunPerformanceTests(const std::vector<TestSpec>& specs);
    Status RunPerformanceTestsInParallel(const std::vector<TestSpec>& specs);
    Status RunPerformanceTestsSequentially(const std::vector<TestSpec>& specs);
    PerformanceTestResult RunSinglePerformanceTest(const TestSpec& spec);

    // Scenario execution
    Status RunPerformanceScenario(const TestScenario& scenario, PerformanceTestResult& result);

    // Step execution
    Status ExecuteGivenStep(const std::string& step);
    Status ExecuteWhenStepWithMeasurement(const std::string& step, PerformanceTestResult& result);
    Status ExecuteThenStep(const std::string& step);

    // Parameter parsing
    std::map<std::string, std::string> ParseStepParameters(const std::string& step);

    // Environment setup
    Status SetupTestEnvironment();
    Status SetupMock(const std::map<std::string, std::string>& params);
    Status SetupDatabase(const std::map<std::string, std::string>& params);
    Status SetupNetwork(const std::map<std::string, std::string>& params);
    Status SetupFileSystem(const std::map<std::string, std::string>& params);

    // API execution
    Status ExecuteAPICall(const std::string& step, const std::map<std::string, std::string>& params);
    Status SendMessage(const std::string& step, const std::map<std::string, std::string>& params);
    Status ReceiveMessage(const std::string& step, const std::map<std::string, std::string>& params);
    Status WriteData(const std::string& step, const std::map<std::string, std::string>& params);
    Status ReadData(const std::string& step, const std::map<std::string, std::string>& params);

    // Validation
    Status ValidateSuccess(const std::map<std::string, std::string>& params);
    Status ValidateFailure(const std::map<std::string, std::string>& params);
    Status ValidateReturnValue(const std::string& step, const std::map<std::string, std::string>& params);
    Status ValidateException(const std::string& step, const std::map<std::string, std::string>& params);
    Status ValidateContent(const std::string& step, const std::map<std::string, std::string>& params);
    Status ValidatePerformanceThreshold(const std::string& step, const std::map<std::string, std::string>& params);

    // Environment management
    Status SetupPerformanceTestEnvironment();
    Status TeardownPerformanceTestEnvironment();
    Status SetupPerformanceServices();
    Status TeardownPerformanceServices();
    Status SetupPerformanceTestData();
    Status TeardownPerformanceTestData();
    Status SetupTestEnvironment(const TestSpec& spec);
    Status TeardownTestEnvironment(const TestSpec& spec);
    Status TeardownTestEnvironment();

    // Metrics collection
    void CollectPerformanceMetrics(PerformanceTestResult& result);
    double GetCurrentCPUUsage();
    double GetCurrentMemoryUsage();
    double GetCurrentNetworkUsage();
    double GetCurrentPowerConsumption();

    // Results and statistics
    std::vector<PerformanceTestResult> GetResults() const;
    PerformanceStatistics GetStatistics() const;
    void ClearResults();

    // Configuration setters
    Status SetMaxParallelTests(uint32_t max_parallel);
    Status SetTimeout(uint32_t timeout_seconds);
    Status SetStopOnFirstFailure(bool stop_on_first_failure);
    Status SetRunParallel(bool run_parallel);
    Status SetPerformanceThresholds(const PerformanceThresholds& thresholds);
    Status SetWarmupIterations(uint32_t warmup_iterations);
    Status SetMeasurementIterations(uint32_t measurement_iterations);
    Status SetCooldownIterations(uint32_t cooldown_iterations);

private:
    PerformanceConfiguration config_;
    TestEnvironment environment_;
    std::vector<PerformanceTestResult> results_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_PERFORMANCE_H
