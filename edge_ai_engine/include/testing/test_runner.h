#ifndef EDGE_AI_ENGINE_TEST_RUNNER_H
#define EDGE_AI_ENGINE_TEST_RUNNER_H

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

// Runner configuration
struct RunnerConfiguration {
    bool run_parallel = true;
    uint32_t max_parallel_tests = 4;
    uint32_t timeout_seconds = 300;
    bool stop_on_first_failure = false;
    uint32_t max_retries = 3;
    std::chrono::milliseconds retry_delay = std::chrono::milliseconds(1000);
    bool enable_coverage = false;
    bool enable_profiling = false;
    bool enable_telemetry = false;
    uint32_t performance_test_iterations = 100;
    std::map<std::string, std::string> environment_variables;
    std::vector<std::string> include_directories;
    std::vector<std::string> library_directories;
    std::vector<std::string> libraries;
    std::vector<std::string> temp_directories;
};

// Using common TestEnvironment and PerformanceThresholds from test_common.h

// Runner statistics
struct RunnerStatistics {
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;
    uint32_t failed_tests = 0;
    uint32_t flaky_tests = 0;
    double success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    double avg_duration_ms = 0.0;
    double avg_coverage_percent = 0.0;
    double avg_stability_score = 0.0;
};

// TestRunner class
class TestRunner {
public:
    TestRunner();
    ~TestRunner();

    // Configuration
    Status SetConfiguration(const RunnerConfiguration& config);
    Status SetTestEnvironment(const TestEnvironment& env);

    // Test execution
    Status RunTests(const std::vector<TestSpec>& specs);
    Status RunTestsInParallel(const std::vector<TestSpec>& specs);
    Status RunTestsSequentially(const std::vector<TestSpec>& specs);
    TestResult RunSingleTest(const TestSpec& spec);

    // Test execution by type
    Status RunTestByType(const TestSpec& spec, TestResult& result);
    Status RunUnitTest(const TestSpec& spec, TestResult& result);
    Status RunIntegrationTest(const TestSpec& spec, TestResult& result);
    Status RunPerformanceTest(const TestSpec& spec, TestResult& result);
    Status RunPropertyTest(const TestSpec& spec, TestResult& result);
    Status RunFuzzTest(const TestSpec& spec, TestResult& result);
    Status RunGenericTest(const TestSpec& spec, TestResult& result);

    // Test executable generation
    std::string GenerateTestExecutable(const TestSpec& spec);
    std::string CreateTempTestFile(const TestSpec& spec);
    std::string GenerateTestCode(const TestSpec& spec);
    std::string CompileTestFile(const std::string& test_file);
    int ExecuteTestExecutable(const std::string& executable);

    // Environment management
    Status SetupTestEnvironment();
    Status TeardownTestEnvironment();
    Status SetupIntegrationTestEnvironment(const TestSpec& spec);
    Status TeardownIntegrationTestEnvironment(const TestSpec& spec);

    // Metrics collection
    void CollectTestMetrics(TestResult& result);
    void CalculatePerformanceMetrics(const std::vector<std::chrono::milliseconds>& durations, TestResult& result);

    // Results and statistics
    std::vector<TestResult> GetResults() const;
    RunnerStatistics GetStatistics() const;
    void ClearResults();

    // Configuration setters
    Status SetMaxParallelTests(uint32_t max_parallel);
    Status SetTimeout(uint32_t timeout_seconds);

private:
    RunnerConfiguration config_;
    TestEnvironment environment_;
    std::vector<TestResult> results_;
    uint32_t max_parallel_tests_;
    uint32_t timeout_seconds_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_RUNNER_H
