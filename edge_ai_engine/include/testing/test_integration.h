#ifndef EDGE_AI_ENGINE_TEST_INTEGRATION_H
#define EDGE_AI_ENGINE_TEST_INTEGRATION_H

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

// Using common TestEnvironment from test_common.h

// Integration configuration
struct IntegrationConfiguration {
    bool run_parallel = true;
    uint32_t max_parallel_tests = 4;
    uint32_t timeout_seconds = 300;
    bool stop_on_first_failure = false;
    uint32_t max_retries = 3;
    std::chrono::milliseconds retry_delay = std::chrono::milliseconds(1000);
    bool enable_coverage = false;
    bool enable_profiling = false;
    bool enable_telemetry = false;
    std::map<std::string, std::string> environment_variables;
    std::vector<std::string> include_directories;
    std::vector<std::string> library_directories;
    std::vector<std::string> libraries;
    std::vector<std::string> temp_directories;
};

// Integration statistics
struct IntegrationStatistics {
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

// TestIntegration class
class TestIntegration {
public:
    TestIntegration();
    ~TestIntegration();

    // Configuration
    Status SetConfiguration(const IntegrationConfiguration& config);
    Status SetTestEnvironment(const TestEnvironment& env);

    // Integration test execution
    Status RunIntegrationTests(const std::vector<TestSpec>& specs);
    Status RunIntegrationTestsInParallel(const std::vector<TestSpec>& specs);
    Status RunIntegrationTestsSequentially(const std::vector<TestSpec>& specs);
    TestResult RunSingleIntegrationTest(const TestSpec& spec);

    // Scenario execution
    Status RunIntegrationScenario(const TestScenario& scenario, TestResult& result);

    // Step execution
    Status ExecuteGivenStep(const std::string& step);
    Status ExecuteWhenStep(const std::string& step);
    Status ExecuteWhenStepWithMeasurement(const std::string& step, TestResult& result);
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

    // Environment management
    Status SetupIntegrationTestEnvironment();
    Status TeardownIntegrationTestEnvironment();
    Status SetupIntegrationServices();
    Status TeardownIntegrationServices();
    Status SetupTestData();
    Status TeardownTestData();
    Status SetupTestEnvironment(const TestSpec& spec);
    Status TeardownTestEnvironment(const TestSpec& spec);
    Status TeardownTestEnvironment();

    // Metrics collection
    void CollectTestMetrics(TestResult& result);

    // Results and statistics
    std::vector<TestResult> GetResults() const;
    IntegrationStatistics GetStatistics() const;
    void ClearResults();

    // Configuration setters
    Status SetMaxParallelTests(uint32_t max_parallel);
    Status SetTimeout(uint32_t timeout_seconds);
    Status SetStopOnFirstFailure(bool stop_on_first_failure);
    Status SetRunParallel(bool run_parallel);

private:
    IntegrationConfiguration config_;
    TestEnvironment environment_;
    std::vector<TestResult> results_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_INTEGRATION_H
