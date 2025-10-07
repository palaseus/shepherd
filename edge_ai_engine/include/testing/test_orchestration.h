#ifndef EDGE_AI_ENGINE_TEST_ORCHESTRATION_H
#define EDGE_AI_ENGINE_TEST_ORCHESTRATION_H

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
#include <memory>
#include <core/types.h>
#include <testing/test_framework.h>
#include <testing/test_common.h>
#include <testing/test_reporter.h>
#include <testing/test_discovery.h>
#include <testing/test_runner.h>
#include <testing/test_integration.h>
#include <testing/test_performance.h>
#include <testing/test_coverage.h>
#include <testing/test_validation.h>

namespace edge_ai {
namespace testing {

// Orchestration configuration
struct OrchestrationConfiguration {
    bool run_parallel = true;
    uint32_t max_parallel_tests = 4;
    uint32_t timeout_seconds = 300;
    bool stop_on_first_failure = false;
    uint32_t max_retries = 3;
    std::chrono::milliseconds retry_delay = std::chrono::milliseconds(1000);
    bool enable_coverage = false;
    bool enable_validation = false;
    bool enable_profiling = false;
    bool enable_telemetry = false;
    std::chrono::milliseconds warmup_time = std::chrono::milliseconds(1000);
    std::chrono::milliseconds cooldown_time = std::chrono::milliseconds(1000);
    std::map<std::string, double> thresholds;
    std::map<std::string, std::string> environment_variables;
    std::vector<std::string> include_directories;
    std::vector<std::string> library_directories;
    std::vector<std::string> libraries;
    std::vector<std::string> temp_directories;
};

// Using common TestStatistics from test_common.h

// TestOrchestration class
class TestOrchestration {
public:
    TestOrchestration();
    ~TestOrchestration();

    // Configuration
    Status SetConfiguration(const OrchestrationConfiguration& config);
    Status SetTestEnvironment(const TestEnvironment& env);

    // Test execution
    Status ExecuteTestSpec(const TestSpec& spec);
    Status ExecuteTestScenario(const TestScenario& scenario, const TestConfig& config);
    Status ExecuteTestSuite(const std::string& suite_name);
    Status ExecuteTestsInParallel(const std::vector<TestSpec>& specs, uint32_t max_parallel_tests = 4);

    // Test discovery
    std::vector<TestSpec> DiscoverTests(const std::string& test_directory);
    std::vector<TestSpec> DiscoverTestsByModule(const std::string& module_name);

    // Results and statistics
    TestResult GetResult(const std::string& test_name) const;
    std::vector<TestResult> GetResultsByModule(const std::string& module_name) const;
    TestStatistics GetStatistics() const;
    void ClearResults();

    // Lifecycle management
    void Shutdown();

private:
    // Internal execution
    Status ExecuteTestInternal(const TestSpec& spec);
    Status SetupTestEnvironment(const TestConfig& config);
    Status TeardownTestEnvironment(const TestConfig& config);
    Status CollectTestMetrics(TestResult& result);

    // Worker thread management
    void StartWorkerThreads();
    void StopWorkerThreads();
    void WorkerThreadFunction();

    // Configuration
    OrchestrationConfiguration config_;
    TestEnvironment environment_;
    std::vector<TestResult> results_;

    // Threading
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> test_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_requested_{false};
    uint32_t max_parallel_tests_;

    // Components
    std::unique_ptr<MockInjector> mock_injector_;
    std::unique_ptr<TestReporter> reporter_;
    std::unique_ptr<TestDiscovery> discovery_;
    std::unique_ptr<TestRunner> runner_;
    std::unique_ptr<TestIntegration> integration_;
    std::unique_ptr<TestPerformance> performance_;
    std::unique_ptr<TestCoverage> coverage_;
    std::unique_ptr<TestValidation> validation_;
};

// TestOrchestrationManager class
class TestOrchestrationManager {
public:
    TestOrchestrationManager();
    ~TestOrchestrationManager();

    // Configuration
    Status SetConfiguration(const OrchestrationConfiguration& config);
    Status SetTestEnvironment(const TestEnvironment& env);

    // Test execution
    Status RunAllTests();
    Status RunTestsByModule(const std::string& module_name);
    Status RunTestsByType(const std::string& test_type);
    Status RunTestsByTag(const std::string& tag);
    Status RunTests(const std::vector<TestSpec>& specs);

    // Test execution by type
    Status RunTestByType(const TestSpec& spec);
    Status RunPropertyTests(const TestSpec& spec);
    Status RunFuzzTests(const TestSpec& spec);

    // Environment management
    Status SetupTestEnvironment();
    Status TeardownTestEnvironment();

    // Results and statistics
    std::vector<TestResult> GetResults() const;
    TestStatistics GetStatistics() const;
    void ClearResults();

    // Report generation
    Status GenerateReport(const std::string& output_file, TestOutputFormat format);
    Status GenerateCoverageReport(const std::string& output_file, CoverageReportFormat format);
    Status GenerateValidationReport(const std::string& output_file, ValidationReportFormat format);

    // Configuration setters
    Status SetMaxParallelTests(uint32_t max_parallel);
    Status SetTimeout(uint32_t timeout_seconds);
    Status SetStopOnFirstFailure(bool stop_on_first_failure);
    Status SetRunParallel(bool run_parallel);
    Status SetEnableCoverage(bool enable_coverage);
    Status SetEnableValidation(bool enable_validation);
    Status SetEnableProfiling(bool enable_profiling);
    Status SetEnableTelemetry(bool enable_telemetry);
    Status SetMaxRetries(uint32_t max_retries);
    Status SetWarmupTime(std::chrono::milliseconds warmup_time);
    Status SetCooldownTime(std::chrono::milliseconds cooldown_time);
    Status SetThresholds(const std::map<std::string, double>& thresholds);
    Status SetEnvironmentVariables(const std::map<std::string, std::string>& env_vars);
    Status SetIncludeDirectories(const std::vector<std::string>& include_dirs);
    Status SetLibraryDirectories(const std::vector<std::string>& library_dirs);
    Status SetLibraries(const std::vector<std::string>& libraries);
    Status SetTempDirectories(const std::vector<std::string>& temp_dirs);

private:
    OrchestrationConfiguration config_;
    TestEnvironment environment_;
    TestDiscovery discovery_;
    TestRunner runner_;
    TestIntegration integration_;
    TestPerformance performance_;
    TestCoverage coverage_;
    TestValidation validation_;
    TestReporter reporter_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_ORCHESTRATION_H
