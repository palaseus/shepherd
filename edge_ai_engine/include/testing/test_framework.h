#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
// #include <yaml-cpp/yaml.h> // TODO: Add yaml-cpp dependency
#include <core/types.h>

namespace edge_ai {
namespace testing {

// Forward declarations
class TestSpec;
class TestScenario;
class TestOrchestrator;
class MockInjector;
class TestReporter;

/**
 * @brief Test execution result with detailed metrics
 */
struct TestResult {
    std::string test_name;
    std::string module_name;
    bool passed{false};
    std::chrono::milliseconds duration{0};
    std::string error_message;
    std::map<std::string, double> metrics;
    std::vector<std::string> warnings;
    std::vector<std::string> telemetry_events;
    std::map<std::string, std::string> metadata;
    std::vector<std::string> tags;
    
    // Performance metrics
    double cpu_usage_percent{0.0};
    double memory_usage_mb{0.0};
    double network_usage_mbps{0.0};
    
    // Coverage metrics
    double code_coverage_percent{0.0};
    double branch_coverage_percent{0.0};
    
    // Stability metrics
    uint32_t retry_count{0};
    bool flaky{false};
    double stability_score{1.0};
};

/**
 * @brief Test configuration and metadata
 */
struct TestConfig {
    std::string test_suite_name;
    std::string module_name;
    std::string test_type; // "unit", "integration", "property", "fuzz", "bdt"
    std::vector<std::string> dependencies;
    std::map<std::string, std::string> environment_vars;
    std::map<std::string, double> thresholds;
    bool enable_profiling{true};
    bool enable_coverage{true};
    bool enable_telemetry{true};
    uint32_t max_retries{3};
    std::chrono::milliseconds timeout{30000};
    std::chrono::milliseconds warmup_time{1000};
    std::chrono::milliseconds cooldown_time{500};
};

/**
 * @brief Declarative test specification in YAML/JSON format
 */
class TestSpec {
public:
    TestSpec() = default;
    explicit TestSpec(const std::string& spec_file);
    
    // Load specification from file
    Status LoadFromFile(const std::string& file_path);
    Status LoadFromYAML(const void* yaml_node);
    Status LoadFromJSON(const std::string& json_string);
    
    // Getters
    const TestConfig& GetConfig() const { return config_; }
    TestConfig& GetConfig() { return config_; }
    const std::vector<TestScenario>& GetScenarios() const { return scenarios_; }
    std::vector<TestScenario>& GetScenarios() { return scenarios_; }
    const std::map<std::string, std::string>& GetMocks() const { return mocks_; }
    std::map<std::string, std::string>& GetMocks() { return mocks_; }
    const std::map<std::string, std::string>& GetFixtures() const { return fixtures_; }
    std::map<std::string, std::string>& GetFixtures() { return fixtures_; }
    
    // Validation
    Status Validate() const;
    std::vector<std::string> GetValidationErrors() const;
    
    // Auto-generation helpers
    static TestSpec GenerateFromModule(const std::string& module_name, 
                                     const std::string& header_file);
    static TestSpec GenerateFromInterface(const std::string& interface_file);

private:
    TestConfig config_;
    std::vector<TestScenario> scenarios_;
    std::map<std::string, std::string> mocks_;
    std::map<std::string, std::string> fixtures_;
    std::string spec_file_path_;
    
    Status ParseConfig(const void* config_node);
    Status ParseScenarios(const void* scenarios_node);
    Status ParseMocks(const void* mocks_node);
    Status ParseFixtures(const void* fixtures_node);
};

/**
 * @brief Behavior-driven test scenario (Given/When/Then style)
 */
class TestScenario {
public:
    TestScenario() = default;
    
    // Scenario definition
    void SetName(const std::string& name) { name_ = name; }
    void SetDescription(const std::string& description) { description_ = description; }
    void SetTags(const std::vector<std::string>& tags) { tags_ = tags; }
    
    // Given/When/Then steps
    void AddGivenStep(const std::string& step) { given_steps_.push_back(step); }
    void AddWhenStep(const std::string& step) { when_steps_.push_back(step); }
    void AddThenStep(const std::string& step) { then_steps_.push_back(step); }
    
    // Execution
    TestResult Execute(const TestConfig& config, MockInjector& mock_injector) const;
    
    // Getters
    const std::string& GetName() const { return name_; }
    const std::string& GetDescription() const { return description_; }
    const std::vector<std::string>& GetTags() const { return tags_; }
    const std::vector<std::string>& GetGivenSteps() const { return given_steps_; }
    const std::vector<std::string>& GetWhenSteps() const { return when_steps_; }
    const std::vector<std::string>& GetThenSteps() const { return then_steps_; }
    
    // Validation
    bool IsValid() const;
    std::vector<std::string> GetValidationErrors() const;

private:
    std::string name_;
    std::string description_;
    std::vector<std::string> tags_;
    std::vector<std::string> given_steps_;
    std::vector<std::string> when_steps_;
    std::vector<std::string> then_steps_;
    
    // Execution helpers
    Status ExecuteGivenStep(const std::string& step, MockInjector& mock_injector) const;
    Status ExecuteWhenStep(const std::string& step, MockInjector& mock_injector) const;
    Status ExecuteThenStep(const std::string& step, MockInjector& mock_injector) const;
    
    // Step parsing and execution
    std::map<std::string, std::string> ParseStepParameters(const std::string& step);
    Status ExecuteAPICall(const std::string& api_call, const std::map<std::string, std::string>& params);
    Status ValidateAssertion(const std::string& assertion, const std::map<std::string, std::string>& params);
};

/**
 * @brief Mock injection and management system
 */
class MockInjector {
public:
    MockInjector() = default;
    
    // Mock registration
    template<typename T>
    void RegisterMock(const std::string& name, std::shared_ptr<T> mock) {
        mocks_[name] = std::static_pointer_cast<void>(mock);
    }
    
    template<typename T>
    std::shared_ptr<T> GetMock(const std::string& name) const {
        auto it = mocks_.find(name);
        if (it != mocks_.end()) {
            return std::static_pointer_cast<T>(it->second);
        }
        return nullptr;
    }
    
    // Mock configuration
    void ConfigureMock(const std::string& name, const std::map<std::string, std::string>& config);
    void SetMockBehavior(const std::string& name, const std::string& method, 
                        const std::string& behavior);
    
    // Mock validation
    bool IsMockRegistered(const std::string& name) const;
    std::vector<std::string> GetRegisteredMocks() const;
    
    // Cleanup
    void ClearMocks();
    void ResetMocks();

private:
    std::map<std::string, std::shared_ptr<void>> mocks_;
    std::map<std::string, std::map<std::string, std::string>> mock_configs_;
    std::map<std::string, std::map<std::string, std::string>> mock_behaviors_;
    mutable std::mutex mocks_mutex_;
};

/**
 * @brief Test orchestrator for managing test execution
 */
class TestOrchestrator {
public:
    TestOrchestrator();
    ~TestOrchestrator();
    
    // Test execution
    Status ExecuteTestSpec(const TestSpec& spec);
    Status ExecuteTestScenario(const TestScenario& scenario, const TestConfig& config);
    Status ExecuteTestSuite(const std::string& suite_name);
    
    // Parallel execution
    Status ExecuteTestsInParallel(const std::vector<TestSpec>& specs, 
                                 uint32_t max_parallel_tests = 4);
    
    // Test discovery
    std::vector<TestSpec> DiscoverTests(const std::string& test_directory);
    std::vector<TestSpec> DiscoverTestsByModule(const std::string& module_name);
    
    // Results management
    const std::vector<TestResult>& GetResults() const { return results_; }
    TestResult GetResult(const std::string& test_name) const;
    std::vector<TestResult> GetResultsByModule(const std::string& module_name) const;
    
    // Statistics
    struct TestStatistics {
        uint32_t total_tests{0};
        uint32_t passed_tests{0};
        uint32_t failed_tests{0};
        uint32_t skipped_tests{0};
        uint32_t flaky_tests{0};
        std::chrono::milliseconds total_duration{0};
        double avg_duration_ms{0.0};
        double success_rate{0.0};
        double avg_coverage_percent{0.0};
        double avg_stability_score{0.0};
    };
    
    TestStatistics GetStatistics() const;
    
    // Configuration
    void SetMaxParallelTests(uint32_t max_parallel) { max_parallel_tests_ = max_parallel; }
    void SetDefaultTimeout(std::chrono::milliseconds timeout) { default_timeout_ = timeout; }
    void EnableProfiling(bool enable) { enable_profiling_ = enable; }
    void EnableCoverage(bool enable) { enable_coverage_ = enable; }
    
    // Cleanup
    void ClearResults();
    void Shutdown();

private:
    std::vector<TestResult> results_;
    std::unique_ptr<MockInjector> mock_injector_;
    std::unique_ptr<TestReporter> reporter_;
    
    // Configuration
    uint32_t max_parallel_tests_{4};
    std::chrono::milliseconds default_timeout_{30000};
    bool enable_profiling_{true};
    bool enable_coverage_{true};
    
    // Threading
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> test_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> shutdown_requested_{false};
    
    // Execution helpers
    Status ExecuteTestInternal(const TestSpec& spec);
    Status SetupTestEnvironment(const TestConfig& config);
    Status TeardownTestEnvironment(const TestConfig& config);
    Status CollectTestMetrics(TestResult& result);
    
    // Worker thread management
    void StartWorkerThreads();
    void StopWorkerThreads();
    void WorkerThreadFunction();
};


// TestUtilities class is defined in test_utilities.h

/**
 * @brief Base class for all test fixtures
 */
class TestFixture {
public:
    TestFixture() = default;
    virtual ~TestFixture() = default;
    
    // Setup/teardown
    virtual void SetUp() {}
    virtual void TearDown() {}
    
    // Test execution
    virtual TestResult ExecuteTest() = 0;
    
    // Configuration
    void SetConfig(const TestConfig& config) { config_ = config; }
    const TestConfig& GetConfig() const { return config_; }
    
    // Mock injection
    void SetMockInjector(std::shared_ptr<MockInjector> injector) { 
        mock_injector_ = injector; 
    }
    std::shared_ptr<MockInjector> GetMockInjector() const { return mock_injector_; }

protected:
    TestConfig config_;
    std::shared_ptr<MockInjector> mock_injector_;
};

/**
 * @brief Property-based test generator
 */
class PropertyTestGenerator {
public:
    PropertyTestGenerator() = default;
    
    // Property definition
    template<typename T>
    void AddProperty(const std::string& name, 
                    std::function<bool(const T&)> property_func,
                    std::function<T()> generator_func) {
        properties_[name] = std::make_pair(
            [property_func](const void* value) {
                return property_func(*static_cast<const T*>(value));
            },
            [generator_func]() -> void* {
                static T value = generator_func();
                return &value;
            }
        );
    }
    
    // Test execution
    Status ExecutePropertyTests(uint32_t num_tests = 1000);
    Status ExecutePropertyTest(const std::string& property_name, uint32_t num_tests = 1000);
    
    // Results
    const std::vector<TestResult>& GetResults() const { return results_; }
    TestResult GetPropertyResult(const std::string& property_name) const;

private:
    std::map<std::string, std::pair<std::function<bool(const void*)>, 
                                   std::function<void*()>>> properties_;
    std::vector<TestResult> results_;
    
    Status ExecuteSinglePropertyTest(const std::string& property_name, 
                                    const void* test_value);
};

/**
 * @brief Fuzz test generator
 */
class FuzzTestGenerator {
public:
    FuzzTestGenerator() = default;
    
    // Fuzz target registration
    void RegisterFuzzTarget(const std::string& name, 
                           std::function<Status(const std::vector<uint8_t>&)> target_func);
    
    // Fuzz execution
    Status ExecuteFuzzTests(uint32_t num_tests = 10000);
    Status ExecuteFuzzTest(const std::string& target_name, uint32_t num_tests = 10000);
    
    // Input generation
    std::vector<uint8_t> GenerateRandomInput(size_t min_size, size_t max_size);
    std::vector<uint8_t> GenerateMutationInput(const std::vector<uint8_t>& base_input);
    std::vector<uint8_t> GenerateCorpusInput(const std::string& corpus_file);
    
    // Results
    const std::vector<TestResult>& GetResults() const { return results_; }
    TestResult GetFuzzResult(const std::string& target_name) const;

private:
    std::map<std::string, std::function<Status(const std::vector<uint8_t>&)>> fuzz_targets_;
    std::vector<TestResult> results_;
    
    Status ExecuteSingleFuzzTest(const std::string& target_name, 
                                const std::vector<uint8_t>& input);
};

} // namespace testing
} // namespace edge_ai
