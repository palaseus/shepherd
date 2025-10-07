#include <testing/test_runner.h>
#include <profiling/profiler.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cstdlib>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cmath>

namespace edge_ai {
namespace testing {

// TestRunner Implementation
TestRunner::TestRunner() : max_parallel_tests_(4), timeout_seconds_(300) {
    // Initialize default configuration
}

TestRunner::~TestRunner() {
    // Cleanup if needed
}

Status TestRunner::SetConfiguration(const RunnerConfiguration& config) {
    config_ = config;
    max_parallel_tests_ = config.max_parallel_tests;
    timeout_seconds_ = config.timeout_seconds;
    return Status::SUCCESS;
}

Status TestRunner::SetTestEnvironment(const TestEnvironment& env) {
    environment_ = env;
    return Status::SUCCESS;
}

Status TestRunner::RunTests(const std::vector<TestSpec>& specs) {
    PROFILER_SCOPED_EVENT(0, "run_tests");
    
    if (specs.empty()) {
        return Status::INVALID_ARGUMENT;
    }
    
    // Setup test environment
    Status status = SetupTestEnvironment();
    if (status != Status::SUCCESS) {
        return status;
    }
    
    // Run tests
    if (config_.run_parallel) {
        status = RunTestsInParallel(specs);
    } else {
        status = RunTestsSequentially(specs);
    }
    
    // Teardown test environment
    TeardownTestEnvironment();
    
    return status;
}

Status TestRunner::RunTestsInParallel(const std::vector<TestSpec>& specs) {
    PROFILER_SCOPED_EVENT(0, "run_tests_parallel");
    
    std::vector<std::thread> threads;
    std::vector<TestResult> results;
    std::mutex results_mutex;
    
    // Create thread pool
    for (size_t i = 0; i < specs.size(); i += max_parallel_tests_) {
        size_t end = std::min(i + max_parallel_tests_, specs.size());
        
        for (size_t j = i; j < end; ++j) {
            threads.emplace_back([this, &specs, j, &results, &results_mutex]() {
                TestResult result = RunSingleTest(specs[j]);
                
                std::lock_guard<std::mutex> lock(results_mutex);
                results.push_back(result);
            });
        }
        
        // Wait for current batch to complete
        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        threads.clear();
    }
    
    // Store results
    results_ = results;
    
    return Status::SUCCESS;
}

Status TestRunner::RunTestsSequentially(const std::vector<TestSpec>& specs) {
    PROFILER_SCOPED_EVENT(0, "run_tests_sequential");
    
    std::vector<TestResult> results;
    
    for (const auto& spec : specs) {
        TestResult result = RunSingleTest(spec);
        results.push_back(result);
        
        // Stop on first failure if configured
        if (config_.stop_on_first_failure && !result.passed) {
            break;
        }
    }
    
    // Store results
    results_ = results;
    
    return Status::SUCCESS;
}

TestResult TestRunner::RunSingleTest(const TestSpec& spec) {
    PROFILER_SCOPED_EVENT(0, "run_single_test");
    
    TestResult result;
    result.test_name = spec.GetConfig().test_suite_name;
    result.module_name = spec.GetConfig().module_name;
    
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        // Run test based on type
        Status status = RunTestByType(spec, result);
        result.passed = (status == Status::SUCCESS);
        
        if (!result.passed) {
            result.error_message = "Test execution failed";
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.error_message = "Exception: " + std::string(e.what());
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Collect metrics
    CollectTestMetrics(result);
    
    return result;
}

Status TestRunner::RunTestByType(const TestSpec& spec, TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "run_test_by_type");
    
    const std::string& test_type = spec.GetConfig().test_type;
    
    if (test_type == "unit") {
        return RunUnitTest(spec, result);
    } else if (test_type == "integration") {
        return RunIntegrationTest(spec, result);
    } else if (test_type == "performance") {
        return RunPerformanceTest(spec, result);
    } else if (test_type == "property") {
        return RunPropertyTest(spec, result);
    } else if (test_type == "fuzz") {
        return RunFuzzTest(spec, result);
    } else {
        return RunGenericTest(spec, result);
    }
}

Status TestRunner::RunUnitTest(const TestSpec& spec, [[maybe_unused]] TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "run_unit_test");
    
    // Generate test executable
    std::string test_executable = GenerateTestExecutable(spec);
    if (test_executable.empty()) {
        return Status::FAILURE;
    }
    
    // Run test executable
    int exit_code = ExecuteTestExecutable(test_executable);
    
    // Cleanup
    std::remove(test_executable.c_str());
    
    return (exit_code == 0) ? Status::SUCCESS : Status::FAILURE;
}

Status TestRunner::RunIntegrationTest(const TestSpec& spec, TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "run_integration_test");
    
    // Setup integration test environment
    Status status = SetupIntegrationTestEnvironment(spec);
    if (status != Status::SUCCESS) {
        return status;
    }
    
    // Run integration test
    status = RunUnitTest(spec, result);
    
    // Teardown integration test environment
    TeardownIntegrationTestEnvironment(spec);
    
    return status;
}

Status TestRunner::RunPerformanceTest(const TestSpec& spec, TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "run_performance_test");
    
    // Run test multiple times for performance measurement
    std::vector<std::chrono::milliseconds> durations;
    
    for (uint32_t i = 0; i < config_.performance_test_iterations; ++i) {
        auto start_time = std::chrono::steady_clock::now();
        
        Status status = RunUnitTest(spec, result);
        if (status != Status::SUCCESS) {
            return status;
        }
        
        auto end_time = std::chrono::steady_clock::now();
        durations.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
    
    // Calculate performance metrics
    CalculatePerformanceMetrics(durations, result);
    
    return Status::SUCCESS;
}

Status TestRunner::RunPropertyTest(const TestSpec& spec, TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "run_property_test");
    
    // TODO: Implement property-based testing
    // For now, run as unit test
    return RunUnitTest(spec, result);
}

Status TestRunner::RunFuzzTest(const TestSpec& spec, TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "run_fuzz_test");
    
    // TODO: Implement fuzz testing
    // For now, run as unit test
    return RunUnitTest(spec, result);
}

Status TestRunner::RunGenericTest(const TestSpec& spec, TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "run_generic_test");
    
    // Run as unit test by default
    return RunUnitTest(spec, result);
}

std::string TestRunner::GenerateTestExecutable(const TestSpec& spec) {
    PROFILER_SCOPED_EVENT(0, "generate_test_executable");
    
    // Create temporary test file
    std::string test_file = CreateTempTestFile(spec);
    if (test_file.empty()) {
        return "";
    }
    
    // Compile test executable
    std::string executable = CompileTestFile(test_file);
    
    // Cleanup test file
    std::remove(test_file.c_str());
    
    return executable;
}

std::string TestRunner::CreateTempTestFile(const TestSpec& spec) {
    PROFILER_SCOPED_EVENT(0, "create_temp_test_file");
    
    // Create temporary file
    char temp_path[] = "/tmp/edge_ai_test_XXXXXX.cpp";
    int fd = mkstemp(temp_path);
    if (fd == -1) {
        return "";
    }
    
    // Generate test code
    std::string test_code = GenerateTestCode(spec);
    
    // Write test code to file
    ssize_t written = write(fd, test_code.c_str(), test_code.size());
    close(fd);
    
    if (written != static_cast<ssize_t>(test_code.size())) {
        std::remove(temp_path);
        return "";
    }
    
    return temp_path;
}

std::string TestRunner::GenerateTestCode(const TestSpec& spec) {
    PROFILER_SCOPED_EVENT(0, "generate_test_code");
    
    std::stringstream code;
    
    // Include headers
    code << "#include <gtest/gtest.h>\n";
    code << "#include <core/types.h>\n";
    code << "#include <profiling/profiler.h>\n";
    code << "\n";
    
    // Generate test class
    code << "class " << spec.GetConfig().test_suite_name << " : public ::testing::Test {\n";
    code << "protected:\n";
    code << "    void SetUp() override {\n";
    code << "        // Setup code\n";
    code << "    }\n";
    code << "    void TearDown() override {\n";
    code << "        // Teardown code\n";
    code << "    }\n";
    code << "};\n";
    code << "\n";
    
    // Generate test cases
    for (const auto& scenario : spec.GetScenarios()) {
        code << "TEST_F(" << spec.GetConfig().test_suite_name << ", " << scenario.GetName() << ") {\n";
        code << "    PROFILER_SCOPED_EVENT(0, \"" << scenario.GetName() << "\");\n";
        code << "    \n";
        code << "    // Given steps\n";
        for (const auto& step : scenario.GetGivenSteps()) {
            code << "    // " << step << "\n";
        }
        code << "    \n";
        code << "    // When steps\n";
        for (const auto& step : scenario.GetWhenSteps()) {
            code << "    // " << step << "\n";
        }
        code << "    \n";
        code << "    // Then steps\n";
        for (const auto& step : scenario.GetThenSteps()) {
            code << "    // " << step << "\n";
        }
        code << "    \n";
        code << "    EXPECT_TRUE(true); // Placeholder assertion\n";
        code << "}\n";
        code << "\n";
    }
    
    // Generate main function
    code << "int main(int argc, char** argv) {\n";
    code << "    ::testing::InitGoogleTest(&argc, argv);\n";
    code << "    return RUN_ALL_TESTS();\n";
    code << "}\n";
    
    return code.str();
}

std::string TestRunner::CompileTestFile(const std::string& test_file) {
    PROFILER_SCOPED_EVENT(0, "compile_test_file");
    
    // Create executable path
    char executable_path[] = "/tmp/edge_ai_test_exec_XXXXXX";
    int fd = mkstemp(executable_path);
    if (fd == -1) {
        return "";
    }
    close(fd);
    
    // Build compile command
    std::stringstream cmd;
    cmd << "g++ -std=c++20 -I" << environment_.include_directories[0] << " ";
    cmd << "-L" << environment_.library_directories[0] << " ";
    cmd << test_file << " -o " << executable_path << " ";
    cmd << "-lgtest -lgtest_main -lpthread ";
    cmd << environment_.libraries[0] << " ";
    cmd << "2>&1";
    
    // Execute compile command
    int compile_result = std::system(cmd.str().c_str());
    if (compile_result != 0) {
        std::remove(executable_path);
        return "";
    }
    
    return executable_path;
}

int TestRunner::ExecuteTestExecutable(const std::string& executable) {
    PROFILER_SCOPED_EVENT(0, "execute_test_executable");
    
    // Fork process
    pid_t pid = fork();
    if (pid == -1) {
        return -1;
    }
    
    if (pid == 0) {
        // Child process
        execl(executable.c_str(), executable.c_str(), nullptr);
        exit(1); // Should not reach here
    } else {
        // Parent process
        int status;
        pid_t result = waitpid(pid, &status, 0);
        
        if (result == -1) {
            return -1;
        }
        
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return -1;
        }
    }
}

Status TestRunner::SetupTestEnvironment() {
    PROFILER_SCOPED_EVENT(0, "setup_test_environment");
    
    // Set environment variables
    for (const auto& var : environment_.environment_variables) {
        setenv(var.first.c_str(), var.second.c_str(), 1);
    }
    
    // Create temporary directories
    for (const auto& dir : environment_.temp_directories) {
        std::filesystem::create_directories(dir);
    }
    
    return Status::SUCCESS;
}

Status TestRunner::TeardownTestEnvironment() {
    PROFILER_SCOPED_EVENT(0, "teardown_test_environment");
    
    // Cleanup environment variables
    for (const auto& var : environment_.environment_variables) {
        unsetenv(var.first.c_str());
    }
    
    // Cleanup temporary directories
    for (const auto& dir : environment_.temp_directories) {
        std::filesystem::remove_all(dir);
    }
    
    return Status::SUCCESS;
}

Status TestRunner::SetupIntegrationTestEnvironment([[maybe_unused]] const TestSpec& spec) {
    PROFILER_SCOPED_EVENT(0, "setup_integration_test_environment");
    
    // TODO: Implement integration test environment setup
    // For now, just return success
    return Status::SUCCESS;
}

Status TestRunner::TeardownIntegrationTestEnvironment([[maybe_unused]] const TestSpec& spec) {
    PROFILER_SCOPED_EVENT(0, "teardown_integration_test_environment");
    
    // TODO: Implement integration test environment teardown
    // For now, just return success
    return Status::SUCCESS;
}

void TestRunner::CollectTestMetrics(TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "collect_test_metrics");
    
    // TODO: Implement actual metrics collection
    // For now, set placeholder values
    result.cpu_usage_percent = 0.0;
    result.memory_usage_mb = 0.0;
    result.network_usage_mbps = 0.0;
    result.code_coverage_percent = 0.0;
    result.stability_score = 1.0;
}

void TestRunner::CalculatePerformanceMetrics(const std::vector<std::chrono::milliseconds>& durations, 
                                           TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "calculate_performance_metrics");
    
    if (durations.empty()) {
        return;
    }
    
    // Calculate average duration
    double total_ms = 0.0;
    for (const auto& duration : durations) {
        total_ms += duration.count();
    }
    double avg_duration = total_ms / durations.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (const auto& duration : durations) {
        double diff = duration.count() - avg_duration;
        variance += diff * diff;
    }
    double std_dev = std::sqrt(variance / durations.size());
    
    // Calculate stability score (lower std dev = higher stability)
    result.stability_score = std::max(0.0, 1.0 - (std_dev / avg_duration));
    
    // Store performance metadata
    result.metadata["avg_duration_ms"] = std::to_string(avg_duration);
    result.metadata["std_dev_ms"] = std::to_string(std_dev);
    result.metadata["min_duration_ms"] = std::to_string(std::min_element(durations.begin(), durations.end())->count());
    result.metadata["max_duration_ms"] = std::to_string(std::max_element(durations.begin(), durations.end())->count());
    result.metadata["iterations"] = std::to_string(durations.size());
}

std::vector<TestResult> TestRunner::GetResults() const {
    return results_;
}

RunnerStatistics TestRunner::GetStatistics() const {
    RunnerStatistics stats;
    
    stats.total_tests = results_.size();
    
    for (const auto& result : results_) {
        if (result.passed) {
            stats.passed_tests++;
        } else {
            stats.failed_tests++;
        }
        
        if (result.flaky) {
            stats.flaky_tests++;
        }
        
        stats.total_duration += result.duration;
        stats.avg_coverage_percent += result.code_coverage_percent;
        stats.avg_stability_score += result.stability_score;
    }
    
    if (stats.total_tests > 0) {
        stats.avg_duration_ms = static_cast<double>(stats.total_duration.count()) / stats.total_tests;
        stats.success_rate = static_cast<double>(stats.passed_tests) / stats.total_tests;
        stats.avg_coverage_percent /= stats.total_tests;
        stats.avg_stability_score /= stats.total_tests;
    }
    
    return stats;
}

void TestRunner::ClearResults() {
    results_.clear();
}

Status TestRunner::SetMaxParallelTests(uint32_t max_parallel) {
    max_parallel_tests_ = max_parallel;
    return Status::SUCCESS;
}

Status TestRunner::SetTimeout(uint32_t timeout_seconds) {
    timeout_seconds_ = timeout_seconds;
    return Status::SUCCESS;
}

} // namespace testing
} // namespace edge_ai
