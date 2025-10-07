#ifndef EDGE_AI_ENGINE_PROPERTY_BASED_TESTING_H
#define EDGE_AI_ENGINE_PROPERTY_BASED_TESTING_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <random>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <type_traits>

#include <core/types.h>
#include <testing/test_common.h>

namespace edge_ai {
namespace testing {

// Forward declarations
class PropertyBasedTestRunner;
class FuzzTestRunner;

// Property-based test configuration
struct PropertyBasedConfig {
    uint32_t num_iterations = 1000;
    uint32_t max_shrink_attempts = 100;
    uint32_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    bool enable_shrinking = true;
    bool enable_parallel_execution = false;
    uint32_t max_parallel_tests = 4;
    std::string output_directory = "property_test_reports/";
    std::map<std::string, std::string> custom_parameters;
};

// Fuzz test configuration
struct FuzzTestConfig {
    uint32_t num_iterations = 10000;
    uint32_t max_input_size = 1024;
    uint32_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    bool enable_corpus_based_fuzzing = true;
    bool enable_mutation_fuzzing = true;
    std::string corpus_directory = "fuzz_corpus/";
    std::string output_directory = "fuzz_reports/";
    std::map<std::string, std::string> custom_parameters;
};

// Property test result
struct PropertyTestResult {
    std::string property_name;
    bool passed = false;
    uint32_t iterations_run = 0;
    uint32_t failures_found = 0;
    std::chrono::milliseconds duration{0};
    std::string failure_message;
    std::map<std::string, std::string> counterexample;
    std::vector<std::string> shrink_history;
};

// Fuzz test result
struct FuzzTestResult {
    std::string test_name;
    bool passed = false;
    uint32_t iterations_run = 0;
    uint32_t crashes_found = 0;
    uint32_t hangs_found = 0;
    std::chrono::milliseconds duration{0};
    std::string crash_input;
    std::string crash_message;
    std::vector<std::string> unique_crashes;
};

// Property-based test statistics
struct PropertyBasedStatistics {
    uint32_t total_properties = 0;
    uint32_t passed_properties = 0;
    uint32_t failed_properties = 0;
    uint32_t total_iterations = 0;
    double success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    std::map<std::string, uint32_t> failure_counts;
};

// Fuzz test statistics
struct FuzzTestStatistics {
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;
    uint32_t failed_tests = 0;
    uint32_t total_iterations = 0;
    uint32_t total_crashes = 0;
    uint32_t total_hangs = 0;
    double success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    std::map<std::string, uint32_t> crash_counts;
};

// Property generator interface
template<typename T>
class PropertyGenerator {
public:
    virtual ~PropertyGenerator() = default;
    virtual T generate(std::mt19937& rng) = 0;
    virtual T shrink(const T& value, std::mt19937& rng) = 0;
    virtual bool is_valid(const T& value) = 0;
};

// Built-in generators
class IntGenerator : public PropertyGenerator<int32_t> {
public:
    IntGenerator(int32_t min = std::numeric_limits<int32_t>::min(), 
                 int32_t max = std::numeric_limits<int32_t>::max());
    
    int32_t generate(std::mt19937& rng) override;
    int32_t shrink(const int32_t& value, std::mt19937& rng) override;
    bool is_valid(const int32_t& value) override;

private:
    int32_t min_, max_;
    std::uniform_int_distribution<int32_t> dist_;
};

class UIntGenerator : public PropertyGenerator<uint32_t> {
public:
    UIntGenerator(uint32_t min = 0, uint32_t max = std::numeric_limits<uint32_t>::max());
    
    uint32_t generate(std::mt19937& rng) override;
    uint32_t shrink(const uint32_t& value, std::mt19937& rng) override;
    bool is_valid(const uint32_t& value) override;

private:
    uint32_t min_, max_;
    std::uniform_int_distribution<uint32_t> dist_;
};

class StringGenerator : public PropertyGenerator<std::string> {
public:
    StringGenerator(size_t min_length = 0, size_t max_length = 100);
    
    std::string generate(std::mt19937& rng) override;
    std::string shrink(const std::string& value, std::mt19937& rng) override;
    bool is_valid(const std::string& value) override;

private:
    size_t min_length_, max_length_;
    std::uniform_int_distribution<size_t> length_dist_;
    std::uniform_int_distribution<char> char_dist_;
};

class VectorGenerator : public PropertyGenerator<std::vector<uint8_t>> {
public:
    VectorGenerator(size_t min_size = 0, size_t max_size = 1024);
    
    std::vector<uint8_t> generate(std::mt19937& rng) override;
    std::vector<uint8_t> shrink(const std::vector<uint8_t>& value, std::mt19937& rng) override;
    bool is_valid(const std::vector<uint8_t>& value) override;

private:
    size_t min_size_, max_size_;
    std::uniform_int_distribution<size_t> size_dist_;
    std::uniform_int_distribution<uint8_t> byte_dist_;
};

// Property-based test runner
class PropertyBasedTestRunner {
public:
    PropertyBasedTestRunner();
    ~PropertyBasedTestRunner();

    // Configuration
    Status SetConfiguration(const PropertyBasedConfig& config);

    // Property registration
    template<typename... Args>
    void RegisterProperty(const std::string& name, 
                         std::function<bool(Args...)> property,
                         PropertyGenerator<Args>*... generators);

    // Test execution
    Status RunAllProperties();
    Status RunProperty(const std::string& property_name);
    Status RunProperties(const std::vector<std::string>& property_names);

    // Results and reporting
    PropertyBasedStatistics GetStatistics() const;
    std::vector<PropertyTestResult> GetResults() const;
    Status GenerateReport(const std::string& output_file, const std::string& format = "html");
    Status ExportResults(const std::string& output_file, const std::string& format = "json");

    // Utility methods
    void ClearResults();
    void SetSeed(uint32_t seed);

private:
    PropertyBasedConfig config_;
    std::mt19937 rng_;
    std::map<std::string, std::function<void()>> properties_;
    std::vector<PropertyTestResult> results_;
    PropertyBasedStatistics statistics_;

    void CalculateStatistics();
    template<typename... Args>
    PropertyTestResult RunSingleProperty(const std::string& name,
                                        std::function<bool(Args...)> property,
                                        PropertyGenerator<Args>*... generators);
};

// Fuzz test runner
class FuzzTestRunner {
public:
    FuzzTestRunner();
    ~FuzzTestRunner();

    // Configuration
    Status SetConfiguration(const FuzzTestConfig& config);

    // Fuzz test registration
    void RegisterFuzzTest(const std::string& name, 
                         std::function<Status(const std::vector<uint8_t>&)> test_function);

    // Test execution
    Status RunAllFuzzTests();
    Status RunFuzzTest(const std::string& test_name);
    Status RunFuzzTests(const std::vector<std::string>& test_names);

    // Results and reporting
    FuzzTestStatistics GetStatistics() const;
    std::vector<FuzzTestResult> GetResults() const;
    Status GenerateReport(const std::string& output_file, const std::string& format = "html");
    Status ExportResults(const std::string& output_file, const std::string& format = "json");

    // Utility methods
    void ClearResults();
    void SetSeed(uint32_t seed);
    void AddToCorpus(const std::string& test_name, const std::vector<uint8_t>& input);

private:
    FuzzTestConfig config_;
    std::mt19937 rng_;
    std::map<std::string, std::function<Status(const std::vector<uint8_t>&)>> fuzz_tests_;
    std::map<std::string, std::vector<std::vector<uint8_t>>> corpus_;
    std::vector<FuzzTestResult> results_;
    FuzzTestStatistics statistics_;

    void CalculateStatistics();
    FuzzTestResult RunSingleFuzzTest(const std::string& name,
                                    std::function<Status(const std::vector<uint8_t>&)> test_function);
    std::vector<uint8_t> GenerateRandomInput();
    std::vector<uint8_t> MutateInput(const std::vector<uint8_t>& input);
    void LoadCorpus();
    void SaveCorpus();
};

// Property-based testing macros
#define PROPERTY(name) \
    static auto property_##name = []() { \
        extern PropertyBasedTestRunner* GetPropertyBasedTestRunner(); \
        GetPropertyBasedTestRunner()->RegisterProperty(#name, \
            [](std::mt19937& rng) -> bool {

#define END_PROPERTY \
            return true; \
        }); \
        return 0; \
    }(); \
    (void)property_##name

// Fuzz testing macros
#define FUZZ_TEST(name) \
    static auto fuzz_test_##name = []() { \
        extern FuzzTestManager* GetFuzzTestRunner(); \
        GetFuzzTestRunner()->RegisterFuzzTest(#name, \
            [](const std::vector<uint8_t>& input) -> Status {

#define END_FUZZ_TEST \
            return Status::SUCCESS; \
        }, __FILE__, __LINE__); \
        return 0; \
    }(); \
    (void)fuzz_test_##name

// Global accessor functions
PropertyBasedTestRunner* GetPropertyBasedTestRunner();
FuzzTestRunner* GetFuzzTestRunner();

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_PROPERTY_BASED_TESTING_H
