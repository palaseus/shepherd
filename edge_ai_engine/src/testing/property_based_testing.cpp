#include <testing/property_based_testing.h>
#include <profiling/profiler.h>
#include <algorithm>
#include <random>
#include <iostream>
#include <filesystem>

namespace edge_ai {
namespace testing {

// IntGenerator Implementation
IntGenerator::IntGenerator(int32_t min, int32_t max) 
    : min_(min), max_(max), dist_(min, max) {
}

int32_t IntGenerator::generate(std::mt19937& rng) {
    return dist_(rng);
}

int32_t IntGenerator::shrink(const int32_t& value, [[maybe_unused]] std::mt19937& rng) {
    if (value == min_) return value;
    
    // Try to shrink towards zero
    if (value > 0) {
        return std::max(min_, value / 2);
    } else if (value < 0) {
        return std::min(max_, value / 2);
    }
    
    return value;
}

bool IntGenerator::is_valid(const int32_t& value) {
    return value >= min_ && value <= max_;
}

// UIntGenerator Implementation
UIntGenerator::UIntGenerator(uint32_t min, uint32_t max) 
    : min_(min), max_(max), dist_(min, max) {
}

uint32_t UIntGenerator::generate(std::mt19937& rng) {
    return dist_(rng);
}

uint32_t UIntGenerator::shrink(const uint32_t& value, [[maybe_unused]] std::mt19937& rng) {
    if (value == min_) return value;
    
    // Try to shrink towards zero
    return std::max(min_, value / 2);
}

bool UIntGenerator::is_valid(const uint32_t& value) {
    return value >= min_ && value <= max_;
}

// StringGenerator Implementation
StringGenerator::StringGenerator(size_t min_length, size_t max_length) 
    : min_length_(min_length), max_length_(max_length),
      length_dist_(min_length, max_length),
      char_dist_(32, 126) { // Printable ASCII characters
}

std::string StringGenerator::generate(std::mt19937& rng) {
    size_t length = length_dist_(rng);
    std::string result;
    result.reserve(length);
    
    for (size_t i = 0; i < length; ++i) {
        result += static_cast<char>(char_dist_(rng));
    }
    
    return result;
}

std::string StringGenerator::shrink(const std::string& value, [[maybe_unused]] std::mt19937& rng) {
    if (value.length() <= min_length_) return value;
    
    // Try to shrink by removing characters
    std::string result = value;
    size_t remove_count = std::min(result.length() - min_length_, result.length() / 2);
    
    for (size_t i = 0; i < remove_count; ++i) {
        size_t pos = rng() % result.length();
        result.erase(pos, 1);
    }
    
    return result;
}

bool StringGenerator::is_valid(const std::string& value) {
    return value.length() >= min_length_ && value.length() <= max_length_;
}

// VectorGenerator Implementation
VectorGenerator::VectorGenerator(size_t min_size, size_t max_size) 
    : min_size_(min_size), max_size_(max_size),
      size_dist_(min_size, max_size),
      byte_dist_(0, 255) {
}

std::vector<uint8_t> VectorGenerator::generate(std::mt19937& rng) {
    size_t size = size_dist_(rng);
    std::vector<uint8_t> result;
    result.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        result.push_back(static_cast<uint8_t>(byte_dist_(rng)));
    }
    
    return result;
}

std::vector<uint8_t> VectorGenerator::shrink(const std::vector<uint8_t>& value, [[maybe_unused]] std::mt19937& rng) {
    if (value.size() <= min_size_) return value;
    
    // Try to shrink by reducing size
    std::vector<uint8_t> result = value;
    size_t remove_count = std::min(result.size() - min_size_, result.size() / 2);
    
    for (size_t i = 0; i < remove_count; ++i) {
        size_t pos = rng() % result.size();
        result.erase(result.begin() + pos);
    }
    
    return result;
}

bool VectorGenerator::is_valid(const std::vector<uint8_t>& value) {
    return value.size() >= min_size_ && value.size() <= max_size_;
}

// PropertyBasedTestRunner Implementation
PropertyBasedTestRunner::PropertyBasedTestRunner() 
    : rng_(config_.seed) {
}

PropertyBasedTestRunner::~PropertyBasedTestRunner() = default;

Status PropertyBasedTestRunner::SetConfiguration(const PropertyBasedConfig& config) {
    config_ = config;
    rng_.seed(config_.seed);
    return Status::SUCCESS;
}

template<typename... Args>
void PropertyBasedTestRunner::RegisterProperty(const std::string& name, 
                                              std::function<bool(Args...)> property,
                                              PropertyGenerator<Args>*... generators) {
    properties_[name] = [this, name, property, generators...]() {
        auto result = RunSingleProperty(name, property, generators...);
        results_.push_back(result);
    };
}

Status PropertyBasedTestRunner::RunAllProperties() {
    PROFILER_SCOPED_EVENT(0, "run_all_properties");
    
    results_.clear();
    statistics_ = PropertyBasedStatistics();
    
    for (const auto& [name, property_func] : properties_) {
        statistics_.total_properties++;
        property_func();
    }
    
    CalculateStatistics();
    return Status::SUCCESS;
}

Status PropertyBasedTestRunner::RunProperty(const std::string& property_name) {
    PROFILER_SCOPED_EVENT(0, "run_single_property");
    
    auto it = properties_.find(property_name);
    if (it == properties_.end()) {
        std::cerr << "Property not found: " << property_name << std::endl;
        return Status::FAILURE;
    }
    
    results_.clear();
    statistics_ = PropertyBasedStatistics();
    
    statistics_.total_properties++;
    it->second();
    
    CalculateStatistics();
    return Status::SUCCESS;
}

Status PropertyBasedTestRunner::RunProperties(const std::vector<std::string>& property_names) {
    PROFILER_SCOPED_EVENT(0, "run_multiple_properties");
    
    results_.clear();
    statistics_ = PropertyBasedStatistics();
    
    for (const std::string& name : property_names) {
        auto it = properties_.find(name);
        if (it != properties_.end()) {
            statistics_.total_properties++;
            it->second();
        }
    }
    
    CalculateStatistics();
    return Status::SUCCESS;
}

PropertyBasedStatistics PropertyBasedTestRunner::GetStatistics() const {
    return statistics_;
}

std::vector<PropertyTestResult> PropertyBasedTestRunner::GetResults() const {
    return results_;
}

Status PropertyBasedTestRunner::GenerateReport(const std::string& output_file, const std::string& format) {
    PROFILER_SCOPED_EVENT(0, "generate_property_report");
    
    // TODO: Implement comprehensive report generation
    std::cout << "Generating property-based test report to " << output_file 
              << " in " << format << " format (placeholder)." << std::endl;
    
    return Status::SUCCESS;
}

Status PropertyBasedTestRunner::ExportResults(const std::string& output_file, const std::string& format) {
    PROFILER_SCOPED_EVENT(0, "export_property_results");
    
    // TODO: Implement results export
    std::cout << "Exporting property-based test results to " << output_file 
              << " in " << format << " format (placeholder)." << std::endl;
    
    return Status::SUCCESS;
}

void PropertyBasedTestRunner::ClearResults() {
    results_.clear();
    statistics_ = PropertyBasedStatistics();
}

void PropertyBasedTestRunner::SetSeed(uint32_t seed) {
    config_.seed = seed;
    rng_.seed(seed);
}

void PropertyBasedTestRunner::CalculateStatistics() {
    statistics_.passed_properties = 0;
    statistics_.failed_properties = 0;
    statistics_.total_iterations = 0;
    statistics_.total_duration = std::chrono::milliseconds(0);
    
    for (const auto& result : results_) {
        statistics_.total_iterations += result.iterations_run;
        statistics_.total_duration += result.duration;
        
        if (result.passed) {
            statistics_.passed_properties++;
        } else {
            statistics_.failed_properties++;
            statistics_.failure_counts[result.property_name]++;
        }
    }
    
    if (statistics_.total_properties > 0) {
        statistics_.success_rate = static_cast<double>(statistics_.passed_properties) 
                                 / statistics_.total_properties * 100.0;
    }
}

template<typename... Args>
PropertyTestResult PropertyBasedTestRunner::RunSingleProperty(const std::string& name,
                                                             std::function<bool(Args...)> property,
                                                             PropertyGenerator<Args>*... generators) {
    PROFILER_SCOPED_EVENT(0, "run_single_property");
    
    PropertyTestResult result;
    result.property_name = name;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        for (uint32_t i = 0; i < config_.num_iterations; ++i) {
            result.iterations_run++;
            
            // Generate test arguments
            auto args = std::make_tuple(generators->generate(rng_)...);
            
            // Check if arguments are valid
            bool valid_args = (generators->is_valid(std::get<0>(args)) && ...);
            if (!valid_args) {
                continue;
            }
            
            // Execute property
            bool property_result = std::apply(property, args);
            
            if (!property_result) {
                result.failures_found++;
                result.failure_message = "Property failed on iteration " + std::to_string(i);
                
                // Try to shrink the counterexample
                if (config_.enable_shrinking) {
                    auto shrunk_args = std::make_tuple(generators->shrink(std::get<0>(args), rng_)...);
                    
                    for (uint32_t shrink_attempt = 0; shrink_attempt < config_.max_shrink_attempts; ++shrink_attempt) {
                        bool shrunk_valid = (generators->is_valid(std::get<0>(shrunk_args)) && ...);
                        if (!shrunk_valid) break;
                        
                        bool shrunk_result = std::apply(property, shrunk_args);
                        if (shrunk_result) break; // Shrinking succeeded
                        
                        // Continue shrinking
                        shrunk_args = std::make_tuple(generators->shrink(std::get<0>(shrunk_args), rng_)...);
                        result.shrink_history.push_back("Shrink attempt " + std::to_string(shrink_attempt));
                    }
                }
                
                result.passed = false;
                break;
            }
        }
        
        if (result.failures_found == 0) {
            result.passed = true;
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.failure_message = "Exception: " + std::string(e.what());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    return result;
}

// FuzzTestRunner Implementation
FuzzTestRunner::FuzzTestRunner() 
    : rng_(config_.seed) {
    LoadCorpus();
}

FuzzTestRunner::~FuzzTestRunner() {
    SaveCorpus();
}

Status FuzzTestRunner::SetConfiguration(const FuzzTestConfig& config) {
    config_ = config;
    rng_.seed(config_.seed);
    return Status::SUCCESS;
}

void FuzzTestRunner::RegisterFuzzTest(const std::string& name, 
                                     std::function<Status(const std::vector<uint8_t>&)> test_function) {
    fuzz_tests_[name] = test_function;
}

Status FuzzTestRunner::RunAllFuzzTests() {
    PROFILER_SCOPED_EVENT(0, "run_all_fuzz_tests");
    
    results_.clear();
    statistics_ = FuzzTestStatistics();
    
    for (const auto& [name, test_func] : fuzz_tests_) {
        statistics_.total_tests++;
        auto result = RunSingleFuzzTest(name, test_func);
        results_.push_back(result);
    }
    
    CalculateStatistics();
    return Status::SUCCESS;
}

Status FuzzTestRunner::RunFuzzTest(const std::string& test_name) {
    PROFILER_SCOPED_EVENT(0, "run_single_fuzz_test");
    
    auto it = fuzz_tests_.find(test_name);
    if (it == fuzz_tests_.end()) {
        std::cerr << "Fuzz test not found: " << test_name << std::endl;
        return Status::FAILURE;
    }
    
    results_.clear();
    statistics_ = FuzzTestStatistics();
    
    statistics_.total_tests++;
    auto result = RunSingleFuzzTest(test_name, it->second);
    results_.push_back(result);
    
    CalculateStatistics();
    return Status::SUCCESS;
}

Status FuzzTestRunner::RunFuzzTests(const std::vector<std::string>& test_names) {
    PROFILER_SCOPED_EVENT(0, "run_multiple_fuzz_tests");
    
    results_.clear();
    statistics_ = FuzzTestStatistics();
    
    for (const std::string& name : test_names) {
        auto it = fuzz_tests_.find(name);
        if (it != fuzz_tests_.end()) {
            statistics_.total_tests++;
            auto result = RunSingleFuzzTest(name, it->second);
            results_.push_back(result);
        }
    }
    
    CalculateStatistics();
    return Status::SUCCESS;
}

FuzzTestStatistics FuzzTestRunner::GetStatistics() const {
    return statistics_;
}

std::vector<FuzzTestResult> FuzzTestRunner::GetResults() const {
    return results_;
}

Status FuzzTestRunner::GenerateReport(const std::string& output_file, const std::string& format) {
    PROFILER_SCOPED_EVENT(0, "generate_fuzz_report");
    
    // TODO: Implement comprehensive report generation
    std::cout << "Generating fuzz test report to " << output_file 
              << " in " << format << " format (placeholder)." << std::endl;
    
    return Status::SUCCESS;
}

Status FuzzTestRunner::ExportResults(const std::string& output_file, const std::string& format) {
    PROFILER_SCOPED_EVENT(0, "export_fuzz_results");
    
    // TODO: Implement results export
    std::cout << "Exporting fuzz test results to " << output_file 
              << " in " << format << " format (placeholder)." << std::endl;
    
    return Status::SUCCESS;
}

void FuzzTestRunner::ClearResults() {
    results_.clear();
    statistics_ = FuzzTestStatistics();
}

void FuzzTestRunner::SetSeed(uint32_t seed) {
    config_.seed = seed;
    rng_.seed(seed);
}

void FuzzTestRunner::AddToCorpus(const std::string& test_name, const std::vector<uint8_t>& input) {
    corpus_[test_name].push_back(input);
}

void FuzzTestRunner::CalculateStatistics() {
    statistics_.passed_tests = 0;
    statistics_.failed_tests = 0;
    statistics_.total_iterations = 0;
    statistics_.total_crashes = 0;
    statistics_.total_hangs = 0;
    statistics_.total_duration = std::chrono::milliseconds(0);
    
    for (const auto& result : results_) {
        statistics_.total_iterations += result.iterations_run;
        statistics_.total_duration += result.duration;
        statistics_.total_crashes += result.crashes_found;
        statistics_.total_hangs += result.hangs_found;
        
        if (result.passed) {
            statistics_.passed_tests++;
        } else {
            statistics_.failed_tests++;
            statistics_.crash_counts[result.test_name]++;
        }
    }
    
    if (statistics_.total_tests > 0) {
        statistics_.success_rate = static_cast<double>(statistics_.passed_tests) 
                                 / statistics_.total_tests * 100.0;
    }
}

FuzzTestResult FuzzTestRunner::RunSingleFuzzTest(const std::string& name,
                                                std::function<Status(const std::vector<uint8_t>&)> test_function) {
    PROFILER_SCOPED_EVENT(0, "run_single_fuzz_test");
    
    FuzzTestResult result;
    result.test_name = name;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        for (uint32_t i = 0; i < config_.num_iterations; ++i) {
            result.iterations_run++;
            
            std::vector<uint8_t> input;
            
            // Use corpus-based fuzzing if enabled and corpus exists
            if (config_.enable_corpus_based_fuzzing && !corpus_[name].empty()) {
                if (rng_() % 2 == 0) {
                    // Use corpus input
                    const auto& corpus_input = corpus_[name][rng_() % corpus_[name].size()];
                    input = corpus_input;
                } else {
                    // Generate random input
                    input = GenerateRandomInput();
                }
            } else {
                // Generate random input
                input = GenerateRandomInput();
            }
            
            // Mutate input if mutation fuzzing is enabled
            if (config_.enable_mutation_fuzzing && !input.empty()) {
                input = MutateInput(input);
            }
            
            // Execute test
            Status test_result = test_function(input);
            
            if (test_result != Status::SUCCESS) {
                result.crashes_found++;
                result.crash_input = std::string(input.begin(), input.end());
                result.crash_message = "Test failed with status: " + std::to_string(static_cast<int>(test_result));
                result.unique_crashes.push_back(result.crash_message);
                result.passed = false;
                break;
            }
        }
        
        if (result.crashes_found == 0) {
            result.passed = true;
        }
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.crashes_found++;
        result.crash_message = "Exception: " + std::string(e.what());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    return result;
}

std::vector<uint8_t> FuzzTestRunner::GenerateRandomInput() {
    std::uniform_int_distribution<size_t> size_dist(0, config_.max_input_size);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    
    size_t size = size_dist(rng_);
    std::vector<uint8_t> input;
    input.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        input.push_back(byte_dist(rng_));
    }
    
    return input;
}

std::vector<uint8_t> FuzzTestRunner::MutateInput(const std::vector<uint8_t>& input) {
    if (input.empty()) return input;
    
    std::vector<uint8_t> mutated = input;
    std::uniform_int_distribution<size_t> pos_dist(0, mutated.size() - 1);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_int_distribution<int> mutation_type(0, 3);
    
    int mutation = mutation_type(rng_);
    
    switch (mutation) {
        case 0: // Flip bit
            {
                size_t pos = pos_dist(rng_);
                mutated[pos] ^= (1 << (rng_() % 8));
            }
            break;
        case 1: // Replace byte
            {
                size_t pos = pos_dist(rng_);
                mutated[pos] = byte_dist(rng_);
            }
            break;
        case 2: // Insert byte
            {
                size_t pos = pos_dist(rng_);
                mutated.insert(mutated.begin() + pos, byte_dist(rng_));
            }
            break;
        case 3: // Delete byte
            {
                size_t pos = pos_dist(rng_);
                mutated.erase(mutated.begin() + pos);
            }
            break;
    }
    
    return mutated;
}

void FuzzTestRunner::LoadCorpus() {
    // TODO: Implement corpus loading from files
    std::cout << "Loading fuzz corpus (placeholder)." << std::endl;
}

void FuzzTestRunner::SaveCorpus() {
    // TODO: Implement corpus saving to files
    std::cout << "Saving fuzz corpus (placeholder)." << std::endl;
}

// Global instances
static std::unique_ptr<PropertyBasedTestRunner> g_property_runner;
static std::unique_ptr<FuzzTestRunner> g_fuzz_runner;

// Global accessor functions
PropertyBasedTestRunner* GetPropertyBasedTestRunner() {
    if (!g_property_runner) {
        g_property_runner = std::make_unique<PropertyBasedTestRunner>();
    }
    return g_property_runner.get();
}

FuzzTestRunner* GetFuzzTestRunner() {
    if (!g_fuzz_runner) {
        g_fuzz_runner = std::make_unique<FuzzTestRunner>();
    }
    return g_fuzz_runner.get();
}

} // namespace testing
} // namespace edge_ai
