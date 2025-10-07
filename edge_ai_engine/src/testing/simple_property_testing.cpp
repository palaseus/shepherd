#include <testing/simple_property_testing.h>
#include <profiling/profiler.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace edge_ai {
namespace testing {

SimplePropertyTestManager::SimplePropertyTestManager() {
    InitializeRNG();
}

SimplePropertyTestManager::~SimplePropertyTestManager() = default;

void SimplePropertyTestManager::InitializeRNG() {
    if (config_.seed == 0) {
        rng_.seed(std::random_device()());
    } else {
        rng_.seed(config_.seed);
    }
}

Status SimplePropertyTestManager::SetConfiguration(const SimplePropertyConfig& config) {
    config_ = config;
    InitializeRNG();
    return Status::SUCCESS;
}

void SimplePropertyTestManager::RegisterProperty(const std::string& name, std::function<bool(std::mt19937&)> property, const std::string& file, uint32_t line) {
    properties_.push_back({name, property, file, line});
}

Status SimplePropertyTestManager::RunAllProperties() {
    PROFILER_SCOPED_EVENT(0, "simple_property_run_all");
    results_.clear();
    
    std::cout << "Running " << properties_.size() << " property-based tests..." << std::endl;
    
    for (const auto& prop : properties_) {
        const std::string& name = prop.name;
        const auto& property = prop.implementation;
        
        std::cout << "Running property: " << name << std::endl;
        
        SimplePropertyResult result;
        result.name = name;
        result.runs = config_.num_runs;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool all_passed = true;
        for (uint32_t i = 0; i < config_.num_runs; ++i) {
            try {
                if (!property(rng_)) {
                    result.failures++;
                    all_passed = false;
                    if (result.failure_message.empty()) {
                        result.failure_message = "Property failed on run " + std::to_string(i + 1);
                    }
                }
            } catch (const std::exception& e) {
                result.failures++;
                all_passed = false;
                if (result.failure_message.empty()) {
                    result.failure_message = "Exception on run " + std::to_string(i + 1) + ": " + e.what();
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.passed = all_passed;
        
        results_.push_back(result);
        
        std::cout << "  " << (result.passed ? "PASSED" : "FAILED") 
                  << " (" << result.failures << "/" << result.runs << " failures)" << std::endl;
    }
    
    return Status::SUCCESS;
}

Status SimplePropertyTestManager::RunProperty(const std::string& name) {
    PROFILER_SCOPED_EVENT(0, "simple_property_run_single");
    
    auto it = std::find_if(properties_.begin(), properties_.end(),
                          [&name](const SimpleProperty& prop) { return prop.name == name; });
    if (it == properties_.end()) {
        std::cerr << "Property '" << name << "' not found" << std::endl;
        return Status::FAILURE;
    }
    
    const auto& property = it->implementation;
    
    std::cout << "Running property: " << name << std::endl;
    
    SimplePropertyResult result;
    result.name = name;
    result.runs = config_.num_runs;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bool all_passed = true;
    for (uint32_t i = 0; i < config_.num_runs; ++i) {
        try {
            if (!property(rng_)) {
                result.failures++;
                all_passed = false;
                if (result.failure_message.empty()) {
                    result.failure_message = "Property failed on run " + std::to_string(i + 1);
                }
            }
        } catch (const std::exception& e) {
            result.failures++;
            all_passed = false;
            if (result.failure_message.empty()) {
                result.failure_message = "Exception on run " + std::to_string(i + 1) + ": " + e.what();
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    result.passed = all_passed;
    
    results_.push_back(result);
    
    std::cout << "  " << (result.passed ? "PASSED" : "FAILED") 
              << " (" << result.failures << "/" << result.runs << " failures)" << std::endl;
    
    return Status::SUCCESS;
}

std::vector<SimplePropertyResult> SimplePropertyTestManager::GetResults() const {
    return results_;
}

void SimplePropertyTestManager::PrintSummary() const {
    std::cout << "\n=== Simple Property-Based Test Summary ===" << std::endl;
    
    uint32_t total_properties = results_.size();
    uint32_t passed_properties = 0;
    uint32_t total_runs = 0;
    uint32_t total_failures = 0;
    std::chrono::milliseconds total_duration{0};
    
    for (const auto& result : results_) {
        if (result.passed) {
            passed_properties++;
        }
        total_runs += result.runs;
        total_failures += result.failures;
        total_duration += result.duration;
    }
    
    std::cout << "Total Properties: " << total_properties << std::endl;
    std::cout << "Passed: " << passed_properties << std::endl;
    std::cout << "Failed: " << (total_properties - passed_properties) << std::endl;
    std::cout << "Total Runs: " << total_runs << std::endl;
    std::cout << "Total Failures: " << total_failures << std::endl;
    std::cout << "Success Rate: " << std::fixed << std::setprecision(2) 
              << (total_runs > 0 ? (100.0 * (total_runs - total_failures) / total_runs) : 0.0) << "%" << std::endl;
    std::cout << "Total Duration: " << total_duration.count() << "ms" << std::endl;
    
    if (total_failures > 0) {
        std::cout << "\nFailed Properties:" << std::endl;
        for (const auto& result : results_) {
            if (!result.passed) {
                std::cout << "  - " << result.name << ": " << result.failure_message << std::endl;
            }
        }
    }
}

Status SimplePropertyTestManager::GenerateReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "simple_property_generate_report");
    
    try {
        // Create output directory if it doesn't exist
        std::filesystem::path output_path(output_file);
        std::filesystem::create_directories(output_path.parent_path());
        
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return Status::FAILURE;
        }
        
        // Generate HTML report
        file << "<!DOCTYPE html>\n";
        file << "<html>\n<head>\n";
        file << "<title>Simple Property-Based Test Report</title>\n";
        file << "<style>\n";
        file << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
        file << "h1 { color: #333; }\n";
        file << "table { border-collapse: collapse; width: 100%; }\n";
        file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
        file << "th { background-color: #f2f2f2; }\n";
        file << ".passed { color: green; }\n";
        file << ".failed { color: red; }\n";
        file << "</style>\n";
        file << "</head>\n<body>\n";
        
        file << "<h1>Simple Property-Based Test Report</h1>\n";
        
        // Summary
        uint32_t total_properties = results_.size();
        uint32_t passed_properties = 0;
        uint32_t total_runs = 0;
        uint32_t total_failures = 0;
        
        for (const auto& result : results_) {
            if (result.passed) passed_properties++;
            total_runs += result.runs;
            total_failures += result.failures;
        }
        
        file << "<h2>Summary</h2>\n";
        file << "<p>Total Properties: " << total_properties << "</p>\n";
        file << "<p>Passed: " << passed_properties << "</p>\n";
        file << "<p>Failed: " << (total_properties - passed_properties) << "</p>\n";
        file << "<p>Total Runs: " << total_runs << "</p>\n";
        file << "<p>Total Failures: " << total_failures << "</p>\n";
        file << "<p>Success Rate: " << std::fixed << std::setprecision(2) 
             << (total_runs > 0 ? (100.0 * (total_runs - total_failures) / total_runs) : 0.0) << "%</p>\n";
        
        // Detailed results
        file << "<h2>Detailed Results</h2>\n";
        file << "<table>\n";
        file << "<tr><th>Property Name</th><th>Status</th><th>Runs</th><th>Failures</th><th>Duration (ms)</th><th>Message</th></tr>\n";
        
        for (const auto& result : results_) {
            file << "<tr>\n";
            file << "<td>" << result.name << "</td>\n";
            file << "<td class=\"" << (result.passed ? "passed" : "failed") << "\">" 
                 << (result.passed ? "PASSED" : "FAILED") << "</td>\n";
            file << "<td>" << result.runs << "</td>\n";
            file << "<td>" << result.failures << "</td>\n";
            file << "<td>" << result.duration.count() << "</td>\n";
            file << "<td>" << result.failure_message << "</td>\n";
            file << "</tr>\n";
        }
        
        file << "</table>\n";
        file << "</body>\n</html>\n";
        
        file.close();
        std::cout << "Report generated: " << output_file << std::endl;
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error generating report: " << e.what() << std::endl;
        return Status::FAILURE;
    }
}

// Global SimplePropertyTestManager instance
static std::unique_ptr<SimplePropertyTestManager> g_simple_property_test_manager;

// Global accessor function
SimplePropertyTestManager* GetSimplePropertyTestManager() {
    if (!g_simple_property_test_manager) {
        g_simple_property_test_manager = std::make_unique<SimplePropertyTestManager>();
    }
    return g_simple_property_test_manager.get();
}

} // namespace testing
} // namespace edge_ai
