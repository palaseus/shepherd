#include <testing/behavior_driven_testing.h>
#include <profiling/profiler.h>
#include <iostream>
#include <filesystem>

using namespace edge_ai;
using namespace edge_ai::testing;

int main(int argc, char* argv[]) {
    PROFILER_SCOPED_EVENT(0, "bdt_test_runner_main");
    
    std::cout << "Edge AI Engine - Behavior-Driven Testing Runner" << std::endl;
    std::cout << "==============================================" << std::endl;
    
    // Initialize BDT Manager
    BDTManager bdt_manager;
    
    // Configure BDT
    BDTConfiguration config;
    config.enable_parallel_execution = true;
    config.max_parallel_scenarios = 4;
    config.default_step_timeout = std::chrono::milliseconds(5000);
    config.default_scenario_timeout = std::chrono::milliseconds(30000);
    config.stop_on_first_failure = false;
    config.enable_step_retry = true;
    config.max_step_retries = 3;
    config.retry_delay = std::chrono::milliseconds(1000);
    config.enable_detailed_logging = true;
    config.enable_step_timing = true;
    config.output_format = "console";
    
    Status status = bdt_manager.SetConfiguration(config);
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to configure BDT Manager" << std::endl;
        return 1;
    }
    
    // Load step definitions
    std::string step_definitions_dir = "tests/step_definitions/";
    if (std::filesystem::exists(step_definitions_dir)) {
        status = bdt_manager.LoadStepDefinitions(step_definitions_dir);
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to load step definitions from: " << step_definitions_dir << std::endl;
        }
    }
    
    // Load feature files
    std::string features_dir = "tests/features/";
    if (std::filesystem::exists(features_dir)) {
        status = bdt_manager.LoadFeatureFiles(features_dir);
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to load feature files from: " << features_dir << std::endl;
        }
    }
    
    // Run specific feature if provided
    if (argc > 1) {
        std::string feature_file = argv[1];
        std::cout << "Running feature file: " << feature_file << std::endl;
        
        status = bdt_manager.RunFeatures({feature_file});
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to run feature file: " << feature_file << std::endl;
            return 1;
        }
    } else {
        // Run all loaded features
        std::cout << "Running all loaded features..." << std::endl;
        
        auto features = bdt_manager.GetLoadedFeatures();
        std::vector<std::string> feature_files;
        
        for (const auto& feature : features) {
            feature_files.push_back(feature.file_path);
        }
        
        if (feature_files.empty()) {
            std::cout << "No feature files loaded. Loading default features..." << std::endl;
            
            // Load default feature files
            std::vector<std::string> default_features = {
                "tests/features/edge_ai_inference.feature"
            };
            
            status = bdt_manager.RunFeatures(default_features);
            if (status != Status::SUCCESS) {
                std::cerr << "Failed to run default features" << std::endl;
                return 1;
            }
        } else {
            status = bdt_manager.RunFeatures(feature_files);
            if (status != Status::SUCCESS) {
                std::cerr << "Failed to run loaded features" << std::endl;
                return 1;
            }
        }
    }
    
    // Get and display statistics
    BDTStatistics stats = bdt_manager.GetStatistics();
    
    std::cout << "\nBDT Test Results Summary" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "Total Features: " << stats.total_features << std::endl;
    std::cout << "Total Scenarios: " << stats.total_scenarios << std::endl;
    std::cout << "Total Steps: " << stats.total_steps << std::endl;
    std::cout << "\nFeature Results:" << std::endl;
    std::cout << "  Passed: " << stats.passed_features << std::endl;
    std::cout << "  Failed: " << stats.failed_features << std::endl;
    std::cout << "  Success Rate: " << std::fixed << std::setprecision(2) 
              << stats.feature_success_rate << "%" << std::endl;
    std::cout << "\nScenario Results:" << std::endl;
    std::cout << "  Passed: " << stats.passed_scenarios << std::endl;
    std::cout << "  Failed: " << stats.failed_scenarios << std::endl;
    std::cout << "  Skipped: " << stats.skipped_scenarios << std::endl;
    std::cout << "  Success Rate: " << std::fixed << std::setprecision(2) 
              << stats.scenario_success_rate << "%" << std::endl;
    std::cout << "\nStep Results:" << std::endl;
    std::cout << "  Passed: " << stats.passed_steps << std::endl;
    std::cout << "  Failed: " << stats.failed_steps << std::endl;
    std::cout << "  Skipped: " << stats.skipped_steps << std::endl;
    std::cout << "  Success Rate: " << std::fixed << std::setprecision(2) 
              << stats.step_success_rate << "%" << std::endl;
    std::cout << "\nTiming Information:" << std::endl;
    std::cout << "  Total Duration: " << stats.total_duration.count() << "ms" << std::endl;
    std::cout << "  Avg Feature Duration: " << std::fixed << std::setprecision(2) 
              << stats.avg_feature_duration_ms << "ms" << std::endl;
    std::cout << "  Avg Scenario Duration: " << std::fixed << std::setprecision(2) 
              << stats.avg_scenario_duration_ms << "ms" << std::endl;
    std::cout << "  Avg Step Duration: " << std::fixed << std::setprecision(2) 
              << stats.avg_step_duration_ms << "ms" << std::endl;
    
    // Generate report
    std::string report_file = "bdt_test_report.html";
    status = bdt_manager.GenerateReport(report_file, "html");
    if (status == Status::SUCCESS) {
        std::cout << "\nDetailed report generated: " << report_file << std::endl;
    }
    
    // Export results
    std::string results_file = "bdt_test_results.json";
    status = bdt_manager.ExportResults(results_file, "json");
    if (status == Status::SUCCESS) {
        std::cout << "Results exported: " << results_file << std::endl;
    }
    
    // Determine exit code
    if (stats.failed_features > 0 || stats.failed_scenarios > 0 || stats.failed_steps > 0) {
        std::cout << "\nSome tests failed!" << std::endl;
        return 1;
    } else {
        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    }
}
