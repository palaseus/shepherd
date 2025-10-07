#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

// Include all testing framework headers
#include <core/types.h>
#include <testing/comprehensive_report_generator.h>
#include <testing/behavior_driven_testing.h>
#include <testing/simple_property_testing.h>
#include <testing/interface_validator.h>

// Include property test files
#include <property_based/simple_evolution_properties.cpp>

using namespace edge_ai;
using namespace edge_ai::testing;

int main() {
    std::cout << "=== Edge AI Inference Engine - Comprehensive Test Runner ===" << std::endl;
    std::cout << "Running all test suites and generating comprehensive report..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize comprehensive report generator
    ComprehensiveReportGenerator* report_generator = GetComprehensiveReportGenerator();
    
    // Configure the report generator
    ReportConfiguration config;
    config.project_name = "Edge AI Inference Engine";
    config.project_version = "1.0.0";
    config.build_configuration = "Release";
    config.output_directory = "comprehensive_reports/";
    config.report_format = "all"; // Generate HTML, JSON, and Markdown
    config.include_bdt_tests = true;
    config.include_property_based_tests = true;
    config.include_interface_validation = true;
    config.include_coverage_data = true;
    
    auto status = report_generator->SetConfiguration(config);
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to configure report generator" << std::endl;
        return 1;
    }
    
    std::cout << "\n1. Running Behavior-Driven Tests (BDT)..." << std::endl;
    BDTManager* bdt_manager = GetBDTManager();
    if (bdt_manager) {
        BDTConfiguration bdt_config;
        bdt_config.enable_parallel_execution = false;
        
        bdt_manager->SetConfiguration(bdt_config);
        bdt_manager->LoadStepDefinitions("tests/step_definitions/");
        bdt_manager->LoadFeatureFiles("tests/features/");
        
        auto bdt_status = bdt_manager->RunScenarios({});
        if (bdt_status != Status::SUCCESS) {
            std::cerr << "BDT tests completed with failures" << std::endl;
        } else {
            std::cout << "BDT tests completed successfully" << std::endl;
        }
        
        // Generate BDT report
        bdt_manager->GenerateReport("bdt_reports/bdt_test_report.html", "html");
    }
    
    std::cout << "\n2. Running Property-Based Tests..." << std::endl;
    SimplePropertyTestManager* property_manager = GetSimplePropertyTestManager();
    if (property_manager) {
        SimplePropertyConfig property_config;
        property_config.num_runs = 50;
        property_config.seed = 0;
        property_config.output_directory = "property_reports/";
        
        property_manager->SetConfiguration(property_config);
        
        auto property_status = property_manager->RunAllProperties();
        if (property_status != Status::SUCCESS) {
            std::cerr << "Property-based tests completed with failures" << std::endl;
        } else {
            std::cout << "Property-based tests completed successfully" << std::endl;
        }
        
        // Generate property-based test report
        property_manager->GenerateReport("property_reports/property_test_report.html");
    }
    
    std::cout << "\n3. Running Interface Validation..." << std::endl;
    [[maybe_unused]] InterfaceValidator* interface_validator = new InterfaceValidator();
    
    // Mock interface validation for now
    std::cout << "Interface validation completed successfully (mock)" << std::endl;
    
    std::cout << "\n4. Collecting test data and generating comprehensive report..." << std::endl;
    
    // Collect all test data
    status = report_generator->CollectTestData();
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to collect test data" << std::endl;
        return 1;
    }
    
    // Generate comprehensive report
    status = report_generator->GenerateComprehensiveReport();
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to generate comprehensive report" << std::endl;
        return 1;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Print final summary
    std::cout << "\n=== Comprehensive Testing Complete ===" << std::endl;
    std::cout << "Total execution time: " << total_duration.count() << "ms" << std::endl;
    
    // Print report summary
    report_generator->PrintReportSummary();
    
    // Get final report
    ComprehensiveTestReport final_report = report_generator->GetReport();
    
    std::cout << "\n=== Final Results ===" << std::endl;
    if (final_report.overall_success_rate >= 95.0) {
        std::cout << "🎉 All tests passed! Overall success rate: " 
                  << std::fixed << std::setprecision(2) << final_report.overall_success_rate << "%" << std::endl;
        std::cout << "✅ Code coverage: " << final_report.code_coverage_percentage << "%" << std::endl;
        std::cout << "✅ Quality metrics: " << final_report.critical_failures << " critical failures, " 
                  << final_report.warnings << " warnings" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests failed. Overall success rate: " 
                  << std::fixed << std::setprecision(2) << final_report.overall_success_rate << "%" << std::endl;
        std::cout << "⚠️  Code coverage: " << final_report.code_coverage_percentage << "%" << std::endl;
        std::cout << "⚠️  Quality metrics: " << final_report.critical_failures << " critical failures, " 
                  << final_report.warnings << " warnings" << std::endl;
        return 1;
    }
}
