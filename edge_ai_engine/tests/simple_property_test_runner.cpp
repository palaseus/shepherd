#include <testing/simple_property_testing.h>
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    std::cout << "=== Simple Property-Based Test Runner ===" << std::endl;
    
    // Get the property test manager
    auto* manager = edge_ai::testing::GetSimplePropertyTestManager();
    if (!manager) {
        std::cerr << "Failed to get property test manager" << std::endl;
        return 1;
    }
    
    // Set configuration
    edge_ai::testing::SimplePropertyConfig config;
    config.num_runs = 50; // Reduced for faster testing
    config.seed = 42; // Fixed seed for reproducible results
    config.output_directory = "simple_property_reports/";
    
    auto status = manager->SetConfiguration(config);
    if (status != edge_ai::Status::SUCCESS) {
        std::cerr << "Failed to set configuration" << std::endl;
        return 1;
    }
    
    // Run all properties
    status = manager->RunAllProperties();
    if (status != edge_ai::Status::SUCCESS) {
        std::cerr << "Failed to run properties" << std::endl;
        return 1;
    }
    
    // Print summary
    manager->PrintSummary();
    
    // Generate report
    status = manager->GenerateReport("simple_property_reports/simple_property_test_report.html");
    if (status != edge_ai::Status::SUCCESS) {
        std::cerr << "Failed to generate report" << std::endl;
        return 1;
    }
    
    std::cout << "\nSimple Property-Based Testing completed successfully!" << std::endl;
    return 0;
}
