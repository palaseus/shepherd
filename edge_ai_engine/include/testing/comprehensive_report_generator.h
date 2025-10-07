#ifndef EDGE_AI_ENGINE_COMPREHENSIVE_REPORT_GENERATOR_H
#define EDGE_AI_ENGINE_COMPREHENSIVE_REPORT_GENERATOR_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

#include <core/types.h>
#include <testing/test_framework.h>
#include <testing/test_common.h>
#include <testing/simple_property_testing.h>
#include <testing/interface_validator.h>

namespace edge_ai {
namespace testing {

// Comprehensive test report structures
struct TestSuiteSummary {
    std::string suite_name;
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;
    uint32_t failed_tests = 0;
    uint32_t skipped_tests = 0;
    double success_rate = 0.0;
    std::chrono::milliseconds duration{0};
    std::string status; // "PASSED", "FAILED", "PARTIAL"
};

struct ComprehensiveTestReport {
    std::string report_title;
    std::string generated_at;
    std::string project_version;
    std::string build_configuration;
    
    // Test suite summaries
    TestSuiteSummary unit_tests;
    TestSuiteSummary integration_tests;
    TestSuiteSummary performance_tests;
    TestSuiteSummary bdt_tests;
    TestSuiteSummary property_based_tests;
    TestSuiteSummary interface_validation;
    
    // Overall statistics
    uint32_t total_test_suites = 0;
    uint32_t passed_test_suites = 0;
    uint32_t failed_test_suites = 0;
    double overall_success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    
    // Coverage information
    double code_coverage_percentage = 0.0;
    uint32_t lines_covered = 0;
    uint32_t total_lines = 0;
    
    // Performance metrics
    double average_test_duration_ms = 0.0;
    double slowest_test_duration_ms = 0.0;
    double fastest_test_duration_ms = 0.0;
    
    // Quality metrics
    uint32_t critical_failures = 0;
    uint32_t warnings = 0;
    uint32_t memory_leaks = 0;
    uint32_t performance_regressions = 0;
};

struct ReportConfiguration {
    std::string output_directory = "comprehensive_test_reports/";
    std::string report_format = "html"; // "html", "json", "markdown"
    bool include_coverage_data = true;
    bool include_performance_data = true;
    bool include_interface_validation = true;
    bool include_property_based_tests = true;
    bool include_bdt_tests = true;
    std::string project_name = "Edge AI Inference Engine";
    std::string project_version = "1.0.0";
    std::string build_configuration = "Release";
};

// Comprehensive Report Generator class
class ComprehensiveReportGenerator {
public:
    ComprehensiveReportGenerator();
    ~ComprehensiveReportGenerator();

    // Configuration
    Status SetConfiguration(const ReportConfiguration& config);
    
    // Data collection
    Status CollectTestData();
    Status CollectUnitTestData();
    Status CollectIntegrationTestData();
    Status CollectPerformanceTestData();
    Status CollectBDTTestData();
    Status CollectPropertyBasedTestData();
    Status CollectInterfaceValidationData();
    Status CollectCoverageData();
    
    // Report generation
    Status GenerateComprehensiveReport();
    Status GenerateHTMLReport(const std::string& output_file);
    Status GenerateJSONReport(const std::string& output_file);
    Status GenerateMarkdownReport(const std::string& output_file);
    
    // Utility functions
    void PrintReportSummary() const;
    ComprehensiveTestReport GetReport() const;
    
private:
    ReportConfiguration config_;
    ComprehensiveTestReport report_;
    
    // Data collection helpers
    TestSuiteSummary CollectTestSuiteSummary(const std::string& suite_name, const std::vector<TestResult>& results);
    void CalculateOverallStatistics();
    void CalculatePerformanceMetrics();
    void CalculateQualityMetrics();
    
    // Report generation helpers
    std::string GenerateHTMLHeader() const;
    std::string GenerateHTMLSummary() const;
    std::string GenerateHTMLTestSuites() const;
    std::string GenerateHTMLCoverage() const;
    std::string GenerateHTMLPerformance() const;
    std::string GenerateHTMLQuality() const;
    std::string GenerateHTMLFooter() const;
    
    std::string GenerateJSONContent() const;
    std::string GenerateMarkdownContent() const;
    
    // Utility functions
    std::string GetCurrentTimestamp() const;
    std::string FormatDuration(std::chrono::milliseconds duration) const;
    std::string GetStatusColor(const std::string& status) const;
    std::string GetStatusIcon(const std::string& status) const;
};

// Global accessor function
ComprehensiveReportGenerator* GetComprehensiveReportGenerator();

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_COMPREHENSIVE_REPORT_GENERATOR_H
