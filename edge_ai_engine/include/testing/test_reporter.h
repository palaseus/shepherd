#ifndef EDGE_AI_ENGINE_TEST_REPORTER_H
#define EDGE_AI_ENGINE_TEST_REPORTER_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <ctime>
#include <algorithm>
#include <core/types.h>
#include <testing/test_framework.h>
#include <testing/test_common.h>

namespace edge_ai {
namespace testing {

// Test output formats
enum class TestOutputFormat {
    CONSOLE,
    JSON,
    HTML,
    XML,
    CSV
};

// Report configuration
struct ReportConfiguration {
    bool include_performance_summary = true;
    bool include_module_breakdown = true;
    bool include_detailed_results = true;
    bool include_coverage_analysis = false;
    bool include_stability_analysis = false;
    bool include_trend_analysis = false;
    std::string report_title = "Edge AI Engine Test Report";
    std::string report_description = "Comprehensive test execution report";
    std::map<std::string, std::string> custom_metadata;
};

// Using common TestStatistics and PerformanceStatistics from test_common.h

// Reporter-specific performance statistics (different from common PerformanceStatistics)
struct ReporterPerformanceStatistics {
    double avg_cpu_usage = 0.0;
    double avg_memory_usage = 0.0;
    double avg_network_usage = 0.0;
    double max_cpu_usage = 0.0;
    double max_memory_usage = 0.0;
    double max_network_usage = 0.0;
};

// Module statistics
struct ModuleStatistics {
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;
    uint32_t failed_tests = 0;
    double success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    double avg_duration_ms = 0.0;
    double avg_coverage_percent = 0.0;
};

// TestReporter class
class TestReporter {
public:
    TestReporter();
    ~TestReporter();

    // Configuration
    Status SetOutputFormat(TestOutputFormat format);
    Status SetOutputFile(const std::string& file_path);
    Status SetReportConfiguration(const ReportConfiguration& config);

    // Report generation
    Status GenerateReport(const std::vector<TestResult>& results);

    // Format-specific report generation
    Status GenerateConsoleReport(const std::vector<TestResult>& results);
    Status GenerateJSONReport(const std::vector<TestResult>& results);
    Status GenerateHTMLReport(const std::vector<TestResult>& results);
    Status GenerateXMLReport(const std::vector<TestResult>& results);
    Status GenerateCSVReport(const std::vector<TestResult>& results);

    // Statistics calculation
    TestStatistics CalculateStatistics(const std::vector<TestResult>& results);
    ReporterPerformanceStatistics CalculatePerformanceStatistics(const std::vector<TestResult>& results);
    std::map<std::string, ModuleStatistics> CalculateModuleStatistics(const std::vector<TestResult>& results);

    // Utility functions
    std::string GetCurrentTimestamp();
    std::string EscapeJSONString(const std::string& str);
    std::string EscapeHTMLString(const std::string& str);
    std::string EscapeXMLString(const std::string& str);
    std::string EscapeCSVString(const std::string& str);

private:
    TestOutputFormat output_format_;
    std::string output_file_path_;
    ReportConfiguration config_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_REPORTER_H
