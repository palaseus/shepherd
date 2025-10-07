#include <testing/test_reporter.h>
#include <profiling/profiler.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <ctime>
#include <algorithm>
#include <iostream>

namespace edge_ai {
namespace testing {

// TestReporter Implementation
TestReporter::TestReporter() : output_format_(TestOutputFormat::CONSOLE) {
    // Initialize default configuration
}

TestReporter::~TestReporter() {
    // Cleanup if needed
}

Status TestReporter::SetOutputFormat(TestOutputFormat format) {
    output_format_ = format;
    return Status::SUCCESS;
}

Status TestReporter::SetOutputFile(const std::string& file_path) {
    output_file_path_ = file_path;
    return Status::SUCCESS;
}

Status TestReporter::SetReportConfiguration(const ReportConfiguration& config) {
    config_ = config;
    return Status::SUCCESS;
}

Status TestReporter::GenerateReport(const std::vector<TestResult>& results) {
    PROFILER_SCOPED_EVENT(0, "generate_test_report");
    
    switch (output_format_) {
        case TestOutputFormat::CONSOLE:
            return GenerateConsoleReport(results);
        case TestOutputFormat::JSON:
            return GenerateJSONReport(results);
        case TestOutputFormat::HTML:
            return GenerateHTMLReport(results);
        case TestOutputFormat::XML:
            return GenerateXMLReport(results);
        case TestOutputFormat::CSV:
            return GenerateCSVReport(results);
        default:
            return Status::INVALID_ARGUMENT;
    }
}

Status TestReporter::GenerateConsoleReport(const std::vector<TestResult>& results) {
    PROFILER_SCOPED_EVENT(0, "generate_console_report");
    
    std::stringstream report;
    
    // Header
    report << "\n=== Edge AI Engine Test Report ===\n";
    report << "Generated: " << GetCurrentTimestamp() << "\n";
    report << "Total Tests: " << results.size() << "\n\n";
    
    // Summary
    TestStatistics stats = CalculateStatistics(results);
    report << "=== Summary ===\n";
    report << "Passed: " << stats.passed_tests << "\n";
    report << "Failed: " << stats.failed_tests << "\n";
    report << "Flaky: " << stats.flaky_tests << "\n";
    report << "Success Rate: " << std::fixed << std::setprecision(2) 
           << (stats.success_rate * 100.0) << "%\n";
    report << "Average Duration: " << stats.avg_duration_ms << " ms\n";
    report << "Total Duration: " << stats.total_duration.count() << " ms\n";
    report << "Average Coverage: " << stats.avg_coverage_percent << "%\n";
    report << "Average Stability: " << stats.avg_stability_score << "\n\n";
    
    // Failed tests
    if (stats.failed_tests > 0) {
        report << "=== Failed Tests ===\n";
        for (const auto& result : results) {
            if (!result.passed) {
                report << "✗ " << result.test_name << " (" << result.module_name << ")\n";
                if (!result.error_message.empty()) {
                    report << "  Error: " << result.error_message << "\n";
                }
                report << "  Duration: " << result.duration.count() << " ms\n";
                if (result.cpu_usage_percent > 0) {
                    report << "  CPU Usage: " << result.cpu_usage_percent << "%\n";
                }
                if (result.memory_usage_mb > 0) {
                    report << "  Memory Usage: " << result.memory_usage_mb << " MB\n";
                }
                report << "\n";
            }
        }
    }
    
    // Flaky tests
    if (stats.flaky_tests > 0) {
        report << "=== Flaky Tests ===\n";
        for (const auto& result : results) {
            if (result.flaky) {
                report << "⚠ " << result.test_name << " (" << result.module_name << ")\n";
                report << "  Duration: " << result.duration.count() << " ms\n";
                report << "  Stability Score: " << result.stability_score << "\n\n";
            }
        }
    }
    
    // Performance summary
    if (config_.include_performance_summary) {
        report << "=== Performance Summary ===\n";
        auto performance_stats = CalculatePerformanceStatistics(results);
        report << "Average CPU Usage: " << performance_stats.avg_cpu_usage << "%\n";
        report << "Average Memory Usage: " << performance_stats.avg_memory_usage << " MB\n";
        report << "Average Network Usage: " << performance_stats.avg_network_usage << " Mbps\n";
        report << "Max CPU Usage: " << performance_stats.max_cpu_usage << "%\n";
        report << "Max Memory Usage: " << performance_stats.max_memory_usage << " MB\n";
        report << "Max Network Usage: " << performance_stats.max_network_usage << " Mbps\n\n";
    }
    
    // Module breakdown
    if (config_.include_module_breakdown) {
        report << "=== Module Breakdown ===\n";
        auto module_stats = CalculateModuleStatistics(results);
        for (const auto& module : module_stats) {
            report << module.first << ":\n";
            report << "  Tests: " << module.second.total_tests << "\n";
            report << "  Passed: " << module.second.passed_tests << "\n";
            report << "  Failed: " << module.second.failed_tests << "\n";
            report << "  Success Rate: " << std::fixed << std::setprecision(2) 
                   << (module.second.success_rate * 100.0) << "%\n";
            report << "  Avg Duration: " << module.second.avg_duration_ms << " ms\n";
            report << "  Avg Coverage: " << module.second.avg_coverage_percent << "%\n\n";
        }
    }
    
    // Detailed results
    if (config_.include_detailed_results) {
        report << "=== Detailed Results ===\n";
        for (const auto& result : results) {
            report << (result.passed ? "✓" : "✗") << " " << result.test_name 
                   << " (" << result.module_name << ")\n";
            report << "  Duration: " << result.duration.count() << " ms\n";
            if (result.code_coverage_percent > 0) {
                report << "  Coverage: " << result.code_coverage_percent << "%\n";
            }
            if (result.stability_score > 0) {
                report << "  Stability: " << result.stability_score << "\n";
            }
            if (!result.metadata.empty()) {
                report << "  Metadata: ";
                for (const auto& meta : result.metadata) {
                    report << meta.first << "=" << meta.second << " ";
                }
                report << "\n";
            }
            if (!result.passed && !result.error_message.empty()) {
                report << "  Error: " << result.error_message << "\n";
            }
            report << "\n";
        }
    }
    
    // Output report
    if (output_file_path_.empty()) {
        std::cout << report.str();
    } else {
        std::ofstream file(output_file_path_);
        if (file.is_open()) {
            file << report.str();
            file.close();
        } else {
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

Status TestReporter::GenerateJSONReport(const std::vector<TestResult>& results) {
    PROFILER_SCOPED_EVENT(0, "generate_json_report");
    
    std::stringstream json;
    
    json << "{\n";
    json << "  \"report_metadata\": {\n";
    json << "    \"generated_at\": \"" << GetCurrentTimestamp() << "\",\n";
    json << "    \"total_tests\": " << results.size() << ",\n";
    json << "    \"report_format\": \"json\"\n";
    json << "  },\n";
    
    // Summary
    TestStatistics stats = CalculateStatistics(results);
    json << "  \"summary\": {\n";
    json << "    \"passed_tests\": " << stats.passed_tests << ",\n";
    json << "    \"failed_tests\": " << stats.failed_tests << ",\n";
    json << "    \"flaky_tests\": " << stats.flaky_tests << ",\n";
    json << "    \"success_rate\": " << std::fixed << std::setprecision(4) << stats.success_rate << ",\n";
    json << "    \"avg_duration_ms\": " << stats.avg_duration_ms << ",\n";
    json << "    \"total_duration_ms\": " << stats.total_duration.count() << ",\n";
    json << "    \"avg_coverage_percent\": " << stats.avg_coverage_percent << ",\n";
    json << "    \"avg_stability_score\": " << stats.avg_stability_score << "\n";
    json << "  },\n";
    
    // Performance summary
    if (config_.include_performance_summary) {
        auto performance_stats = CalculatePerformanceStatistics(results);
        json << "  \"performance_summary\": {\n";
        json << "    \"avg_cpu_usage_percent\": " << performance_stats.avg_cpu_usage << ",\n";
        json << "    \"avg_memory_usage_mb\": " << performance_stats.avg_memory_usage << ",\n";
        json << "    \"avg_network_usage_mbps\": " << performance_stats.avg_network_usage << ",\n";
        json << "    \"max_cpu_usage_percent\": " << performance_stats.max_cpu_usage << ",\n";
        json << "    \"max_memory_usage_mb\": " << performance_stats.max_memory_usage << ",\n";
        json << "    \"max_network_usage_mbps\": " << performance_stats.max_network_usage << "\n";
        json << "  },\n";
    }
    
    // Module breakdown
    if (config_.include_module_breakdown) {
        auto module_stats = CalculateModuleStatistics(results);
        json << "  \"module_breakdown\": {\n";
        bool first = true;
        for (const auto& module : module_stats) {
            if (!first) json << ",\n";
            json << "    \"" << module.first << "\": {\n";
            json << "      \"total_tests\": " << module.second.total_tests << ",\n";
            json << "      \"passed_tests\": " << module.second.passed_tests << ",\n";
            json << "      \"failed_tests\": " << module.second.failed_tests << ",\n";
            json << "      \"success_rate\": " << std::fixed << std::setprecision(4) << module.second.success_rate << ",\n";
            json << "      \"avg_duration_ms\": " << module.second.avg_duration_ms << ",\n";
            json << "      \"avg_coverage_percent\": " << module.second.avg_coverage_percent << "\n";
            json << "    }";
            first = false;
        }
        json << "\n  },\n";
    }
    
    // Test results
    json << "  \"test_results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        json << "    {\n";
        json << "      \"test_name\": \"" << result.test_name << "\",\n";
        json << "      \"module_name\": \"" << result.module_name << "\",\n";
        json << "      \"passed\": " << (result.passed ? "true" : "false") << ",\n";
        json << "      \"flaky\": " << (result.flaky ? "true" : "false") << ",\n";
        json << "      \"duration_ms\": " << result.duration.count() << ",\n";
        json << "      \"code_coverage_percent\": " << result.code_coverage_percent << ",\n";
        json << "      \"stability_score\": " << result.stability_score << ",\n";
        json << "      \"cpu_usage_percent\": " << result.cpu_usage_percent << ",\n";
        json << "      \"memory_usage_mb\": " << result.memory_usage_mb << ",\n";
        json << "      \"network_usage_mbps\": " << result.network_usage_mbps << ",\n";
        
        if (!result.error_message.empty()) {
            json << "      \"error_message\": \"" << EscapeJSONString(result.error_message) << "\",\n";
        }
        
        if (!result.metadata.empty()) {
            json << "      \"metadata\": {\n";
            bool first_meta = true;
            for (const auto& meta : result.metadata) {
                if (!first_meta) json << ",\n";
                json << "        \"" << meta.first << "\": \"" << EscapeJSONString(meta.second) << "\"";
                first_meta = false;
            }
            json << "\n      },\n";
        }
        
        json << "      \"tags\": [";
        for (size_t j = 0; j < result.tags.size(); ++j) {
            if (j > 0) json << ", ";
            json << "\"" << EscapeJSONString(result.tags[j]) << "\"";
        }
        json << "]\n";
        
        json << "    }";
        if (i < results.size() - 1) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    
    // Output report
    if (output_file_path_.empty()) {
        std::cout << json.str();
    } else {
        std::ofstream file(output_file_path_);
        if (file.is_open()) {
            file << json.str();
            file.close();
        } else {
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

Status TestReporter::GenerateHTMLReport(const std::vector<TestResult>& results) {
    PROFILER_SCOPED_EVENT(0, "generate_html_report");
    
    std::stringstream html;
    
    html << "<!DOCTYPE html>\n";
    html << "<html>\n";
    html << "<head>\n";
    html << "  <title>Edge AI Engine Test Report</title>\n";
    html << "  <style>\n";
    html << "    body { font-family: Arial, sans-serif; margin: 20px; }\n";
    html << "    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }\n";
    html << "    .summary { background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }\n";
    html << "    .failed { color: #d32f2f; }\n";
    html << "    .passed { color: #388e3c; }\n";
    html << "    .flaky { color: #f57c00; }\n";
    html << "    .test-result { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }\n";
    html << "    .test-passed { background-color: #e8f5e8; }\n";
    html << "    .test-failed { background-color: #ffeaea; }\n";
    html << "    .test-flaky { background-color: #fff3e0; }\n";
    html << "    .metadata { font-size: 0.9em; color: #666; }\n";
    html << "    .error { color: #d32f2f; font-weight: bold; }\n";
    html << "    table { border-collapse: collapse; width: 100%; }\n";
    html << "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
    html << "    th { background-color: #f2f2f2; }\n";
    html << "  </style>\n";
    html << "</head>\n";
    html << "<body>\n";
    
    // Header
    html << "  <div class=\"header\">\n";
    html << "    <h1>Edge AI Engine Test Report</h1>\n";
    html << "    <p>Generated: " << GetCurrentTimestamp() << "</p>\n";
    html << "    <p>Total Tests: " << results.size() << "</p>\n";
    html << "  </div>\n";
    
    // Summary
    TestStatistics stats = CalculateStatistics(results);
    html << "  <div class=\"summary\">\n";
    html << "    <h2>Summary</h2>\n";
    html << "    <p><span class=\"passed\">Passed: " << stats.passed_tests << "</span></p>\n";
    html << "    <p><span class=\"failed\">Failed: " << stats.failed_tests << "</span></p>\n";
    html << "    <p><span class=\"flaky\">Flaky: " << stats.flaky_tests << "</span></p>\n";
    html << "    <p>Success Rate: " << std::fixed << std::setprecision(2) 
         << (stats.success_rate * 100.0) << "%</p>\n";
    html << "    <p>Average Duration: " << stats.avg_duration_ms << " ms</p>\n";
    html << "    <p>Total Duration: " << stats.total_duration.count() << " ms</p>\n";
    html << "    <p>Average Coverage: " << stats.avg_coverage_percent << "%</p>\n";
    html << "    <p>Average Stability: " << stats.avg_stability_score << "</p>\n";
    html << "  </div>\n";
    
    // Performance summary
    if (config_.include_performance_summary) {
        auto performance_stats = CalculatePerformanceStatistics(results);
        html << "  <div class=\"summary\">\n";
        html << "    <h2>Performance Summary</h2>\n";
        html << "    <table>\n";
        html << "      <tr><th>Metric</th><th>Average</th><th>Maximum</th></tr>\n";
        html << "      <tr><td>CPU Usage</td><td>" << performance_stats.avg_cpu_usage 
             << "%</td><td>" << performance_stats.max_cpu_usage << "%</td></tr>\n";
        html << "      <tr><td>Memory Usage</td><td>" << performance_stats.avg_memory_usage 
             << " MB</td><td>" << performance_stats.max_memory_usage << " MB</td></tr>\n";
        html << "      <tr><td>Network Usage</td><td>" << performance_stats.avg_network_usage 
             << " Mbps</td><td>" << performance_stats.max_network_usage << " Mbps</td></tr>\n";
        html << "    </table>\n";
        html << "  </div>\n";
    }
    
    // Module breakdown
    if (config_.include_module_breakdown) {
        auto module_stats = CalculateModuleStatistics(results);
        html << "  <div class=\"summary\">\n";
        html << "    <h2>Module Breakdown</h2>\n";
        html << "    <table>\n";
        html << "      <tr><th>Module</th><th>Tests</th><th>Passed</th><th>Failed</th><th>Success Rate</th><th>Avg Duration</th><th>Avg Coverage</th></tr>\n";
        for (const auto& module : module_stats) {
            html << "      <tr>\n";
            html << "        <td>" << module.first << "</td>\n";
            html << "        <td>" << module.second.total_tests << "</td>\n";
            html << "        <td>" << module.second.passed_tests << "</td>\n";
            html << "        <td>" << module.second.failed_tests << "</td>\n";
            html << "        <td>" << std::fixed << std::setprecision(2) 
                 << (module.second.success_rate * 100.0) << "%</td>\n";
            html << "        <td>" << module.second.avg_duration_ms << " ms</td>\n";
            html << "        <td>" << module.second.avg_coverage_percent << "%</td>\n";
            html << "      </tr>\n";
        }
        html << "    </table>\n";
        html << "  </div>\n";
    }
    
    // Test results
    html << "  <h2>Test Results</h2>\n";
    for (const auto& result : results) {
        std::string css_class = "test-result ";
        if (result.passed) {
            css_class += "test-passed";
        } else if (result.flaky) {
            css_class += "test-flaky";
        } else {
            css_class += "test-failed";
        }
        
        html << "  <div class=\"" << css_class << "\">\n";
        html << "    <h3>" << (result.passed ? "✓" : (result.flaky ? "⚠" : "✗")) 
             << " " << result.test_name << " (" << result.module_name << ")</h3>\n";
        html << "    <p>Duration: " << result.duration.count() << " ms</p>\n";
        
        if (result.code_coverage_percent > 0) {
            html << "    <p>Coverage: " << result.code_coverage_percent << "%</p>\n";
        }
        
        if (result.stability_score > 0) {
            html << "    <p>Stability: " << result.stability_score << "</p>\n";
        }
        
        if (result.cpu_usage_percent > 0 || result.memory_usage_mb > 0 || result.network_usage_mbps > 0) {
            html << "    <div class=\"metadata\">\n";
            html << "      <p>Performance: ";
            if (result.cpu_usage_percent > 0) {
                html << "CPU " << result.cpu_usage_percent << "% ";
            }
            if (result.memory_usage_mb > 0) {
                html << "Memory " << result.memory_usage_mb << " MB ";
            }
            if (result.network_usage_mbps > 0) {
                html << "Network " << result.network_usage_mbps << " Mbps ";
            }
            html << "</p>\n";
            html << "    </div>\n";
        }
        
        if (!result.metadata.empty()) {
            html << "    <div class=\"metadata\">\n";
            html << "      <p>Metadata: ";
            for (const auto& meta : result.metadata) {
                html << meta.first << "=" << meta.second << " ";
            }
            html << "</p>\n";
            html << "    </div>\n";
        }
        
        if (!result.tags.empty()) {
            html << "    <div class=\"metadata\">\n";
            html << "      <p>Tags: ";
            for (size_t i = 0; i < result.tags.size(); ++i) {
                if (i > 0) html << ", ";
                html << result.tags[i];
            }
            html << "</p>\n";
            html << "    </div>\n";
        }
        
        if (!result.passed && !result.error_message.empty()) {
            html << "    <p class=\"error\">Error: " << EscapeHTMLString(result.error_message) << "</p>\n";
        }
        
        html << "  </div>\n";
    }
    
    html << "</body>\n";
    html << "</html>\n";
    
    // Output report
    if (output_file_path_.empty()) {
        std::cout << html.str();
    } else {
        std::ofstream file(output_file_path_);
        if (file.is_open()) {
            file << html.str();
            file.close();
        } else {
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

Status TestReporter::GenerateXMLReport(const std::vector<TestResult>& results) {
    PROFILER_SCOPED_EVENT(0, "generate_xml_report");
    
    std::stringstream xml;
    
    xml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    xml << "<testreport>\n";
    xml << "  <metadata>\n";
    xml << "    <generated_at>" << GetCurrentTimestamp() << "</generated_at>\n";
    xml << "    <total_tests>" << results.size() << "</total_tests>\n";
    xml << "    <report_format>xml</report_format>\n";
    xml << "  </metadata>\n";
    
    // Summary
    TestStatistics stats = CalculateStatistics(results);
    xml << "  <summary>\n";
    xml << "    <passed_tests>" << stats.passed_tests << "</passed_tests>\n";
    xml << "    <failed_tests>" << stats.failed_tests << "</failed_tests>\n";
    xml << "    <flaky_tests>" << stats.flaky_tests << "</flaky_tests>\n";
    xml << "    <success_rate>" << std::fixed << std::setprecision(4) << stats.success_rate << "</success_rate>\n";
    xml << "    <avg_duration_ms>" << stats.avg_duration_ms << "</avg_duration_ms>\n";
    xml << "    <total_duration_ms>" << stats.total_duration.count() << "</total_duration_ms>\n";
    xml << "    <avg_coverage_percent>" << stats.avg_coverage_percent << "</avg_coverage_percent>\n";
    xml << "    <avg_stability_score>" << stats.avg_stability_score << "</avg_stability_score>\n";
    xml << "  </summary>\n";
    
    // Performance summary
    if (config_.include_performance_summary) {
        auto performance_stats = CalculatePerformanceStatistics(results);
        xml << "  <performance_summary>\n";
        xml << "    <avg_cpu_usage_percent>" << performance_stats.avg_cpu_usage << "</avg_cpu_usage_percent>\n";
        xml << "    <avg_memory_usage_mb>" << performance_stats.avg_memory_usage << "</avg_memory_usage_mb>\n";
        xml << "    <avg_network_usage_mbps>" << performance_stats.avg_network_usage << "</avg_network_usage_mbps>\n";
        xml << "    <max_cpu_usage_percent>" << performance_stats.max_cpu_usage << "</max_cpu_usage_percent>\n";
        xml << "    <max_memory_usage_mb>" << performance_stats.max_memory_usage << "</max_memory_usage_mb>\n";
        xml << "    <max_network_usage_mbps>" << performance_stats.max_network_usage << "</max_network_usage_mbps>\n";
        xml << "  </performance_summary>\n";
    }
    
    // Module breakdown
    if (config_.include_module_breakdown) {
        auto module_stats = CalculateModuleStatistics(results);
        xml << "  <module_breakdown>\n";
        for (const auto& module : module_stats) {
            xml << "    <module name=\"" << EscapeXMLString(module.first) << "\">\n";
            xml << "      <total_tests>" << module.second.total_tests << "</total_tests>\n";
            xml << "      <passed_tests>" << module.second.passed_tests << "</passed_tests>\n";
            xml << "      <failed_tests>" << module.second.failed_tests << "</failed_tests>\n";
            xml << "      <success_rate>" << std::fixed << std::setprecision(4) << module.second.success_rate << "</success_rate>\n";
            xml << "      <avg_duration_ms>" << module.second.avg_duration_ms << "</avg_duration_ms>\n";
            xml << "      <avg_coverage_percent>" << module.second.avg_coverage_percent << "</avg_coverage_percent>\n";
            xml << "    </module>\n";
        }
        xml << "  </module_breakdown>\n";
    }
    
    // Test results
    xml << "  <test_results>\n";
    for (const auto& result : results) {
        xml << "    <test>\n";
        xml << "      <name>" << EscapeXMLString(result.test_name) << "</name>\n";
        xml << "      <module>" << EscapeXMLString(result.module_name) << "</module>\n";
        xml << "      <passed>" << (result.passed ? "true" : "false") << "</passed>\n";
        xml << "      <flaky>" << (result.flaky ? "true" : "false") << "</flaky>\n";
        xml << "      <duration_ms>" << result.duration.count() << "</duration_ms>\n";
        xml << "      <code_coverage_percent>" << result.code_coverage_percent << "</code_coverage_percent>\n";
        xml << "      <stability_score>" << result.stability_score << "</stability_score>\n";
        xml << "      <cpu_usage_percent>" << result.cpu_usage_percent << "</cpu_usage_percent>\n";
        xml << "      <memory_usage_mb>" << result.memory_usage_mb << "</memory_usage_mb>\n";
        xml << "      <network_usage_mbps>" << result.network_usage_mbps << "</network_usage_mbps>\n";
        
        if (!result.error_message.empty()) {
            xml << "      <error_message>" << EscapeXMLString(result.error_message) << "</error_message>\n";
        }
        
        if (!result.metadata.empty()) {
            xml << "      <metadata>\n";
            for (const auto& meta : result.metadata) {
                xml << "        <item key=\"" << EscapeXMLString(meta.first) << "\" value=\"" 
                    << EscapeXMLString(meta.second) << "\"/>\n";
            }
            xml << "      </metadata>\n";
        }
        
        if (!result.tags.empty()) {
            xml << "      <tags>\n";
            for (const auto& tag : result.tags) {
                xml << "        <tag>" << EscapeXMLString(tag) << "</tag>\n";
            }
            xml << "      </tags>\n";
        }
        
        xml << "    </test>\n";
    }
    xml << "  </test_results>\n";
    xml << "</testreport>\n";
    
    // Output report
    if (output_file_path_.empty()) {
        std::cout << xml.str();
    } else {
        std::ofstream file(output_file_path_);
        if (file.is_open()) {
            file << xml.str();
            file.close();
        } else {
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

Status TestReporter::GenerateCSVReport(const std::vector<TestResult>& results) {
    PROFILER_SCOPED_EVENT(0, "generate_csv_report");
    
    std::stringstream csv;
    
    // Header
    csv << "test_name,module_name,passed,flaky,duration_ms,code_coverage_percent,stability_score,"
        << "cpu_usage_percent,memory_usage_mb,network_usage_mbps,error_message,tags,metadata\n";
    
    // Data rows
    for (const auto& result : results) {
        csv << EscapeCSVString(result.test_name) << ",";
        csv << EscapeCSVString(result.module_name) << ",";
        csv << (result.passed ? "true" : "false") << ",";
        csv << (result.flaky ? "true" : "false") << ",";
        csv << result.duration.count() << ",";
        csv << result.code_coverage_percent << ",";
        csv << result.stability_score << ",";
        csv << result.cpu_usage_percent << ",";
        csv << result.memory_usage_mb << ",";
        csv << result.network_usage_mbps << ",";
        csv << EscapeCSVString(result.error_message) << ",";
        
        // Tags
        std::string tags_str;
        for (size_t i = 0; i < result.tags.size(); ++i) {
            if (i > 0) tags_str += ";";
            tags_str += result.tags[i];
        }
        csv << EscapeCSVString(tags_str) << ",";
        
        // Metadata
        std::string metadata_str;
        for (const auto& meta : result.metadata) {
            if (!metadata_str.empty()) metadata_str += ";";
            metadata_str += meta.first + "=" + meta.second;
        }
        csv << EscapeCSVString(metadata_str) << "\n";
    }
    
    // Output report
    if (output_file_path_.empty()) {
        std::cout << csv.str();
    } else {
        std::ofstream file(output_file_path_);
        if (file.is_open()) {
            file << csv.str();
            file.close();
        } else {
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

TestStatistics TestReporter::CalculateStatistics(const std::vector<TestResult>& results) {
    TestStatistics stats;
    
    stats.total_tests = results.size();
    
    for (const auto& result : results) {
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

ReporterPerformanceStatistics TestReporter::CalculatePerformanceStatistics(const std::vector<TestResult>& results) {
    ReporterPerformanceStatistics stats;
    
    for (const auto& result : results) {
        stats.avg_cpu_usage += result.cpu_usage_percent;
        stats.avg_memory_usage += result.memory_usage_mb;
        stats.avg_network_usage += result.network_usage_mbps;
        
        stats.max_cpu_usage = std::max(stats.max_cpu_usage, result.cpu_usage_percent);
        stats.max_memory_usage = std::max(stats.max_memory_usage, result.memory_usage_mb);
        stats.max_network_usage = std::max(stats.max_network_usage, result.network_usage_mbps);
    }
    
    if (!results.empty()) {
        stats.avg_cpu_usage /= results.size();
        stats.avg_memory_usage /= results.size();
        stats.avg_network_usage /= results.size();
    }
    
    return stats;
}

std::map<std::string, ModuleStatistics> TestReporter::CalculateModuleStatistics(const std::vector<TestResult>& results) {
    std::map<std::string, ModuleStatistics> module_stats;
    
    for (const auto& result : results) {
        auto& stats = module_stats[result.module_name];
        stats.total_tests++;
        
        if (result.passed) {
            stats.passed_tests++;
        } else {
            stats.failed_tests++;
        }
        
        stats.total_duration += result.duration;
        stats.avg_coverage_percent += result.code_coverage_percent;
    }
    
    for (auto& module : module_stats) {
        auto& stats = module.second;
        if (stats.total_tests > 0) {
            stats.avg_duration_ms = static_cast<double>(stats.total_duration.count()) / stats.total_tests;
            stats.success_rate = static_cast<double>(stats.passed_tests) / stats.total_tests;
            stats.avg_coverage_percent /= stats.total_tests;
        }
    }
    
    return module_stats;
}

std::string TestReporter::GetCurrentTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string TestReporter::EscapeJSONString(const std::string& str) {
    std::string escaped;
    escaped.reserve(str.length() + 10);
    
    for (char c : str) {
        switch (c) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\b': escaped += "\\b"; break;
            case '\f': escaped += "\\f"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c; break;
        }
    }
    
    return escaped;
}

std::string TestReporter::EscapeHTMLString(const std::string& str) {
    std::string escaped;
    escaped.reserve(str.length() + 10);
    
    for (char c : str) {
        switch (c) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&#39;"; break;
            default: escaped += c; break;
        }
    }
    
    return escaped;
}

std::string TestReporter::EscapeXMLString(const std::string& str) {
    std::string escaped;
    escaped.reserve(str.length() + 10);
    
    for (char c : str) {
        switch (c) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&apos;"; break;
            default: escaped += c; break;
        }
    }
    
    return escaped;
}

std::string TestReporter::EscapeCSVString(const std::string& str) {
    if (str.find(',') != std::string::npos || str.find('"') != std::string::npos || 
        str.find('\n') != std::string::npos || str.find('\r') != std::string::npos) {
        std::string escaped = "\"";
        for (char c : str) {
            if (c == '"') {
                escaped += "\"\"";
            } else {
                escaped += c;
            }
        }
        escaped += "\"";
        return escaped;
    }
    return str;
}

} // namespace testing
} // namespace edge_ai
