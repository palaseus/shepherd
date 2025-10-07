#ifndef EDGE_AI_ENGINE_TEST_VALIDATION_H
#define EDGE_AI_ENGINE_TEST_VALIDATION_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <filesystem>
#include <regex>
#include <core/types.h>
#include <testing/test_framework.h>
#include <testing/test_performance.h>
#include <testing/test_coverage.h>
#include <testing/test_integration.h>

namespace edge_ai {
namespace testing {

// Validation configuration
struct ValidationConfiguration {
    bool strict_mode = true;
    bool validate_dependencies = true;
    bool validate_environment = true;
    bool validate_thresholds = true;
    bool validate_metadata = true;
    std::vector<std::string> required_fields;
    std::vector<std::string> optional_fields;
    std::map<std::string, std::string> field_patterns;
    std::map<std::string, double> field_ranges;
    std::vector<std::string> allowed_test_types;
    std::vector<std::string> allowed_tags;
    std::vector<std::string> forbidden_patterns;
};

// Validation rule
struct ValidationRule {
    std::string rule_name;
    std::string description;
    std::function<bool(const TestSpec&)> validator;
    std::string error_message;
    bool is_warning = false;
};

// Validation result
struct ValidationResult {
    std::string spec_name;
    bool is_valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::map<std::string, std::string> metadata;
};

// Validation report formats
enum class ValidationReportFormat {
    HTML,
    XML,
    JSON,
    TEXT
};

// Validation statistics
struct ValidationStatistics {
    uint32_t total_validations = 0;
    uint32_t valid_count = 0;
    uint32_t invalid_count = 0;
    double success_rate = 0.0;
    uint32_t total_errors = 0;
    uint32_t total_warnings = 0;
};

// TestValidation class
class TestValidation {
public:
    TestValidation();
    ~TestValidation();

    // Configuration
    Status SetConfiguration(const ValidationConfiguration& config);
    Status SetValidationRules(const std::vector<ValidationRule>& rules);

    // Test spec validation
    Status ValidateTestSpec(const TestSpec& spec);
    Status ValidateConfiguration(const TestConfig& config, ValidationResult& result);
    Status ValidateScenarios(const std::vector<TestScenario>& scenarios, ValidationResult& result);
    Status ValidateScenario(const TestScenario& scenario, ValidationResult& result, size_t index);
    Status ValidateMocks(const std::map<std::string, std::string>& mocks, ValidationResult& result);
    Status ValidateFixtures(const std::map<std::string, std::string>& fixtures, ValidationResult& result);

    // Test result validation
    Status ValidateTestResult(const TestResult& result);
    Status ValidatePerformanceResult(const PerformanceTestResult& result);
    Status ValidateCoverageData(const FileCoverageData& coverage_data);
    Status ValidateTestEnvironment(const TestEnvironment& env);

    // Utility methods
    bool IsValidIdentifier(const std::string& identifier);
    bool IsValidMockConfiguration(const std::string& config);
    bool IsValidFixtureConfiguration(const std::string& config);

    // Results and statistics
    std::vector<ValidationResult> GetValidationResults() const;
    ValidationResult GetValidationResult(const std::string& spec_name) const;
    ValidationStatistics GetValidationStatistics() const;
    void ClearValidationResults();

    // Report generation
    Status ExportValidationReport(const std::string& output_file, ValidationReportFormat format);
    Status ExportHTMLReport(const std::string& output_file);
    Status ExportXMLReport(const std::string& output_file);
    Status ExportJSONReport(const std::string& output_file);
    Status ExportTextReport(const std::string& output_file);

    // Utility functions
    std::string GetCurrentTimestamp();
    std::string EscapeXMLString(const std::string& str);
    std::string EscapeJSONString(const std::string& str);

private:
    ValidationConfiguration config_;
    std::vector<ValidationRule> validation_rules_;
    std::map<std::string, ValidationResult> validation_results_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_VALIDATION_H
