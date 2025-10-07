#include <testing/test_validation.h>
#include <profiling/profiler.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <cstring>

namespace edge_ai {
namespace testing {

// TestValidation Implementation
TestValidation::TestValidation() {
    // Initialize default configuration
}

TestValidation::~TestValidation() {
    // Cleanup if needed
}

Status TestValidation::SetConfiguration(const ValidationConfiguration& config) {
    config_ = config;
    return Status::SUCCESS;
}

Status TestValidation::SetValidationRules(const std::vector<ValidationRule>& rules) {
    validation_rules_ = rules;
    return Status::SUCCESS;
}

Status TestValidation::ValidateTestSpec(const TestSpec& spec) {
    PROFILER_SCOPED_EVENT(0, "validate_test_spec");
    
    ValidationResult result;
    result.spec_name = spec.GetConfig().test_suite_name;
    result.is_valid = true;
    
    // Validate configuration
    Status status = ValidateConfiguration(spec.GetConfig(), result);
    if (status != Status::SUCCESS) {
        result.is_valid = false;
    }
    
    // Validate scenarios
    status = ValidateScenarios(spec.GetScenarios(), result);
    if (status != Status::SUCCESS) {
        result.is_valid = false;
    }
    
    // Validate mocks
    status = ValidateMocks(spec.GetMocks(), result);
    if (status != Status::SUCCESS) {
        result.is_valid = false;
    }
    
    // Validate fixtures
    status = ValidateFixtures(spec.GetFixtures(), result);
    if (status != Status::SUCCESS) {
        result.is_valid = false;
    }
    
    // Store result
    validation_results_[spec.GetConfig().test_suite_name] = result;
    
    return result.is_valid ? Status::SUCCESS : Status::FAILURE;
}

Status TestValidation::ValidateConfiguration(const TestConfig& config, ValidationResult& result) {
    PROFILER_SCOPED_EVENT(0, "validate_configuration");
    
    // Validate test suite name
    if (config.test_suite_name.empty()) {
        result.errors.push_back("Test suite name is required");
        return Status::FAILURE;
    }
    
    // Validate module name
    if (config.module_name.empty()) {
        result.errors.push_back("Module name is required");
        return Status::FAILURE;
    }
    
    // Validate test type
    if (config.test_type.empty()) {
        result.errors.push_back("Test type is required");
        return Status::FAILURE;
    }
    
    // Validate test type is supported
    std::vector<std::string> supported_types = {"unit", "integration", "performance", "property", "fuzz"};
    if (std::find(supported_types.begin(), supported_types.end(), config.test_type) == supported_types.end()) {
        result.errors.push_back("Unsupported test type: " + config.test_type);
        return Status::FAILURE;
    }
    
    // Validate dependencies
    for (const auto& dep : config.dependencies) {
        if (dep.empty()) {
            result.errors.push_back("Empty dependency found");
            return Status::FAILURE;
        }
    }
    
    // Validate environment variables
    for (const auto& var : config.environment_vars) {
        if (var.first.empty() || var.second.empty()) {
            result.errors.push_back("Invalid environment variable: " + var.first);
            return Status::FAILURE;
        }
    }
    
    // Validate thresholds
    for (const auto& threshold : config.thresholds) {
        if (threshold.second < 0.0 || threshold.second > 1.0) {
            result.errors.push_back("Invalid threshold value for " + threshold.first + ": " + std::to_string(threshold.second));
            return Status::FAILURE;
        }
    }
    
    // Validate timeout
    if (config.timeout.count() <= 0) {
        result.errors.push_back("Invalid timeout value");
        return Status::FAILURE;
    }
    
    // Validate warmup time
    if (config.warmup_time.count() < 0) {
        result.errors.push_back("Invalid warmup time value");
        return Status::FAILURE;
    }
    
    // Validate cooldown time
    if (config.cooldown_time.count() < 0) {
        result.errors.push_back("Invalid cooldown time value");
        return Status::FAILURE;
    }
    
    return Status::SUCCESS;
}

Status TestValidation::ValidateScenarios(const std::vector<TestScenario>& scenarios, ValidationResult& result) {
    PROFILER_SCOPED_EVENT(0, "validate_scenarios");
    
    if (scenarios.empty()) {
        result.errors.push_back("At least one test scenario is required");
        return Status::FAILURE;
    }
    
    for (size_t i = 0; i < scenarios.size(); ++i) {
        const auto& scenario = scenarios[i];
        Status status = ValidateScenario(scenario, result, i);
        if (status != Status::SUCCESS) {
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

Status TestValidation::ValidateScenario(const TestScenario& scenario, ValidationResult& result, size_t index) {
    PROFILER_SCOPED_EVENT(0, "validate_scenario");
    
    std::string prefix = "Scenario[" + std::to_string(index) + "]";
    
    // Validate scenario name
    if (scenario.GetName().empty()) {
        result.errors.push_back(prefix + ": Scenario name is required");
        return Status::FAILURE;
    }
    
    // Validate scenario name format
    if (!IsValidIdentifier(scenario.GetName())) {
        result.errors.push_back(prefix + ": Invalid scenario name format: " + scenario.GetName());
        return Status::FAILURE;
    }
    
    // Validate Given steps
    if (scenario.GetGivenSteps().empty()) {
        result.warnings.push_back(prefix + ": No Given steps found");
    }
    
    // Validate When steps
    if (scenario.GetWhenSteps().empty()) {
        result.errors.push_back(prefix + ": At least one When step is required");
        return Status::FAILURE;
    }
    
    // Validate Then steps
    if (scenario.GetThenSteps().empty()) {
        result.errors.push_back(prefix + ": At least one Then step is required");
        return Status::FAILURE;
    }
    
    // Validate step content
    for (size_t j = 0; j < scenario.GetGivenSteps().size(); ++j) {
        const auto& step = scenario.GetGivenSteps()[j];
        if (step.empty()) {
            result.errors.push_back(prefix + ": Empty Given step at index " + std::to_string(j));
            return Status::FAILURE;
        }
    }
    
    for (size_t j = 0; j < scenario.GetWhenSteps().size(); ++j) {
        const auto& step = scenario.GetWhenSteps()[j];
        if (step.empty()) {
            result.errors.push_back(prefix + ": Empty When step at index " + std::to_string(j));
            return Status::FAILURE;
        }
    }
    
    for (size_t j = 0; j < scenario.GetThenSteps().size(); ++j) {
        const auto& step = scenario.GetThenSteps()[j];
        if (step.empty()) {
            result.errors.push_back(prefix + ": Empty Then step at index " + std::to_string(j));
            return Status::FAILURE;
        }
    }
    
    // Validate tags
    for (const auto& tag : scenario.GetTags()) {
        if (tag.empty()) {
            result.errors.push_back(prefix + ": Empty tag found");
            return Status::FAILURE;
        }
        
        if (!IsValidIdentifier(tag)) {
            result.errors.push_back(prefix + ": Invalid tag format: " + tag);
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

Status TestValidation::ValidateMocks(const std::map<std::string, std::string>& mocks, ValidationResult& result) {
    PROFILER_SCOPED_EVENT(0, "validate_mocks");
    
    for (const auto& mock : mocks) {
        // Validate mock name
        if (mock.first.empty()) {
            result.errors.push_back("Empty mock name found");
            return Status::FAILURE;
        }
        
        if (!IsValidIdentifier(mock.first)) {
            result.errors.push_back("Invalid mock name format: " + mock.first);
            return Status::FAILURE;
        }
        
        // Validate mock configuration
        if (mock.second.empty()) {
            result.errors.push_back("Empty mock configuration for: " + mock.first);
            return Status::FAILURE;
        }
        
        // Validate mock configuration format
        if (!IsValidMockConfiguration(mock.second)) {
            result.errors.push_back("Invalid mock configuration for: " + mock.first);
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

Status TestValidation::ValidateFixtures(const std::map<std::string, std::string>& fixtures, ValidationResult& result) {
    PROFILER_SCOPED_EVENT(0, "validate_fixtures");
    
    for (const auto& fixture : fixtures) {
        // Validate fixture name
        if (fixture.first.empty()) {
            result.errors.push_back("Empty fixture name found");
            return Status::FAILURE;
        }
        
        if (!IsValidIdentifier(fixture.first)) {
            result.errors.push_back("Invalid fixture name format: " + fixture.first);
            return Status::FAILURE;
        }
        
        // Validate fixture configuration
        if (fixture.second.empty()) {
            result.errors.push_back("Empty fixture configuration for: " + fixture.first);
            return Status::FAILURE;
        }
        
        // Validate fixture configuration format
        if (!IsValidFixtureConfiguration(fixture.second)) {
            result.errors.push_back("Invalid fixture configuration for: " + fixture.first);
            return Status::FAILURE;
        }
    }
    
    return Status::SUCCESS;
}

bool TestValidation::IsValidIdentifier(const std::string& identifier) {
    if (identifier.empty()) {
        return false;
    }
    
    // Check first character
    if (!std::isalpha(identifier[0]) && identifier[0] != '_') {
        return false;
    }
    
    // Check remaining characters
    for (size_t i = 1; i < identifier.length(); ++i) {
        if (!std::isalnum(identifier[i]) && identifier[i] != '_') {
            return false;
        }
    }
    
    return true;
}

bool TestValidation::IsValidMockConfiguration(const std::string& config) {
    // TODO: Implement mock configuration validation
    // For now, just check if it's not empty
    return !config.empty();
}

bool TestValidation::IsValidFixtureConfiguration(const std::string& config) {
    // TODO: Implement fixture configuration validation
    // For now, just check if it's not empty
    return !config.empty();
}

Status TestValidation::ValidateTestResult(const TestResult& result) {
    PROFILER_SCOPED_EVENT(0, "validate_test_result");
    
    ValidationResult validation_result;
    validation_result.spec_name = result.test_name;
    validation_result.is_valid = true;
    
    // Validate test name
    if (result.test_name.empty()) {
        validation_result.errors.push_back("Test name is required");
        validation_result.is_valid = false;
    }
    
    // Validate module name
    if (result.module_name.empty()) {
        validation_result.errors.push_back("Module name is required");
        validation_result.is_valid = false;
    }
    
    // Validate duration
    if (result.duration.count() < 0) {
        validation_result.errors.push_back("Invalid duration value");
        validation_result.is_valid = false;
    }
    
    // Validate code coverage
    if (result.code_coverage_percent < 0.0 || result.code_coverage_percent > 100.0) {
        validation_result.errors.push_back("Invalid code coverage percentage");
        validation_result.is_valid = false;
    }
    
    // Validate stability score
    if (result.stability_score < 0.0 || result.stability_score > 1.0) {
        validation_result.errors.push_back("Invalid stability score");
        validation_result.is_valid = false;
    }
    
    // Validate CPU usage
    if (result.cpu_usage_percent < 0.0 || result.cpu_usage_percent > 100.0) {
        validation_result.errors.push_back("Invalid CPU usage percentage");
        validation_result.is_valid = false;
    }
    
    // Validate memory usage
    if (result.memory_usage_mb < 0.0) {
        validation_result.errors.push_back("Invalid memory usage value");
        validation_result.is_valid = false;
    }
    
    // Validate network usage
    if (result.network_usage_mbps < 0.0) {
        validation_result.errors.push_back("Invalid network usage value");
        validation_result.is_valid = false;
    }
    
    // Store result
    validation_results_[result.test_name] = validation_result;
    
    return validation_result.is_valid ? Status::SUCCESS : Status::FAILURE;
}

Status TestValidation::ValidatePerformanceResult(const PerformanceTestResult& result) {
    PROFILER_SCOPED_EVENT(0, "validate_performance_result");
    
    ValidationResult validation_result;
    validation_result.spec_name = result.test_name;
    validation_result.is_valid = true;
    
    // Validate base test result
    TestResult base_result;
    base_result.test_name = result.test_name;
    base_result.module_name = result.module_name;
    base_result.duration = result.duration;
    base_result.code_coverage_percent = result.code_coverage_percent;
    base_result.stability_score = result.stability_score;
    base_result.cpu_usage_percent = result.cpu_usage_percent;
    base_result.memory_usage_mb = result.memory_usage_mb;
    base_result.network_usage_mbps = result.network_usage_mbps;
    
    Status status = ValidateTestResult(base_result);
    if (status != Status::SUCCESS) {
        validation_result.is_valid = false;
    }
    
    // Validate performance-specific fields
    if (result.avg_execution_time_us < 0.0) {
        validation_result.errors.push_back("Invalid average execution time");
        validation_result.is_valid = false;
    }
    
    if (result.min_execution_time_us < 0.0) {
        validation_result.errors.push_back("Invalid minimum execution time");
        validation_result.is_valid = false;
    }
    
    if (result.max_execution_time_us < 0.0) {
        validation_result.errors.push_back("Invalid maximum execution time");
        validation_result.is_valid = false;
    }
    
    if (result.std_dev_execution_time_us < 0.0) {
        validation_result.errors.push_back("Invalid standard deviation of execution time");
        validation_result.is_valid = false;
    }
    
    if (result.throughput_ops_per_sec < 0.0) {
        validation_result.errors.push_back("Invalid throughput value");
        validation_result.is_valid = false;
    }
    
    if (result.power_consumption_watts < 0.0) {
        validation_result.errors.push_back("Invalid power consumption value");
        validation_result.is_valid = false;
    }
    
    // Store result
    validation_results_[result.test_name] = validation_result;
    
    return validation_result.is_valid ? Status::SUCCESS : Status::FAILURE;
}

Status TestValidation::ValidateCoverageData(const FileCoverageData& coverage_data) {
    PROFILER_SCOPED_EVENT(0, "validate_coverage_data");
    
    ValidationResult validation_result;
    validation_result.spec_name = coverage_data.file_name;
    validation_result.is_valid = true;
    
    // Validate file path
    if (coverage_data.file_path.empty()) {
        validation_result.errors.push_back("File path is required");
        validation_result.is_valid = false;
    }
    
    // Validate file name
    if (coverage_data.file_name.empty()) {
        validation_result.errors.push_back("File name is required");
        validation_result.is_valid = false;
    }
    
    // Validate line coverage percentage
    if (coverage_data.line_coverage_percent < 0.0 || coverage_data.line_coverage_percent > 100.0) {
        validation_result.errors.push_back("Invalid line coverage percentage");
        validation_result.is_valid = false;
    }
    
    // Validate function coverage percentage
    if (coverage_data.function_coverage_percent < 0.0 || coverage_data.function_coverage_percent > 100.0) {
        validation_result.errors.push_back("Invalid function coverage percentage");
        validation_result.is_valid = false;
    }
    
    // Validate branch coverage percentage
    if (coverage_data.branch_coverage_percent < 0.0 || coverage_data.branch_coverage_percent > 100.0) {
        validation_result.errors.push_back("Invalid branch coverage percentage");
        validation_result.is_valid = false;
    }
    
    // Validate lines
    for (size_t i = 0; i < coverage_data.lines.size(); ++i) {
        const auto& line = coverage_data.lines[i];
        if (line.line_number == 0) {
            validation_result.errors.push_back("Invalid line number at index " + std::to_string(i));
            validation_result.is_valid = false;
        }
        
        // execution_count is uint32_t, so it can't be negative
    }
    
    // Validate functions
    for (size_t i = 0; i < coverage_data.functions.size(); ++i) {
        const auto& function = coverage_data.functions[i];
        if (function.function_name.empty()) {
            validation_result.errors.push_back("Empty function name at index " + std::to_string(i));
            validation_result.is_valid = false;
        }
        
        if (function.line_number == 0) {
            validation_result.errors.push_back("Invalid function line number at index " + std::to_string(i));
            validation_result.is_valid = false;
        }
        
        // execution_count is uint32_t, so it can't be negative
    }
    
    // Validate branches
    for (size_t i = 0; i < coverage_data.branches.size(); ++i) {
        const auto& branch = coverage_data.branches[i];
        if (branch.branch_type.empty()) {
            validation_result.errors.push_back("Empty branch type at index " + std::to_string(i));
            validation_result.is_valid = false;
        }
        
        if (branch.line_number == 0) {
            validation_result.errors.push_back("Invalid branch line number at index " + std::to_string(i));
            validation_result.is_valid = false;
        }
        
        // execution_count is uint32_t, so it can't be negative
    }
    
    // Store result
    validation_results_[coverage_data.file_name] = validation_result;
    
    return validation_result.is_valid ? Status::SUCCESS : Status::FAILURE;
}

Status TestValidation::ValidateTestEnvironment(const TestEnvironment& env) {
    PROFILER_SCOPED_EVENT(0, "validate_test_environment");
    
    ValidationResult validation_result;
    validation_result.spec_name = "test_environment";
    validation_result.is_valid = true;
    
    // Validate include directories
    for (const auto& dir : env.include_directories) {
        if (dir.empty()) {
            validation_result.errors.push_back("Empty include directory found");
            validation_result.is_valid = false;
        }
        
        if (!std::filesystem::exists(dir)) {
            validation_result.warnings.push_back("Include directory does not exist: " + dir);
        }
    }
    
    // Validate library directories
    for (const auto& dir : env.library_directories) {
        if (dir.empty()) {
            validation_result.errors.push_back("Empty library directory found");
            validation_result.is_valid = false;
        }
        
        if (!std::filesystem::exists(dir)) {
            validation_result.warnings.push_back("Library directory does not exist: " + dir);
        }
    }
    
    // Validate libraries
    for (const auto& lib : env.libraries) {
        if (lib.empty()) {
            validation_result.errors.push_back("Empty library found");
            validation_result.is_valid = false;
        }
    }
    
    // Validate environment variables
    for (const auto& var : env.environment_variables) {
        if (var.first.empty()) {
            validation_result.errors.push_back("Empty environment variable name found");
            validation_result.is_valid = false;
        }
    }
    
    // Validate temporary directories
    for (const auto& dir : env.temp_directories) {
        if (dir.empty()) {
            validation_result.errors.push_back("Empty temporary directory found");
            validation_result.is_valid = false;
        }
    }
    
    // Store result
    validation_results_["test_environment"] = validation_result;
    
    return validation_result.is_valid ? Status::SUCCESS : Status::FAILURE;
}

std::vector<ValidationResult> TestValidation::GetValidationResults() const {
    std::vector<ValidationResult> results;
    for (const auto& result : validation_results_) {
        results.push_back(result.second);
    }
    return results;
}

ValidationResult TestValidation::GetValidationResult(const std::string& spec_name) const {
    auto it = validation_results_.find(spec_name);
    if (it != validation_results_.end()) {
        return it->second;
    }
    return ValidationResult{};
}

ValidationStatistics TestValidation::GetValidationStatistics() const {
    ValidationStatistics stats;
    
    stats.total_validations = validation_results_.size();
    
    for (const auto& result : validation_results_) {
        if (result.second.is_valid) {
            stats.valid_count++;
        } else {
            stats.invalid_count++;
        }
        
        stats.total_errors += result.second.errors.size();
        stats.total_warnings += result.second.warnings.size();
    }
    
    if (stats.total_validations > 0) {
        stats.success_rate = static_cast<double>(stats.valid_count) / stats.total_validations;
    }
    
    return stats;
}

void TestValidation::ClearValidationResults() {
    validation_results_.clear();
}

Status TestValidation::ExportValidationReport(const std::string& output_file, ValidationReportFormat format) {
    PROFILER_SCOPED_EVENT(0, "export_validation_report");
    
    switch (format) {
        case ValidationReportFormat::HTML:
            return ExportHTMLReport(output_file);
        case ValidationReportFormat::XML:
            return ExportXMLReport(output_file);
        case ValidationReportFormat::JSON:
            return ExportJSONReport(output_file);
        case ValidationReportFormat::TEXT:
            return ExportTextReport(output_file);
        default:
            return Status::INVALID_ARGUMENT;
    }
}

Status TestValidation::ExportHTMLReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_html_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate HTML report
    file << "<!DOCTYPE html>\n";
    file << "<html>\n";
    file << "<head>\n";
    file << "  <title>Edge AI Engine Validation Report</title>\n";
    file << "  <style>\n";
    file << "    body { font-family: Arial, sans-serif; margin: 20px; }\n";
    file << "    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }\n";
    file << "    .summary { background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }\n";
    file << "    .error { color: #d32f2f; }\n";
    file << "    .warning { color: #f57c00; }\n";
    file << "    .success { color: #388e3c; }\n";
    file << "    .validation-result { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }\n";
    file << "    .validation-valid { background-color: #e8f5e8; }\n";
    file << "    .validation-invalid { background-color: #ffeaea; }\n";
    file << "    .validation-errors { margin: 5px 0; }\n";
    file << "    .validation-warnings { margin: 5px 0; }\n";
    file << "  </style>\n";
    file << "</head>\n";
    file << "<body>\n";
    
    // Header
    file << "  <div class=\"header\">\n";
    file << "    <h1>Edge AI Engine Validation Report</h1>\n";
    file << "    <p>Generated: " << GetCurrentTimestamp() << "</p>\n";
    file << "  </div>\n";
    
    // Summary
    auto stats = GetValidationStatistics();
    file << "  <div class=\"summary\">\n";
    file << "    <h2>Validation Summary</h2>\n";
    file << "    <p>Total Validations: " << stats.total_validations << "</p>\n";
    file << "    <p><span class=\"success\">Valid: " << stats.valid_count << "</span></p>\n";
    file << "    <p><span class=\"error\">Invalid: " << stats.invalid_count << "</span></p>\n";
    file << "    <p>Success Rate: " << std::fixed << std::setprecision(2) 
         << (stats.success_rate * 100.0) << "%</p>\n";
    file << "    <p><span class=\"error\">Total Errors: " << stats.total_errors << "</span></p>\n";
    file << "    <p><span class=\"warning\">Total Warnings: " << stats.total_warnings << "</span></p>\n";
    file << "  </div>\n";
    
    // Validation results
    file << "  <h2>Validation Results</h2>\n";
    for (const auto& result : validation_results_) {
        const auto& validation_result = result.second;
        std::string css_class = "validation-result ";
        if (validation_result.is_valid) {
            css_class += "validation-valid";
        } else {
            css_class += "validation-invalid";
        }
        
        file << "  <div class=\"" << css_class << "\">\n";
        file << "    <h3>" << (validation_result.is_valid ? "✓" : "✗") 
             << " " << validation_result.spec_name << "</h3>\n";
        
        if (!validation_result.errors.empty()) {
            file << "    <div class=\"validation-errors\">\n";
            file << "      <h4>Errors:</h4>\n";
            file << "      <ul>\n";
            for (const auto& error : validation_result.errors) {
                file << "        <li class=\"error\">" << error << "</li>\n";
            }
            file << "      </ul>\n";
            file << "    </div>\n";
        }
        
        if (!validation_result.warnings.empty()) {
            file << "    <div class=\"validation-warnings\">\n";
            file << "      <h4>Warnings:</h4>\n";
            file << "      <ul>\n";
            for (const auto& warning : validation_result.warnings) {
                file << "        <li class=\"warning\">" << warning << "</li>\n";
            }
            file << "      </ul>\n";
            file << "    </div>\n";
        }
        
        file << "  </div>\n";
    }
    
    file << "</body>\n";
    file << "</html>\n";
    
    file.close();
    return Status::SUCCESS;
}

Status TestValidation::ExportXMLReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_xml_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate XML report
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<validation_report>\n";
    file << "  <metadata>\n";
    file << "    <generated_at>" << GetCurrentTimestamp() << "</generated_at>\n";
    file << "  </metadata>\n";
    
    // Summary
    auto stats = GetValidationStatistics();
    file << "  <summary>\n";
    file << "    <total_validations>" << stats.total_validations << "</total_validations>\n";
    file << "    <valid_count>" << stats.valid_count << "</valid_count>\n";
    file << "    <invalid_count>" << stats.invalid_count << "</invalid_count>\n";
    file << "    <success_rate>" << std::fixed << std::setprecision(4) << stats.success_rate << "</success_rate>\n";
    file << "    <total_errors>" << stats.total_errors << "</total_errors>\n";
    file << "    <total_warnings>" << stats.total_warnings << "</total_warnings>\n";
    file << "  </summary>\n";
    
    // Validation results
    file << "  <validation_results>\n";
    for (const auto& result : validation_results_) {
        const auto& validation_result = result.second;
        file << "    <validation_result spec_name=\"" << validation_result.spec_name << "\" valid=\"" 
             << (validation_result.is_valid ? "true" : "false") << "\">\n";
        
        if (!validation_result.errors.empty()) {
            file << "      <errors>\n";
            for (const auto& error : validation_result.errors) {
                file << "        <error>" << EscapeXMLString(error) << "</error>\n";
            }
            file << "      </errors>\n";
        }
        
        if (!validation_result.warnings.empty()) {
            file << "      <warnings>\n";
            for (const auto& warning : validation_result.warnings) {
                file << "        <warning>" << EscapeXMLString(warning) << "</warning>\n";
            }
            file << "      </warnings>\n";
        }
        
        file << "    </validation_result>\n";
    }
    file << "  </validation_results>\n";
    file << "</validation_report>\n";
    
    file.close();
    return Status::SUCCESS;
}

Status TestValidation::ExportJSONReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_json_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate JSON report
    file << "{\n";
    file << "  \"metadata\": {\n";
    file << "    \"generated_at\": \"" << GetCurrentTimestamp() << "\"\n";
    file << "  },\n";
    
    // Summary
    auto stats = GetValidationStatistics();
    file << "  \"summary\": {\n";
    file << "    \"total_validations\": " << stats.total_validations << ",\n";
    file << "    \"valid_count\": " << stats.valid_count << ",\n";
    file << "    \"invalid_count\": " << stats.invalid_count << ",\n";
    file << "    \"success_rate\": " << std::fixed << std::setprecision(4) << stats.success_rate << ",\n";
    file << "    \"total_errors\": " << stats.total_errors << ",\n";
    file << "    \"total_warnings\": " << stats.total_warnings << "\n";
    file << "  },\n";
    
    // Validation results
    file << "  \"validation_results\": [\n";
    bool first = true;
    for (const auto& result : validation_results_) {
        if (!first) file << ",\n";
        const auto& validation_result = result.second;
        file << "    {\n";
        file << "      \"spec_name\": \"" << validation_result.spec_name << "\",\n";
        file << "      \"valid\": " << (validation_result.is_valid ? "true" : "false") << ",\n";
        
        if (!validation_result.errors.empty()) {
            file << "      \"errors\": [\n";
            for (size_t i = 0; i < validation_result.errors.size(); ++i) {
                if (i > 0) file << ",\n";
                file << "        \"" << EscapeJSONString(validation_result.errors[i]) << "\"";
            }
            file << "\n      ],\n";
        }
        
        if (!validation_result.warnings.empty()) {
            file << "      \"warnings\": [\n";
            for (size_t i = 0; i < validation_result.warnings.size(); ++i) {
                if (i > 0) file << ",\n";
                file << "        \"" << EscapeJSONString(validation_result.warnings[i]) << "\"";
            }
            file << "\n      ],\n";
        }
        
        file << "    }";
        first = false;
    }
    file << "\n  ]\n";
    file << "}\n";
    
    file.close();
    return Status::SUCCESS;
}

Status TestValidation::ExportTextReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_text_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate text report
    file << "Edge AI Engine Validation Report\n";
    file << "Generated: " << GetCurrentTimestamp() << "\n\n";
    
    // Summary
    auto stats = GetValidationStatistics();
    file << "=== Validation Summary ===\n";
    file << "Total Validations: " << stats.total_validations << "\n";
    file << "Valid: " << stats.valid_count << "\n";
    file << "Invalid: " << stats.invalid_count << "\n";
    file << "Success Rate: " << std::fixed << std::setprecision(2) 
         << (stats.success_rate * 100.0) << "%\n";
    file << "Total Errors: " << stats.total_errors << "\n";
    file << "Total Warnings: " << stats.total_warnings << "\n\n";
    
    // Validation results
    file << "=== Validation Results ===\n";
    for (const auto& result : validation_results_) {
        const auto& validation_result = result.second;
        file << (validation_result.is_valid ? "✓" : "✗") 
             << " " << validation_result.spec_name << "\n";
        
        if (!validation_result.errors.empty()) {
            file << "  Errors:\n";
            for (const auto& error : validation_result.errors) {
                file << "    - " << error << "\n";
            }
        }
        
        if (!validation_result.warnings.empty()) {
            file << "  Warnings:\n";
            for (const auto& warning : validation_result.warnings) {
                file << "    - " << warning << "\n";
            }
        }
        
        file << "\n";
    }
    
    file.close();
    return Status::SUCCESS;
}

std::string TestValidation::GetCurrentTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string TestValidation::EscapeXMLString(const std::string& str) {
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

std::string TestValidation::EscapeJSONString(const std::string& str) {
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

} // namespace testing
} // namespace edge_ai
