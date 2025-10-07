#ifndef EDGE_AI_ENGINE_TEST_COVERAGE_H
#define EDGE_AI_ENGINE_TEST_COVERAGE_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <filesystem>
#include <regex>
#include <core/types.h>

namespace edge_ai {
namespace testing {

// Coverage configuration
struct CoverageConfiguration {
    bool enable_line_coverage = true;
    bool enable_function_coverage = true;
    bool enable_branch_coverage = true;
    bool enable_condition_coverage = false;
    bool enable_path_coverage = false;
    double coverage_threshold = 80.0;
    std::vector<std::string> source_directories;
    std::vector<std::string> test_directories;
    std::vector<std::string> exclude_patterns;
    std::vector<std::string> include_patterns;
    std::string coverage_tool = "gcov";
    std::string coverage_format = "html";
    std::string output_directory = "coverage/";
};

// Coverage report formats
enum class CoverageReportFormat {
    HTML,
    XML,
    JSON,
    LCOV
};

// Line coverage data
struct LineCoverageData {
    uint32_t line_number = 0;
    std::string content;
    bool is_executable = false;
    bool is_covered = false;
    uint32_t execution_count = 0;
    std::vector<uint32_t> branch_conditions;
};

// Function coverage data
struct FunctionCoverageData {
    std::string function_name;
    std::string return_type;
    uint32_t line_number = 0;
    bool is_covered = false;
    uint32_t execution_count = 0;
    std::vector<uint32_t> call_sites;
};

// Branch coverage data
struct BranchCoverageData {
    std::string branch_type;
    uint32_t line_number = 0;
    bool is_covered = false;
    uint32_t execution_count = 0;
    std::vector<uint32_t> conditions;
};

// File coverage data
struct FileCoverageData {
    std::string file_path;
    std::string file_name;
    double line_coverage_percent = 0.0;
    double function_coverage_percent = 0.0;
    double branch_coverage_percent = 0.0;
    std::vector<LineCoverageData> lines;
    std::vector<FunctionCoverageData> functions;
    std::vector<BranchCoverageData> branches;
};

// Module coverage data
struct ModuleCoverageData {
    uint32_t total_lines = 0;
    uint32_t covered_lines = 0;
    uint32_t total_functions = 0;
    uint32_t covered_functions = 0;
    uint32_t total_branches = 0;
    uint32_t covered_branches = 0;
    double line_coverage_percent = 0.0;
    double function_coverage_percent = 0.0;
    double branch_coverage_percent = 0.0;
};

// Overall coverage data
struct OverallCoverageData {
    uint32_t total_lines = 0;
    uint32_t covered_lines = 0;
    uint32_t total_functions = 0;
    uint32_t covered_functions = 0;
    uint32_t total_branches = 0;
    uint32_t covered_branches = 0;
    double line_coverage_percent = 0.0;
    double function_coverage_percent = 0.0;
    double branch_coverage_percent = 0.0;
};

// Coverage statistics
struct CoverageStatistics {
    uint32_t total_files = 0;
    uint32_t total_modules = 0;
    double overall_line_coverage = 0.0;
    double overall_function_coverage = 0.0;
    double overall_branch_coverage = 0.0;
    double avg_file_coverage = 0.0;
    double avg_module_coverage = 0.0;
};

// TestCoverage class
class TestCoverage {
public:
    TestCoverage();
    ~TestCoverage();

    // Configuration
    Status SetConfiguration(const CoverageConfiguration& config);
    Status SetSourceDirectories(const std::vector<std::string>& directories);
    Status SetTestDirectories(const std::vector<std::string>& directories);
    Status SetExcludePatterns(const std::vector<std::string>& patterns);

    // Coverage collection
    Status CollectCoverage();
    Status CollectCoverageForDirectory(const std::string& directory);
    Status CollectCoverageForFile(const std::string& file_path);
    Status CollectCoverageForCppFile(const std::string& file_path);
    Status CollectCoverageForHeaderFile(const std::string& file_path);

    // Coverage processing
    void ProcessCoverageData();
    void CalculateOverallCoverage();
    void CalculatePerFileCoverage();
    void CalculatePerModuleCoverage();

    // Coverage data access
    OverallCoverageData GetOverallCoverage() const;
    std::map<std::string, FileCoverageData> GetFileCoverage() const;
    std::map<std::string, ModuleCoverageData> GetModuleCoverage() const;
    CoverageStatistics GetCoverageStatistics() const;

    // Coverage report generation
    Status ExportCoverageReport(const std::string& output_file, CoverageReportFormat format);
    Status ExportHTMLReport(const std::string& output_file);
    Status ExportXMLReport(const std::string& output_file);
    Status ExportJSONReport(const std::string& output_file);
    Status ExportLCOVReport(const std::string& output_file);

    // Utility functions
    std::string GetCurrentTimestamp();

private:
    // Parsing methods
    void ParseLinesForCoverage(const std::string& content, FileCoverageData& file_coverage);
    void ParseFunctionsForCoverage(const std::string& content, FileCoverageData& file_coverage);
    void ParseBranchesForCoverage(const std::string& content, FileCoverageData& file_coverage);
    void ParseInlineFunctionsForCoverage(const std::string& content, FileCoverageData& file_coverage);
    void ParseTemplateFunctionsForCoverage(const std::string& content, FileCoverageData& file_coverage);

    // Utility methods
    bool ShouldIncludeFile(const std::string& file_path);
    std::string GetFileExtension(const std::string& file_path);
    std::string GetFileName(const std::string& file_path);
    std::string GetModuleNameFromPath(const std::string& file_path);
    bool IsExecutableLine(const std::string& line);
    uint32_t GetLineNumberForMatch(const std::string& content, size_t position);

    // Configuration
    CoverageConfiguration config_;
    std::vector<std::string> source_directories_;
    std::vector<std::string> test_directories_;
    std::vector<std::string> exclude_patterns_;

    // Coverage data
    std::map<std::string, FileCoverageData> coverage_data_;
    std::map<std::string, ModuleCoverageData> module_coverage_;
    OverallCoverageData overall_coverage_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_COVERAGE_H
