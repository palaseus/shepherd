#include <testing/test_coverage.h>
#include <profiling/profiler.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <cstdlib>

namespace edge_ai {
namespace testing {

// TestCoverage Implementation
TestCoverage::TestCoverage() {
    // Initialize default configuration
}

TestCoverage::~TestCoverage() {
    // Cleanup if needed
}

Status TestCoverage::SetConfiguration(const CoverageConfiguration& config) {
    config_ = config;
    return Status::SUCCESS;
}

Status TestCoverage::SetSourceDirectories(const std::vector<std::string>& directories) {
    source_directories_ = directories;
    return Status::SUCCESS;
}

Status TestCoverage::SetTestDirectories(const std::vector<std::string>& directories) {
    test_directories_ = directories;
    return Status::SUCCESS;
}

Status TestCoverage::SetExcludePatterns(const std::vector<std::string>& patterns) {
    exclude_patterns_ = patterns;
    return Status::SUCCESS;
}

Status TestCoverage::CollectCoverage() {
    PROFILER_SCOPED_EVENT(0, "collect_coverage");
    
    // Initialize coverage data
    coverage_data_.clear();
    
    // Collect coverage for each source directory
    for (const auto& source_dir : source_directories_) {
        Status status = CollectCoverageForDirectory(source_dir);
        if (status != Status::SUCCESS) {
            return status;
        }
    }
    
    // Process coverage data
    ProcessCoverageData();
    
    return Status::SUCCESS;
}

Status TestCoverage::CollectCoverageForDirectory(const std::string& directory) {
    PROFILER_SCOPED_EVENT(0, "collect_coverage_for_directory");
    
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                
                // Check if file should be included
                if (ShouldIncludeFile(file_path)) {
                    Status status = CollectCoverageForFile(file_path);
                    if (status != Status::SUCCESS) {
                        return status;
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        return Status::FAILURE;
    }
    
    return Status::SUCCESS;
}

Status TestCoverage::CollectCoverageForFile(const std::string& file_path) {
    PROFILER_SCOPED_EVENT(0, "collect_coverage_for_file");
    
    // Determine file type and collect coverage accordingly
    std::string extension = GetFileExtension(file_path);
    
    if (extension == "cpp" || extension == "cc" || extension == "cxx") {
        return CollectCoverageForCppFile(file_path);
    } else if (extension == "h" || extension == "hpp") {
        return CollectCoverageForHeaderFile(file_path);
    }
    
    return Status::SUCCESS;
}

Status TestCoverage::CollectCoverageForCppFile(const std::string& file_path) {
    PROFILER_SCOPED_EVENT(0, "collect_coverage_for_cpp_file");
    
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return Status::FAILURE;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // Parse file for coverage information
        FileCoverageData file_coverage;
        file_coverage.file_path = file_path;
        file_coverage.file_name = GetFileName(file_path);
        
        // Parse lines
        ParseLinesForCoverage(content, file_coverage);
        
        // Parse functions
        ParseFunctionsForCoverage(content, file_coverage);
        
        // Parse branches
        ParseBranchesForCoverage(content, file_coverage);
        
        // Store coverage data
        coverage_data_[file_path] = file_coverage;
        
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
    
    return Status::SUCCESS;
}

Status TestCoverage::CollectCoverageForHeaderFile(const std::string& file_path) {
    PROFILER_SCOPED_EVENT(0, "collect_coverage_for_header_file");
    
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return Status::FAILURE;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // Parse header file for coverage information
        FileCoverageData file_coverage;
        file_coverage.file_path = file_path;
        file_coverage.file_name = GetFileName(file_path);
        
        // Parse inline functions
        ParseInlineFunctionsForCoverage(content, file_coverage);
        
        // Parse template functions
        ParseTemplateFunctionsForCoverage(content, file_coverage);
        
        // Store coverage data
        coverage_data_[file_path] = file_coverage;
        
    } catch (const std::exception& e) {
        return Status::FAILURE;
    }
    
    return Status::SUCCESS;
}

void TestCoverage::ParseLinesForCoverage(const std::string& content, FileCoverageData& file_coverage) {
    PROFILER_SCOPED_EVENT(0, "parse_lines_for_coverage");
    
    std::istringstream stream(content);
    std::string line;
    uint32_t line_number = 1;
    
    while (std::getline(stream, line)) {
        LineCoverageData line_coverage;
        line_coverage.line_number = line_number;
        line_coverage.content = line;
        line_coverage.is_executable = IsExecutableLine(line);
        line_coverage.is_covered = false; // Will be updated by coverage tools
        line_coverage.execution_count = 0;
        
        file_coverage.lines.push_back(line_coverage);
        line_number++;
    }
}

void TestCoverage::ParseFunctionsForCoverage(const std::string& content, FileCoverageData& file_coverage) {
    PROFILER_SCOPED_EVENT(0, "parse_functions_for_coverage");
    
    // Regular expression for function definitions
    std::regex function_regex(R"((\w+(?:\s*::\s*\w+)*)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:\{|;))");
    
    std::sregex_iterator iter(content.begin(), content.end(), function_regex);
    std::sregex_iterator end;
    
    for (auto it = iter; it != end; ++it) {
        FunctionCoverageData function_coverage;
        function_coverage.function_name = (*it)[2].str();
        function_coverage.return_type = (*it)[1].str();
        function_coverage.is_covered = false; // Will be updated by coverage tools
        function_coverage.execution_count = 0;
        function_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.functions.push_back(function_coverage);
    }
}

void TestCoverage::ParseBranchesForCoverage(const std::string& content, FileCoverageData& file_coverage) {
    PROFILER_SCOPED_EVENT(0, "parse_branches_for_coverage");
    
    // Regular expressions for different branch types
    std::regex if_regex(R"(\bif\s*\()");
    std::regex else_regex(R"(\belse\b)");
    std::regex switch_regex(R"(\bswitch\s*\()");
    std::regex case_regex(R"(\bcase\s+)");
    std::regex default_regex(R"(\bdefault\s*:)");
    std::regex ternary_regex(R"(\?\s*[^:]+\s*:)");
    
    // Find if statements
    std::sregex_iterator if_iter(content.begin(), content.end(), if_regex);
    std::sregex_iterator end;
    
    for (auto it = if_iter; it != end; ++it) {
        BranchCoverageData branch_coverage;
        branch_coverage.branch_type = "if";
        branch_coverage.is_covered = false; // Will be updated by coverage tools
        branch_coverage.execution_count = 0;
        branch_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.branches.push_back(branch_coverage);
    }
    
    // Find else statements
    std::sregex_iterator else_iter(content.begin(), content.end(), else_regex);
    for (auto it = else_iter; it != end; ++it) {
        BranchCoverageData branch_coverage;
        branch_coverage.branch_type = "else";
        branch_coverage.is_covered = false; // Will be updated by coverage tools
        branch_coverage.execution_count = 0;
        branch_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.branches.push_back(branch_coverage);
    }
    
    // Find switch statements
    std::sregex_iterator switch_iter(content.begin(), content.end(), switch_regex);
    for (auto it = switch_iter; it != end; ++it) {
        BranchCoverageData branch_coverage;
        branch_coverage.branch_type = "switch";
        branch_coverage.is_covered = false; // Will be updated by coverage tools
        branch_coverage.execution_count = 0;
        branch_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.branches.push_back(branch_coverage);
    }
    
    // Find case statements
    std::sregex_iterator case_iter(content.begin(), content.end(), case_regex);
    for (auto it = case_iter; it != end; ++it) {
        BranchCoverageData branch_coverage;
        branch_coverage.branch_type = "case";
        branch_coverage.is_covered = false; // Will be updated by coverage tools
        branch_coverage.execution_count = 0;
        branch_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.branches.push_back(branch_coverage);
    }
    
    // Find default statements
    std::sregex_iterator default_iter(content.begin(), content.end(), default_regex);
    for (auto it = default_iter; it != end; ++it) {
        BranchCoverageData branch_coverage;
        branch_coverage.branch_type = "default";
        branch_coverage.is_covered = false; // Will be updated by coverage tools
        branch_coverage.execution_count = 0;
        branch_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.branches.push_back(branch_coverage);
    }
    
    // Find ternary operators
    std::sregex_iterator ternary_iter(content.begin(), content.end(), ternary_regex);
    for (auto it = ternary_iter; it != end; ++it) {
        BranchCoverageData branch_coverage;
        branch_coverage.branch_type = "ternary";
        branch_coverage.is_covered = false; // Will be updated by coverage tools
        branch_coverage.execution_count = 0;
        branch_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.branches.push_back(branch_coverage);
    }
}

void TestCoverage::ParseInlineFunctionsForCoverage(const std::string& content, FileCoverageData& file_coverage) {
    PROFILER_SCOPED_EVENT(0, "parse_inline_functions_for_coverage");
    
    // Regular expression for inline functions
    std::regex inline_regex(R"(\binline\s+(\w+(?:\s*::\s*\w+)*)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:\{|;))");
    
    std::sregex_iterator iter(content.begin(), content.end(), inline_regex);
    std::sregex_iterator end;
    
    for (auto it = iter; it != end; ++it) {
        FunctionCoverageData function_coverage;
        function_coverage.function_name = (*it)[2].str();
        function_coverage.return_type = (*it)[1].str();
        function_coverage.is_covered = false; // Will be updated by coverage tools
        function_coverage.execution_count = 0;
        function_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.functions.push_back(function_coverage);
    }
}

void TestCoverage::ParseTemplateFunctionsForCoverage(const std::string& content, FileCoverageData& file_coverage) {
    PROFILER_SCOPED_EVENT(0, "parse_template_functions_for_coverage");
    
    // Regular expression for template functions
    std::regex template_regex(R"(\btemplate\s*<[^>]*>\s*(\w+(?:\s*::\s*\w+)*)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:\{|;))");
    
    std::sregex_iterator iter(content.begin(), content.end(), template_regex);
    std::sregex_iterator end;
    
    for (auto it = iter; it != end; ++it) {
        FunctionCoverageData function_coverage;
        function_coverage.function_name = (*it)[2].str();
        function_coverage.return_type = (*it)[1].str();
        function_coverage.is_covered = false; // Will be updated by coverage tools
        function_coverage.execution_count = 0;
        function_coverage.line_number = GetLineNumberForMatch(content, it->position());
        
        file_coverage.functions.push_back(function_coverage);
    }
}

void TestCoverage::ProcessCoverageData() {
    PROFILER_SCOPED_EVENT(0, "process_coverage_data");
    
    // Calculate overall coverage statistics
    CalculateOverallCoverage();
    
    // Calculate per-file coverage statistics
    CalculatePerFileCoverage();
    
    // Calculate per-module coverage statistics
    CalculatePerModuleCoverage();
}

void TestCoverage::CalculateOverallCoverage() {
    PROFILER_SCOPED_EVENT(0, "calculate_overall_coverage");
    
    overall_coverage_.total_lines = 0;
    overall_coverage_.covered_lines = 0;
    overall_coverage_.total_functions = 0;
    overall_coverage_.covered_functions = 0;
    overall_coverage_.total_branches = 0;
    overall_coverage_.covered_branches = 0;
    
    for (const auto& file_pair : coverage_data_) {
        const auto& file_coverage = file_pair.second;
        
        // Count lines
        for (const auto& line : file_coverage.lines) {
            if (line.is_executable) {
                overall_coverage_.total_lines++;
                if (line.is_covered) {
                    overall_coverage_.covered_lines++;
                }
            }
        }
        
        // Count functions
        overall_coverage_.total_functions += file_coverage.functions.size();
        for (const auto& function : file_coverage.functions) {
            if (function.is_covered) {
                overall_coverage_.covered_functions++;
            }
        }
        
        // Count branches
        overall_coverage_.total_branches += file_coverage.branches.size();
        for (const auto& branch : file_coverage.branches) {
            if (branch.is_covered) {
                overall_coverage_.covered_branches++;
            }
        }
    }
    
    // Calculate percentages
    if (overall_coverage_.total_lines > 0) {
        overall_coverage_.line_coverage_percent = 
            (static_cast<double>(overall_coverage_.covered_lines) / overall_coverage_.total_lines) * 100.0;
    }
    
    if (overall_coverage_.total_functions > 0) {
        overall_coverage_.function_coverage_percent = 
            (static_cast<double>(overall_coverage_.covered_functions) / overall_coverage_.total_functions) * 100.0;
    }
    
    if (overall_coverage_.total_branches > 0) {
        overall_coverage_.branch_coverage_percent = 
            (static_cast<double>(overall_coverage_.covered_branches) / overall_coverage_.total_branches) * 100.0;
    }
}

void TestCoverage::CalculatePerFileCoverage() {
    PROFILER_SCOPED_EVENT(0, "calculate_per_file_coverage");
    
    for (auto& file_pair : coverage_data_) {
        auto& file_coverage = file_pair.second;
        
        // Calculate line coverage
        uint32_t total_lines = 0;
        uint32_t covered_lines = 0;
        
        for (const auto& line : file_coverage.lines) {
            if (line.is_executable) {
                total_lines++;
                if (line.is_covered) {
                    covered_lines++;
                }
            }
        }
        
        if (total_lines > 0) {
            file_coverage.line_coverage_percent = (static_cast<double>(covered_lines) / total_lines) * 100.0;
        }
        
        // Calculate function coverage
        uint32_t total_functions = file_coverage.functions.size();
        uint32_t covered_functions = 0;
        
        for (const auto& function : file_coverage.functions) {
            if (function.is_covered) {
                covered_functions++;
            }
        }
        
        if (total_functions > 0) {
            file_coverage.function_coverage_percent = (static_cast<double>(covered_functions) / total_functions) * 100.0;
        }
        
        // Calculate branch coverage
        uint32_t total_branches = file_coverage.branches.size();
        uint32_t covered_branches = 0;
        
        for (const auto& branch : file_coverage.branches) {
            if (branch.is_covered) {
                covered_branches++;
            }
        }
        
        if (total_branches > 0) {
            file_coverage.branch_coverage_percent = (static_cast<double>(covered_branches) / total_branches) * 100.0;
        }
    }
}

void TestCoverage::CalculatePerModuleCoverage() {
    PROFILER_SCOPED_EVENT(0, "calculate_per_module_coverage");
    
    module_coverage_.clear();
    
    for (const auto& file_pair : coverage_data_) {
        const auto& file_coverage = file_pair.second;
        std::string module_name = GetModuleNameFromPath(file_coverage.file_path);
        
        auto& module_stats = module_coverage_[module_name];
        
        // Count lines
        for (const auto& line : file_coverage.lines) {
            if (line.is_executable) {
                module_stats.total_lines++;
                if (line.is_covered) {
                    module_stats.covered_lines++;
                }
            }
        }
        
        // Count functions
        module_stats.total_functions += file_coverage.functions.size();
        for (const auto& function : file_coverage.functions) {
            if (function.is_covered) {
                module_stats.covered_functions++;
            }
        }
        
        // Count branches
        module_stats.total_branches += file_coverage.branches.size();
        for (const auto& branch : file_coverage.branches) {
            if (branch.is_covered) {
                module_stats.covered_branches++;
            }
        }
    }
    
    // Calculate percentages for each module
    for (auto& module_pair : module_coverage_) {
        auto& module_stats = module_pair.second;
        
        if (module_stats.total_lines > 0) {
            module_stats.line_coverage_percent = 
                (static_cast<double>(module_stats.covered_lines) / module_stats.total_lines) * 100.0;
        }
        
        if (module_stats.total_functions > 0) {
            module_stats.function_coverage_percent = 
                (static_cast<double>(module_stats.covered_functions) / module_stats.total_functions) * 100.0;
        }
        
        if (module_stats.total_branches > 0) {
            module_stats.branch_coverage_percent = 
                (static_cast<double>(module_stats.covered_branches) / module_stats.total_branches) * 100.0;
        }
    }
}

bool TestCoverage::ShouldIncludeFile(const std::string& file_path) {
    // Check exclude patterns
    for (const auto& pattern : exclude_patterns_) {
        if (std::regex_match(file_path, std::regex(pattern))) {
            return false;
        }
    }
    
    // Check file extension
    std::string extension = GetFileExtension(file_path);
    return (extension == "cpp" || extension == "cc" || extension == "cxx" || 
            extension == "h" || extension == "hpp");
}

std::string TestCoverage::GetFileExtension(const std::string& file_path) {
    size_t dot_pos = file_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        std::string extension = file_path.substr(dot_pos + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        return extension;
    }
    return "";
}

std::string TestCoverage::GetFileName(const std::string& file_path) {
    std::filesystem::path path(file_path);
    return path.filename().string();
}

std::string TestCoverage::GetModuleNameFromPath(const std::string& file_path) {
    std::filesystem::path path(file_path);
    std::string filename = path.stem().string();
    
    // Remove common suffixes
    std::vector<std::string> suffixes = {"_impl", "_internal", "Impl", "Internal"};
    for (const auto& suffix : suffixes) {
        if (filename.length() > suffix.length() && 
            filename.substr(filename.length() - suffix.length()) == suffix) {
            filename = filename.substr(0, filename.length() - suffix.length());
            break;
        }
    }
    
    return filename;
}

bool TestCoverage::IsExecutableLine(const std::string& line) {
    // Remove leading whitespace
    std::string trimmed = line;
    trimmed.erase(0, trimmed.find_first_not_of(" \t"));
    
    // Skip empty lines
    if (trimmed.empty()) {
        return false;
    }
    
    // Skip comments
    if (trimmed[0] == '/' && (trimmed[1] == '/' || trimmed[1] == '*')) {
        return false;
    }
    
    // Skip preprocessor directives
    if (trimmed[0] == '#') {
        return false;
    }
    
    // Skip closing braces
    if (trimmed == "}" || trimmed == "};") {
        return false;
    }
    
    // Skip opening braces
    if (trimmed == "{") {
        return false;
    }
    
    // Skip namespace declarations
    if (trimmed.substr(0, 9) == "namespace") {
        return false;
    }
    
    // Skip using declarations
    if (trimmed.substr(0, 5) == "using") {
        return false;
    }
    
    // Skip include statements
    if (trimmed.substr(0, 8) == "#include") {
        return false;
    }
    
    return true;
}

uint32_t TestCoverage::GetLineNumberForMatch(const std::string& content, size_t position) {
    uint32_t line_number = 1;
    for (size_t i = 0; i < position && i < content.length(); ++i) {
        if (content[i] == '\n') {
            line_number++;
        }
    }
    return line_number;
}

OverallCoverageData TestCoverage::GetOverallCoverage() const {
    return overall_coverage_;
}

std::map<std::string, FileCoverageData> TestCoverage::GetFileCoverage() const {
    return coverage_data_;
}

std::map<std::string, ModuleCoverageData> TestCoverage::GetModuleCoverage() const {
    return module_coverage_;
}

CoverageStatistics TestCoverage::GetCoverageStatistics() const {
    CoverageStatistics stats;
    
    stats.total_files = coverage_data_.size();
    stats.total_modules = module_coverage_.size();
    
    stats.overall_line_coverage = overall_coverage_.line_coverage_percent;
    stats.overall_function_coverage = overall_coverage_.function_coverage_percent;
    stats.overall_branch_coverage = overall_coverage_.branch_coverage_percent;
    
    // Calculate average coverage per file
    double total_file_coverage = 0.0;
    for (const auto& file_pair : coverage_data_) {
        total_file_coverage += file_pair.second.line_coverage_percent;
    }
    stats.avg_file_coverage = (stats.total_files > 0) ? (total_file_coverage / stats.total_files) : 0.0;
    
    // Calculate average coverage per module
    double total_module_coverage = 0.0;
    for (const auto& module_pair : module_coverage_) {
        total_module_coverage += module_pair.second.line_coverage_percent;
    }
    stats.avg_module_coverage = (stats.total_modules > 0) ? (total_module_coverage / stats.total_modules) : 0.0;
    
    return stats;
}

Status TestCoverage::ExportCoverageReport(const std::string& output_file, CoverageReportFormat format) {
    PROFILER_SCOPED_EVENT(0, "export_coverage_report");
    
    switch (format) {
        case CoverageReportFormat::HTML:
            return ExportHTMLReport(output_file);
        case CoverageReportFormat::XML:
            return ExportXMLReport(output_file);
        case CoverageReportFormat::JSON:
            return ExportJSONReport(output_file);
        case CoverageReportFormat::LCOV:
            return ExportLCOVReport(output_file);
        default:
            return Status::INVALID_ARGUMENT;
    }
}

Status TestCoverage::ExportHTMLReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_html_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate HTML report
    file << "<!DOCTYPE html>\n";
    file << "<html>\n";
    file << "<head>\n";
    file << "  <title>Edge AI Engine Coverage Report</title>\n";
    file << "  <style>\n";
    file << "    body { font-family: Arial, sans-serif; margin: 20px; }\n";
    file << "    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }\n";
    file << "    .summary { background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }\n";
    file << "    .coverage-good { color: #388e3c; }\n";
    file << "    .coverage-warning { color: #f57c00; }\n";
    file << "    .coverage-bad { color: #d32f2f; }\n";
    file << "    table { border-collapse: collapse; width: 100%; }\n";
    file << "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
    file << "    th { background-color: #f2f2f2; }\n";
    file << "  </style>\n";
    file << "</head>\n";
    file << "<body>\n";
    
    // Header
    file << "  <div class=\"header\">\n";
    file << "    <h1>Edge AI Engine Coverage Report</h1>\n";
    file << "    <p>Generated: " << GetCurrentTimestamp() << "</p>\n";
    file << "  </div>\n";
    
    // Summary
    file << "  <div class=\"summary\">\n";
    file << "    <h2>Overall Coverage</h2>\n";
    file << "    <p>Line Coverage: " << std::fixed << std::setprecision(2) 
         << overall_coverage_.line_coverage_percent << "%</p>\n";
    file << "    <p>Function Coverage: " << std::fixed << std::setprecision(2) 
         << overall_coverage_.function_coverage_percent << "%</p>\n";
    file << "    <p>Branch Coverage: " << std::fixed << std::setprecision(2) 
         << overall_coverage_.branch_coverage_percent << "%</p>\n";
    file << "  </div>\n";
    
    // Module breakdown
    file << "  <h2>Module Coverage</h2>\n";
    file << "  <table>\n";
    file << "    <tr><th>Module</th><th>Line Coverage</th><th>Function Coverage</th><th>Branch Coverage</th></tr>\n";
    for (const auto& module_pair : module_coverage_) {
        const auto& module_stats = module_pair.second;
        file << "    <tr>\n";
        file << "      <td>" << module_pair.first << "</td>\n";
        file << "      <td>" << std::fixed << std::setprecision(2) << module_stats.line_coverage_percent << "%</td>\n";
        file << "      <td>" << std::fixed << std::setprecision(2) << module_stats.function_coverage_percent << "%</td>\n";
        file << "      <td>" << std::fixed << std::setprecision(2) << module_stats.branch_coverage_percent << "%</td>\n";
        file << "    </tr>\n";
    }
    file << "  </table>\n";
    
    file << "</body>\n";
    file << "</html>\n";
    
    file.close();
    return Status::SUCCESS;
}

Status TestCoverage::ExportXMLReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_xml_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate XML report
    file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    file << "<coverage>\n";
    file << "  <overall>\n";
    file << "    <line_coverage>" << std::fixed << std::setprecision(2) 
         << overall_coverage_.line_coverage_percent << "</line_coverage>\n";
    file << "    <function_coverage>" << std::fixed << std::setprecision(2) 
         << overall_coverage_.function_coverage_percent << "</function_coverage>\n";
    file << "    <branch_coverage>" << std::fixed << std::setprecision(2) 
         << overall_coverage_.branch_coverage_percent << "</branch_coverage>\n";
    file << "  </overall>\n";
    
    // Module coverage
    file << "  <modules>\n";
    for (const auto& module_pair : module_coverage_) {
        const auto& module_stats = module_pair.second;
        file << "    <module name=\"" << module_pair.first << "\">\n";
        file << "      <line_coverage>" << std::fixed << std::setprecision(2) 
             << module_stats.line_coverage_percent << "</line_coverage>\n";
        file << "      <function_coverage>" << std::fixed << std::setprecision(2) 
             << module_stats.function_coverage_percent << "</function_coverage>\n";
        file << "      <branch_coverage>" << std::fixed << std::setprecision(2) 
             << module_stats.branch_coverage_percent << "</branch_coverage>\n";
        file << "    </module>\n";
    }
    file << "  </modules>\n";
    
    file << "</coverage>\n";
    
    file.close();
    return Status::SUCCESS;
}

Status TestCoverage::ExportJSONReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_json_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate JSON report
    file << "{\n";
    file << "  \"overall_coverage\": {\n";
    file << "    \"line_coverage\": " << std::fixed << std::setprecision(2) 
         << overall_coverage_.line_coverage_percent << ",\n";
    file << "    \"function_coverage\": " << std::fixed << std::setprecision(2) 
         << overall_coverage_.function_coverage_percent << ",\n";
    file << "    \"branch_coverage\": " << std::fixed << std::setprecision(2) 
         << overall_coverage_.branch_coverage_percent << "\n";
    file << "  },\n";
    
    // Module coverage
    file << "  \"module_coverage\": {\n";
    bool first = true;
    for (const auto& module_pair : module_coverage_) {
        if (!first) file << ",\n";
        const auto& module_stats = module_pair.second;
        file << "    \"" << module_pair.first << "\": {\n";
        file << "      \"line_coverage\": " << std::fixed << std::setprecision(2) 
             << module_stats.line_coverage_percent << ",\n";
        file << "      \"function_coverage\": " << std::fixed << std::setprecision(2) 
             << module_stats.function_coverage_percent << ",\n";
        file << "      \"branch_coverage\": " << std::fixed << std::setprecision(2) 
             << module_stats.branch_coverage_percent << "\n";
        file << "    }";
        first = false;
    }
    file << "\n  }\n";
    file << "}\n";
    
    file.close();
    return Status::SUCCESS;
}

Status TestCoverage::ExportLCOVReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "export_lcov_report");
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        return Status::FAILURE;
    }
    
    // Generate LCOV report
    for (const auto& file_pair : coverage_data_) {
        const auto& file_coverage = file_pair.second;
        
        file << "TN:\n"; // Test name
        file << "SF:" << file_coverage.file_path << "\n"; // Source file
        
        // Function coverage
        for (const auto& function : file_coverage.functions) {
            file << "FN:" << function.line_number << "," << function.function_name << "\n";
            if (function.is_covered) {
                file << "FNDA:" << function.execution_count << "," << function.function_name << "\n";
            }
        }
        file << "FNF:" << file_coverage.functions.size() << "\n"; // Functions found
        file << "FNH:" << std::count_if(file_coverage.functions.begin(), file_coverage.functions.end(),
                                       [](const FunctionCoverageData& f) { return f.is_covered; }) << "\n"; // Functions hit
        
        // Line coverage
        for (const auto& line : file_coverage.lines) {
            if (line.is_executable) {
                file << "DA:" << line.line_number << "," << line.execution_count << "\n";
            }
        }
        file << "LF:" << std::count_if(file_coverage.lines.begin(), file_coverage.lines.end(),
                                      [](const LineCoverageData& l) { return l.is_executable; }) << "\n"; // Lines found
        file << "LH:" << std::count_if(file_coverage.lines.begin(), file_coverage.lines.end(),
                                      [](const LineCoverageData& l) { return l.is_executable && l.is_covered; }) << "\n"; // Lines hit
        
        // Branch coverage
        for (const auto& branch : file_coverage.branches) {
            file << "BRDA:" << branch.line_number << ",0,0," << (branch.is_covered ? "1" : "0") << "\n";
        }
        file << "BRF:" << file_coverage.branches.size() << "\n"; // Branches found
        file << "BRH:" << std::count_if(file_coverage.branches.begin(), file_coverage.branches.end(),
                                       [](const BranchCoverageData& b) { return b.is_covered; }) << "\n"; // Branches hit
        
        file << "end_of_record\n";
    }
    
    file.close();
    return Status::SUCCESS;
}

std::string TestCoverage::GetCurrentTimestamp() {
    auto now = std::time(nullptr);
    auto tm = *std::localtime(&now);
    
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

} // namespace testing
} // namespace edge_ai
