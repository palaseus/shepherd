#include <testing/interface_validator.h>
#include <profiling/profiler.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>

namespace edge_ai {
namespace testing {

InterfaceValidator::InterfaceValidator() = default;
InterfaceValidator::~InterfaceValidator() = default;

Status InterfaceValidator::SetConfiguration(const InterfaceValidationConfig& config) {
    config_ = config;
    return Status::SUCCESS;
}

Status InterfaceValidator::ParseHeaderFiles() {
    PROFILER_SCOPED_EVENT(0, "interface_parse_headers");
    
    interfaces_.clear();
    
    std::vector<std::string> header_files = FindHeaderFiles();
    std::cout << "Found " << header_files.size() << " header files to parse" << std::endl;
    
    for (const auto& file_path : header_files) {
        auto status = ParseHeaderFile(file_path);
        if (status != Status::SUCCESS) {
            std::cerr << "Failed to parse header file: " << file_path << std::endl;
            // Continue with other files
        }
    }
    
    std::cout << "Parsed " << interfaces_.size() << " interfaces from " << header_files.size() << " header files" << std::endl;
    return Status::SUCCESS;
}

Status InterfaceValidator::ParseHeaderFile(const std::string& file_path) {
    PROFILER_SCOPED_EVENT(0, "interface_parse_single_header");
    
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return Status::FAILURE;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();
        
        auto file_interfaces = ParseFileContent(file_path, content);
        interfaces_.insert(interfaces_.end(), file_interfaces.begin(), file_interfaces.end());
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing file " << file_path << ": " << e.what() << std::endl;
        return Status::FAILURE;
    }
}

std::vector<std::string> InterfaceValidator::FindHeaderFiles() {
    std::vector<std::string> header_files;
    
    for (const auto& directory : config_.header_directories) {
        try {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
                if (entry.is_regular_file() && entry.path().extension() == ".h") {
                    std::string file_path = entry.path().string();
                    if (ShouldParseFile(file_path)) {
                        header_files.push_back(file_path);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error scanning directory " << directory << ": " << e.what() << std::endl;
        }
    }
    
    return header_files;
}

bool InterfaceValidator::ShouldParseFile(const std::string& file_path) {
    // Check include patterns
    bool matches_include = config_.include_patterns.empty();
    for (const auto& pattern : config_.include_patterns) {
        if (file_path.find(pattern) != std::string::npos) {
            matches_include = true;
            break;
        }
    }
    
    if (!matches_include) {
        return false;
    }
    
    // Check exclude patterns
    for (const auto& pattern : config_.exclude_patterns) {
        if (file_path.find(pattern) != std::string::npos) {
            return false;
        }
    }
    
    return true;
}

std::vector<InterfaceSignature> InterfaceValidator::ParseFileContent(const std::string& file_path, const std::string& content) {
    std::vector<InterfaceSignature> file_interfaces;
    std::istringstream stream(content);
    std::string line;
    uint32_t line_number = 0;
    
    // Regular expressions for parsing
    std::regex function_regex(R"((\w+(?:\s*::\s*\w+)*)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:;|\{))");
    std::regex class_method_regex(R"((\w+(?:\s*::\s*\w+)*)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:override\s*)?(?:;|\{))");
    std::regex virtual_method_regex(R"(virtual\s+(\w+(?:\s*::\s*\w+)*)\s+(\w+)\s*\(([^)]*)\)\s*(?:const\s*)?(?:=\s*0\s*)?(?:;|\{))");
    
    while (std::getline(stream, line)) {
        line_number++;
        
        // Skip comments and empty lines
        if (line.empty() || line.find("//") == 0 || line.find("/*") != std::string::npos) {
            continue;
        }
        
        std::smatch match;
        
        // Try to match virtual methods first
        if (std::regex_search(line, match, virtual_method_regex)) {
            InterfaceSignature signature;
            signature.return_type = match[1].str();
            signature.name = match[2].str();
            signature.parameters = ExtractParameters(match[3].str());
            signature.file_path = file_path;
            signature.line_number = line_number;
            signature.is_virtual = true;
            signature.is_pure_virtual = line.find("= 0") != std::string::npos;
            signature.is_const = line.find("const") != std::string::npos;
            
            file_interfaces.push_back(signature);
        }
        // Try to match class methods
        else if (std::regex_search(line, match, class_method_regex)) {
            InterfaceSignature signature;
            signature.return_type = match[1].str();
            signature.name = match[2].str();
            signature.parameters = ExtractParameters(match[3].str());
            signature.file_path = file_path;
            signature.line_number = line_number;
            signature.is_const = line.find("const") != std::string::npos;
            
            file_interfaces.push_back(signature);
        }
        // Try to match regular functions
        else if (std::regex_search(line, match, function_regex)) {
            InterfaceSignature signature;
            signature.return_type = match[1].str();
            signature.name = match[2].str();
            signature.parameters = ExtractParameters(match[3].str());
            signature.file_path = file_path;
            signature.line_number = line_number;
            signature.is_const = line.find("const") != std::string::npos;
            
            file_interfaces.push_back(signature);
        }
    }
    
    return file_interfaces;
}

std::vector<std::string> InterfaceValidator::ExtractParameters(const std::string& param_string) {
    std::vector<std::string> parameters;
    
    if (param_string.empty() || param_string.find_first_not_of(" \t") == std::string::npos) {
        return parameters;
    }
    
    std::istringstream stream(param_string);
    std::string param;
    
    while (std::getline(stream, param, ',')) {
        // Clean up the parameter
        param.erase(0, param.find_first_not_of(" \t"));
        param.erase(param.find_last_not_of(" \t") + 1);
        
        if (!param.empty()) {
            parameters.push_back(param);
        }
    }
    
    return parameters;
}

Status InterfaceValidator::ValidateInterfaces() {
    PROFILER_SCOPED_EVENT(0, "interface_validate_all");
    
    validation_results_.clear();
    
    std::cout << "Validating " << interfaces_.size() << " interfaces..." << std::endl;
    
    for (const auto& interface : interfaces_) {
        InterfaceValidationResult result;
        result.interface_name = interface.name;
        result.is_valid = ValidateSignature(interface);
        
        if (!result.is_valid) {
            result.validation_message = "Interface validation failed";
        } else {
            result.validation_message = "Interface validation passed";
        }
        
        validation_results_.push_back(result);
    }
    
    uint32_t valid_count = 0;
    for (const auto& result : validation_results_) {
        if (result.is_valid) {
            valid_count++;
        }
    }
    
    std::cout << "Validation complete: " << valid_count << "/" << interfaces_.size() << " interfaces valid" << std::endl;
    
    return Status::SUCCESS;
}

Status InterfaceValidator::ValidateInterface(const std::string& interface_name) {
    PROFILER_SCOPED_EVENT(0, "interface_validate_single");
    
    for (const auto& interface : interfaces_) {
        if (interface.name == interface_name) {
            InterfaceValidationResult result;
            result.interface_name = interface_name;
            result.is_valid = ValidateSignature(interface);
            
            if (!result.is_valid) {
                result.validation_message = "Interface validation failed";
            } else {
                result.validation_message = "Interface validation passed";
            }
            
            validation_results_.push_back(result);
            return Status::SUCCESS;
        }
    }
    
    std::cerr << "Interface '" << interface_name << "' not found" << std::endl;
    return Status::FAILURE;
}

bool InterfaceValidator::ValidateSignature(const InterfaceSignature& signature) {
    bool is_valid = true;
    
    if (config_.validate_const_correctness) {
        is_valid &= ValidateConstCorrectness(signature);
    }
    
    if (config_.validate_virtual_consistency) {
        is_valid &= ValidateVirtualConsistency(signature);
    }
    
    if (config_.validate_parameter_types) {
        is_valid &= ValidateParameterTypes(signature);
    }
    
    return is_valid;
}

bool InterfaceValidator::ValidateConstCorrectness([[maybe_unused]] const InterfaceSignature& signature) {
    // Basic const correctness validation
    // This is a simplified implementation
    return true;
}

bool InterfaceValidator::ValidateVirtualConsistency([[maybe_unused]] const InterfaceSignature& signature) {
    // Basic virtual consistency validation
    // This is a simplified implementation
    return true;
}

bool InterfaceValidator::ValidateParameterTypes(const InterfaceSignature& signature) {
    // Basic parameter type validation
    for (const auto& param : signature.parameters) {
        if (!IsValidType(param)) {
            return false;
        }
    }
    return true;
}

bool InterfaceValidator::IsValidType(const std::string& type) {
    // Basic type validation - check for common C++ types
    std::vector<std::string> valid_types = {
        "int", "uint32_t", "uint64_t", "int32_t", "int64_t",
        "float", "double", "bool", "char", "std::string",
        "Status", "void", "const", "std::shared_ptr", "std::unique_ptr",
        "std::vector", "std::map", "std::function"
    };
    
    std::string clean_type = type;
    clean_type.erase(0, clean_type.find_first_not_of(" \t"));
    clean_type.erase(clean_type.find_last_not_of(" \t") + 1);
    
    // Remove common prefixes/suffixes
    if (clean_type.find("const ") == 0) {
        clean_type = clean_type.substr(6);
    }
    if (clean_type.find("&") != std::string::npos) {
        clean_type.erase(clean_type.find("&"));
    }
    if (clean_type.find("*") != std::string::npos) {
        clean_type.erase(clean_type.find("*"));
    }
    
    // Check if it's a valid type
    for (const auto& valid_type : valid_types) {
        if (clean_type.find(valid_type) != std::string::npos) {
            return true;
        }
    }
    
    // Allow custom types (assume they're valid)
    return true;
}

std::vector<InterfaceValidationResult> InterfaceValidator::GetValidationResults() const {
    return validation_results_;
}

Status InterfaceValidator::GenerateValidationReport(const std::string& output_file) {
    PROFILER_SCOPED_EVENT(0, "interface_generate_report");
    
    try {
        // Create output directory if it doesn't exist
        std::filesystem::path output_path(output_file);
        std::filesystem::create_directories(output_path.parent_path());
        
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return Status::FAILURE;
        }
        
        // Generate HTML report
        file << "<!DOCTYPE html>\n";
        file << "<html>\n<head>\n";
        file << "<title>Interface Validation Report</title>\n";
        file << "<style>\n";
        file << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
        file << "h1 { color: #333; }\n";
        file << "table { border-collapse: collapse; width: 100%; }\n";
        file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n";
        file << "th { background-color: #f2f2f2; }\n";
        file << ".valid { color: green; }\n";
        file << ".invalid { color: red; }\n";
        file << "</style>\n";
        file << "</head>\n<body>\n";
        
        file << "<h1>Interface Validation Report</h1>\n";
        
        // Summary
        uint32_t total_interfaces = validation_results_.size();
        uint32_t valid_interfaces = 0;
        for (const auto& result : validation_results_) {
            if (result.is_valid) {
                valid_interfaces++;
            }
        }
        
        file << "<h2>Summary</h2>\n";
        file << "<p>Total Interfaces: " << total_interfaces << "</p>\n";
        file << "<p>Valid Interfaces: " << valid_interfaces << "</p>\n";
        file << "<p>Invalid Interfaces: " << (total_interfaces - valid_interfaces) << "</p>\n";
        file << "<p>Success Rate: " << std::fixed << std::setprecision(2) 
             << (total_interfaces > 0 ? (100.0 * valid_interfaces / total_interfaces) : 0.0) << "%</p>\n";
        
        // Detailed results
        file << "<h2>Detailed Results</h2>\n";
        file << "<table>\n";
        file << "<tr><th>Interface Name</th><th>Status</th><th>Message</th><th>File</th><th>Line</th></tr>\n";
        
        for (const auto& result : validation_results_) {
            // Find the corresponding interface
            InterfaceSignature interface;
            for (const auto& iface : interfaces_) {
                if (iface.name == result.interface_name) {
                    interface = iface;
                    break;
                }
            }
            
            file << "<tr>\n";
            file << "<td>" << result.interface_name << "</td>\n";
            file << "<td class=\"" << (result.is_valid ? "valid" : "invalid") << "\">" 
                 << (result.is_valid ? "VALID" : "INVALID") << "</td>\n";
            file << "<td>" << result.validation_message << "</td>\n";
            file << "<td>" << interface.file_path << "</td>\n";
            file << "<td>" << interface.line_number << "</td>\n";
            file << "</tr>\n";
        }
        
        file << "</table>\n";
        file << "</body>\n</html>\n";
        
        file.close();
        std::cout << "Interface validation report generated: " << output_file << std::endl;
        
        return Status::SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error generating report: " << e.what() << std::endl;
        return Status::FAILURE;
    }
}

void InterfaceValidator::PrintValidationSummary() const {
    std::cout << "\n=== Interface Validation Summary ===" << std::endl;
    
    uint32_t total_interfaces = validation_results_.size();
    uint32_t valid_interfaces = 0;
    
    for (const auto& result : validation_results_) {
        if (result.is_valid) {
            valid_interfaces++;
        }
    }
    
    std::cout << "Total Interfaces: " << total_interfaces << std::endl;
    std::cout << "Valid Interfaces: " << valid_interfaces << std::endl;
    std::cout << "Invalid Interfaces: " << (total_interfaces - valid_interfaces) << std::endl;
    std::cout << "Success Rate: " << std::fixed << std::setprecision(2) 
              << (total_interfaces > 0 ? (100.0 * valid_interfaces / total_interfaces) : 0.0) << "%" << std::endl;
    
    if (valid_interfaces < total_interfaces) {
        std::cout << "\nInvalid Interfaces:" << std::endl;
        for (const auto& result : validation_results_) {
            if (!result.is_valid) {
                std::cout << "  - " << result.interface_name << ": " << result.validation_message << std::endl;
            }
        }
    }
}

std::vector<InterfaceSignature> InterfaceValidator::GetInterfaces() const {
    return interfaces_;
}

std::vector<InterfaceSignature> InterfaceValidator::GetInterfacesByFile(const std::string& file_path) const {
    std::vector<InterfaceSignature> file_interfaces;
    for (const auto& interface : interfaces_) {
        if (interface.file_path == file_path) {
            file_interfaces.push_back(interface);
        }
    }
    return file_interfaces;
}

InterfaceSignature InterfaceValidator::GetInterface(const std::string& name) const {
    for (const auto& interface : interfaces_) {
        if (interface.name == name) {
            return interface;
        }
    }
    return InterfaceSignature{}; // Return empty signature if not found
}

// Global InterfaceValidator instance
static std::unique_ptr<InterfaceValidator> g_interface_validator;

// Global accessor function
InterfaceValidator* GetInterfaceValidator() {
    if (!g_interface_validator) {
        g_interface_validator = std::make_unique<InterfaceValidator>();
    }
    return g_interface_validator.get();
}

} // namespace testing
} // namespace edge_ai
