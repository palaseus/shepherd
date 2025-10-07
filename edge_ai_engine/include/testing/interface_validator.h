#ifndef EDGE_AI_ENGINE_INTERFACE_VALIDATOR_H
#define EDGE_AI_ENGINE_INTERFACE_VALIDATOR_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <regex>
#include <fstream>

#include <core/types.h>

namespace edge_ai {
namespace testing {

// Interface validation structures
struct InterfaceSignature {
    std::string name;
    std::string return_type;
    std::vector<std::string> parameters;
    std::string file_path;
    uint32_t line_number;
    bool is_const = false;
    bool is_virtual = false;
    bool is_pure_virtual = false;
};

struct InterfaceValidationResult {
    std::string interface_name;
    bool is_valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    std::string validation_message;
};

struct InterfaceValidationConfig {
    std::vector<std::string> header_directories;
    std::vector<std::string> include_patterns;
    std::vector<std::string> exclude_patterns;
    bool validate_const_correctness = true;
    bool validate_virtual_consistency = true;
    bool validate_parameter_types = true;
    std::string output_directory = "interface_validation_reports/";
};

// Interface Validator class
class InterfaceValidator {
public:
    InterfaceValidator();
    ~InterfaceValidator();

    // Configuration
    Status SetConfiguration(const InterfaceValidationConfig& config);
    
    // Interface parsing
    Status ParseHeaderFiles();
    Status ParseHeaderFile(const std::string& file_path);
    
    // Interface validation
    Status ValidateInterfaces();
    Status ValidateInterface(const std::string& interface_name);
    
    // Results and reporting
    std::vector<InterfaceValidationResult> GetValidationResults() const;
    Status GenerateValidationReport(const std::string& output_file);
    void PrintValidationSummary() const;
    
    // Interface management
    std::vector<InterfaceSignature> GetInterfaces() const;
    std::vector<InterfaceSignature> GetInterfacesByFile(const std::string& file_path) const;
    InterfaceSignature GetInterface(const std::string& name) const;

private:
    InterfaceValidationConfig config_;
    std::vector<InterfaceSignature> interfaces_;
    std::vector<InterfaceValidationResult> validation_results_;
    
    // Parsing helpers
    std::vector<std::string> FindHeaderFiles();
    bool ShouldParseFile(const std::string& file_path);
    std::vector<InterfaceSignature> ParseFileContent(const std::string& file_path, const std::string& content);
    InterfaceSignature ParseFunctionSignature(const std::string& signature, const std::string& file_path, uint32_t line_number);
    InterfaceSignature ParseClassMethod(const std::string& method, const std::string& file_path, uint32_t line_number);
    
    // Validation helpers
    bool ValidateSignature(const InterfaceSignature& signature);
    bool ValidateConstCorrectness(const InterfaceSignature& signature);
    bool ValidateVirtualConsistency(const InterfaceSignature& signature);
    bool ValidateParameterTypes(const InterfaceSignature& signature);
    
    // Utility functions
    std::string CleanSignature(const std::string& signature);
    std::vector<std::string> ExtractParameters(const std::string& signature);
    bool IsValidType(const std::string& type);
    std::string GetReturnType(const std::string& signature);
};

// Global accessor function
InterfaceValidator* GetInterfaceValidator();

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_INTERFACE_VALIDATOR_H
