#include <testing/test_discovery.h>
#include <profiling/profiler.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <regex>
#include <cstring>

namespace edge_ai {
namespace testing {

// TestDiscovery Implementation
TestDiscovery::TestDiscovery() {
    // Initialize default configuration
}

TestDiscovery::~TestDiscovery() {
    // Cleanup if needed
}

Status TestDiscovery::SetSearchPaths(const std::vector<std::string>& paths) {
    search_paths_ = paths;
    return Status::SUCCESS;
}

Status TestDiscovery::SetFilePatterns(const std::vector<std::string>& patterns) {
    file_patterns_ = patterns;
    return Status::SUCCESS;
}

Status TestDiscovery::SetExcludePatterns(const std::vector<std::string>& patterns) {
    exclude_patterns_ = patterns;
    return Status::SUCCESS;
}

Status TestDiscovery::SetDiscoveryConfiguration(const DiscoveryConfiguration& config) {
    config_ = config;
    return Status::SUCCESS;
}

std::vector<TestSpec> TestDiscovery::DiscoverTests() {
    PROFILER_SCOPED_EVENT(0, "discover_tests");
    
    std::vector<TestSpec> specs;
    
    for (const auto& path : search_paths_) {
        auto path_specs = DiscoverTestsInPath(path);
        specs.insert(specs.end(), path_specs.begin(), path_specs.end());
    }
    
    return specs;
}

std::vector<TestSpec> TestDiscovery::DiscoverTestsInPath(const std::string& path) {
    PROFILER_SCOPED_EVENT(0, "discover_tests_in_path");
    
    std::vector<TestSpec> specs;
    
    try {
        if (std::filesystem::is_directory(path)) {
            specs = DiscoverTestsInDirectory(path);
        } else if (std::filesystem::is_regular_file(path)) {
            auto spec = DiscoverTestFromFile(path);
            if (spec.Validate() == Status::SUCCESS) {
                specs.push_back(spec);
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        // Path doesn't exist or can't be accessed
    }
    
    return specs;
}

std::vector<TestSpec> TestDiscovery::DiscoverTestsInDirectory(const std::string& directory) {
    PROFILER_SCOPED_EVENT(0, "discover_tests_in_directory");
    
    std::vector<TestSpec> specs;
    
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                
                // Check if file matches patterns
                if (MatchesFilePattern(file_path) && !MatchesExcludePattern(file_path)) {
                    auto spec = DiscoverTestFromFile(file_path);
                    if (spec.Validate() == Status::SUCCESS) {
                        specs.push_back(spec);
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        // Directory doesn't exist or can't be accessed
    }
    
    return specs;
}

TestSpec TestDiscovery::DiscoverTestFromFile(const std::string& file_path) {
    PROFILER_SCOPED_EVENT(0, "discover_test_from_file");
    
    TestSpec spec;
    
    // Determine file type and parse accordingly
    std::string extension = GetFileExtension(file_path);
    
    if (extension == "yaml" || extension == "yml") {
        spec = TestSpec(file_path);
    } else if (extension == "json") {
        // TODO: Implement JSON parsing
        // For now, create a basic spec
        spec = CreateBasicSpecFromFile(file_path);
    } else if (extension == "cpp" || extension == "cc" || extension == "cxx") {
        spec = DiscoverTestFromCppFile(file_path);
    } else if (extension == "h" || extension == "hpp") {
        spec = DiscoverTestFromHeaderFile(file_path);
    }
    
    return spec;
}

TestSpec TestDiscovery::DiscoverTestFromCppFile(const std::string& file_path) {
    PROFILER_SCOPED_EVENT(0, "discover_test_from_cpp_file");
    
    TestSpec spec;
    
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return spec;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // Parse Google Test macros
        auto test_cases = ParseGoogleTestMacros(content);
        
        // Create test spec from parsed test cases
        spec = CreateSpecFromTestCases(test_cases, file_path);
        
    } catch (const std::exception& e) {
        // Error reading file
    }
    
    return spec;
}

TestSpec TestDiscovery::DiscoverTestFromHeaderFile(const std::string& file_path) {
    PROFILER_SCOPED_EVENT(0, "discover_test_from_header_file");
    
    TestSpec spec;
    
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            return spec;
        }
        
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();
        
        // Parse class definitions and method signatures
        auto classes = ParseClassDefinitions(content);
        auto methods = ParseMethodSignatures(content);
        
        // Create test spec from parsed classes and methods
        spec = CreateSpecFromInterface(classes, methods, file_path);
        
    } catch (const std::exception& e) {
        // Error reading file
    }
    
    return spec;
}

std::vector<TestCase> TestDiscovery::ParseGoogleTestMacros(const std::string& content) {
    std::vector<TestCase> test_cases;
    
    // Regular expressions for Google Test macros
    std::regex test_regex(R"(TEST\s*\(\s*(\w+)\s*,\s*(\w+)\s*\))");
    std::regex test_f_regex(R"(TEST_F\s*\(\s*(\w+)\s*,\s*(\w+)\s*\))");
    std::regex test_p_regex(R"(TEST_P\s*\(\s*(\w+)\s*,\s*(\w+)\s*\))");
    
    std::sregex_iterator test_iter(content.begin(), content.end(), test_regex);
    std::sregex_iterator test_f_iter(content.begin(), content.end(), test_f_regex);
    std::sregex_iterator test_p_iter(content.begin(), content.end(), test_p_regex);
    std::sregex_iterator end;
    
    // Parse TEST macros
    for (auto it = test_iter; it != end; ++it) {
        TestCase test_case;
        test_case.test_suite = (*it)[1].str();
        test_case.test_name = (*it)[2].str();
        test_case.test_type = "TEST";
        test_cases.push_back(test_case);
    }
    
    // Parse TEST_F macros
    for (auto it = test_f_iter; it != end; ++it) {
        TestCase test_case;
        test_case.test_suite = (*it)[1].str();
        test_case.test_name = (*it)[2].str();
        test_case.test_type = "TEST_F";
        test_cases.push_back(test_case);
    }
    
    // Parse TEST_P macros
    for (auto it = test_p_iter; it != end; ++it) {
        TestCase test_case;
        test_case.test_suite = (*it)[1].str();
        test_case.test_name = (*it)[2].str();
        test_case.test_type = "TEST_P";
        test_cases.push_back(test_case);
    }
    
    return test_cases;
}

std::vector<ClassDefinition> TestDiscovery::ParseClassDefinitions(const std::string& content) {
    std::vector<ClassDefinition> classes;
    
    // Regular expression for class definitions
    std::regex class_regex(R"(class\s+(\w+)(?:\s*:\s*public\s+(\w+))?\s*\{)");
    
    std::sregex_iterator iter(content.begin(), content.end(), class_regex);
    std::sregex_iterator end;
    
    for (auto it = iter; it != end; ++it) {
        ClassDefinition class_def;
        class_def.class_name = (*it)[1].str();
        if ((*it)[2].matched) {
            class_def.base_class = (*it)[2].str();
        }
        classes.push_back(class_def);
    }
    
    return classes;
}

std::vector<MethodSignature> TestDiscovery::ParseMethodSignatures(const std::string& content) {
    std::vector<MethodSignature> methods;
    
    // Regular expression for method signatures
    std::regex method_regex(R"((\w+)\s+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:\{|;))");
    
    std::sregex_iterator iter(content.begin(), content.end(), method_regex);
    std::sregex_iterator end;
    
    for (auto it = iter; it != end; ++it) {
        MethodSignature method;
        method.return_type = (*it)[1].str();
        method.method_name = (*it)[2].str();
        methods.push_back(method);
    }
    
    return methods;
}

TestSpec TestDiscovery::CreateSpecFromTestCases(const std::vector<TestCase>& test_cases, 
                                               const std::string& file_path) {
    TestSpec spec;
    
    // Set basic configuration
    spec.GetConfig().test_suite_name = GetModuleNameFromPath(file_path) + "_test_suite";
    spec.GetConfig().module_name = GetModuleNameFromPath(file_path);
    spec.GetConfig().test_type = "unit";
    spec.GetConfig().dependencies = {"gtest"};
    
    // Create scenarios from test cases
    for (const auto& test_case : test_cases) {
        TestScenario scenario;
        scenario.SetName(test_case.test_name);
        scenario.SetDescription("Test case: " + test_case.test_name);
        
        // Add Given step
        scenario.AddGivenStep("Test environment is set up");
        
        // Add When step
        scenario.AddWhenStep("Execute " + test_case.test_name);
        
        // Add Then step
        scenario.AddThenStep("Test should pass");
        
        spec.GetScenarios().push_back(scenario);
    }
    
    return spec;
}

TestSpec TestDiscovery::CreateSpecFromInterface(const std::vector<ClassDefinition>& classes,
                                               const std::vector<MethodSignature>& methods,
                                               const std::string& file_path) {
    TestSpec spec;
    
    // Set basic configuration
    spec.GetConfig().test_suite_name = GetModuleNameFromPath(file_path) + "_interface_test_suite";
    spec.GetConfig().module_name = GetModuleNameFromPath(file_path);
    spec.GetConfig().test_type = "integration";
    
    // Create scenarios from classes and methods
    for (const auto& class_def : classes) {
        for (const auto& method : methods) {
            TestScenario scenario;
            scenario.SetName("test_" + class_def.class_name + "_" + method.method_name);
            scenario.SetDescription("Test " + class_def.class_name + "::" + method.method_name);
            
            // Add Given step
            scenario.AddGivenStep("Create instance of " + class_def.class_name);
            
            // Add When step
            scenario.AddWhenStep("Call " + method.method_name);
            
            // Add Then step
            scenario.AddThenStep("Method should execute successfully");
            
            spec.GetScenarios().push_back(scenario);
        }
    }
    
    return spec;
}

TestSpec TestDiscovery::CreateBasicSpecFromFile(const std::string& file_path) {
    TestSpec spec;
    
    // Set basic configuration
    spec.GetConfig().test_suite_name = GetModuleNameFromPath(file_path) + "_test_suite";
    spec.GetConfig().module_name = GetModuleNameFromPath(file_path);
    spec.GetConfig().test_type = "basic";
    
    // Create a basic scenario
    TestScenario scenario;
    scenario.SetName("basic_test");
    scenario.SetDescription("Basic test for " + GetModuleNameFromPath(file_path));
    
    scenario.AddGivenStep("Test environment is set up");
    scenario.AddWhenStep("Execute basic test");
    scenario.AddThenStep("Test should pass");
    
    spec.GetScenarios().push_back(scenario);
    
    return spec;
}

bool TestDiscovery::MatchesFilePattern(const std::string& file_path) {
    if (file_patterns_.empty()) {
        return true; // No patterns specified, match all
    }
    
    for (const auto& pattern : file_patterns_) {
        if (std::regex_match(file_path, std::regex(pattern))) {
            return true;
        }
    }
    
    return false;
}

bool TestDiscovery::MatchesExcludePattern(const std::string& file_path) {
    for (const auto& pattern : exclude_patterns_) {
        if (std::regex_match(file_path, std::regex(pattern))) {
            return true;
        }
    }
    
    return false;
}

std::string TestDiscovery::GetFileExtension(const std::string& file_path) {
    size_t dot_pos = file_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        std::string extension = file_path.substr(dot_pos + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        return extension;
    }
    return "";
}

std::string TestDiscovery::GetModuleNameFromPath(const std::string& file_path) {
    std::filesystem::path path(file_path);
    std::string filename = path.stem().string();
    
    // Remove common test suffixes
    std::vector<std::string> suffixes = {"_test", "_tests", "Test", "Tests"};
    for (const auto& suffix : suffixes) {
        if (filename.length() > suffix.length() && 
            filename.substr(filename.length() - suffix.length()) == suffix) {
            filename = filename.substr(0, filename.length() - suffix.length());
            break;
        }
    }
    
    return filename;
}

std::vector<TestSpec> TestDiscovery::DiscoverTestsByModule(const std::string& module_name) {
    PROFILER_SCOPED_EVENT(0, "discover_tests_by_module");
    
    std::vector<TestSpec> specs;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        if (spec.GetConfig().module_name == module_name) {
            specs.push_back(spec);
        }
    }
    
    return specs;
}

std::vector<TestSpec> TestDiscovery::DiscoverTestsByType(const std::string& test_type) {
    PROFILER_SCOPED_EVENT(0, "discover_tests_by_type");
    
    std::vector<TestSpec> specs;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        if (spec.GetConfig().test_type == test_type) {
            specs.push_back(spec);
        }
    }
    
    return specs;
}

std::vector<TestSpec> TestDiscovery::DiscoverTestsByTag(const std::string& tag) {
    PROFILER_SCOPED_EVENT(0, "discover_tests_by_tag");
    
    std::vector<TestSpec> specs;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        for (const auto& scenario : spec.GetScenarios()) {
            const auto& tags = scenario.GetTags();
            if (std::find(tags.begin(), tags.end(), tag) != tags.end()) {
                specs.push_back(spec);
                break;
            }
        }
    }
    
    return specs;
}

std::map<std::string, std::vector<TestSpec>> TestDiscovery::DiscoverTestsByModule() {
    PROFILER_SCOPED_EVENT(0, "discover_tests_by_module_map");
    
    std::map<std::string, std::vector<TestSpec>> module_specs;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        module_specs[spec.GetConfig().module_name].push_back(spec);
    }
    
    return module_specs;
}

std::map<std::string, std::vector<TestSpec>> TestDiscovery::DiscoverTestsByType() {
    PROFILER_SCOPED_EVENT(0, "discover_tests_by_type_map");
    
    std::map<std::string, std::vector<TestSpec>> type_specs;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        type_specs[spec.GetConfig().test_type].push_back(spec);
    }
    
    return type_specs;
}

std::vector<std::string> TestDiscovery::GetDiscoveredModules() {
    PROFILER_SCOPED_EVENT(0, "get_discovered_modules");
    
    std::set<std::string> modules;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        modules.insert(spec.GetConfig().module_name);
    }
    
    return std::vector<std::string>(modules.begin(), modules.end());
}

std::vector<std::string> TestDiscovery::GetDiscoveredTestTypes() {
    PROFILER_SCOPED_EVENT(0, "get_discovered_test_types");
    
    std::set<std::string> types;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        types.insert(spec.GetConfig().test_type);
    }
    
    return std::vector<std::string>(types.begin(), types.end());
}

std::vector<std::string> TestDiscovery::GetDiscoveredTags() {
    PROFILER_SCOPED_EVENT(0, "get_discovered_tags");
    
    std::set<std::string> tags;
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        for (const auto& scenario : spec.GetScenarios()) {
            const auto& scenario_tags = scenario.GetTags();
            tags.insert(scenario_tags.begin(), scenario_tags.end());
        }
    }
    
    return std::vector<std::string>(tags.begin(), tags.end());
}

DiscoveryStatistics TestDiscovery::GetDiscoveryStatistics() {
    PROFILER_SCOPED_EVENT(0, "get_discovery_statistics");
    
    DiscoveryStatistics stats;
    auto all_specs = DiscoverTests();
    
    stats.total_specs = all_specs.size();
    stats.total_scenarios = 0;
    
    std::set<std::string> modules;
    std::set<std::string> types;
    std::set<std::string> tags;
    
    for (const auto& spec : all_specs) {
        stats.total_scenarios += spec.GetScenarios().size();
        modules.insert(spec.GetConfig().module_name);
        types.insert(spec.GetConfig().test_type);
        
        for (const auto& scenario : spec.GetScenarios()) {
            const auto& scenario_tags = scenario.GetTags();
            tags.insert(scenario_tags.begin(), scenario_tags.end());
        }
    }
    
    stats.unique_modules = modules.size();
    stats.unique_test_types = types.size();
    stats.unique_tags = tags.size();
    
    return stats;
}

Status TestDiscovery::ValidateDiscoveredTests() {
    PROFILER_SCOPED_EVENT(0, "validate_discovered_tests");
    
    auto all_specs = DiscoverTests();
    
    for (const auto& spec : all_specs) {
        Status status = spec.Validate();
        if (status != Status::SUCCESS) {
            return status;
        }
    }
    
    return Status::SUCCESS;
}

std::vector<std::string> TestDiscovery::GetValidationErrors() {
    PROFILER_SCOPED_EVENT(0, "get_validation_errors");
    
    std::vector<std::string> errors;
    auto all_specs = DiscoverTests();
    
    for (size_t i = 0; i < all_specs.size(); ++i) {
        auto spec_errors = all_specs[i].GetValidationErrors();
        for (const auto& error : spec_errors) {
            errors.push_back("Spec[" + std::to_string(i) + "]: " + error);
        }
    }
    
    return errors;
}

} // namespace testing
} // namespace edge_ai
