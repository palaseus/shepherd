#ifndef EDGE_AI_ENGINE_TEST_DISCOVERY_H
#define EDGE_AI_ENGINE_TEST_DISCOVERY_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <filesystem>
#include <regex>
#include <core/types.h>
#include <testing/test_framework.h>

namespace edge_ai {
namespace testing {

// Discovery configuration
struct DiscoveryConfiguration {
    bool recursive_search = true;
    bool follow_symlinks = false;
    bool case_sensitive = true;
    uint32_t max_depth = 10;
    std::vector<std::string> default_search_paths = {"tests/", "test/"};
    std::vector<std::string> default_file_patterns = {"*_test.cpp", "*_test.cc", "*_test.cxx", "test_*.cpp", "test_*.cc", "test_*.cxx"};
    std::vector<std::string> default_exclude_patterns = {"*/build/*", "*/cmake-build-*/*", "*/.*"};
};

// Test case information
struct TestCase {
    std::string test_suite;
    std::string test_name;
    std::string test_type;
    uint32_t line_number = 0;
    std::string file_path;
};

// Class definition
struct ClassDefinition {
    std::string class_name;
    std::string base_class;
    uint32_t line_number = 0;
    std::string file_path;
};

// Method signature
struct MethodSignature {
    std::string return_type;
    std::string method_name;
    uint32_t line_number = 0;
    std::string file_path;
};

// Discovery statistics
struct DiscoveryStatistics {
    uint32_t total_specs = 0;
    uint32_t total_scenarios = 0;
    uint32_t unique_modules = 0;
    uint32_t unique_test_types = 0;
    uint32_t unique_tags = 0;
};

// TestDiscovery class
class TestDiscovery {
public:
    TestDiscovery();
    ~TestDiscovery();

    // Configuration
    Status SetSearchPaths(const std::vector<std::string>& paths);
    Status SetFilePatterns(const std::vector<std::string>& patterns);
    Status SetExcludePatterns(const std::vector<std::string>& patterns);
    Status SetDiscoveryConfiguration(const DiscoveryConfiguration& config);

    // Test discovery
    std::vector<TestSpec> DiscoverTests();
    std::vector<TestSpec> DiscoverTestsInPath(const std::string& path);
    std::vector<TestSpec> DiscoverTestsInDirectory(const std::string& directory);
    TestSpec DiscoverTestFromFile(const std::string& file_path);
    TestSpec DiscoverTestFromCppFile(const std::string& file_path);
    TestSpec DiscoverTestFromHeaderFile(const std::string& file_path);

    // Discovery by criteria
    std::vector<TestSpec> DiscoverTestsByModule(const std::string& module_name);
    std::vector<TestSpec> DiscoverTestsByType(const std::string& test_type);
    std::vector<TestSpec> DiscoverTestsByTag(const std::string& tag);
    std::map<std::string, std::vector<TestSpec>> DiscoverTestsByModule();
    std::map<std::string, std::vector<TestSpec>> DiscoverTestsByType();

    // Discovery information
    std::vector<std::string> GetDiscoveredModules();
    std::vector<std::string> GetDiscoveredTestTypes();
    std::vector<std::string> GetDiscoveredTags();
    DiscoveryStatistics GetDiscoveryStatistics();

    // Validation
    Status ValidateDiscoveredTests();
    std::vector<std::string> GetValidationErrors();

private:
    // Parsing methods
    std::vector<TestCase> ParseGoogleTestMacros(const std::string& content);
    std::vector<ClassDefinition> ParseClassDefinitions(const std::string& content);
    std::vector<MethodSignature> ParseMethodSignatures(const std::string& content);

    // Test spec creation
    TestSpec CreateSpecFromTestCases(const std::vector<TestCase>& test_cases, const std::string& file_path);
    TestSpec CreateSpecFromInterface(const std::vector<ClassDefinition>& classes, const std::vector<MethodSignature>& methods, const std::string& file_path);
    TestSpec CreateBasicSpecFromFile(const std::string& file_path);

    // Utility methods
    bool MatchesFilePattern(const std::string& file_path);
    bool MatchesExcludePattern(const std::string& file_path);
    std::string GetFileExtension(const std::string& file_path);
    std::string GetModuleNameFromPath(const std::string& file_path);

    // Configuration
    std::vector<std::string> search_paths_;
    std::vector<std::string> file_patterns_;
    std::vector<std::string> exclude_patterns_;
    DiscoveryConfiguration config_;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_DISCOVERY_H
