#ifndef EDGE_AI_ENGINE_BEHAVIOR_DRIVEN_TESTING_H
#define EDGE_AI_ENGINE_BEHAVIOR_DRIVEN_TESTING_H

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <regex>
#include <chrono>
#include <core/types.h>
#include <testing/test_common.h>

namespace edge_ai {
namespace testing {

// BDT Step types
enum class BDTStepType {
    GIVEN,
    WHEN,
    THEN,
    AND,
    BUT
};

// BDT Step definition
struct BDTStep {
    BDTStepType type;
    std::string description;
    std::string regex_pattern;
    std::function<Status(const std::map<std::string, std::string>&)> step_function;
    std::vector<std::string> parameters;
    std::string file_path;
    uint32_t line_number = 0;
    std::chrono::milliseconds timeout{5000};
    bool is_async = false;
    std::map<std::string, std::string> metadata;
};

// BDT Scenario definition
struct BDTScenario {
    std::string name;
    std::string description;
    std::vector<std::string> tags;
    std::vector<BDTStep> steps;
    std::map<std::string, std::string> background_data;
    std::map<std::string, std::string> scenario_context;
    std::string file_path;
    uint32_t line_number = 0;
    bool is_outline = false;
    std::vector<std::map<std::string, std::string>> examples;
    std::chrono::milliseconds timeout{30000};
    std::map<std::string, std::string> metadata;
};

// BDT Feature definition
struct BDTFeature {
    std::string name;
    std::string description;
    std::vector<std::string> tags;
    std::vector<BDTScenario> scenarios;
    std::vector<BDTStep> background_steps;
    std::map<std::string, std::string> feature_context;
    std::string file_path;
    std::string language = "en";
    std::map<std::string, std::string> metadata;
};

// BDT Step result
struct BDTStepResult {
    std::string step_description;
    BDTStepType step_type;
    Status status = Status::SUCCESS;
    std::chrono::milliseconds duration{0};
    std::string error_message;
    std::map<std::string, std::string> captured_data;
    std::vector<std::string> logs;
    bool skipped = false;
    std::map<std::string, std::string> metadata;
};

// BDT Scenario result
struct BDTScenarioResult {
    std::string scenario_name;
    Status status = Status::SUCCESS;
    std::chrono::milliseconds duration{0};
    std::vector<BDTStepResult> step_results;
    std::string error_message;
    std::map<std::string, std::string> scenario_data;
    std::vector<std::string> logs;
    bool skipped = false;
    std::map<std::string, std::string> metadata;
};

// BDT Feature result
struct BDTFeatureResult {
    std::string feature_name;
    Status status = Status::SUCCESS;
    std::chrono::milliseconds duration{0};
    std::vector<BDTScenarioResult> scenario_results;
    std::string error_message;
    std::map<std::string, std::string> feature_data;
    std::vector<std::string> logs;
    std::map<std::string, std::string> metadata;
};

// BDT Configuration
struct BDTConfiguration {
    bool enable_parallel_execution = true;
    uint32_t max_parallel_scenarios = 4;
    std::chrono::milliseconds default_step_timeout{5000};
    std::chrono::milliseconds default_scenario_timeout{30000};
    bool stop_on_first_failure = false;
    bool enable_step_retry = true;
    uint32_t max_step_retries = 3;
    std::chrono::milliseconds retry_delay{1000};
    bool enable_detailed_logging = true;
    bool enable_step_timing = true;
    std::vector<std::string> include_tags;
    std::vector<std::string> exclude_tags;
    std::map<std::string, std::string> global_context;
    std::string output_format = "console";
    std::string output_file;
    std::map<std::string, std::string> metadata;
};

// BDT Statistics
struct BDTStatistics {
    uint32_t total_features = 0;
    uint32_t total_scenarios = 0;
    uint32_t total_steps = 0;
    uint32_t passed_features = 0;
    uint32_t failed_features = 0;
    uint32_t passed_scenarios = 0;
    uint32_t failed_scenarios = 0;
    uint32_t skipped_scenarios = 0;
    uint32_t passed_steps = 0;
    uint32_t failed_steps = 0;
    uint32_t skipped_steps = 0;
    double feature_success_rate = 0.0;
    double scenario_success_rate = 0.0;
    double step_success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    double avg_feature_duration_ms = 0.0;
    double avg_scenario_duration_ms = 0.0;
    double avg_step_duration_ms = 0.0;
    std::map<std::string, uint32_t> tag_statistics;
    std::map<std::string, double> step_type_statistics;
};

// BDT Step Registry
class BDTStepRegistry {
public:
    BDTStepRegistry() = default;
    ~BDTStepRegistry() = default;

    // Step registration
    void RegisterGivenStep(const std::string& pattern, 
                          std::function<Status(const std::map<std::string, std::string>&)> step_func,
                          const std::string& file_path = "", uint32_t line_number = 0);
    
    void RegisterWhenStep(const std::string& pattern, 
                         std::function<Status(const std::map<std::string, std::string>&)> step_func,
                         const std::string& file_path = "", uint32_t line_number = 0);
    
    void RegisterThenStep(const std::string& pattern, 
                         std::function<Status(const std::map<std::string, std::string>&)> step_func,
                         const std::string& file_path = "", uint32_t line_number = 0);
    
    void RegisterAndStep(const std::string& pattern, 
                        std::function<Status(const std::map<std::string, std::string>&)> step_func,
                        const std::string& file_path = "", uint32_t line_number = 0);
    
    void RegisterButStep(const std::string& pattern, 
                        std::function<Status(const std::map<std::string, std::string>&)> step_func,
                        const std::string& file_path = "", uint32_t line_number = 0);

    // Step lookup
    BDTStep* FindMatchingStep(const std::string& step_description, BDTStepType expected_type);
    std::vector<BDTStep*> FindStepsByType(BDTStepType type);
    std::vector<BDTStep*> FindStepsByPattern(const std::string& pattern);

    // Registry management
    void ClearRegistry();
    size_t GetStepCount() const;
    size_t GetStepCountByType(BDTStepType type) const;
    std::vector<std::string> GetRegisteredPatterns() const;

private:
    std::vector<BDTStep> registered_steps_;
    std::map<std::string, size_t> pattern_to_step_index_;
    
    void RegisterStep(BDTStepType type, const std::string& pattern, 
                     std::function<Status(const std::map<std::string, std::string>&)> step_func,
                     const std::string& file_path, uint32_t line_number);
};

// BDT Parser
class BDTParser {
public:
    BDTParser() = default;
    ~BDTParser() = default;

    // Feature parsing
    Status ParseFeatureFile(const std::string& file_path, BDTFeature& feature);
    Status ParseFeatureContent(const std::string& content, BDTFeature& feature);
    Status ParseScenario(const std::string& content, BDTScenario& scenario);
    Status ParseStep(const std::string& content, BDTStep& step);

    // Gherkin syntax parsing
    Status ParseGherkinFeature(const std::string& content, BDTFeature& feature);
    Status ParseGherkinScenario(const std::string& content, BDTScenario& scenario);
    Status ParseGherkinStep(const std::string& content, BDTStep& step);

    // Parameter extraction
    std::map<std::string, std::string> ExtractStepParameters(const std::string& step_description, 
                                                           const std::string& pattern);
    std::vector<std::string> ExtractTableData(const std::string& content);
    std::vector<std::map<std::string, std::string>> ParseExamplesTable(const std::string& content);

    // Validation
    bool ValidateFeature(const BDTFeature& feature);
    bool ValidateScenario(const BDTScenario& scenario);
    bool ValidateStep(const BDTStep& step);

private:
    std::regex step_regex_;
    std::regex scenario_regex_;
    std::regex feature_regex_;
    std::regex table_regex_;
    
    void InitializeRegexPatterns();
    std::string CleanStepDescription(const std::string& description);
    BDTStepType DetermineStepType(const std::string& step_text);
};

// BDT Executor
class BDTExecutor {
public:
    BDTExecutor();
    ~BDTExecutor();

    // Configuration
    Status SetConfiguration(const BDTConfiguration& config);
    Status SetStepRegistry(std::shared_ptr<BDTStepRegistry> registry);

    // Execution
    Status ExecuteFeature(const BDTFeature& feature, BDTFeatureResult& result);
    Status ExecuteScenario(const BDTScenario& scenario, BDTScenarioResult& result);
    Status ExecuteStep(const BDTStep& step, const std::map<std::string, std::string>& context, 
                      BDTStepResult& result);

    // Parallel execution
    Status ExecuteFeaturesInParallel(const std::vector<BDTFeature>& features, 
                                   std::vector<BDTFeatureResult>& results);
    Status ExecuteScenariosInParallel(const std::vector<BDTScenario>& scenarios, 
                                     std::vector<BDTScenarioResult>& results);

    // Context management
    void SetGlobalContext(const std::map<std::string, std::string>& context);
    void UpdateContext(const std::map<std::string, std::string>& updates);
    std::map<std::string, std::string> GetContext() const;
    void ClearContext();

    // Results and statistics
    BDTStatistics GetStatistics() const;
    std::vector<BDTFeatureResult> GetFeatureResults() const;
    std::vector<BDTScenarioResult> GetScenarioResults() const;
    std::vector<BDTStepResult> GetStepResults() const;
    void ClearResults();

private:
    BDTConfiguration config_;
    std::shared_ptr<BDTStepRegistry> step_registry_;
    std::map<std::string, std::string> global_context_;
    std::vector<BDTFeatureResult> feature_results_;
    std::vector<BDTScenarioResult> scenario_results_;
    std::vector<BDTStepResult> step_results_;
    
    Status ExecuteStepWithRetry(const BDTStep& step, const std::map<std::string, std::string>& context, 
                               BDTStepResult& result);
    Status ExecuteStepAsync(const BDTStep& step, const std::map<std::string, std::string>& context, 
                           BDTStepResult& result);
    void UpdateStatistics(const BDTFeatureResult& result);
    void UpdateStatistics(const BDTScenarioResult& result);
    void UpdateStatistics(const BDTStepResult& result);
};

// BDT Manager - Main interface for BDT functionality
class BDTManager {
public:
    BDTManager();
    ~BDTManager();

    // Configuration
    Status SetConfiguration(const BDTConfiguration& config);
    Status LoadStepDefinitions(const std::string& directory);
    Status LoadFeatureFiles(const std::string& directory);

    // Step registration helpers
    void RegisterCommonSteps();
    void RegisterEdgeAISteps();
    void RegisterNetworkSteps();
    void RegisterDatabaseSteps();
    void RegisterAPISteps();

    // Execution
    Status RunFeatures(const std::vector<std::string>& feature_files);
    Status RunScenarios(const std::vector<std::string>& scenario_names);
    Status RunTags(const std::vector<std::string>& tags);

    // Results and reporting
    BDTStatistics GetStatistics() const;
    Status GenerateReport(const std::string& output_file, const std::string& format = "html");
    Status ExportResults(const std::string& output_file, const std::string& format = "json");

    // Utility methods
    std::vector<BDTFeature> GetLoadedFeatures() const;
    std::vector<BDTScenario> GetLoadedScenarios() const;
    void ClearLoadedFeatures();
    void ClearResults();
    
    // Access to step registry
    BDTStepRegistry* GetStepRegistry() const;

private:
    BDTConfiguration config_;
    std::shared_ptr<BDTStepRegistry> step_registry_;
    std::unique_ptr<BDTParser> parser_;
    std::unique_ptr<BDTExecutor> executor_;
    std::vector<BDTFeature> loaded_features_;
    
    Status LoadStepDefinitionFile(const std::string& file_path);
    Status LoadFeatureFile(const std::string& file_path);
    void RegisterDefaultSteps();
};

// Global BDT Manager accessor function
BDTManager* GetBDTManager();

// BDT Macros for easy step registration
#define BDT_GIVEN(pattern) \
    static auto bdt_given_##__LINE__ = []() { \
        extern BDTManager* GetBDTManager(); \
        GetBDTManager()->GetStepRegistry()->RegisterGivenStep(pattern, \
            []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status

#define BDT_WHEN(pattern) \
    static auto bdt_when_##__LINE__ = []() { \
        extern BDTManager* GetBDTManager(); \
        GetBDTManager()->GetStepRegistry()->RegisterWhenStep(pattern, \
            []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status

#define BDT_THEN(pattern) \
    static auto bdt_then_##__LINE__ = []() { \
        extern BDTManager* GetBDTManager(); \
        GetBDTManager()->GetStepRegistry()->RegisterThenStep(pattern, \
            []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status

#define BDT_AND(pattern) \
    static auto bdt_and_##__LINE__ = []() { \
        extern BDTManager* GetBDTManager(); \
        GetBDTManager()->GetStepRegistry()->RegisterAndStep(pattern, \
            []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status

#define BDT_BUT(pattern) \
    static auto bdt_but_##__LINE__ = []() { \
        extern BDTManager* GetBDTManager(); \
        GetBDTManager()->GetStepRegistry()->RegisterButStep(pattern, \
            []([[maybe_unused]] const std::map<std::string, std::string>& params) -> Status

#define BDT_END_STEP \
        ; \
        return Status::SUCCESS; \
    }, __FILE__, __LINE__); \
    return 0; \
    }(); \
    (void)bdt_given_##__LINE__

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_BEHAVIOR_DRIVEN_TESTING_H
