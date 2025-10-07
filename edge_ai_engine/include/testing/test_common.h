#ifndef EDGE_AI_ENGINE_TEST_COMMON_H
#define EDGE_AI_ENGINE_TEST_COMMON_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <core/types.h>

namespace edge_ai {
namespace testing {

// Common test environment structure
struct TestEnvironment {
    std::string environment_name;
    std::map<std::string, std::string> environment_variables;
    std::vector<std::string> required_services;
    std::vector<std::string> required_databases;
    std::vector<std::string> required_apis;
    std::map<std::string, std::string> service_endpoints;
    std::map<std::string, std::string> database_connections;
    std::map<std::string, std::string> api_keys;
    std::vector<std::string> temp_directories;
    std::vector<std::string> include_directories;
    std::vector<std::string> library_directories;
    std::vector<std::string> libraries;
    std::string working_directory;
    std::string output_directory;
    std::string log_directory;
    bool is_production = false;
    bool is_isolated = true;
    std::chrono::milliseconds setup_timeout{30000};
    std::chrono::milliseconds teardown_timeout{10000};
};

// Common performance thresholds structure
struct PerformanceThresholds {
    double max_cpu_usage_percent = 80.0;
    double max_memory_usage_mb = 1024.0;
    double max_network_usage_mbps = 100.0;
    std::chrono::milliseconds max_duration{5000};
    double min_throughput_rps = 100.0;
    double max_latency_ms = 100.0;
    double min_accuracy_percent = 95.0;
    double max_error_rate_percent = 1.0;
    double max_power_consumption_watts = 100.0;
    std::map<std::string, double> custom_thresholds;
};

// Common test statistics structure
struct TestStatistics {
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;
    uint32_t failed_tests = 0;
    uint32_t flaky_tests = 0;
    double success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    double avg_duration_ms = 0.0;
    double avg_coverage_percent = 0.0;
    double avg_stability_score = 0.0;
};

// Common performance statistics structure
struct PerformanceStatistics {
    uint32_t total_tests = 0;
    uint32_t passed_tests = 0;
    uint32_t failed_tests = 0;
    uint32_t flaky_tests = 0;
    double success_rate = 0.0;
    std::chrono::milliseconds total_duration{0};
    double avg_duration_ms = 0.0;
    double avg_coverage_percent = 0.0;
    double avg_stability_score = 0.0;
    double avg_execution_time_us = 0.0;
    double avg_throughput_ops_per_sec = 0.0;
    double avg_cpu_usage_percent = 0.0;
    double avg_memory_usage_mb = 0.0;
    double avg_network_usage_mbps = 0.0;
    double avg_power_consumption_watts = 0.0;
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_COMMON_H
