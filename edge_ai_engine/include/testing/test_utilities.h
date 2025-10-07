#ifndef EDGE_AI_ENGINE_TEST_UTILITIES_H
#define EDGE_AI_ENGINE_TEST_UTILITIES_H

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <random>
#include <filesystem>
#include <core/types.h>
#include <testing/test_framework.h>

namespace edge_ai {
namespace testing {

// TestUtilities class
class TestUtilities {
public:
    // Environment management
    static Status SetupTestEnvironment(const TestConfig& config);
    static Status TeardownTestEnvironment(const TestConfig& config);

    // Data generation
    static std::vector<uint8_t> GenerateRandomData(size_t size);
    static std::string GenerateRandomString(size_t length);
    static double GenerateRandomDouble(double min, double max);
    static int32_t GenerateRandomInt(int32_t min, int32_t max);

    // File operations
    static Status CreateTempFile(const std::string& content, std::string& file_path);
    static Status DeleteTempFile(const std::string& file_path);
    static Status CreateTempDirectory(std::string& dir_path);
    static Status DeleteTempDirectory(const std::string& dir_path);

    // Network simulation
    static Status SimulateNetworkLatency(std::chrono::milliseconds latency);
    static Status SimulateNetworkFailure(double failure_rate);
    static Status SimulateNetworkBandwidth(double bandwidth_mbps);

    // System monitoring
    static double GetCurrentCPUUsage();
    static double GetCurrentMemoryUsage();
    static double GetCurrentNetworkUsage();

    // Assertions
    static void AssertWithinRange(double value, double min, double max, const std::string& message = "");
    static void AssertApproximatelyEqual(double expected, double actual, double tolerance, const std::string& message = "");
    static void AssertStringContains(const std::string& haystack, const std::string& needle, const std::string& message = "");
    static void AssertFileExists(const std::string& file_path, const std::string& message = "");
    static void AssertDirectoryExists(const std::string& dir_path, const std::string& message = "");

    // Time utilities
    static std::chrono::steady_clock::time_point GetCurrentTime();
    static std::chrono::milliseconds GetElapsedTime(const std::chrono::steady_clock::time_point& start_time);

    // Logging
    static void LogTestStart(const std::string& test_name);
    static void LogTestEnd(const std::string& test_name, bool passed, std::chrono::milliseconds duration);
    static void LogTestWarning(const std::string& test_name, const std::string& warning);
    static void LogTestError(const std::string& test_name, const std::string& error);
};

} // namespace testing
} // namespace edge_ai

#endif // EDGE_AI_ENGINE_TEST_UTILITIES_H
