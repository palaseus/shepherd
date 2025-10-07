/**
 * @file benchmark_profiler_overhead.cpp
 * @brief Simple benchmark for profiler overhead measurement
 * @author AI Co-Developer
 * @date 2024
 */

#include "profiling/profiler.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>
#include <thread>

namespace edge_ai {

/**
 * @struct BenchmarkResult
 * @brief Results from a benchmark run
 */
struct BenchmarkResult {
    std::vector<double> latencies_ms;
    double mean_ms;
    double p50_ms;
    double p90_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
    double total_time_ms;
    
    BenchmarkResult() : mean_ms(0), p50_ms(0), p90_ms(0), p99_ms(0), 
                       min_ms(0), max_ms(0), total_time_ms(0) {}
};

/**
 * @brief Calculate percentiles from latency data
 */
BenchmarkResult CalculateStats(const std::vector<double>& latencies) {
    BenchmarkResult result;
    result.latencies_ms = latencies;
    
    if (latencies.empty()) {
        return result;
    }
    
    // Sort latencies for percentile calculation
    std::vector<double> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());
    
    // Calculate statistics
    double sum = 0.0;
    result.min_ms = sorted_latencies[0];
    result.max_ms = sorted_latencies[sorted_latencies.size() - 1];
    
    for (double latency : sorted_latencies) {
        sum += latency;
    }
    result.mean_ms = sum / sorted_latencies.size();
    
    // Calculate percentiles
    size_t n = sorted_latencies.size();
    result.p50_ms = sorted_latencies[static_cast<size_t>(n * 0.50)];
    result.p90_ms = sorted_latencies[static_cast<size_t>(n * 0.90)];
    result.p99_ms = sorted_latencies[static_cast<size_t>(n * 0.99)];
    
    return result;
}

/**
 * @brief Simulate some work
 */
void SimulateWork(int iterations = 10000) {
    int sum = 0;
    for (int i = 0; i < iterations; ++i) {
        sum = sum + i * i;
        // Add some memory access to make it more realistic
        if (i % 100 == 0) {
            sum = sum % 1000;
        }
    }
    // Prevent optimization
    (void)sum;
}

/**
 * @brief Run profiler overhead benchmark
 */
BenchmarkResult RunProfilerBenchmark(bool profiler_enabled, int num_requests = 10000) {
    BenchmarkResult result;
    std::vector<double> latencies;
    
    // Initialize profiler
    Profiler& profiler = Profiler::GetInstance();
    profiler.Initialize();
    profiler.SetEnabled(profiler_enabled);
    
    // Start global session if profiler is enabled
    if (profiler_enabled) {
        profiler.StartGlobalSession("benchmark_session");
    }
    
    // Benchmark
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_requests; ++i) {
        auto request_start = std::chrono::high_resolution_clock::now();
        
        // Simulate inference work with profiler hooks
        PROFILER_SCOPED_EVENT(i, "inference_total");
        PROFILER_MARK_EVENT(i, "inference_start");
        
        // Simulate some work
        SimulateWork(50000);
        
        // Add a small delay to make timing more realistic
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        
        PROFILER_MARK_EVENT(i, "inference_end");
        
        auto request_end = std::chrono::high_resolution_clock::now();
        
        auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
            request_end - request_start).count();
        latencies.push_back(latency / 1000.0); // Convert to milliseconds
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    result = CalculateStats(latencies);
    
    // Stop global session and cleanup
    if (profiler_enabled) {
        profiler.StopGlobalSession();
    }
    profiler.Shutdown();
    
    return result;
}

/**
 * @brief Print benchmark results in a formatted table
 */
void PrintBenchmarkResults(const BenchmarkResult& disabled_result, 
                          const BenchmarkResult& enabled_result) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    PROFILER OVERHEAD BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << "\n" << std::setw(20) << "Metric" 
              << std::setw(15) << "Profiler OFF" 
              << std::setw(15) << "Profiler ON" 
              << std::setw(15) << "Overhead %" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    // Mean latency
    double mean_overhead = 0.0;
    if (disabled_result.mean_ms > 0.0) {
        mean_overhead = ((enabled_result.mean_ms - disabled_result.mean_ms) / 
                        disabled_result.mean_ms) * 100.0;
    }
    std::cout << std::setw(20) << "Mean (ms)" 
              << std::setw(15) << disabled_result.mean_ms 
              << std::setw(15) << enabled_result.mean_ms 
              << std::setw(15) << mean_overhead << std::endl;
    
    // P50 latency
    double p50_overhead = 0.0;
    if (disabled_result.p50_ms > 0.0) {
        p50_overhead = ((enabled_result.p50_ms - disabled_result.p50_ms) / 
                       disabled_result.p50_ms) * 100.0;
    }
    std::cout << std::setw(20) << "P50 (ms)" 
              << std::setw(15) << disabled_result.p50_ms 
              << std::setw(15) << enabled_result.p50_ms 
              << std::setw(15) << p50_overhead << std::endl;
    
    // P90 latency
    double p90_overhead = 0.0;
    if (disabled_result.p90_ms > 0.0) {
        p90_overhead = ((enabled_result.p90_ms - disabled_result.p90_ms) / 
                       disabled_result.p90_ms) * 100.0;
    }
    std::cout << std::setw(20) << "P90 (ms)" 
              << std::setw(15) << disabled_result.p90_ms 
              << std::setw(15) << enabled_result.p90_ms 
              << std::setw(15) << p90_overhead << std::endl;
    
    // P99 latency
    double p99_overhead = 0.0;
    if (disabled_result.p99_ms > 0.0) {
        p99_overhead = ((enabled_result.p99_ms - disabled_result.p99_ms) / 
                       disabled_result.p99_ms) * 100.0;
    }
    std::cout << std::setw(20) << "P99 (ms)" 
              << std::setw(15) << disabled_result.p99_ms 
              << std::setw(15) << enabled_result.p99_ms 
              << std::setw(15) << p99_overhead << std::endl;
    
    // Min latency
    std::cout << std::setw(20) << "Min (ms)" 
              << std::setw(15) << disabled_result.min_ms 
              << std::setw(15) << enabled_result.min_ms 
              << std::setw(15) << "-" << std::endl;
    
    // Max latency
    std::cout << std::setw(20) << "Max (ms)" 
              << std::setw(15) << disabled_result.max_ms 
              << std::setw(15) << enabled_result.max_ms 
              << std::setw(15) << "-" << std::endl;
    
    // Total time
    double total_overhead = 0.0;
    if (disabled_result.total_time_ms > 0.0) {
        total_overhead = ((enabled_result.total_time_ms - disabled_result.total_time_ms) / 
                         disabled_result.total_time_ms) * 100.0;
    }
    std::cout << std::setw(20) << "Total Time (ms)" 
              << std::setw(15) << disabled_result.total_time_ms 
              << std::setw(15) << enabled_result.total_time_ms 
              << std::setw(15) << total_overhead << std::endl;
    
    std::cout << std::string(65, '-') << std::endl;
    
    // Summary
    std::cout << "\nSUMMARY:" << std::endl;
    if (mean_overhead <= 10.0) {
        std::cout << "✅ Profiler overhead is within acceptable limits (< 10%)" << std::endl;
    } else {
        std::cout << "⚠️  Profiler overhead exceeds 10% - optimization needed" << std::endl;
    }
    
    std::cout << "📊 Mean overhead: " << mean_overhead << "%" << std::endl;
    std::cout << "📊 P99 overhead: " << p99_overhead << "%" << std::endl;
    
    std::cout << std::string(80, '=') << std::endl;
}

} // namespace edge_ai

int main(int argc, char* argv[]) {
    int num_requests = 10000;
    
    // Parse command line arguments
    if (argc > 1) {
        num_requests = std::atoi(argv[1]);
        if (num_requests <= 0) {
            std::cerr << "Invalid number of requests. Using default: 10000" << std::endl;
            num_requests = 10000;
        }
    }
    
    std::cout << "🚀 Starting Profiler Overhead Benchmark" << std::endl;
    std::cout << "📊 Number of requests: " << num_requests << std::endl;
    std::cout << "🎯 Target overhead: < 10%" << std::endl;
    
    // Run benchmark with profiler disabled
    std::cout << "\n⏱️  Running benchmark with profiler DISABLED..." << std::endl;
    auto disabled_result = edge_ai::RunProfilerBenchmark(false, num_requests);
    
    // Run benchmark with profiler enabled
    std::cout << "⏱️  Running benchmark with profiler ENABLED..." << std::endl;
    auto enabled_result = edge_ai::RunProfilerBenchmark(true, num_requests);
    
    // Print results
    edge_ai::PrintBenchmarkResults(disabled_result, enabled_result);
    
    return 0;
}
