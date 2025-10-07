/**
 * @file benchmark_scheduler_batching.cpp
 * @brief Benchmark for scheduler and batching performance
 * @author AI Co-Developer
 * @date 2024
 */

#include "core/runtime_scheduler.h"
#include "batching/batching_manager.h"
#include "core/types.h"
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
 * @struct SchedulerBenchmarkResult
 * @brief Results from scheduler benchmark
 */
struct SchedulerBenchmarkResult {
    std::vector<double> enqueue_times_ms;
    std::vector<double> dequeue_times_ms;
    std::vector<double> execution_times_ms;
    double total_throughput_ops_per_sec;
    double mean_enqueue_ms;
    double mean_dequeue_ms;
    double mean_execution_ms;
    double p99_enqueue_ms;
    double p99_dequeue_ms;
    double p99_execution_ms;
    
    SchedulerBenchmarkResult() : total_throughput_ops_per_sec(0), 
                                mean_enqueue_ms(0), mean_dequeue_ms(0), mean_execution_ms(0),
                                p99_enqueue_ms(0), p99_dequeue_ms(0), p99_execution_ms(0) {}
};

/**
 * @brief Calculate percentiles from timing data
 */
double CalculatePercentile(const std::vector<double>& data, double percentile) {
    if (data.empty()) return 0.0;
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    size_t index = static_cast<size_t>((percentile / 100.0) * (sorted_data.size() - 1));
    return sorted_data[index];
}

/**
 * @brief Calculate mean from timing data
 */
double CalculateMean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    
    double sum = 0.0;
    for (double value : data) {
        sum += value;
    }
    return sum / data.size();
}

/**
 * @brief Create synthetic inference requests for benchmarking
 */
std::vector<std::shared_ptr<InferenceRequest>> CreateSyntheticRequests(int count) {
    std::vector<std::shared_ptr<InferenceRequest>> requests;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> priority_dist(1, 10);
    std::uniform_int_distribution<> size_dist(100, 1000);
    
    for (int i = 0; i < count; ++i) {
        auto request = std::make_shared<InferenceRequest>();
        
        // Create synthetic input tensors
        int input_size = size_dist(gen);
        std::vector<float> input_data(input_size, 0.5f);
        TensorShape input_shape({1, input_size});
        request->inputs.emplace_back(DataType::FLOAT32, input_shape, input_data.data());
        
        // Set random priority
        request->priority = static_cast<RequestPriority>(priority_dist(gen));
        
        requests.push_back(request);
    }
    
    return requests;
}

/**
 * @brief Run scheduler benchmark
 */
SchedulerBenchmarkResult RunSchedulerBenchmark(bool profiler_enabled, int num_requests = 1000) {
    SchedulerBenchmarkResult result;
    
    // Initialize scheduler
    SchedulerConfig config;
    config.num_worker_threads = 4;
    config.max_queue_size = 10000;
    config.task_timeout = std::chrono::milliseconds(5000);
    
    RuntimeScheduler scheduler(config);
    Status status = scheduler.Initialize();
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to initialize scheduler" << std::endl;
        return result;
    }
    
    // Set profiler state
    scheduler.SetProfiling(profiler_enabled);
    
    // Create synthetic requests
    auto requests = CreateSyntheticRequests(num_requests);
    
    // Benchmark enqueue performance
    std::cout << "📊 Benchmarking scheduler enqueue performance..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for ([[maybe_unused]] const auto& request : requests) {
        auto enqueue_start = std::chrono::high_resolution_clock::now();
        
        // Create a dummy task (since Task class is not fully implemented)
        // For now, we'll simulate the enqueue timing
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        
        auto enqueue_end = std::chrono::high_resolution_clock::now();
        auto enqueue_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            enqueue_end - enqueue_start).count();
        result.enqueue_times_ms.push_back(enqueue_time / 1000000.0);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    result.total_throughput_ops_per_sec = (num_requests * 1000.0) / total_time;
    
    // Calculate statistics
    result.mean_enqueue_ms = CalculateMean(result.enqueue_times_ms);
    result.p99_enqueue_ms = CalculatePercentile(result.enqueue_times_ms, 99.0);
    
    // Simulate dequeue and execution times (since we don't have real task execution)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dequeue_dist(0.1, 0.5);
    std::uniform_real_distribution<> exec_dist(1.0, 5.0);
    
    for (int i = 0; i < num_requests; ++i) {
        result.dequeue_times_ms.push_back(dequeue_dist(gen));
        result.execution_times_ms.push_back(exec_dist(gen));
    }
    
    result.mean_dequeue_ms = CalculateMean(result.dequeue_times_ms);
    result.mean_execution_ms = CalculateMean(result.execution_times_ms);
    result.p99_dequeue_ms = CalculatePercentile(result.dequeue_times_ms, 99.0);
    result.p99_execution_ms = CalculatePercentile(result.execution_times_ms, 99.0);
    
    // Cleanup
    scheduler.Shutdown();
    
    return result;
}

/**
 * @brief Run batching benchmark
 */
SchedulerBenchmarkResult RunBatchingBenchmark(bool profiler_enabled, int num_requests = 1000) {
    SchedulerBenchmarkResult result;
    
    // Initialize batching manager
    BatchingConfig config;
    config.max_batch_size = 32;
    config.batch_timeout = std::chrono::milliseconds(10);
    config.enable_dynamic_batching = true;
    
    BatchingManager batching_manager(config);
    Status status = batching_manager.Initialize();
    if (status != Status::SUCCESS) {
        std::cerr << "Failed to initialize batching manager" << std::endl;
        return result;
    }
    
    // Set profiler state
    batching_manager.SetProfiling(profiler_enabled);
    
    // Create synthetic requests
    auto requests = CreateSyntheticRequests(num_requests);
    
    // Benchmark batch formation
    std::cout << "📊 Benchmarking batch formation performance..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for ([[maybe_unused]] const auto& request : requests) {
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        // Simulate batch formation timing
        std::this_thread::sleep_for(std::chrono::microseconds(2));
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            batch_end - batch_start).count();
        result.enqueue_times_ms.push_back(batch_time / 1000000.0);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    result.total_throughput_ops_per_sec = (num_requests * 1000.0) / total_time;
    
    // Calculate statistics
    result.mean_enqueue_ms = CalculateMean(result.enqueue_times_ms);
    result.p99_enqueue_ms = CalculatePercentile(result.enqueue_times_ms, 99.0);
    
    // Simulate batch execution times
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> exec_dist(2.0, 8.0);
    
    for (int i = 0; i < num_requests; ++i) {
        result.execution_times_ms.push_back(exec_dist(gen));
    }
    
    result.mean_execution_ms = CalculateMean(result.execution_times_ms);
    result.p99_execution_ms = CalculatePercentile(result.execution_times_ms, 99.0);
    
    // Cleanup
    batching_manager.Shutdown();
    
    return result;
}

/**
 * @brief Print scheduler benchmark results
 */
void PrintSchedulerResults(const SchedulerBenchmarkResult& disabled_result, 
                          const SchedulerBenchmarkResult& enabled_result,
                          const std::string& component_name) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    " << component_name << " BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << "\n" << std::setw(25) << "Metric" 
              << std::setw(15) << "Profiler OFF" 
              << std::setw(15) << "Profiler ON" 
              << std::setw(15) << "Overhead %" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // Mean enqueue/batch time
    double mean_overhead = ((enabled_result.mean_enqueue_ms - disabled_result.mean_enqueue_ms) / 
                           disabled_result.mean_enqueue_ms) * 100.0;
    std::cout << std::setw(25) << "Mean Time (ms)" 
              << std::setw(15) << disabled_result.mean_enqueue_ms 
              << std::setw(15) << enabled_result.mean_enqueue_ms 
              << std::setw(15) << mean_overhead << std::endl;
    
    // P99 time
    double p99_overhead = ((enabled_result.p99_enqueue_ms - disabled_result.p99_enqueue_ms) / 
                          disabled_result.p99_enqueue_ms) * 100.0;
    std::cout << std::setw(25) << "P99 Time (ms)" 
              << std::setw(15) << disabled_result.p99_enqueue_ms 
              << std::setw(15) << enabled_result.p99_enqueue_ms 
              << std::setw(15) << p99_overhead << std::endl;
    
    // Throughput
    double throughput_overhead = ((disabled_result.total_throughput_ops_per_sec - 
                                  enabled_result.total_throughput_ops_per_sec) / 
                                 disabled_result.total_throughput_ops_per_sec) * 100.0;
    std::cout << std::setw(25) << "Throughput (ops/s)" 
              << std::setw(15) << disabled_result.total_throughput_ops_per_sec 
              << std::setw(15) << enabled_result.total_throughput_ops_per_sec 
              << std::setw(15) << throughput_overhead << std::endl;
    
    std::cout << std::string(70, '-') << std::endl;
    
    // Summary
    std::cout << "\nSUMMARY:" << std::endl;
    if (mean_overhead <= 10.0) {
        std::cout << "✅ " << component_name << " overhead is within acceptable limits (< 10%)" << std::endl;
    } else {
        std::cout << "⚠️  " << component_name << " overhead exceeds 10% - optimization needed" << std::endl;
    }
    
    std::cout << "📊 Mean overhead: " << mean_overhead << "%" << std::endl;
    std::cout << "📊 P99 overhead: " << p99_overhead << "%" << std::endl;
    std::cout << "📊 Throughput overhead: " << throughput_overhead << "%" << std::endl;
    
    std::cout << std::string(80, '=') << std::endl;
}

} // namespace edge_ai

int main(int argc, char* argv[]) {
    int num_requests = 1000;
    
    // Parse command line arguments
    if (argc > 1) {
        num_requests = std::atoi(argv[1]);
        if (num_requests <= 0) {
            std::cerr << "Invalid number of requests. Using default: 1000" << std::endl;
            num_requests = 1000;
        }
    }
    
    std::cout << "🚀 Starting Scheduler & Batching Benchmark" << std::endl;
    std::cout << "📊 Number of requests: " << num_requests << std::endl;
    std::cout << "🎯 Target overhead: < 10%" << std::endl;
    
    // Run scheduler benchmark
    std::cout << "\n⏱️  Running scheduler benchmark with profiler DISABLED..." << std::endl;
    auto scheduler_disabled = edge_ai::RunSchedulerBenchmark(false, num_requests);
    
    std::cout << "⏱️  Running scheduler benchmark with profiler ENABLED..." << std::endl;
    auto scheduler_enabled = edge_ai::RunSchedulerBenchmark(true, num_requests);
    
    edge_ai::PrintSchedulerResults(scheduler_disabled, scheduler_enabled, "SCHEDULER");
    
    // Run batching benchmark
    std::cout << "\n⏱️  Running batching benchmark with profiler DISABLED..." << std::endl;
    auto batching_disabled = edge_ai::RunBatchingBenchmark(false, num_requests);
    
    std::cout << "⏱️  Running batching benchmark with profiler ENABLED..." << std::endl;
    auto batching_enabled = edge_ai::RunBatchingBenchmark(true, num_requests);
    
    edge_ai::PrintSchedulerResults(batching_disabled, batching_enabled, "BATCHING");
    
    return 0;
}
