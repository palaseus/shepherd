/**
 * @file benchmark_optimization_system.cpp
 * @brief Benchmark for the optimization system
 * @author AI Co-Developer
 * @date 2024
 */

#include "optimization/optimization_manager.h"
#include "batching/batching_manager.h"
#include "core/runtime_scheduler.h"
#include "core/inference_engine.h"
#include "core/cpu_device.h"
#include "memory/memory_manager.h"
#include "profiling/profiler.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>

using namespace edge_ai;

class OptimizationBenchmark {
public:
    OptimizationBenchmark() : rng_(std::random_device{}()) {
        // Initialize profiler
        Profiler::GetInstance().Initialize();
        
        // Create test components
        batching_manager_ = std::make_shared<BatchingManager>();
        scheduler_ = std::make_shared<RuntimeScheduler>();
        auto device = std::make_shared<CPUDevice>(0);
        auto memory_manager = std::make_shared<MemoryManager>();
        inference_engine_ = std::make_shared<InferenceEngine>(device, scheduler_, memory_manager, &Profiler::GetInstance());
        
        // Create optimization manager
        config_.optimization_interval = std::chrono::milliseconds(10);
        config_.convergence_timeout = std::chrono::milliseconds(100);
        optimization_manager_ = std::make_unique<OptimizationManager>(config_);
        
        optimization_manager_->Initialize();
        optimization_manager_->RegisterComponents(batching_manager_, scheduler_, inference_engine_);
    }
    
    void RunOptimizationLatencyBenchmark() {
        std::cout << "🚀 Starting Optimization Latency Benchmark\n";
        std::cout << "📊 Number of optimization cycles: 1000\n";
        std::cout << "🎯 Target: < 1ms per optimization cycle\n\n";
        
        const int num_cycles = 1000;
        std::vector<double> latencies;
        latencies.reserve(num_cycles);
        
        // Start optimization
        optimization_manager_->StartOptimization();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_cycles; ++i) {
            auto cycle_start = std::chrono::high_resolution_clock::now();
            
            // Simulate metric updates
            OptimizationMetrics metrics = GenerateRandomMetrics();
            optimization_manager_->UpdateMetrics(metrics);
            
            // Simulate optimization decision
            OptimizationDecision decision = GenerateRandomDecision();
            optimization_manager_->ApplyOptimization(decision);
            
            auto cycle_end = std::chrono::high_resolution_clock::now();
            auto cycle_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                cycle_end - cycle_start).count();
            
            latencies.push_back(cycle_duration / 1000.0); // Convert to milliseconds
            
            // Small delay to prevent overwhelming the system
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        // Stop optimization
        optimization_manager_->StopOptimization();
        
        // Calculate statistics
        std::sort(latencies.begin(), latencies.end());
        double mean_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double p50_latency = latencies[latencies.size() * 0.5];
        double p90_latency = latencies[latencies.size() * 0.9];
        double p99_latency = latencies[latencies.size() * 0.99];
        
        // Print results
        std::cout << "================================================================================\n";
        std::cout << "                    OPTIMIZATION LATENCY BENCHMARK RESULTS\n";
        std::cout << "================================================================================\n\n";
        
        std::cout << "              Metric   Value (ms)   Target (ms)   Status\n";
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "           Mean (ms)   " << std::setw(10) << std::fixed << std::setprecision(3) 
                  << mean_latency << "   " << std::setw(10) << "1.000" << "   " 
                  << (mean_latency < 1.0 ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "            P50 (ms)   " << std::setw(10) << std::fixed << std::setprecision(3) 
                  << p50_latency << "   " << std::setw(10) << "1.000" << "   " 
                  << (p50_latency < 1.0 ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "            P90 (ms)   " << std::setw(10) << std::fixed << std::setprecision(3) 
                  << p90_latency << "   " << std::setw(10) << "1.000" << "   " 
                  << (p90_latency < 1.0 ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "            P99 (ms)   " << std::setw(10) << std::fixed << std::setprecision(3) 
                  << p99_latency << "   " << std::setw(10) << "1.000" << "   " 
                  << (p99_latency < 1.0 ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "        Total Time     " << std::setw(10) << std::fixed << std::setprecision(3) 
                  << total_duration << "   " << std::setw(10) << "N/A" << "   " << "📊 INFO" << "\n";
        
        std::cout << "\n";
    }
    
    void RunOptimizationThroughputBenchmark() {
        std::cout << "🚀 Starting Optimization Throughput Benchmark\n";
        std::cout << "📊 Duration: 5 seconds\n";
        std::cout << "🎯 Target: > 1000 optimization decisions per second\n\n";
        
        const int duration_seconds = 5;
        const auto end_time = std::chrono::steady_clock::now() + std::chrono::seconds(duration_seconds);
        
        int decisions_made = 0;
        int metrics_updated = 0;
        
        // Start optimization
        optimization_manager_->StartOptimization();
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (std::chrono::steady_clock::now() < end_time) {
            // Update metrics
            OptimizationMetrics metrics = GenerateRandomMetrics();
            optimization_manager_->UpdateMetrics(metrics);
            metrics_updated++;
            
            // Apply optimization decision
            OptimizationDecision decision = GenerateRandomDecision();
            optimization_manager_->ApplyOptimization(decision);
            decisions_made++;
            
            // Small delay to prevent overwhelming the system
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        
        auto actual_end_time = std::chrono::steady_clock::now();
        auto actual_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            actual_end_time - start_time).count();
        
        // Stop optimization
        optimization_manager_->StopOptimization();
        
        // Calculate throughput
        double decisions_per_second = (decisions_made * 1000.0) / actual_duration;
        double metrics_per_second = (metrics_updated * 1000.0) / actual_duration;
        
        // Print results
        std::cout << "================================================================================\n";
        std::cout << "                   OPTIMIZATION THROUGHPUT BENCHMARK RESULTS\n";
        std::cout << "================================================================================\n\n";
        
        std::cout << "              Metric   Value (ops/sec)   Target (ops/sec)   Status\n";
        std::cout << "---------------------------------------------------------------------\n";
        std::cout << "    Decisions/sec     " << std::setw(15) << std::fixed << std::setprecision(1) 
                  << decisions_per_second << "   " << std::setw(15) << "1000.0" << "   " 
                  << (decisions_per_second > 1000.0 ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "     Metrics/sec      " << std::setw(15) << std::fixed << std::setprecision(1) 
                  << metrics_per_second << "   " << std::setw(15) << "1000.0" << "   " 
                  << (metrics_per_second > 1000.0 ? "✅ PASS" : "❌ FAIL") << "\n";
        std::cout << "    Total Decisions   " << std::setw(15) << std::fixed << std::setprecision(0) 
                  << decisions_made << "   " << std::setw(15) << "N/A" << "   " << "📊 INFO" << "\n";
        std::cout << "   Total Metrics      " << std::setw(15) << std::fixed << std::setprecision(0) 
                  << metrics_updated << "   " << std::setw(15) << "N/A" << "   " << "📊 INFO" << "\n";
        std::cout << "    Duration (ms)     " << std::setw(15) << std::fixed << std::setprecision(1) 
                  << actual_duration << "   " << std::setw(15) << "N/A" << "   " << "📊 INFO" << "\n";
        
        std::cout << "\n";
    }
    
    void RunOptimizationConvergenceBenchmark() {
        std::cout << "🚀 Starting Optimization Convergence Benchmark\n";
        std::cout << "📊 Simulating load changes and measuring convergence time\n";
        std::cout << "🎯 Target: < 2 seconds convergence time\n\n";
        
        const int num_load_changes = 5;
        std::vector<double> convergence_times;
        
        for (int i = 0; i < num_load_changes; ++i) {
            std::cout << "📈 Load Change " << (i + 1) << "/" << num_load_changes << "\n";
            
            // Start optimization
            optimization_manager_->StartOptimization();
            
            auto start_time = std::chrono::steady_clock::now();
            
            // Simulate load change
            OptimizationMetrics initial_metrics = GenerateRandomMetrics();
            optimization_manager_->UpdateMetrics(initial_metrics);
            
            // Simulate gradual load increase
            for (int j = 0; j < 20; ++j) {
                OptimizationMetrics metrics = GenerateRandomMetrics();
                metrics.avg_latency_ms += j * 5.0; // Gradually increase latency
                metrics.queue_depth += j * 2; // Gradually increase queue depth
                
                optimization_manager_->UpdateMetrics(metrics);
                
                // Check if optimization decisions are being made
                auto stats = optimization_manager_->GetStats();
                if (stats.total_decisions > 0) {
                    auto convergence_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        convergence_time - start_time).count();
                    convergence_times.push_back(duration);
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            // Stop optimization
            optimization_manager_->StopOptimization();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        // Calculate statistics
        if (!convergence_times.empty()) {
            std::sort(convergence_times.begin(), convergence_times.end());
            double mean_convergence = std::accumulate(convergence_times.begin(), convergence_times.end(), 0.0) / convergence_times.size();
            double p50_convergence = convergence_times[convergence_times.size() * 0.5];
            double p90_convergence = convergence_times[convergence_times.size() * 0.9];
            double p99_convergence = convergence_times[convergence_times.size() * 0.99];
            
            // Print results
            std::cout << "\n================================================================================\n";
            std::cout << "                   OPTIMIZATION CONVERGENCE BENCHMARK RESULTS\n";
            std::cout << "================================================================================\n\n";
            
            std::cout << "              Metric   Value (ms)   Target (ms)   Status\n";
            std::cout << "-----------------------------------------------------------------\n";
            std::cout << "           Mean (ms)   " << std::setw(10) << std::fixed << std::setprecision(1) 
                      << mean_convergence << "   " << std::setw(10) << "2000.0" << "   " 
                      << (mean_convergence < 2000.0 ? "✅ PASS" : "❌ FAIL") << "\n";
            std::cout << "            P50 (ms)   " << std::setw(10) << std::fixed << std::setprecision(1) 
                      << p50_convergence << "   " << std::setw(10) << "2000.0" << "   " 
                      << (p50_convergence < 2000.0 ? "✅ PASS" : "❌ FAIL") << "\n";
            std::cout << "            P90 (ms)   " << std::setw(10) << std::fixed << std::setprecision(1) 
                      << p90_convergence << "   " << std::setw(10) << "2000.0" << "   " 
                      << (p90_convergence < 2000.0 ? "✅ PASS" : "❌ FAIL") << "\n";
            std::cout << "            P99 (ms)   " << std::setw(10) << std::fixed << std::setprecision(1) 
                      << p99_convergence << "   " << std::setw(10) << "2000.0" << "   " 
                      << (p99_convergence < 2000.0 ? "✅ PASS" : "❌ FAIL") << "\n";
            std::cout << "    Load Changes      " << std::setw(10) << std::fixed << std::setprecision(0) 
                      << num_load_changes << "   " << std::setw(10) << "N/A" << "   " << "📊 INFO" << "\n";
            
            std::cout << "\n";
        } else {
            std::cout << "❌ No convergence times recorded - optimization may not be working properly\n\n";
        }
    }

private:
    OptimizationMetrics GenerateRandomMetrics() {
        OptimizationMetrics metrics;
        metrics.avg_latency_ms = 50.0 + (rng_() % 100); // 50-150ms
        metrics.p99_latency_ms = metrics.avg_latency_ms * (1.5 + (rng_() % 50) / 100.0);
        metrics.throughput_ops_per_sec = 800.0 + (rng_() % 400); // 800-1200 ops/sec
        metrics.memory_usage_percent = 0.3 + (rng_() % 50) / 100.0; // 30-80%
        metrics.cpu_utilization_percent = 0.4 + (rng_() % 40) / 100.0; // 40-80%
        metrics.queue_depth = rng_() % 100; // 0-100
        metrics.batch_efficiency = 0.7 + (rng_() % 20) / 100.0; // 70-90%
        metrics.total_requests = 10000 + rng_() % 5000; // 10k-15k
        metrics.failed_requests = rng_() % 100; // 0-100
        return metrics;
    }
    
    OptimizationDecision GenerateRandomDecision() {
        OptimizationDecision decision;
        decision.action = static_cast<OptimizationAction>(rng_() % 6);
        decision.trigger = static_cast<OptimizationTrigger>(rng_() % 6);
        decision.parameter_name = "test_parameter";
        decision.old_value = "old_value";
        decision.new_value = "new_value";
        decision.expected_improvement = 0.1 + (rng_() % 20) / 100.0; // 10-30%
        decision.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        decision.request_id = rng_();
        return decision;
    }
    
    std::mt19937 rng_;
    AdaptiveOptimizationConfig config_;
    std::unique_ptr<OptimizationManager> optimization_manager_;
    std::shared_ptr<BatchingManager> batching_manager_;
    std::shared_ptr<RuntimeScheduler> scheduler_;
    std::shared_ptr<InferenceEngine> inference_engine_;
};

int main() {
    std::cout << "🎯 Edge AI Engine - Optimization System Benchmark\n";
    std::cout << "==================================================\n\n";
    
    try {
        OptimizationBenchmark benchmark;
        
        // Run benchmarks
        benchmark.RunOptimizationLatencyBenchmark();
        benchmark.RunOptimizationThroughputBenchmark();
        benchmark.RunOptimizationConvergenceBenchmark();
        
        std::cout << "✅ All optimization benchmarks completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
