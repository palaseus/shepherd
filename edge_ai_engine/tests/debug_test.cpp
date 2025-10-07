#include "optimization/ml_based_policy.h"
#include "types.h"
#include <iostream>

int main() {
    auto ml_policy = std::make_shared<edge_ai::MLBasedPolicy>();
    
    edge_ai::OptimizationMetrics metrics;
    metrics.avg_latency_ms = 80.0;
    metrics.memory_usage_percent = 50.0;
    metrics.queue_depth = 20;
    metrics.cpu_utilization_percent = 80.0;
    metrics.throughput_ops_per_sec = 5.0;
    
    edge_ai::AdaptiveOptimizationConfig config;
    config.enable_adaptive_batching = true;
    
    auto decisions = ml_policy->AnalyzeAndDecide(metrics, config);
    
    std::cout << "Number of decisions: " << decisions.size() << std::endl;
    for (const auto& decision : decisions) {
        std::cout << "Trigger: " << static_cast<int>(decision.trigger) << std::endl;
        std::cout << "Action: " << static_cast<int>(decision.action) << std::endl;
        std::cout << "Parameter: " << decision.parameter_name << std::endl;
        std::cout << "New value: " << decision.new_value << std::endl;
        std::cout << "---" << std::endl;
    }
    
    return 0;
}
