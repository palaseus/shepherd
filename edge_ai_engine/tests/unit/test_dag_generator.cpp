#include <gtest/gtest.h>
#include <autonomous/dag_generator.h>
#include <core/types.h>
#include <chrono>
#include <thread>

using namespace edge_ai;

class DAGGeneratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.strategy = DAGGenerationStrategy::BALANCED;
        config_.max_generation_iterations = 100;
        config_.max_population_size = 50;
        config_.mutation_rate = 0.1;
        config_.crossover_rate = 0.8;
        config_.selection_pressure = 2.0;
        config_.max_generation_time_ms = 5000.0;
        config_.min_improvement_threshold = 0.01;
        config_.max_stagnation_generations = 20;
        config_.enable_parallel_execution = true;
        config_.max_parallel_branches = 4;
        config_.parallelism_threshold = 0.5;
        config_.enable_cost_modeling = true;
        config_.max_total_cost = 1000.0;
        config_.enable_validation = true;
        config_.enable_simulation = true;
        config_.simulation_iterations = 10;
        
        task_spec_.task_id = "test_task_001";
        task_spec_.task_name = "test_task";
        task_spec_.task_description = "Test inference task";
        task_spec_.input_shapes = {TensorShape({1, 224, 224, 3})};
        task_spec_.input_types = {DataType::FLOAT32};
        task_spec_.output_shapes = {TensorShape({1, 1000})};
        task_spec_.output_types = {DataType::FLOAT32};
        task_spec_.max_latency_ms = 50.0;
        task_spec_.min_throughput_rps = 500.0;
        task_spec_.max_memory_mb = 200.0;
        task_spec_.max_power_watts = 100.0;
        task_spec_.min_accuracy = 0.95;
        task_spec_.max_error_rate = 0.05;
        task_spec_.quality_metric = "accuracy";
        task_spec_.available_devices = {"cpu", "gpu"};
        task_spec_.preferred_backends = {"tensorflow", "onnx"};
        task_spec_.resource_limits["cpu_cores"] = 4.0;
        task_spec_.resource_limits["memory_mb"] = 512.0;
        task_spec_.resource_limits["gpu_memory_mb"] = 1024.0;
        task_spec_.optimization_priorities = {"latency", "throughput", "accuracy"};
        task_spec_.objective_weights["latency"] = 0.3;
        task_spec_.objective_weights["throughput"] = 0.3;
        task_spec_.objective_weights["accuracy"] = 0.4;
    }

    void TearDown() override {
        if (generator_.IsInitialized()) {
            generator_.Shutdown();
        }
    }

    DAGGenerationConfig config_;
    TaskSpecification task_spec_;
    DAGGenerator generator_;
};

TEST_F(DAGGeneratorTest, Initialize) {
    EXPECT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(generator_.IsInitialized());
}

TEST_F(DAGGeneratorTest, InitializeInvalidConfig) {
    DAGGeneratorConfig invalid_config;
    invalid_config.max_nodes = 0; // Invalid
    EXPECT_EQ(generator_.Initialize(invalid_config), Status::INVALID_ARGUMENT);
    EXPECT_FALSE(generator_.IsInitialized());
}

TEST_F(DAGGeneratorTest, GenerateDAG) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<GeneratedDAG> dags;
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    EXPECT_GT(dags.size(), 0);
    
    const auto& dag = dags[0];
    EXPECT_FALSE(dag.graph_id.empty());
    EXPECT_GT(dag.node_ids.size(), 0);
    EXPECT_GE(dag.edge_ids.size(), 0);
    EXPECT_GT(dag.estimated_latency_ms, 0.0);
    EXPECT_GT(dag.estimated_throughput_rps, 0.0);
    EXPECT_GT(dag.estimated_memory_mb, 0.0);
    EXPECT_GE(dag.estimated_accuracy, 0.0);
    EXPECT_LE(dag.estimated_accuracy, 1.0);
    EXPECT_GT(dag.optimization_score, 0.0);
    EXPECT_LE(dag.optimization_score, 1.0);
    EXPECT_FALSE(dag.generation_metadata.empty());
}

TEST_F(DAGGeneratorTest, GenerateDAGNotInitialized) {
    std::vector<GeneratedDAG> dags;
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorTest, OptimizeDAG) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<GeneratedDAG> dags;
    ASSERT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    EXPECT_GT(dags.size(), 0);
    
    GeneratedDAG& dag = dags[0];
    double original_score = dag.optimization_score;
    EXPECT_EQ(generator_.OptimizeDAG(dag, task_spec_), Status::SUCCESS);
    EXPECT_FALSE(dag.graph_id.empty());
    EXPECT_GE(dag.optimization_score, original_score);
}

TEST_F(DAGGeneratorTest, OptimizeDAGNotInitialized) {
    GeneratedDAG dag;
    EXPECT_EQ(generator_.OptimizeDAG(dag, task_spec_), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorTest, AnalyzeDependencies) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<ModelDependencyAnalysis> analyses;
    EXPECT_EQ(generator_.AnalyzeTaskRequirements(task_spec_, analyses), Status::SUCCESS);
    EXPECT_GT(analyses.size(), 0);
    
    const auto& analysis = analyses[0];
    EXPECT_FALSE(analysis.dependency_graph.empty());
    EXPECT_GT(analysis.critical_path_length, 0);
    EXPECT_GT(analysis.parallelism_opportunities, 0);
    EXPECT_GT(analysis.bottleneck_nodes.size(), 0);
    EXPECT_GT(analysis.optimization_opportunities.size(), 0);
}

TEST_F(DAGGeneratorTest, AnalyzeDependenciesNotInitialized) {
    std::vector<ModelDependencyAnalysis> analyses;
    EXPECT_EQ(generator_.AnalyzeTaskRequirements(task_spec_, analyses), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorTest, EstimatePerformance) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<GeneratedDAG> dags;
    ASSERT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    EXPECT_GT(dags.size(), 0);
    
    const auto& dag = dags[0];
    double latency, throughput, memory, power;
    EXPECT_EQ(generator_.PredictDAGLatency(dag, latency), Status::SUCCESS);
    EXPECT_EQ(generator_.PredictDAGThroughput(dag, throughput), Status::SUCCESS);
    EXPECT_EQ(generator_.PredictDAGMemoryUsage(dag, memory), Status::SUCCESS);
    EXPECT_EQ(generator_.PredictDAGPowerConsumption(dag, power), Status::SUCCESS);
    
    EXPECT_GT(latency, 0.0);
    EXPECT_GT(throughput, 0.0);
    EXPECT_GT(memory, 0.0);
    EXPECT_GT(power, 0.0);
}

TEST_F(DAGGeneratorTest, EstimatePerformanceNotInitialized) {
    GeneratedDAG dag;
    double latency;
    EXPECT_EQ(generator_.PredictDAGLatency(dag, latency), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorTest, ValidateDAG) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<GeneratedDAG> dags;
    ASSERT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    EXPECT_GT(dags.size(), 0);
    
    const auto& dag = dags[0];
    EXPECT_EQ(generator_.ValidateDAG(dag, task_spec_), Status::SUCCESS);
}

TEST_F(DAGGeneratorTest, ValidateDAGNotInitialized) {
    GeneratedDAG dag;
    EXPECT_EQ(generator_.ValidateDAG(dag, task_spec_), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorTest, GetStats) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    auto stats = generator_.GetStats();
    EXPECT_GE(stats.total_generations, 0);
    EXPECT_GE(stats.successful_generations, 0);
    EXPECT_GE(stats.failed_generations, 0);
    EXPECT_GE(stats.total_candidates_generated, 0);
    EXPECT_GE(stats.avg_generation_time_ms, 0.0);
    EXPECT_GE(stats.avg_fitness_score, 0.0);
    EXPECT_GE(stats.avg_latency_improvement, 0.0);
    EXPECT_GE(stats.avg_throughput_improvement, 0.0);
    EXPECT_GE(stats.avg_accuracy, 0.0);
    EXPECT_LE(stats.avg_accuracy, 1.0);
    EXPECT_GE(stats.avg_cost_efficiency, 0.0);
    EXPECT_GE(stats.avg_complexity_score, 0.0);
    EXPECT_GE(stats.total_mutations, 0);
    EXPECT_GE(stats.total_crossovers, 0);
    EXPECT_GE(stats.total_selections, 0);
    EXPECT_GE(stats.diversity_score, 0.0);
    EXPECT_LE(stats.diversity_score, 1.0);
    EXPECT_GE(stats.avg_memory_usage_mb, 0.0);
    EXPECT_GE(stats.avg_cpu_usage_percent, 0.0);
    EXPECT_LE(stats.avg_cpu_usage_percent, 100.0);
    EXPECT_GE(stats.avg_gpu_usage_percent, 0.0);
    EXPECT_LE(stats.avg_gpu_usage_percent, 100.0);
}

TEST_F(DAGGeneratorTest, Shutdown) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(generator_.IsInitialized());
    
    EXPECT_EQ(generator_.Shutdown(), Status::SUCCESS);
    EXPECT_FALSE(generator_.IsInitialized());
}

TEST_F(DAGGeneratorTest, ShutdownNotInitialized) {
    EXPECT_EQ(generator_.Shutdown(), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorTest, NeuralArchitectureSearch) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    NASConfig nas_config;
    nas_config.search_strategy = "evolutionary";
    nas_config.population_size = 20;
    nas_config.generations = 10;
    nas_config.mutation_rate = 0.1;
    nas_config.crossover_rate = 0.8;
    nas_config.selection_pressure = 2.0;
    nas_config.min_layers = 2;
    nas_config.max_layers = 8;
    nas_config.min_width = 64;
    nas_config.max_width = 512;
    nas_config.target_latency_ms = 50.0;
    nas_config.target_throughput_rps = 1000.0;
    nas_config.target_memory_mb = 200.0;
    nas_config.target_accuracy = 0.95;
    nas_config.fitness_weight_latency = 0.3;
    nas_config.fitness_weight_throughput = 0.3;
    nas_config.fitness_weight_memory = 0.2;
    nas_config.fitness_weight_accuracy = 0.2;
    nas_config.patience_generations = 5;
    nas_config.improvement_threshold = 0.01;
    nas_config.enable_pareto_optimization = true;
    nas_config.pareto_front_size = 5;
    
    EXPECT_EQ(generator_.InitializeNAS(nas_config), Status::SUCCESS);
    
    std::vector<ArchitectureCandidate> candidates;
    EXPECT_EQ(generator_.RunNeuralArchitectureSearch(task_spec_, nas_config, candidates), Status::SUCCESS);
    EXPECT_GT(candidates.size(), 0);
    
    // Verify candidate properties
    for (const auto& candidate : candidates) {
        EXPECT_FALSE(candidate.architecture_id.empty());
        EXPECT_GT(candidate.layers.size(), 0);
        EXPECT_GT(candidate.fitness_score, 0.0);
        EXPECT_LE(candidate.fitness_score, 1.0);
        EXPECT_GT(candidate.predicted_latency_ms, 0.0);
        EXPECT_GT(candidate.predicted_throughput_rps, 0.0);
        EXPECT_GT(candidate.predicted_memory_mb, 0.0);
        EXPECT_GE(candidate.predicted_accuracy, 0.0);
        EXPECT_LE(candidate.predicted_accuracy, 1.0);
        EXPECT_TRUE(candidate.is_evaluated);
    }
}

TEST_F(DAGGeneratorTest, NeuralArchitectureSearchNotInitialized) {
    NASConfig nas_config;
    std::vector<ArchitectureCandidate> candidates;
    EXPECT_EQ(generator_.RunNeuralArchitectureSearch(task_spec_, nas_config, candidates), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorTest, ConcurrentGeneration) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    const int num_threads = 4;
    const int dags_per_thread = 5;
    std::vector<std::thread> threads;
    std::vector<Status> results(num_threads);
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            Status result = Status::SUCCESS;
            for (int j = 0; j < dags_per_thread; ++j) {
                std::vector<GeneratedDAG> dags;
                if (generator_.GenerateDAG(task_spec_, dags) != Status::SUCCESS) {
                    result = Status::FAILURE;
                    break;
                }
            }
            results[i] = result;
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    for (const auto& result : results) {
        EXPECT_EQ(result, Status::SUCCESS);
    }
    
    auto stats = generator_.GetStats();
    EXPECT_GE(stats.total_candidates_generated, num_threads * dags_per_thread);
}

TEST_F(DAGGeneratorTest, MemoryUsage) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    // Generate multiple DAGs to test memory usage
    for (int i = 0; i < 10; ++i) {
        std::vector<GeneratedDAG> dags;
        EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    }
    
    auto stats = generator_.GetStats();
    EXPECT_GT(stats.avg_memory_usage_mb, 0.0);
    EXPECT_LT(stats.avg_memory_usage_mb, 1000.0); // Should be reasonable
}

TEST_F(DAGGeneratorTest, PerformanceUnderLoad) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Generate many DAGs quickly
    for (int i = 0; i < 50; ++i) {
        std::vector<GeneratedDAG> dags;
        EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete within reasonable time
    EXPECT_LT(duration.count(), 5000); // 5 seconds
    
    auto stats = generator_.GetStats();
    EXPECT_GT(stats.avg_generation_time_ms, 0.0);
    EXPECT_LT(stats.avg_generation_time_ms, 100.0); // Should be fast
}

