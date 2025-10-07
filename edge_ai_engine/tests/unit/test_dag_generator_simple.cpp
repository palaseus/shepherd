#include <gtest/gtest.h>
#include <autonomous/dag_generator.h>
#include <core/types.h>
#include <set>

using namespace edge_ai;

class DAGGeneratorSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.strategy = DAGGenerationStrategy::BALANCED;
        config_.max_generation_iterations = 10;
        config_.max_population_size = 20;
        config_.mutation_rate = 0.1;
        config_.crossover_rate = 0.8;
        config_.selection_pressure = 2.0;
        config_.max_generation_time_ms = 1000.0;
        config_.min_improvement_threshold = 0.01;
        config_.max_stagnation_generations = 5;
        config_.enable_parallel_execution = true;
        config_.max_parallel_branches = 2;
        config_.parallelism_threshold = 0.5;
        config_.enable_cost_modeling = true;
        config_.max_total_cost = 100.0;
        config_.enable_validation = true;
        config_.enable_simulation = true;
        config_.simulation_iterations = 5;
        
        task_spec_.task_id = "test_task_001";
        task_spec_.task_name = "test_task";
        task_spec_.task_description = "Test inference task";
        task_spec_.required_models = {"resnet50", "mobilenet"};
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

TEST_F(DAGGeneratorSimpleTest, Initialize) {
    EXPECT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(generator_.IsInitialized());
}

TEST_F(DAGGeneratorSimpleTest, InitializeInvalidConfig) {
    DAGGenerationConfig invalid_config;
    invalid_config.max_generation_iterations = 0; // Invalid
    EXPECT_EQ(generator_.Initialize(invalid_config), Status::INVALID_ARGUMENT);
    EXPECT_FALSE(generator_.IsInitialized());
}

TEST_F(DAGGeneratorSimpleTest, GenerateDAG) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<GeneratedDAG> dags;
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    EXPECT_GT(dags.size(), 0);
    
    const auto& dag = dags[0];
    EXPECT_FALSE(dag.graph_id.empty());
    EXPECT_GT(dag.node_ids.size(), 0);
    EXPECT_GE(dag.edge_ids.size(), 0);
    
    // Additional comprehensive validation
    EXPECT_GT(dag.estimated_latency_ms, 0.0);
    EXPECT_GT(dag.estimated_throughput_rps, 0.0);
    EXPECT_GT(dag.estimated_memory_mb, 0.0);
    EXPECT_GE(dag.estimated_accuracy, 0.0);
    EXPECT_LE(dag.estimated_accuracy, 1.0);
    EXPECT_GT(dag.optimization_score, 0.0);
    EXPECT_LE(dag.optimization_score, 1.0);
    EXPECT_FALSE(dag.generation_metadata.empty());
    
    // Validate DAG structure
    EXPECT_EQ(dag.task_id, task_spec_.task_id);
    EXPECT_FALSE(dag.dag_id.empty());
    EXPECT_EQ(dag.generation_strategy, "initial");
    EXPECT_EQ(dag.generation_iteration, 0);
}

TEST_F(DAGGeneratorSimpleTest, GenerateDAGNotInitialized) {
    std::vector<GeneratedDAG> dags;
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorSimpleTest, GetStats) {
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

TEST_F(DAGGeneratorSimpleTest, Shutdown) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    EXPECT_TRUE(generator_.IsInitialized());
    
    EXPECT_EQ(generator_.Shutdown(), Status::SUCCESS);
    EXPECT_FALSE(generator_.IsInitialized());
}

TEST_F(DAGGeneratorSimpleTest, ShutdownNotInitialized) {
    EXPECT_EQ(generator_.Shutdown(), Status::NOT_INITIALIZED);
}

TEST_F(DAGGeneratorSimpleTest, MultipleDAGGeneration) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    // Generate multiple DAGs
    std::vector<GeneratedDAG> dags1, dags2, dags3;
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags1), Status::SUCCESS);
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags2), Status::SUCCESS);
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags3), Status::SUCCESS);
    
    EXPECT_GT(dags1.size(), 0);
    EXPECT_GT(dags2.size(), 0);
    EXPECT_GT(dags3.size(), 0);
    
    // Verify each DAG has unique IDs
    EXPECT_NE(dags1[0].dag_id, dags2[0].dag_id);
    EXPECT_NE(dags2[0].dag_id, dags3[0].dag_id);
    EXPECT_NE(dags1[0].dag_id, dags3[0].dag_id);
}

TEST_F(DAGGeneratorSimpleTest, DAGPerformanceValidation) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<GeneratedDAG> dags;
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    EXPECT_GT(dags.size(), 0);
    
    const auto& dag = dags[0];
    
    // Validate performance estimates are reasonable
    EXPECT_GT(dag.estimated_latency_ms, 0.0);
    EXPECT_LT(dag.estimated_latency_ms, 1000.0); // Should be reasonable
    
    EXPECT_GT(dag.estimated_throughput_rps, 0.0);
    EXPECT_LT(dag.estimated_throughput_rps, 10000.0); // Should be reasonable
    
    EXPECT_GT(dag.estimated_memory_mb, 0.0);
    EXPECT_LT(dag.estimated_memory_mb, 10000.0); // Should be reasonable
    
    EXPECT_GE(dag.estimated_accuracy, 0.0);
    EXPECT_LE(dag.estimated_accuracy, 1.0);
    
    EXPECT_GT(dag.optimization_score, 0.0);
    EXPECT_LE(dag.optimization_score, 1.0);
}

TEST_F(DAGGeneratorSimpleTest, DAGStructureValidation) {
    ASSERT_EQ(generator_.Initialize(config_), Status::SUCCESS);
    
    std::vector<GeneratedDAG> dags;
    EXPECT_EQ(generator_.GenerateDAG(task_spec_, dags), Status::SUCCESS);
    EXPECT_GT(dags.size(), 0);
    
    const auto& dag = dags[0];
    
    // Validate DAG structure
    EXPECT_FALSE(dag.graph_id.empty());
    EXPECT_FALSE(dag.dag_id.empty());
    EXPECT_EQ(dag.task_id, task_spec_.task_id);
    EXPECT_EQ(dag.generation_strategy, "initial");
    EXPECT_EQ(dag.generation_iteration, 0);
    
    // Validate nodes and edges
    EXPECT_GT(dag.node_ids.size(), 0);
    EXPECT_GE(dag.edge_ids.size(), 0);
    
    // Each node should have a unique ID
    std::set<std::string> unique_node_ids(dag.node_ids.begin(), dag.node_ids.end());
    EXPECT_EQ(unique_node_ids.size(), dag.node_ids.size());
    
    // Each edge should have a unique ID
    std::set<std::string> unique_edge_ids(dag.edge_ids.begin(), dag.edge_ids.end());
    EXPECT_EQ(unique_edge_ids.size(), dag.edge_ids.size());
}
