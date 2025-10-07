# Distributed orchestration and auto-sharding layer
add_library(edge_ai_engine_distributed STATIC
    cluster_manager.cpp
    graph_partitioner.cpp
    distributed_scheduler.cpp
    migration_manager.cpp
    transport_layer.cpp
    reactive_scheduler.cpp
)

target_include_directories(edge_ai_engine_distributed PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(edge_ai_engine_distributed
    edge_ai_engine_core
    edge_ai_engine_graph
    edge_ai_engine_profiling
    edge_ai_engine_optimization
    edge_ai_engine_ml_policy
    Threads::Threads
)

# Compiler definitions
target_compile_definitions(edge_ai_engine_distributed PRIVATE
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

# Compiler options
target_compile_options(edge_ai_engine_distributed PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -march=native
    -fPIC
)

# C++ standard
target_compile_features(edge_ai_engine_distributed PUBLIC cxx_std_20)
