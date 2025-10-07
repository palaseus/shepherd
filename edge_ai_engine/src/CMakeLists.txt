# Source CMakeLists.txt for Edge AI Engine

        # Core library
        add_library(edge_ai_engine_core
            core/edge_ai_engine.cpp
            core/model_loader.cpp
            core/inference_engine.cpp
            core/runtime_scheduler.cpp
            core/types.cpp
            core/model.cpp
            core/device.cpp
            core/cpu_device.cpp
            core/gpu_device.cpp
        )

# Optimization library
add_library(edge_ai_engine_optimization
    optimization/optimization_manager.cpp
)

# ML Policy library
add_library(edge_ai_engine_ml_policy
    ml_policy/ml_based_policy.cpp
)

# Graph library
add_library(edge_ai_engine_graph
    graph/graph.cpp
    graph/graph_compiler.cpp
    graph/graph_scheduler.cpp
    graph/graph_executor.cpp
)

# Distributed orchestration library
add_library(edge_ai_engine_distributed
    distributed/cluster_manager.cpp
    distributed/graph_partitioner.cpp
    distributed/distributed_scheduler.cpp
    distributed/migration_manager.cpp
    distributed/transport_layer.cpp
    distributed/reactive_scheduler.cpp
    distributed/predictive_failover.cpp
    distributed/streaming_migration.cpp
    distributed/self_healing_graphs.cpp
    distributed/advanced_telemetry.cpp
    distributed/multi_tenant_execution.cpp
)

# Governance library
add_library(edge_ai_engine_governance
    governance/governance_manager.cpp
)

# Federation library
add_library(edge_ai_engine_federation
    federation/federation_manager.cpp
)

# Evolution library
add_library(edge_ai_engine_evolution
    evolution/evolution_manager.cpp
)

# Analytics library
add_library(edge_ai_engine_analytics
    analytics/telemetry_analytics.cpp
)

# Security library
add_library(edge_ai_engine_security
    security/security_manager.cpp
)

# Autonomous library
add_library(edge_ai_engine_autonomous
    autonomous/dag_generator.cpp
    autonomous/synthetic_testbed.cpp
    autonomous/autonomous_optimizer.cpp
)

# Testing library
add_library(edge_ai_engine_testing
    testing/test_framework.cpp
    testing/test_reporter.cpp
    testing/test_discovery.cpp
    testing/test_runner.cpp
    testing/test_coverage.cpp
    testing/test_integration.cpp
    testing/test_performance.cpp
    testing/test_validation.cpp
    testing/test_orchestration.cpp
    testing/test_utilities.cpp
    testing/behavior_driven_testing.cpp
    testing/property_based_testing.cpp
    testing/simple_property_testing.cpp
    testing/interface_validator.cpp
    testing/comprehensive_report_generator.cpp
)

# Memory management library
add_library(edge_ai_engine_memory
    memory/memory_manager.cpp
    memory/memory_pool.cpp
    memory/memory_allocator.cpp
)

# Batching library
add_library(edge_ai_engine_batching
    batching/batching_manager.cpp
)

# Profiling library
add_library(edge_ai_engine_profiling
    profiling/profiler.cpp
    profiling/profiler_session.cpp
    profiling/metrics_collector.cpp
    profiling/performance_counter.cpp
)

# Hardware acceleration library
add_library(edge_ai_engine_hardware
    hardware/device_manager.cpp
    hardware/cpu_accelerator.cpp
    hardware/gpu_accelerator.cpp
    hardware/npu_accelerator.cpp
)

# Backend execution library
add_library(edge_ai_engine_backend
    backend/execution_backend.cpp
    backend/cpu_backend.cpp
    backend/mock_gpu_backend.cpp
    backend/backend_registry.cpp
)

# Utility library
add_library(edge_ai_engine_utils
    utils/logger.cpp
    utils/error_handler.cpp
    utils/config_parser.cpp
    utils/file_utils.cpp
)

# Main library
add_library(edge_ai_engine
    $<TARGET_OBJECTS:edge_ai_engine_core>
    $<TARGET_OBJECTS:edge_ai_engine_optimization>
    $<TARGET_OBJECTS:edge_ai_engine_memory>
    $<TARGET_OBJECTS:edge_ai_engine_batching>
    $<TARGET_OBJECTS:edge_ai_engine_profiling>
    $<TARGET_OBJECTS:edge_ai_engine_hardware>
    $<TARGET_OBJECTS:edge_ai_engine_backend>
    $<TARGET_OBJECTS:edge_ai_engine_utils>
)

# Set target properties
set_target_properties(edge_ai_engine PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(edge_ai_engine_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_optimization PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_memory PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_batching PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_profiling PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_hardware PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_backend PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_utils PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_distributed PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_governance PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_federation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_evolution PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_analytics PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_security PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_autonomous PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

target_include_directories(edge_ai_engine_testing PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Link libraries
target_link_libraries(edge_ai_engine_core
    Threads::Threads
)

target_link_libraries(edge_ai_engine_optimization
    edge_ai_engine_core
    edge_ai_engine_batching
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_ml_policy
    edge_ai_engine_core
    edge_ai_engine_optimization
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_graph
    edge_ai_engine_core
    edge_ai_engine_backend
    edge_ai_engine_optimization
    edge_ai_engine_ml_policy
    edge_ai_engine_profiling
    edge_ai_engine_memory
    edge_ai_engine_batching
    Threads::Threads
)

target_link_libraries(edge_ai_engine_memory
    edge_ai_engine_core
    edge_ai_engine_utils
)

target_link_libraries(edge_ai_engine_batching
    edge_ai_engine_core
    edge_ai_engine_utils
)

target_link_libraries(edge_ai_engine_profiling
    edge_ai_engine_core
    edge_ai_engine_utils
)

target_link_libraries(edge_ai_engine_hardware
    edge_ai_engine_core
    edge_ai_engine_utils
)

target_link_libraries(edge_ai_engine_backend
    edge_ai_engine_core
    edge_ai_engine_utils
)

target_link_libraries(edge_ai_engine_utils
    Threads::Threads
)

target_link_libraries(edge_ai_engine_distributed
    edge_ai_engine_core
    edge_ai_engine_graph
    edge_ai_engine_profiling
    edge_ai_engine_optimization
    Threads::Threads
)

target_link_libraries(edge_ai_engine_governance
    edge_ai_engine_core
    edge_ai_engine_distributed
    edge_ai_engine_ml_policy
    edge_ai_engine_optimization
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_federation
    edge_ai_engine_core
    edge_ai_engine_distributed
    edge_ai_engine_ml_policy
    edge_ai_engine_governance
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_evolution
    edge_ai_engine_core
    edge_ai_engine_distributed
    edge_ai_engine_ml_policy
    edge_ai_engine_governance
    edge_ai_engine_federation
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_analytics
    edge_ai_engine_core
    edge_ai_engine_distributed
    edge_ai_engine_ml_policy
    edge_ai_engine_governance
    edge_ai_engine_federation
    edge_ai_engine_evolution
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_security
    edge_ai_engine_core
    edge_ai_engine_distributed
    edge_ai_engine_governance
    edge_ai_engine_federation
    edge_ai_engine_analytics
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_autonomous
    edge_ai_engine_core
    edge_ai_engine_graph
    edge_ai_engine_distributed
    edge_ai_engine_evolution
    edge_ai_engine_analytics
    edge_ai_engine_governance
    edge_ai_engine_profiling
    Threads::Threads
)

target_link_libraries(edge_ai_engine_testing
    edge_ai_engine_core
    edge_ai_engine_profiling
    Threads::Threads
)

# Main library dependencies
target_link_libraries(edge_ai_engine
    edge_ai_engine_core
    edge_ai_engine_optimization
    edge_ai_engine_ml_policy
    edge_ai_engine_graph
    edge_ai_engine_distributed
    edge_ai_engine_governance
    edge_ai_engine_federation
    edge_ai_engine_evolution
    edge_ai_engine_analytics
    edge_ai_engine_security
    edge_ai_engine_autonomous
    edge_ai_engine_testing
    edge_ai_engine_memory
    edge_ai_engine_batching
    edge_ai_engine_profiling
    edge_ai_engine_hardware
    edge_ai_engine_backend
    edge_ai_engine_utils
    Threads::Threads
)

# Optional dependencies
if(OpenCL_FOUND)
    target_link_libraries(edge_ai_engine_hardware OpenCL::OpenCL)
endif()

if(Vulkan_FOUND)
    target_link_libraries(edge_ai_engine_hardware Vulkan::Vulkan)
endif()

if(Eigen3_FOUND)
    target_link_libraries(edge_ai_engine_optimization Eigen3::Eigen)
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(edge_ai_engine PRIVATE /W4 /WX)
else()
    target_compile_options(edge_ai_engine PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Debug/Release specific options
target_compile_definitions(edge_ai_engine_core PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(edge_ai_engine_optimization PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(edge_ai_engine_memory PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(edge_ai_engine_batching PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(edge_ai_engine_profiling PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_hardware PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(edge_ai_engine_backend PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(edge_ai_engine_utils PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(edge_ai_engine_distributed PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_governance PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_federation PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_evolution PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_analytics PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_security PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_autonomous PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

target_compile_definitions(edge_ai_engine_testing PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<BOOL:${PROFILER_ENABLED}>:PROFILER_ENABLED>
)

# Install targets
install(TARGETS edge_ai_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS edge_ai_engine_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_optimization
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_memory
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_batching
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_profiling
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_hardware
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_utils
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_distributed
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_governance
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_federation
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_evolution
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_analytics
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_security
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_autonomous
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS edge_ai_engine_testing
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
