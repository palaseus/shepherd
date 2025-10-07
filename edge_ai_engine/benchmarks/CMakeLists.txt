# Benchmarks CMakeLists.txt for Edge AI Engine

# Profiler overhead benchmark (simpler, more reliable)
add_executable(benchmark_profiler_overhead
    benchmark_profiler_overhead.cpp
)

target_link_libraries(benchmark_profiler_overhead
    edge_ai_engine_profiling
    Threads::Threads
)

target_include_directories(benchmark_profiler_overhead PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Scheduler and batching benchmark
add_executable(benchmark_scheduler_batching
    benchmark_scheduler_batching.cpp
)

target_link_libraries(benchmark_scheduler_batching
    edge_ai_engine
    Threads::Threads
)

target_include_directories(benchmark_scheduler_batching PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Optimization system benchmark
add_executable(benchmark_optimization_system
    benchmark_optimization_system.cpp
)

target_link_libraries(benchmark_optimization_system
    edge_ai_engine
    Threads::Threads
)

target_include_directories(benchmark_optimization_system PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Compiler-specific options for benchmarks
if(MSVC)
    target_compile_options(benchmark_profiler_overhead PRIVATE /W4 /WX)
    target_compile_options(benchmark_scheduler_batching PRIVATE /W4 /WX)
    target_compile_options(benchmark_optimization_system PRIVATE /W4 /WX)
else()
    target_compile_options(benchmark_profiler_overhead PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(benchmark_scheduler_batching PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(benchmark_optimization_system PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Debug/Release specific options for benchmarks
target_compile_definitions(benchmark_profiler_overhead PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(benchmark_scheduler_batching PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(benchmark_optimization_system PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Install benchmark executables
install(TARGETS benchmark_profiler_overhead benchmark_scheduler_batching benchmark_optimization_system
    RUNTIME DESTINATION bin
)
