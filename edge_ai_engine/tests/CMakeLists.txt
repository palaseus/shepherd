# Tests CMakeLists.txt for Edge AI Engine

# Enable testing
enable_testing()

# Test executable
add_executable(edge_ai_engine_tests
    # Unit tests
    unit/test_edge_ai_engine.cpp
    unit/test_profiler.cpp
    unit/test_optimization_manager.cpp
    unit/test_ml_policy.cpp
    unit/test_reactive_scheduler.cpp
    unit/test_dag_generator_simple.cpp
    unit/test_synthetic_testbed_simple.cpp
    unit/test_autonomous_optimizer_simple.cpp
)

# BDT Test Runner
add_executable(bdt_test_runner
    bdt_test_runner.cpp
    step_definitions/edge_ai_steps_simple.cpp
)

# Simple Property-Based Test Runner
add_executable(simple_property_test_runner
    simple_property_test_runner.cpp
    property_based/basic_evolution_properties.cpp
)

# Comprehensive Test Runner
add_executable(comprehensive_test_runner
    comprehensive_test_runner.cpp
    property_based/simple_evolution_properties.cpp
)

# Link test executable
target_link_libraries(edge_ai_engine_tests
    edge_ai_engine
    gtest
    gtest_main
    gmock
    gmock_main
    Threads::Threads
)

# Link BDT test runner
target_link_libraries(bdt_test_runner
    edge_ai_engine
    edge_ai_engine_testing
    Threads::Threads
)

# Link Simple Property-Based test runner
target_link_libraries(simple_property_test_runner
    edge_ai_engine
    edge_ai_engine_testing
    Threads::Threads
)

# Link Comprehensive test runner
target_link_libraries(comprehensive_test_runner
    edge_ai_engine
    edge_ai_engine_testing
    Threads::Threads
)

# Include directories for tests
target_include_directories(edge_ai_engine_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/tests
)

# Include directories for BDT test runner
target_include_directories(bdt_test_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/tests
)

# Include directories for Simple Property-Based test runner
target_include_directories(simple_property_test_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/tests
)

# Include directories for Comprehensive test runner
target_include_directories(comprehensive_test_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/tests
)

# Compiler-specific options for tests
if(MSVC)
    target_compile_options(edge_ai_engine_tests PRIVATE /W4 /WX)
    target_compile_options(bdt_test_runner PRIVATE /W4 /WX)
    target_compile_options(simple_property_test_runner PRIVATE /W4 /WX)
    target_compile_options(comprehensive_test_runner PRIVATE /W4 /WX)
else()
    target_compile_options(edge_ai_engine_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(bdt_test_runner PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(simple_property_test_runner PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(comprehensive_test_runner PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Debug/Release specific options for tests
target_compile_definitions(edge_ai_engine_tests PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(bdt_test_runner PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(simple_property_test_runner PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(comprehensive_test_runner PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Add tests to CTest
add_test(NAME EdgeAIEngineTests COMMAND edge_ai_engine_tests)
add_test(NAME BDTTests COMMAND bdt_test_runner)
add_test(NAME SimplePropertyBasedTests COMMAND simple_property_test_runner)
add_test(NAME ComprehensiveTests COMMAND comprehensive_test_runner)

# Set test properties
set_tests_properties(EdgeAIEngineTests PROPERTIES
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(BDTTests PROPERTIES
    TIMEOUT 600
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(SimplePropertyBasedTests PROPERTIES
    TIMEOUT 600
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_tests_properties(ComprehensiveTests PROPERTIES
    TIMEOUT 1200
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Coverage configuration (if enabled)
if(ENABLE_COVERAGE)
    target_compile_options(edge_ai_engine_tests PRIVATE --coverage)
    target_link_libraries(edge_ai_engine_tests PRIVATE --coverage)
endif()

# Memory sanitizer configuration (if enabled)
if(ENABLE_MEMORY_SANITIZER)
    target_compile_options(edge_ai_engine_tests PRIVATE -fsanitize=memory)
    target_link_libraries(edge_ai_engine_tests PRIVATE -fsanitize=memory)
endif()

# Address sanitizer configuration (if enabled)
if(ENABLE_ADDRESS_SANITIZER)
    target_compile_options(edge_ai_engine_tests PRIVATE -fsanitize=address)
    target_link_libraries(edge_ai_engine_tests PRIVATE -fsanitize=address)
endif()

# Thread sanitizer configuration (if enabled)
if(ENABLE_THREAD_SANITIZER)
    target_compile_options(edge_ai_engine_tests PRIVATE -fsanitize=thread)
    target_link_libraries(edge_ai_engine_tests PRIVATE -fsanitize=thread)
endif()

# Install test executables
install(TARGETS edge_ai_engine_tests bdt_test_runner simple_property_test_runner comprehensive_test_runner
    RUNTIME DESTINATION bin
)
