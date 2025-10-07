# ML Policy module CMakeLists.txt
cmake_minimum_required(VERSION 3.16)

# Create ML policy library
add_library(edge_ai_engine_ml_policy
    ml_based_policy.cpp
)

# Set target properties
set_target_properties(edge_ai_engine_ml_policy PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(edge_ai_engine_ml_policy
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(edge_ai_engine_ml_policy
    PUBLIC
        edge_ai_engine_core
        edge_ai_engine_optimization
        edge_ai_engine_profiling
    PRIVATE
        Threads::Threads
)

# Compiler options
target_compile_options(edge_ai_engine_ml_policy PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Install targets
install(TARGETS edge_ai_engine_ml_policy
    EXPORT EdgeAIEngineTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)
