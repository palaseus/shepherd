# Graph module CMakeLists.txt for Edge AI Engine

# Graph library sources
set(GRAPH_SOURCES
    graph.cpp
    graph_compiler.cpp
    graph_scheduler.cpp
    graph_executor.cpp
)

# Create graph library
add_library(edge_ai_engine_graph STATIC ${GRAPH_SOURCES})

# Include directories
target_include_directories(edge_ai_engine_graph PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Link libraries
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

# Compiler-specific options
if(MSVC)
    target_compile_options(edge_ai_engine_graph PRIVATE /W4 /WX)
else()
    target_compile_options(edge_ai_engine_graph PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Debug/Release specific options
target_compile_definitions(edge_ai_engine_graph PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Optional dependencies
if(OpenCL_FOUND)
    target_link_libraries(edge_ai_engine_graph OpenCL::OpenCL)
endif()

# Install targets
install(TARGETS edge_ai_engine_graph
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/graph/
    DESTINATION include/graph
    FILES_MATCHING PATTERN "*.h"
)
