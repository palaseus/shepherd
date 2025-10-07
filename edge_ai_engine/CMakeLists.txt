cmake_minimum_required(VERSION 3.20)
project(EdgeAIEngine VERSION 1.0.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Profiler option
option(PROFILER_ENABLED "Enable high-performance profiler" ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Find required packages
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Optional packages
find_package(CUDA QUIET)
find_package(OpenCL QUIET)
find_package(Vulkan QUIET)
find_package(Eigen3 QUIET)

# Third-party dependencies
include(FetchContent)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

# For now, we'll build without the heavy dependencies to get the basic structure working
# These can be added back later when implementing the actual model loading functionality
# 
# # ONNX Runtime
# FetchContent_Declare(
#     onnxruntime
#     GIT_REPOSITORY https://github.com/Microsoft/onnxruntime.git
#     GIT_TAG v1.15.1
# )
# FetchContent_MakeAvailable(onnxruntime)
# 
# # TensorFlow Lite
# FetchContent_Declare(
#     tensorflow
#     GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
#     GIT_TAG v2.13.0
# )
# FetchContent_MakeAvailable(tensorflow)
# 
# # PyTorch
# FetchContent_Declare(
#     pytorch
#     GIT_REPOSITORY https://github.com/pytorch/pytorch.git
#     GIT_TAG v2.0.1
# )
# FetchContent_MakeAvailable(pytorch)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)

# Benchmarks
add_subdirectory(benchmarks)

# Tools
add_subdirectory(tools)

add_subdirectory(examples)

# Enable testing
enable_testing()

# Custom targets
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS edge_ai_engine_tests
    COMMENT "Running all tests"
)

add_custom_target(coverage
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS edge_ai_engine_tests
    COMMENT "Running coverage analysis"
)

# Installation
install(TARGETS edge_ai_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
