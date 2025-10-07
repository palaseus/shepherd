# Testing module CMakeLists.txt

# Set testing source files
set(TESTING_SOURCES
    test_framework.cpp
    test_reporter.cpp
    test_discovery.cpp
    test_runner.cpp
    test_coverage.cpp
    test_integration.cpp
    test_performance.cpp
    test_validation.cpp
    test_orchestration.cpp
    test_utilities.cpp
    behavior_driven_testing.cpp
    property_based_testing.cpp
    simple_property_testing.cpp
    interface_validator.cpp
    comprehensive_report_generator.cpp
)

# Set testing header files
set(TESTING_HEADERS
    ${CMAKE_SOURCE_DIR}/include/testing/test_framework.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_reporter.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_discovery.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_runner.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_coverage.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_integration.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_performance.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_validation.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_orchestration.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_utilities.h
    ${CMAKE_SOURCE_DIR}/include/testing/test_common.h
    ${CMAKE_SOURCE_DIR}/include/testing/behavior_driven_testing.h
    ${CMAKE_SOURCE_DIR}/include/testing/property_based_testing.h
    ${CMAKE_SOURCE_DIR}/include/testing/simple_property_testing.h
    ${CMAKE_SOURCE_DIR}/include/testing/interface_validator.h
    ${CMAKE_SOURCE_DIR}/include/testing/comprehensive_report_generator.h
)

# Create testing library
add_library(edge_ai_engine_testing STATIC ${TESTING_SOURCES})

# Set target properties
set_target_properties(edge_ai_engine_testing PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Include directories
target_include_directories(edge_ai_engine_testing PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/testing
    ${CMAKE_SOURCE_DIR}/include/core
)

# Link libraries
target_link_libraries(edge_ai_engine_testing
    edge_ai_core
    ${CMAKE_THREAD_LIBS_INIT}
)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# Link YAML library
target_link_libraries(edge_ai_engine_testing ${YAML_CPP_LIBRARIES})
target_include_directories(edge_ai_engine_testing PRIVATE ${YAML_CPP_INCLUDE_DIRS})

# Compiler flags
target_compile_options(edge_ai_engine_testing PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -O2
    -g
)

# Add testing to main library
target_link_libraries(edge_ai_engine edge_ai_engine_testing)

# Install testing library
install(TARGETS edge_ai_engine_testing
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install testing headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/testing/
    DESTINATION include/testing
    FILES_MATCHING PATTERN "*.h"
)
