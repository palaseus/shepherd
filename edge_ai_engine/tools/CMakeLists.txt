# Tools CMakeLists.txt for Edge AI Engine

# Trace viewer tool
add_executable(trace_viewer
    trace_viewer_simple.cpp
)

target_link_libraries(trace_viewer
    Threads::Threads
)

target_include_directories(trace_viewer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Validation report generator tool
add_executable(validation_report_generator
    validation_report_generator.cpp
)

target_link_libraries(validation_report_generator
    Threads::Threads
)

target_include_directories(validation_report_generator PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Compiler-specific options for tools
if(MSVC)
    target_compile_options(trace_viewer PRIVATE /W4 /WX)
    target_compile_options(validation_report_generator PRIVATE /W4 /WX)
else()
    target_compile_options(trace_viewer PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(validation_report_generator PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Debug/Release specific options for tools
target_compile_definitions(trace_viewer PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

target_compile_definitions(validation_report_generator PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# Install tool executables
install(TARGETS trace_viewer validation_report_generator
    RUNTIME DESTINATION bin
)
