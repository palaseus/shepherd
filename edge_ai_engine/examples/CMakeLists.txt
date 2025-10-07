# Examples CMakeLists.txt for Edge AI Engine

# Basic inference example (commented out due to Tensor copy issues)
# add_executable(basic_inference
#     basic_inference.cpp
# )

# # Link with main library
# target_link_libraries(basic_inference
#     edge_ai_engine
#     Threads::Threads
# )

# # Include directories
# target_include_directories(basic_inference PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
#     ${CMAKE_SOURCE_DIR}/third_party
# )

# # Set target properties
# set_target_properties(basic_inference PROPERTIES
#     CXX_STANDARD 20
#     CXX_STANDARD_REQUIRED ON
#     CXX_EXTENSIONS OFF
# )

# # Compiler-specific options
# if(MSVC)
#     target_compile_options(basic_inference PRIVATE /W4 /WX)
# else()
#     target_compile_options(basic_inference PRIVATE -Wall -Wextra -Wpedantic -Werror)
# endif()

# # Debug/Release specific options
# target_compile_definitions(basic_inference PRIVATE
#     $<$<CONFIG:Debug>:DEBUG>
#     $<$<CONFIG:Release>:NDEBUG>
# )

# # Install example
# install(TARGETS basic_inference
#     RUNTIME DESTINATION bin
# )
