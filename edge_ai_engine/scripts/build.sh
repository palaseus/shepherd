#!/bin/bash
# Build script for Edge AI Engine
# Author: AI Co-Developer
# Date: 2024

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required commands
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    fi
    
    if ! command_exists make; then
        missing_deps+=("make")
    fi
    
    if ! command_exists g++; then
        missing_deps+=("g++")
    fi
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists pip3; then
        missing_deps+=("pip3")
    fi
    
    # Check for optional commands
    if ! command_exists nvcc; then
        print_warning "CUDA not found - GPU acceleration will be disabled"
    fi
    
    if ! command_exists clang++; then
        print_warning "Clang++ not found - using g++"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        print_error "Please install the missing dependencies and try again"
        exit 1
    fi
    
    print_success "All required dependencies found"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
    else
        # Install basic dependencies
        pip3 install numpy matplotlib seaborn pytest psutil
    fi
    
    print_success "Python dependencies installed"
}

# Function to create build directory
create_build_dir() {
    local build_type=$1
    local build_dir="build/${build_type}"
    
    print_status "Creating build directory: ${build_dir}"
    
    if [ -d "$build_dir" ]; then
        print_warning "Build directory already exists, cleaning..."
        rm -rf "$build_dir"
    fi
    
    mkdir -p "$build_dir"
    cd "$build_dir"
}

# Function to configure CMake
configure_cmake() {
    local build_type=$1
    local enable_tests=$2
    local enable_coverage=$3
    local enable_sanitizers=$4
    
    print_status "Configuring CMake..."
    
    local cmake_args=(
        "-DCMAKE_BUILD_TYPE=${build_type}"
        "-DCMAKE_CXX_STANDARD=20"
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
        "-DCMAKE_CXX_EXTENSIONS=OFF"
    )
    
    if [ "$enable_tests" = "true" ]; then
        cmake_args+=("-DENABLE_TESTS=ON")
    fi
    
    if [ "$enable_coverage" = "true" ]; then
        cmake_args+=("-DENABLE_COVERAGE=ON")
    fi
    
    if [ "$enable_sanitizers" = "true" ]; then
        cmake_args+=("-DENABLE_ADDRESS_SANITIZER=ON")
        cmake_args+=("-DENABLE_MEMORY_SANITIZER=ON")
        cmake_args+=("-DENABLE_THREAD_SANITIZER=ON")
    fi
    
    # Check for CUDA
    if command_exists nvcc; then
        cmake_args+=("-DENABLE_CUDA=ON")
    fi
    
    # Check for OpenCL
    if [ -f "/usr/include/CL/cl.h" ] || [ -f "/usr/local/include/CL/cl.h" ]; then
        cmake_args+=("-DENABLE_OPENCL=ON")
    fi
    
    # Check for Vulkan
    if [ -f "/usr/include/vulkan/vulkan.h" ] || [ -f "/usr/local/include/vulkan/vulkan.h" ]; then
        cmake_args+=("-DENABLE_VULKAN=ON")
    fi
    
    cmake "${cmake_args[@]}" ../..
    
    print_success "CMake configuration completed"
}

# Function to build the project
build_project() {
    local num_jobs=$1
    
    print_status "Building project with ${num_jobs} jobs..."
    
    make -j"$num_jobs"
    
    print_success "Build completed"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Run C++ tests
    if [ -f "edge_ai_engine_tests" ]; then
        ./edge_ai_engine_tests
        print_success "C++ tests completed"
    else
        print_warning "C++ test executable not found"
    fi
    
    # Run Python tests
    if [ -f "../../python/testing/test_runner.py" ]; then
        python3 ../../python/testing/test_runner.py --test-type all
        print_success "Python tests completed"
    else
        print_warning "Python test runner not found"
    fi
}

# Function to run benchmarks
run_benchmarks() {
    print_status "Running benchmarks..."
    
    if [ -f "../../python/testing/test_runner.py" ]; then
        python3 ../../python/testing/test_runner.py --benchmark --report ../../benchmark_results.json
        print_success "Benchmarks completed"
    else
        print_warning "Python test runner not found"
    fi
}

# Function to generate coverage report
generate_coverage() {
    print_status "Generating coverage report..."
    
    if command_exists gcov; then
        gcov -r ../../src/**/*.cpp
        print_success "Coverage report generated"
    else
        print_warning "gcov not found - coverage report not generated"
    fi
}

# Function to install the project
install_project() {
    print_status "Installing project..."
    
    make install
    
    print_success "Project installed"
}

# Function to clean build
clean_build() {
    print_status "Cleaning build..."
    
    if [ -d "build" ]; then
        rm -rf build
        print_success "Build cleaned"
    else
        print_warning "No build directory found"
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -t, --build-type TYPE   Build type (Debug, Release, RelWithDebInfo) [default: Release]"
    echo "  -j, --jobs NUM          Number of parallel jobs [default: auto]"
    echo "  -c, --clean             Clean build before building"
    echo "  -T, --tests             Run tests after building"
    echo "  -B, --benchmarks        Run benchmarks after building"
    echo "  -C, --coverage          Enable coverage reporting"
    echo "  -S, --sanitizers        Enable sanitizers"
    echo "  -I, --install           Install after building"
    echo "  --clean-only            Only clean build directory"
    echo "  --deps-only             Only install dependencies"
    echo ""
    echo "Examples:"
    echo "  $0                      # Build in Release mode"
    echo "  $0 -t Debug -T          # Build in Debug mode and run tests"
    echo "  $0 -c -T -B             # Clean build, run tests and benchmarks"
    echo "  $0 --clean-only         # Only clean build directory"
}

# Main function
main() {
    local build_type="Release"
    local num_jobs=$(nproc 2>/dev/null || echo "4")
    local clean_build_flag=false
    local run_tests_flag=false
    local run_benchmarks_flag=false
    local enable_coverage=false
    local enable_sanitizers=false
    local install_flag=false
    local clean_only=false
    local deps_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -t|--build-type)
                build_type="$2"
                shift 2
                ;;
            -j|--jobs)
                num_jobs="$2"
                shift 2
                ;;
            -c|--clean)
                clean_build_flag=true
                shift
                ;;
            -T|--tests)
                run_tests_flag=true
                shift
                ;;
            -B|--benchmarks)
                run_benchmarks_flag=true
                shift
                ;;
            -C|--coverage)
                enable_coverage=true
                shift
                ;;
            -S|--sanitizers)
                enable_sanitizers=true
                shift
                ;;
            -I|--install)
                install_flag=true
                shift
                ;;
            --clean-only)
                clean_only=true
                shift
                ;;
            --deps-only)
                deps_only=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Validate build type
    if [[ ! "$build_type" =~ ^(Debug|Release|RelWithDebInfo)$ ]]; then
        print_error "Invalid build type: $build_type"
        print_error "Valid build types: Debug, Release, RelWithDebInfo"
        exit 1
    fi
    
    # Change to project root directory
    cd "$(dirname "$0")/.."
    
    print_status "Starting build process..."
    print_status "Build type: $build_type"
    print_status "Number of jobs: $num_jobs"
    
    # Check dependencies
    check_dependencies
    
    # Install Python dependencies
    install_python_deps
    
    # If only installing dependencies, exit here
    if [ "$deps_only" = "true" ]; then
        print_success "Dependencies installed successfully"
        exit 0
    fi
    
    # Clean build if requested
    if [ "$clean_build_flag" = "true" ] || [ "$clean_only" = "true" ]; then
        clean_build
        if [ "$clean_only" = "true" ]; then
            exit 0
        fi
    fi
    
    # Create build directory
    create_build_dir "$build_type"
    
    # Configure CMake
    configure_cmake "$build_type" "$run_tests_flag" "$enable_coverage" "$enable_sanitizers"
    
    # Build project
    build_project "$num_jobs"
    
    # Run tests if requested
    if [ "$run_tests_flag" = "true" ]; then
        run_tests
    fi
    
    # Run benchmarks if requested
    if [ "$run_benchmarks_flag" = "true" ]; then
        run_benchmarks
    fi
    
    # Generate coverage report if requested
    if [ "$enable_coverage" = "true" ]; then
        generate_coverage
    fi
    
    # Install if requested
    if [ "$install_flag" = "true" ]; then
        install_project
    fi
    
    print_success "Build process completed successfully!"
}

# Run main function
main "$@"
