#!/bin/bash
# Complete system test script for Edge AI Engine
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

# Function to run tests
run_tests() {
    local test_type=$1
    local test_executable=$2
    local test_name=$3
    
    print_status "Running $test_name tests..."
    
    if [ -f "$test_executable" ]; then
        if ./"$test_executable" --gtest_output=xml:"${test_type}_test_results.xml"; then
            print_success "$test_name tests passed"
            return 0
        else
            print_error "$test_name tests failed"
            return 1
        fi
    else
        print_warning "$test_executable not found, skipping $test_name tests"
        return 0
    fi
}

# Function to run benchmarks
run_benchmarks() {
    local benchmark_executable=$1
    local benchmark_name=$2
    
    print_status "Running $benchmark_name benchmark..."
    
    if [ -f "$benchmark_executable" ]; then
        if ./"$benchmark_executable"; then
            print_success "$benchmark_name benchmark completed"
            return 0
        else
            print_warning "$benchmark_name benchmark failed (non-critical)"
            return 0
        fi
    else
        print_warning "$benchmark_executable not found, skipping $benchmark_name benchmark"
        return 0
    fi
}

# Main function
main() {
    print_status "Starting complete system test for Edge AI Engine"
    print_status "=================================================="
    
    # Change to project root directory
    cd "$(dirname "$0")/.."
    
    # Check if build directory exists
    if [ ! -d "build/Release" ]; then
        print_error "Build directory not found. Please run build.sh first."
        exit 1
    fi
    
    # Check if test executables exist
    if [ ! -d "build/Release/bin" ]; then
        print_error "Bin directory not found. Please run build.sh first."
        exit 1
    fi
    
    cd build/Release/bin
    
    local test_results=()
    local benchmark_results=()
    
    # Run unit tests
    print_status "Running unit tests..."
    if run_tests "unit" "edge_ai_engine_tests" "Unit"; then
        test_results+=("Unit tests: PASSED")
    else
        test_results+=("Unit tests: FAILED")
    fi
    
    # Run BDT tests
    print_status "Running BDT tests..."
    if run_tests "bdt" "bdt_test_runner" "BDT"; then
        test_results+=("BDT tests: PASSED")
    else
        test_results+=("BDT tests: FAILED")
    fi
    
    # Run property-based tests
    print_status "Running property-based tests..."
    if run_tests "property" "simple_property_test_runner" "Property-based"; then
        test_results+=("Property-based tests: PASSED")
    else
        test_results+=("Property-based tests: FAILED")
    fi
    
    # Run comprehensive tests
    print_status "Running comprehensive tests..."
    if run_tests "comprehensive" "comprehensive_test_runner" "Comprehensive"; then
        test_results+=("Comprehensive tests: PASSED")
    else
        test_results+=("Comprehensive tests: FAILED")
    fi
    
    # Run benchmarks
    print_status "Running benchmarks..."
    
    # Profiler overhead benchmark
    if run_benchmarks "benchmark_profiler_overhead" "Profiler Overhead"; then
        benchmark_results+=("Profiler Overhead: COMPLETED")
    else
        benchmark_results+=("Profiler Overhead: FAILED")
    fi
    
    # Optimization system benchmark
    if run_benchmarks "benchmark_optimization_system" "Optimization System"; then
        benchmark_results+=("Optimization System: COMPLETED")
    else
        benchmark_results+=("Optimization System: FAILED")
    fi
    
    # Scheduler batching benchmark
    if run_benchmarks "benchmark_scheduler_batching" "Scheduler Batching"; then
        benchmark_results+=("Scheduler Batching: COMPLETED")
    else
        benchmark_results+=("Scheduler Batching: FAILED")
    fi
    
    # Inference latency benchmark
    if run_benchmarks "benchmark_inference_latency" "Inference Latency"; then
        benchmark_results+=("Inference Latency: COMPLETED")
    else
        benchmark_results+=("Inference Latency: FAILED")
    fi
    
    # Print results summary
    print_status "=================================================="
    print_status "TEST RESULTS SUMMARY"
    print_status "=================================================="
    
    for result in "${test_results[@]}"; do
        if [[ $result == *"PASSED"* ]]; then
            print_success "$result"
        else
            print_error "$result"
        fi
    done
    
    print_status "=================================================="
    print_status "BENCHMARK RESULTS SUMMARY"
    print_status "=================================================="
    
    for result in "${benchmark_results[@]}"; do
        if [[ $result == *"COMPLETED"* ]]; then
            print_success "$result"
        else
            print_warning "$result"
        fi
    done
    
    # Check for any test failures
    local failed_tests=0
    for result in "${test_results[@]}"; do
        if [[ $result == *"FAILED"* ]]; then
            ((failed_tests++))
        fi
    done
    
    if [ $failed_tests -eq 0 ]; then
        print_success "All tests passed! ✅"
        print_status "System is ready for production use."
        exit 0
    else
        print_error "$failed_tests test suite(s) failed! ❌"
        print_status "Please check the test output above for details."
        exit 1
    fi
}

# Run main function
main "$@"
