#!/usr/bin/env python3
"""
Test runner for Edge AI Engine
Author: AI Co-Developer
Date: 2024

This module provides utilities for running tests and benchmarks for the Edge AI Engine.
"""

import os
import sys
import argparse
import logging
import json
import time
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    import psutil
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Please install required packages: pip install pytest psutil matplotlib seaborn")

logger = logging.getLogger(__name__)

class TestRunner:
    """Test runner utility class"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.project_root = project_root
        self.test_results = {}
    
    def run_unit_tests(self, test_path: Optional[str] = None) -> bool:
        """
        Run unit tests
        
        Args:
            test_path: Path to specific test file or directory
            
        Returns:
            True if all tests passed, False otherwise
        """
        try:
            self.logger.info("Running unit tests")
            
            if test_path is None:
                test_path = self.project_root / "tests" / "unit"
            
            # Run pytest
            cmd = ["python", "-m", "pytest", str(test_path), "-v", "--tb=short"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            self.test_results['unit_tests'] = {
                'passed': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Error running unit tests: {e}")
            return False
    
    def run_integration_tests(self, test_path: Optional[str] = None) -> bool:
        """
        Run integration tests
        
        Args:
            test_path: Path to specific test file or directory
            
        Returns:
            True if all tests passed, False otherwise
        """
        try:
            self.logger.info("Running integration tests")
            
            if test_path is None:
                test_path = self.project_root / "tests" / "integration"
            
            # Run pytest
            cmd = ["python", "-m", "pytest", str(test_path), "-v", "--tb=short"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            self.test_results['integration_tests'] = {
                'passed': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Error running integration tests: {e}")
            return False
    
    def run_performance_tests(self, test_path: Optional[str] = None) -> bool:
        """
        Run performance tests
        
        Args:
            test_path: Path to specific test file or directory
            
        Returns:
            True if all tests passed, False otherwise
        """
        try:
            self.logger.info("Running performance tests")
            
            if test_path is None:
                test_path = self.project_root / "tests" / "performance"
            
            # Run pytest with performance markers
            cmd = ["python", "-m", "pytest", str(test_path), "-v", "--tb=short", "-m", "performance"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            self.test_results['performance_tests'] = {
                'passed': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            return result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Error running performance tests: {e}")
            return False
    
    def run_benchmarks(self, benchmark_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run benchmarks
        
        Args:
            benchmark_config: Benchmark configuration
            
        Returns:
            Dictionary containing benchmark results
        """
        try:
            self.logger.info("Running benchmarks")
            
            if benchmark_config is None:
                benchmark_config = self._get_default_benchmark_config()
            
            results = {}
            
            # Run different types of benchmarks
            if benchmark_config.get('inference_benchmark', True):
                results['inference'] = self._run_inference_benchmark(benchmark_config)
            
            if benchmark_config.get('memory_benchmark', True):
                results['memory'] = self._run_memory_benchmark(benchmark_config)
            
            if benchmark_config.get('latency_benchmark', True):
                results['latency'] = self._run_latency_benchmark(benchmark_config)
            
            self.test_results['benchmarks'] = results
            return results
            
        except Exception as e:
            self.logger.error(f"Error running benchmarks: {e}")
            return {}
    
    def _get_default_benchmark_config(self) -> Dict[str, Any]:
        """Get default benchmark configuration"""
        return {
            'inference_benchmark': True,
            'memory_benchmark': True,
            'latency_benchmark': True,
            'num_iterations': 100,
            'batch_sizes': [1, 4, 8, 16, 32],
            'input_shapes': [(1, 3, 224, 224), (1, 3, 512, 512)],
            'model_types': ['onnx', 'tflite', 'pt']
        }
    
    def _run_inference_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference benchmark"""
        results = {
            'throughput': {},
            'latency': {},
            'memory_usage': {}
        }
        
        try:
            # This is a placeholder implementation
            # In practice, you would load models and run actual inference
            for batch_size in config.get('batch_sizes', [1, 4, 8, 16, 32]):
                # Simulate inference timing
                start_time = time.time()
                time.sleep(0.001)  # Simulate inference time
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                throughput = batch_size / latency * 1000  # Samples per second
                
                results['throughput'][batch_size] = throughput
                results['latency'][batch_size] = latency
                results['memory_usage'][batch_size] = batch_size * 1024 * 1024  # Simulate memory usage
            
        except Exception as e:
            self.logger.error(f"Error running inference benchmark: {e}")
        
        return results
    
    def _run_memory_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run memory benchmark"""
        results = {
            'peak_memory': 0,
            'average_memory': 0,
            'memory_growth': []
        }
        
        try:
            # Monitor memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_samples = []
            for i in range(config.get('num_iterations', 100)):
                # Simulate memory allocation
                data = np.random.rand(1000, 1000)
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
                time.sleep(0.01)
            
            results['peak_memory'] = max(memory_samples)
            results['average_memory'] = sum(memory_samples) / len(memory_samples)
            results['memory_growth'] = memory_samples
            
        except Exception as e:
            self.logger.error(f"Error running memory benchmark: {e}")
        
        return results
    
    def _run_latency_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run latency benchmark"""
        results = {
            'min_latency': float('inf'),
            'max_latency': 0,
            'average_latency': 0,
            'p95_latency': 0,
            'p99_latency': 0,
            'latency_samples': []
        }
        
        try:
            latencies = []
            for i in range(config.get('num_iterations', 100)):
                start_time = time.perf_counter()
                # Simulate inference
                time.sleep(0.001)
                end_time = time.perf_counter()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            latencies.sort()
            results['min_latency'] = latencies[0]
            results['max_latency'] = latencies[-1]
            results['average_latency'] = sum(latencies) / len(latencies)
            results['p95_latency'] = latencies[int(len(latencies) * 0.95)]
            results['p99_latency'] = latencies[int(len(latencies) * 0.99)]
            results['latency_samples'] = latencies
            
        except Exception as e:
            self.logger.error(f"Error running latency benchmark: {e}")
        
        return results
    
    def generate_report(self, output_path: str) -> bool:
        """
        Generate test report
        
        Args:
            output_path: Path to output report file
            
        Returns:
            True if report generated successfully, False otherwise
        """
        try:
            self.logger.info(f"Generating test report: {output_path}")
            
            report = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_results': self.test_results,
                'summary': self._generate_summary()
            }
            
            # Save JSON report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate HTML report
            html_path = output_path.replace('.json', '.html')
            self._generate_html_report(report, html_path)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return False
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_categories': {}
        }
        
        for category, results in self.test_results.items():
            if isinstance(results, dict) and 'passed' in results:
                summary['total_tests'] += 1
                if results['passed']:
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                
                summary['test_categories'][category] = {
                    'passed': results['passed'],
                    'return_code': results.get('return_code', -1)
                }
        
        return summary
    
    def _generate_html_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Generate HTML report"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Edge AI Engine Test Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .summary {{ margin: 20px 0; }}
                    .test-category {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                    .passed {{ background-color: #d4edda; }}
                    .failed {{ background-color: #f8d7da; }}
                    .benchmark {{ margin: 20px 0; }}
                    .chart {{ margin: 20px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Edge AI Engine Test Report</h1>
                    <p>Generated: {report['timestamp']}</p>
                </div>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total Tests: {report['summary']['total_tests']}</p>
                    <p>Passed: {report['summary']['passed_tests']}</p>
                    <p>Failed: {report['summary']['failed_tests']}</p>
                </div>
                
                <div class="test-categories">
                    <h2>Test Categories</h2>
            """
            
            for category, results in report['summary']['test_categories'].items():
                status_class = 'passed' if results['passed'] else 'failed'
                status_text = 'PASSED' if results['passed'] else 'FAILED'
                html_content += f"""
                    <div class="test-category {status_class}">
                        <h3>{category.replace('_', ' ').title()}</h3>
                        <p>Status: {status_text}</p>
                        <p>Return Code: {results['return_code']}</p>
                    </div>
                """
            
            html_content += """
                </div>
                
                <div class="benchmarks">
                    <h2>Benchmarks</h2>
                    <p>Benchmark results are available in the JSON report.</p>
                </div>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Test runner for Edge AI Engine')
    parser.add_argument('--test-type', choices=['unit', 'integration', 'performance', 'all'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--test-path', help='Path to specific test file or directory')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarks')
    parser.add_argument('--report', help='Path to output report file')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create test runner
    test_runner = TestRunner()
    
    # Run tests
    success = True
    
    if args.test_type in ['unit', 'all']:
        if not test_runner.run_unit_tests(args.test_path):
            success = False
    
    if args.test_type in ['integration', 'all']:
        if not test_runner.run_integration_tests(args.test_path):
            success = False
    
    if args.test_type in ['performance', 'all']:
        if not test_runner.run_performance_tests(args.test_path):
            success = False
    
    # Run benchmarks
    if args.benchmark:
        benchmark_results = test_runner.run_benchmarks()
        if not benchmark_results:
            success = False
    
    # Generate report
    if args.report:
        if not test_runner.generate_report(args.report):
            success = False
    
    if success:
        print("All tests completed successfully")
    else:
        print("Some tests failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
