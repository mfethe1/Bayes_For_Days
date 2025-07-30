"""
Comprehensive test runner for Bayes For Days platform.

This module provides utilities for running the complete test suite
with performance validation, coverage analysis, and benchmark testing.

Features:
- Unit test execution with coverage reporting
- Benchmark problem validation
- Performance regression testing
- Integration test coordination
- Test result analysis and reporting
"""

import pytest
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)


class TestRunner:
    """
    Comprehensive test runner for the Bayes For Days platform.
    
    Coordinates execution of all test types and provides
    detailed reporting on test results and performance.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize test runner.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.src_dir = self.project_root / "src"
        self.results_dir = self.project_root / "test_results"
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_config = {
            'unit_tests': True,
            'integration_tests': True,
            'benchmark_tests': False,  # Slow tests, disabled by default
            'coverage_threshold': 85.0,
            'performance_regression_threshold': 1.5,  # 50% slowdown threshold
        }
        
        logger.info(f"Initialized test runner for project: {self.project_root}")
    
    def run_all_tests(
        self,
        include_benchmarks: bool = False,
        coverage: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run all tests with comprehensive reporting.
        
        Args:
            include_benchmarks: Whether to run benchmark tests
            coverage: Whether to generate coverage report
            verbose: Whether to use verbose output
            
        Returns:
            Dictionary with test results and metrics
        """
        start_time = time.time()
        results = {
            'start_time': start_time,
            'test_results': {},
            'coverage': {},
            'performance': {},
            'summary': {},
        }
        
        try:
            # Run unit tests
            if self.test_config['unit_tests']:
                logger.info("Running unit tests...")
                unit_results = self._run_unit_tests(coverage=coverage, verbose=verbose)
                results['test_results']['unit'] = unit_results
            
            # Run integration tests
            if self.test_config['integration_tests']:
                logger.info("Running integration tests...")
                integration_results = self._run_integration_tests(verbose=verbose)
                results['test_results']['integration'] = integration_results
            
            # Run benchmark tests
            if include_benchmarks and self.test_config['benchmark_tests']:
                logger.info("Running benchmark tests...")
                benchmark_results = self._run_benchmark_tests(verbose=verbose)
                results['test_results']['benchmark'] = benchmark_results
            
            # Generate coverage report
            if coverage:
                logger.info("Generating coverage report...")
                coverage_results = self._generate_coverage_report()
                results['coverage'] = coverage_results
            
            # Performance analysis
            logger.info("Analyzing performance...")
            performance_results = self._analyze_performance()
            results['performance'] = performance_results
            
            # Generate summary
            results['summary'] = self._generate_summary(results)
            results['execution_time'] = time.time() - start_time
            
            # Save results
            self._save_results(results)
            
            # Print summary
            if verbose:
                self._print_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            results['error'] = str(e)
            results['execution_time'] = time.time() - start_time
            return results
    
    def _run_unit_tests(self, coverage: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """Run unit tests with pytest."""
        unit_test_dir = self.test_dir / "unit"
        
        # Build pytest command
        cmd = ["python", "-m", "pytest", str(unit_test_dir)]
        
        if coverage:
            cmd.extend([
                "--cov=bayes_for_days",
                f"--cov-report=html:{self.results_dir}/coverage_html",
                f"--cov-report=json:{self.results_dir}/coverage.json",
                "--cov-report=term-missing"
            ])
        
        if verbose:
            cmd.append("-v")
        
        # Add output options
        cmd.extend([
            f"--junitxml={self.results_dir}/unit_tests.xml",
            "--tb=short"
        ])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'return_code': -1,
                'error': 'Unit tests timed out after 5 minutes',
                'success': False
            }
        except Exception as e:
            return {
                'return_code': -1,
                'error': str(e),
                'success': False
            }
    
    def _run_integration_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run integration tests."""
        integration_test_dir = self.test_dir / "integration"
        
        if not integration_test_dir.exists():
            return {
                'return_code': 0,
                'message': 'No integration tests found',
                'success': True
            }
        
        cmd = ["python", "-m", "pytest", str(integration_test_dir)]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            f"--junitxml={self.results_dir}/integration_tests.xml",
            "--tb=short"
        ])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'return_code': -1,
                'error': 'Integration tests timed out after 10 minutes',
                'success': False
            }
        except Exception as e:
            return {
                'return_code': -1,
                'error': str(e),
                'success': False
            }
    
    def _run_benchmark_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run benchmark tests."""
        benchmark_test_dir = self.test_dir / "benchmarks"
        
        cmd = ["python", "-m", "pytest", str(benchmark_test_dir), "-m", "not slow"]
        
        if verbose:
            cmd.append("-v")
        
        cmd.extend([
            f"--junitxml={self.results_dir}/benchmark_tests.xml",
            "--tb=short"
        ])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            return {
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'return_code': -1,
                'error': 'Benchmark tests timed out after 30 minutes',
                'success': False
            }
        except Exception as e:
            return {
                'return_code': -1,
                'error': str(e),
                'success': False
            }
    
    def _generate_coverage_report(self) -> Dict[str, Any]:
        """Generate and analyze coverage report."""
        coverage_file = self.results_dir / "coverage.json"
        
        if not coverage_file.exists():
            return {'error': 'Coverage file not found'}
        
        try:
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
            
            # Extract key metrics
            total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
            
            # Analyze by module
            module_coverage = {}
            for filename, file_data in coverage_data.get('files', {}).items():
                if 'bayes_for_days' in filename:
                    module_name = self._extract_module_name(filename)
                    coverage_pct = file_data.get('summary', {}).get('percent_covered', 0)
                    module_coverage[module_name] = coverage_pct
            
            return {
                'total_coverage': total_coverage,
                'module_coverage': module_coverage,
                'meets_threshold': total_coverage >= self.test_config['coverage_threshold'],
                'threshold': self.test_config['coverage_threshold'],
                'coverage_file': str(coverage_file)
            }
            
        except Exception as e:
            return {'error': f'Failed to parse coverage report: {e}'}
    
    def _extract_module_name(self, filename: str) -> str:
        """Extract module name from filename."""
        # Convert path to module name
        if 'bayes_for_days' in filename:
            parts = filename.split('bayes_for_days')[1].split('/')
            parts = [p for p in parts if p and not p.endswith('.py')]
            return '.'.join(parts) if parts else 'core'
        return 'unknown'
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics from test results."""
        # This is a simplified performance analysis
        # In practice, would analyze timing data from tests
        
        performance_data = {
            'test_execution_times': {},
            'memory_usage': {},
            'regression_detected': False,
            'performance_summary': 'Performance analysis not fully implemented'
        }
        
        # Check for performance regression (simplified)
        # Would compare against baseline performance data
        
        return performance_data
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test execution summary."""
        summary = {
            'total_tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'overall_success': True,
            'issues': []
        }
        
        # Analyze test results
        for test_type, test_result in results.get('test_results', {}).items():
            if test_result.get('success', False):
                summary['tests_passed'] += 1
            else:
                summary['tests_failed'] += 1
                summary['overall_success'] = False
                summary['issues'].append(f"{test_type} tests failed")
            
            summary['total_tests_run'] += 1
        
        # Check coverage
        coverage = results.get('coverage', {})
        if coverage.get('meets_threshold', True) is False:
            summary['issues'].append(
                f"Coverage {coverage.get('total_coverage', 0):.1f}% below threshold "
                f"{coverage.get('threshold', 0):.1f}%"
            )
        
        # Check performance
        performance = results.get('performance', {})
        if performance.get('regression_detected', False):
            summary['issues'].append("Performance regression detected")
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        results_file = self.results_dir / "test_results.json"
        
        try:
            # Make results JSON serializable
            serializable_results = self._make_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Test results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test execution summary."""
        print("\n" + "="*60)
        print("BAYES FOR DAYS - TEST EXECUTION SUMMARY")
        print("="*60)
        
        summary = results.get('summary', {})
        
        print(f"Total Tests Run: {summary.get('total_tests_run', 0)}")
        print(f"Tests Passed: {summary.get('tests_passed', 0)}")
        print(f"Tests Failed: {summary.get('tests_failed', 0)}")
        print(f"Overall Success: {'✓' if summary.get('overall_success', False) else '✗'}")
        
        # Coverage information
        coverage = results.get('coverage', {})
        if coverage:
            total_cov = coverage.get('total_coverage', 0)
            threshold = coverage.get('threshold', 0)
            meets_threshold = coverage.get('meets_threshold', False)
            
            print(f"\nCode Coverage: {total_cov:.1f}% {'✓' if meets_threshold else '✗'}")
            print(f"Coverage Threshold: {threshold:.1f}%")
        
        # Issues
        issues = summary.get('issues', [])
        if issues:
            print(f"\nIssues Found ({len(issues)}):")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\nNo issues found! ✓")
        
        # Execution time
        exec_time = results.get('execution_time', 0)
        print(f"\nTotal Execution Time: {exec_time:.2f} seconds")
        
        print("="*60)


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Bayes For Days test suite")
    parser.add_argument("--benchmarks", action="store_true", help="Include benchmark tests")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage analysis")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if not args.quiet else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    runner = TestRunner()
    results = runner.run_all_tests(
        include_benchmarks=args.benchmarks,
        coverage=not args.no_coverage,
        verbose=not args.quiet
    )
    
    # Exit with appropriate code
    if results.get('summary', {}).get('overall_success', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
