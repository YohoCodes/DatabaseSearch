"""
Script to run tests for the SearchEngine application.

This script discovers and runs all tests for the SearchEngine application.
It uses the unittest framework to find and execute test cases defined in the 'tests' directory.

The script:
1. Sets up the Python path to ensure imports work correctly
2. Discovers all test cases in the 'tests' directory
3. Runs the tests with detailed output
4. Returns an appropriate exit code based on test results (0 if all pass, 1 if any fail)

Usage:
    python run_tests.py
"""
import os
import sys
import unittest

if __name__ == "__main__":
    # Add the parent directory to the path to import modules
    # This ensures that imports in test files work correctly
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Find and run all tests
    # The discover method automatically finds all test modules in the specified directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests")
    
    # Run the tests with verbosity level 2 for detailed output
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    # This is useful for CI/CD pipelines to detect test failures
    sys.exit(not result.wasSuccessful())
