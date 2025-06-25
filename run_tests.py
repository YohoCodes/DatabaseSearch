"""
Script to run tests for the SearchEngine application.
"""
import os
import sys
import unittest

if __name__ == "__main__":
    # Add the parent directory to the path to import modules
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Find and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover("tests")
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())
