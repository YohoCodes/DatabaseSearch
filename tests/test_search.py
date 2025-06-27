"""
Tests for the Search class.

This module contains unit tests for the Search class, which is the core component
of the semantic search engine. The tests verify that the Search class correctly
initializes, validates inputs, and returns appropriate search results.

The test suite covers:
- Input validation during initialization
- Search behavior with different column configurations
- Handling of empty or invalid queries
- Proper result formatting for different search configurations
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
# This allows importing from the parent directory without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from operations.search import Search


class TestSearch(unittest.TestCase):
    """
    Test cases for the Search class.
    
    This class tests the functionality of the Search class with different
    configurations and input scenarios. It verifies that the search engine
    behaves correctly under various conditions.
    """
    
    def setUp(self):
        """
        Set up test data before each test.
        
        This method creates a test DataFrame with sample data and initializes
        several Search instances with different configurations for testing.
        """
        # Create a test DataFrame with predictable test data
        # The data contains titles, content, and authors that will be used for searching
        self.data = pd.DataFrame({
            'title': [
                'Machine learning for beginners',
                'Advanced deep learning techniques',
                'Introduction to natural language processing',
                'Python programming tutorial'
            ],
            'content': [
                'Learn the basics of machine learning algorithms and applications.',
                'Explore cutting-edge deep learning models and architectures.',
                'Understand how computers process and analyze human language.',
                'Get started with Python programming from the ground up.'
            ],
            'author': [
                'John Smith',
                'Jane Doe',
                'Alex Johnson',
                'Maria Garcia'
            ]
        })
        
        # Create Search instances for different configurations to test various scenarios
        
        # Configuration 1: Search and return the same column (title)
        self.search_same_column = Search(
            database=self.data,
            search_by='title',
            search_res='title',
            res_is_by=True,
            cache_embeddings=False
        )
        
        # Configuration 2: Search by title, return content (string column)
        self.search_string_column = Search(
            database=self.data,
            search_by='title',
            search_res='content',
            res_is_by=False,
            cache_embeddings=False
        )
        
        # Configuration 3: Search by title, return multiple columns (content and author)
        self.search_list_columns = Search(
            database=self.data,
            search_by='title',
            search_res=['content', 'author'],
            res_is_by=False,
            cache_embeddings=False
        )
    
    def test_init_validation(self):
        """
        Test that initialization validates inputs correctly.
        
        This test verifies that the Search class constructor properly validates
        input parameters and raises appropriate exceptions when invalid inputs
        are provided.
        """
        # Test case 1: Invalid search_by column should raise ValueError
        with self.assertRaises(ValueError):
            Search(
                database=self.data,
                search_by='nonexistent_column',  # Column that doesn't exist
                search_res='title',
                cache_embeddings=False
            )
        
        # Test case 2: Invalid search_res column (string) should raise ValueError
        with self.assertRaises(ValueError):
            Search(
                database=self.data,
                search_by='title',
                search_res='nonexistent_column',  # Column that doesn't exist
                cache_embeddings=False
            )
        
        # Test case 3: Invalid search_res column (in list) should raise ValueError
        with self.assertRaises(ValueError):
            Search(
                database=self.data,
                search_by='title',
                search_res=['content', 'nonexistent_column'],  # List with invalid column
                cache_embeddings=False
            )
    
    def test_top_results_same_column(self):
        """
        Test top_results when search_by and search_res are the same.
        
        This test verifies that when searching and returning the same column,
        the Search.top_results method returns a numpy array of values from that
        column, and that the results are semantically relevant to the query.
        """
        # Perform a search for "machine learning" and get top 2 results
        results = self.search_same_column.top_results('machine learning', n_results=2)
        
        # Verify the result type is a numpy array
        self.assertIsInstance(results, np.ndarray)
        
        # Verify we get exactly the requested number of results
        self.assertEqual(len(results), 2)
        
        # Verify the results are relevant (first result should contain "machine learning")
        self.assertTrue('machine learning' in results[0].lower())
    
    def test_top_results_string_column(self):
        """
        Test top_results when search_res is a single column.
        
        This test verifies that when searching by one column and returning a different
        column, the Search.top_results method returns an iterator of tuples where
        each tuple contains the search column value and the result column value.
        """
        # Perform a search for "machine learning" and get top 2 results
        # Convert the iterator to a list to examine the results
        results = list(self.search_string_column.top_results('machine learning', n_results=2))
        
        # Verify we get exactly the requested number of results
        self.assertEqual(len(results), 2)
        
        # Verify each result is a tuple with 2 elements (title, content)
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
    
    def test_top_results_list_columns(self):
        """
        Test top_results when search_res is a list of columns.
        
        This test verifies that when searching by one column and returning multiple
        columns, the Search.top_results method returns an iterator of tuples where
        each tuple contains the search column value followed by values from each
        of the result columns.
        """
        # Perform a search for "machine learning" and get top 2 results
        # Convert the iterator to a list to examine the results
        results = list(self.search_list_columns.top_results('machine learning', n_results=2))
        
        # Verify we get exactly the requested number of results
        self.assertEqual(len(results), 2)
        
        # Verify each result is a tuple with 3 elements (title, content, author)
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
    
    def test_empty_query(self):
        """
        Test that an empty query raises a ValueError.
        
        This test verifies that the Search.top_results method properly validates
        the query string and raises a ValueError if the query is empty or
        contains only whitespace.
        """
        # Test case 1: Empty string should raise ValueError
        with self.assertRaises(ValueError):
            self.search_same_column.top_results('', n_results=2)
        
        # Test case 2: Whitespace-only string should raise ValueError
        with self.assertRaises(ValueError):
            self.search_same_column.top_results('   ', n_results=2)


if __name__ == '__main__':
    unittest.main()
