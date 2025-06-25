"""
Tests for the Search class.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from operations.search import Search


class TestSearch(unittest.TestCase):
    """
    Test cases for the Search class.
    """
    
    def setUp(self):
        """
        Set up test data before each test.
        """
        # Create a test DataFrame
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
        
        # Create Search instances for different configurations
        self.search_same_column = Search(
            database=self.data,
            search_by='title',
            search_res='title',
            res_is_by=True,
            cache_embeddings=False
        )
        
        self.search_string_column = Search(
            database=self.data,
            search_by='title',
            search_res='content',
            res_is_by=False,
            cache_embeddings=False
        )
        
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
        """
        # Test invalid search_by column
        with self.assertRaises(ValueError):
            Search(
                database=self.data,
                search_by='nonexistent_column',
                search_res='title',
                cache_embeddings=False
            )
        
        # Test invalid search_res column
        with self.assertRaises(ValueError):
            Search(
                database=self.data,
                search_by='title',
                search_res='nonexistent_column',
                cache_embeddings=False
            )
        
        # Test invalid search_res list
        with self.assertRaises(ValueError):
            Search(
                database=self.data,
                search_by='title',
                search_res=['content', 'nonexistent_column'],
                cache_embeddings=False
            )
    
    def test_top_results_same_column(self):
        """
        Test top_results when search_by and search_res are the same.
        """
        results = self.search_same_column.top_results('machine learning', n_results=2)
        
        # Check that results is a numpy array
        self.assertIsInstance(results, np.ndarray)
        
        # Check that we get the expected number of results
        self.assertEqual(len(results), 2)
        
        # Check that the first result contains 'machine learning'
        self.assertTrue('machine learning' in results[0].lower())
    
    def test_top_results_string_column(self):
        """
        Test top_results when search_res is a single column.
        """
        results = list(self.search_string_column.top_results('machine learning', n_results=2))
        
        # Check that we get the expected number of results
        self.assertEqual(len(results), 2)
        
        # Check that each result is a tuple of (title, content)
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
    
    def test_top_results_list_columns(self):
        """
        Test top_results when search_res is a list of columns.
        """
        results = list(self.search_list_columns.top_results('machine learning', n_results=2))
        
        # Check that we get the expected number of results
        self.assertEqual(len(results), 2)
        
        # Check that each result is a tuple of (title, content, author)
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
    
    def test_empty_query(self):
        """
        Test that an empty query raises a ValueError.
        """
        with self.assertRaises(ValueError):
            self.search_same_column.top_results('', n_results=2)
        
        with self.assertRaises(ValueError):
            self.search_same_column.top_results('   ', n_results=2)


if __name__ == '__main__':
    unittest.main()
