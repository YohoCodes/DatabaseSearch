"""
Core operations package for the SearchEngine application.

This package contains the main functionality for semantic search operations.
It provides the Search class which is responsible for creating embeddings
and finding semantically similar items in a dataset.

Classes:
    Search: The main search engine class that creates embeddings and performs
            semantic similarity searches.

Usage:
    from operations import Search
    
    search_engine = Search(
        database=my_dataframe,
        search_by='title_column',
        search_res=['content_column', 'author_column']
    )
    
    results = search_engine.top_results('my search query')
"""
# Import the Search class for direct access from operations module
from .search import Search

# Define public API
__all__ = ["Search"]
