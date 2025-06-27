"""
Search module for semantic search on tabular datasets.

This module provides the Search class which implements semantic search functionality
using sentence transformers. It converts text data to vector embeddings and computes
similarity between queries and data to find the most relevant matches.

The module supports:
- Searching by one column and returning results from the same or different columns
- Caching embeddings to disk for better performance
- Customizable models and search parameters
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Iterator, Any, Optional
from sentence_transformers import SentenceTransformer, util

from utils.logging import logger


class Search:
    """
    Semantic search engine for tabular data.
    
    This class provides functionality to search through tabular data using semantic similarity
    with transformer-based embeddings. It encodes text data into vector space and finds
    the most semantically similar entries to a given query.
    
    The search process works by:
    1. Converting the search column data to vector embeddings using a transformer model
    2. Converting the search query to a vector embedding
    3. Computing cosine similarity between the query and all data embeddings
    4. Returning the top N most similar results
    """
    
    def __init__(
        self, 
        database: pd.DataFrame, 
        search_by: str, 
        search_res: Union[str, List[str]], 
        res_is_by: bool = True,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = False,
        cache_dir: str = ".cache"
    ):
        """
        Initialize the search engine.
        
        Args:
            database: Pandas DataFrame containing the data to search
            search_by: Column name to search by (will be embedded)
            search_res: Column name(s) to return in results
            res_is_by: If True, search_res is the same as search_by
            model_name: Name of the sentence transformer model to use
            cache_embeddings: Whether to cache embeddings to disk
            cache_dir: Directory to store cached embeddings
            
        Raises:
            ValueError: If search_by column is not in the database
            ValueError: If any search_res column is not in the database
        """
        # Validate that the search_by column exists in the database
        if search_by not in database.columns:
            raise ValueError(f"Column '{search_by}' not found in database")
        
        # Validate the result columns based on their type
        if isinstance(search_res, str):
            # If search_res is a single column name (string)
            if search_res not in database.columns:
                raise ValueError(f"Column '{search_res}' not found in database")
        elif isinstance(search_res, list):
            # If search_res is a list of column names
            for col in search_res:
                if col not in database.columns:
                    raise ValueError(f"Column '{col}' not found in database")
        else:
            # If search_res is neither a string nor a list
            raise TypeError("search_res must be a string or list of strings")
            
        # Store configuration parameters
        self.model_name = model_name
        self.database = database 
        self.search_by = search_by
        
        # Extract the values from the search_by column for embedding
        self.by_arr = database[search_by].values  # 1-D array of values to search by
        
        # Store configuration for result format
        self.res_is_by = res_is_by  # Whether result column is same as search_by column
        self.search_res = search_res  # Column(s) to return in results
        self.res_is_str = isinstance(self.search_res, str)  # Whether search_res is a single column
        
        # Cache configuration
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir
        
        # Set up the result array based on configuration
        if res_is_by:
            # If searching and returning the same column, use the by_arr directly
            self.res_arr = self.by_arr
        else:
            # If returning different columns, extract those values
            self.res_arr = database[search_res].values

        # Load or compute embeddings for the search_by column
        self.by_embeddings = self._get_embeddings()
        
    def _get_embeddings(self) -> np.ndarray:
        """
        Get embeddings for the search_by column, either from cache or by computing them.
        
        This method attempts to load pre-computed embeddings from a cache file if caching
        is enabled. If cached embeddings aren't available, it loads the model and
        computes new embeddings.
        
        Returns:
            Numpy array of embeddings with shape (n_items, embedding_dim)
        """
        # Try to load from cache if caching is enabled
        if self.cache_embeddings:
            # Attempt to load previously cached embeddings
            embeddings = self._load_cached_embeddings()
            if embeddings is not None:
                logger.info(f"Loaded cached embeddings for '{self.search_by}'")
                return embeddings
        
        # If we couldn't load from cache, initialize the model
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Compute new embeddings for all items in the search_by column
        logger.info(f"Computing embeddings for {len(self.by_arr)} items")
        embeddings = self.model.encode(self.by_arr, show_progress_bar=True)
        
        # Cache the computed embeddings if caching is enabled
        if self.cache_embeddings:
            self._save_cached_embeddings(embeddings)
            
        return embeddings
    
    def _get_cache_path(self) -> str:
        """
        Get the path for the cached embeddings file.
        
        This method creates a unique filename for the cache file based on the model name
        and the data being embedded. This ensures that if either the model or the data
        changes, a new cache file will be created.
        
        Returns:
            Path string for the cache file
        """
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Create a unique cache filename based on the model and data hash
        # This ensures a new cache file is created if either changes
        # Create a unique cache filename based on the model and data hash
        # This ensures a new cache file is created if either changes
        cache_hash = hash((self.model_name, tuple(self.by_arr)))
        return os.path.join(self.cache_dir, f"embeddings_{abs(cache_hash)}.pkl")
    
    def _load_cached_embeddings(self) -> Optional[np.ndarray]:
        """
        Load embeddings from cache if available.
        
        This method attempts to load previously computed embeddings from a cache file.
        If the file doesn't exist or can't be loaded, it returns None.
        
        Returns:
            Numpy array of embeddings or None if cache doesn't exist or loading fails
        """
        # Get the path to the cache file
        cache_path = self._get_cache_path()
        
        # Check if the cache file exists
        if os.path.exists(cache_path):
            try:
                # Attempt to load the embeddings from the cache file
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                # Log a warning if loading fails
                logger.warning(f"Failed to load cached embeddings: {str(e)}")
                return None
        return None
    
    def _save_cached_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Save embeddings to cache.
        
        This method serializes and saves the computed embeddings to a cache file
        for faster loading in the future.
        
        Args:
            embeddings: Numpy array of embeddings to cache
        """
        # Get the path to the cache file
        cache_path = self._get_cache_path()
        
        try:
            # Attempt to save the embeddings to the cache file
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to cache: {cache_path}")
        except Exception as e:
            # Log a warning if saving fails
            logger.warning(f"Failed to save embeddings to cache: {str(e)}")

    def top_results(self, query: str, n_results: int = 5) -> Union[np.ndarray, Iterator[Tuple]]:
        """
        Get the top N results for a query.
        
        This method performs the core search functionality by:
        1. Encoding the query text into a vector embedding
        2. Computing similarity between the query and all data embeddings
        3. Finding the indices of the top N most similar items
        4. Returning the corresponding data items
        
        Args:
            query: Search query string to find similar items for
            n_results: Number of top results to return
            
        Returns:
            If res_is_by is True, returns an array of top results.
            Otherwise, returns an iterator of tuples containing (by_value, res_value1, res_value2, ...)
            
        Raises:
            ValueError: If query is empty
        """
        # Validate that the query is not empty
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Load model if not already loaded (in case we loaded from cache)
        if not hasattr(self, 'model'):
            self.model = SentenceTransformer(self.model_name)
            
        # Encode the query string into a vector embedding
        logger.debug(f"Encoding query: '{query}'")
        query_embedding = self.model.encode(query) 

        # Calculate cosine similarities between query embedding and all data embeddings
        # This gives a measure of semantic similarity between the query and each data item
        similarities = util.cos_sim(query_embedding, self.by_embeddings)[0].numpy()

        # Get indices of the top N most similar items
        # np.argsort returns indices that would sort the array
        # [::-1] reverses to get descending order (highest similarity first)
        # [:n_results] slices to get only the top N indices
        top_idx = np.argsort(similarities)[::-1][:n_results]
        
        # Prepare the results based on the configuration
        if self.res_is_str:
            # If result column is a string (single column), directly index into result array
            top_res = self.res_arr[top_idx]
        else:
            # If result columns are a list, get results for each column
            # The .T (transpose) converts from row-oriented to column-oriented
            top_res = [arr[top_idx] for arr in self.res_arr.T]
                
        if self.res_is_by:
            # If search_by == search_res, return top result array directly
            return top_res
        else:
            # If different columns, need to combine search_by column with result columns
            # First, reshape the by_arr to be 2D for concatenation
            by_arr_2d = np.expand_dims(self.by_arr[top_idx], axis=0)
            
            # Concatenate search_by values with result values
            # This creates a 2D array where:
            # - First row contains the search_by values
            # - Remaining rows contain the result values
            arrays = np.concatenate((by_arr_2d, top_res), axis=0)

            # Transpose and convert to tuples for each result
            # Each tuple will be (by_value, res_value1, res_value2, ...)
            return zip(*arrays)


# Test code that runs when this module is executed directly
if __name__ == '__main__':
    # Import pandas here for the test case
    import pandas as pd
    
    print('Test block executed')
    
    # Load the test dataset
    df = pd.read_csv('../SethGodin.csv').dropna()

    # Set up test parameters
    by_column = 'title'
    res_column = ['url', 'publication-date']  # Testing list of columns

    # Determine if result is the same as search column
    if by_column == res_column:
        by_equal_res = True
    else:
        by_equal_res = False

    # Create a search instance
    title_search = Search(df, search_by=by_column, search_res=res_column, res_is_by=by_equal_res)

    # Test a search and print the results
    print(list(title_search.top_results('Work is hard')))

    if by_column == res_column:
        by_equal_res = True
    else:
        by_equal_res = False

    title_search = Search(df, search_by=by_column, search_res=res_column, res_is_by=by_equal_res)

    print(list(title_search.top_results('Work is hard')))