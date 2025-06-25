"""
Search module for semantic search on tabular datasets.
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
    with transformer-based embeddings.
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
        # Validate inputs
        if search_by not in database.columns:
            raise ValueError(f"Column '{search_by}' not found in database")
        
        if isinstance(search_res, str):
            if search_res not in database.columns:
                raise ValueError(f"Column '{search_res}' not found in database")
        elif isinstance(search_res, list):
            for col in search_res:
                if col not in database.columns:
                    raise ValueError(f"Column '{col}' not found in database")
        else:
            raise TypeError("search_res must be a string or list of strings")
            
        # Initialize instance variables
        self.model_name = model_name
        self.database = database 
        self.search_by = search_by
        self.by_arr = database[search_by].values
        self.res_is_by = res_is_by
        self.search_res = search_res 
        self.res_is_str = isinstance(self.search_res, str)
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir
        
        # Set up result array based on configuration
        if res_is_by:
            # If true, search returns data from by column
            self.res_arr = self.by_arr
        else:           
            # If false, search returns data from selected columns but still based by column embeddings
            self.res_arr = database[search_res].values

        # Load or create embeddings
        self.by_embeddings = self._get_embeddings()
        
    def _get_embeddings(self) -> np.ndarray:
        """
        Get embeddings for the search_by column, either from cache or by computing them.
        
        Returns:
            Numpy array of embeddings
        """
        # Try to load from cache if enabled
        if self.cache_embeddings:
            embeddings = self._load_cached_embeddings()
            if embeddings is not None:
                logger.info(f"Loaded cached embeddings for '{self.search_by}'")
                return embeddings
        
        # Initialize the model
        logger.info(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Compute embeddings
        logger.info(f"Computing embeddings for {len(self.by_arr)} items")
        embeddings = self.model.encode(self.by_arr, show_progress_bar=True)
        
        # Cache embeddings if enabled
        if self.cache_embeddings:
            self._save_cached_embeddings(embeddings)
            
        return embeddings
    
    def _get_cache_path(self) -> str:
        """
        Get the path for the cached embeddings file.
        
        Returns:
            Path string for the cache file
        """
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Create a unique cache filename based on the data and model
        cache_hash = hash((self.model_name, tuple(self.by_arr)))
        return os.path.join(self.cache_dir, f"embeddings_{abs(cache_hash)}.pkl")
    
    def _load_cached_embeddings(self) -> Optional[np.ndarray]:
        """
        Load embeddings from cache if available.
        
        Returns:
            Numpy array of embeddings or None if cache doesn't exist
        """
        cache_path = self._get_cache_path()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {str(e)}")
                return None
        return None
    
    def _save_cached_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Save embeddings to cache.
        
        Args:
            embeddings: Numpy array of embeddings to cache
        """
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save embeddings to cache: {str(e)}")

    def top_results(self, query: str, n_results: int = 5) -> Union[np.ndarray, Iterator[Tuple]]:
        """
        Get the top N results for a query.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            
        Returns:
            If res_is_by is True, returns an array of top results.
            Otherwise, returns an iterator of tuples containing (by_value, res_value1, res_value2, ...)
            
        Raises:
            ValueError: If query is empty
        """
        # Validate input
        if not query.strip():
            raise ValueError("Query cannot be empty")
            
        # Load model if not already loaded (in case we loaded from cache)
        if not hasattr(self, 'model'):
            self.model = SentenceTransformer(self.model_name)
            
        # Calculate the query vector
        logger.debug(f"Encoding query: '{query}'")
        query_embedding = self.model.encode(query) 

        # Calculate similarities
        similarities = util.cos_sim(query_embedding, self.by_embeddings)[0].numpy()

        # Get indices of top results
        top_idx = np.argsort(similarities)[::-1][:n_results]
        
        # Return results in the appropriate format
        if self.res_is_str:
            # Return top results that correspond with these indices
            top_res = self.res_arr[top_idx]
        else:
            # Return a list of arrays yielding top results based on search_by
            top_res = [arr[top_idx] for arr in self.res_arr.T]
                
        if self.res_is_by:
            # If search_by == search_res, return top result only
            return top_res
        else:
            # Ensure both arrays are 2D for concatenation
            by_arr_2d = np.expand_dims(self.by_arr[top_idx], axis=0)
            arrays = np.concatenate((by_arr_2d, top_res), axis=0)

            return zip(*arrays)  # Unpack arrays into tuples for each result


if __name__ == '__main__':
    import pandas as pd
    
    print('Test block executed')
    df = pd.read_csv('../SethGodin.csv').dropna()

    by_column = 'title'

    # To test: Can res_column take list of strings?
    res_column = ['url', 'publication-date']

    if by_column == res_column:
        by_equal_res = True
    else:
        by_equal_res = False

    title_search = Search(df, search_by=by_column, search_res=res_column, res_is_by=by_equal_res)

    print(list(title_search.top_results('Work is hard')))