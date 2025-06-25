"""
Main module for the SearchEngine application.

This module provides a Streamlit web interface for the semantic search engine.
"""
import os
import sys
import pandas as pd
import streamlit as st

# Import local modules
from operations import Search
from utils import Config, logger

# Constants
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_RESULTS = 5


def load_configuration(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
    """
    Load the application configuration.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object
        
    Raises:
        FileNotFoundError: If the configuration file is not found
    """
    try:
        return Config(config_path)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame
        
    Raises:
        FileNotFoundError: If the file is not found
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        logger.error(f"Error loading dataset: {str(e)}")
        sys.exit(1)


def create_search_engine(config: Config, dataset_name: str) -> Search:
    """
    Create a Search engine instance based on configuration.
    
    Args:
        config: Configuration object
        dataset_name: Name of the dataset to use
        
    Returns:
        Search engine instance
        
    Raises:
        KeyError: If the dataset is not found in configuration
        ValueError: If the by_column or res_column is not valid
    """
    try:
        # Get dataset configuration
        dataset_config = config.get_dataset_config(dataset_name)
        
        # Load dataset
        data = load_dataset(dataset_config["file"])
        
        # Get search configuration
        by_column = config.get("search_engine.by_column", dataset_config["columns"][0])
        res_columns = dataset_config.get("default_result_columns", [by_column])
        
        # Check if result is same as by_column
        by_equal_res = by_column == res_columns
        
        # Create and return search engine
        return Search(
            database=data,
            search_by=by_column,
            search_res=res_columns,
            res_is_by=by_equal_res,
            model_name=config.get_model_name(),
            cache_embeddings=config.get("model.cache_embeddings", False),
            cache_dir=config.get("model.cache_dir", ".cache")
        )
    except Exception as e:
        st.error(f"Error creating search engine: {str(e)}")
        logger.error(f"Error creating search engine: {str(e)}")
        sys.exit(1)


def render_search_ui(engine: Search, dataset_name: str, config: Config) -> None:
    """
    Render the Streamlit UI for the search engine.
    
    Args:
        engine: Search engine instance
        dataset_name: Name of the dataset being searched
        config: Configuration object
    """
    # Set page title
    st.title(f"{dataset_name} Semantic Search")
    
    # Add a description
    st.markdown("""
    This application uses semantic search to find the most relevant entries
    in the dataset based on your query. Enter a natural language query below
    to get started.
    """)
    
    # Create search input
    query = st.text_input("Enter your search query:", "")
    
    # Set number of results
    num_results = st.sidebar.slider(
        "Number of results", 
        min_value=1, 
        max_value=20, 
        value=config.get("search_engine.num_results", DEFAULT_RESULTS)
    )
    
    # Show model information
    with st.sidebar.expander("Model Information"):
        st.write(f"Model: {config.get_model_name()}")
        st.write(f"Search by column: {engine.search_by}")
        if isinstance(engine.search_res, list):
            st.write(f"Result columns: {', '.join(engine.search_res)}")
        else:
            st.write(f"Result column: {engine.search_res}")
    
    # Process search if query is provided
    if query:
        try:
            with st.spinner('Searching...'):
                results = engine.top_results(query, n_results=num_results)
                
            # Display results
            if engine.res_is_by:
                for i, res in enumerate(results):
                    with st.container():
                        st.subheader(f"Result {i+1}")
                        st.write(res)
                        st.divider()
            else:
                for i, res in enumerate(results):
                    with st.container():
                        st.subheader(f"Result {i+1}")
                        
                        # Display the by_column value as the main result
                        st.markdown(f"**{engine.search_by}**")
                        st.write(res[0])
                        st.markdown("---")
                        
                        # Display other result columns
                        if engine.res_is_str:
                            st.markdown(f"**{engine.search_res}**")
                            st.write(res[1])
                        else:
                            for j, col in enumerate(engine.search_res):
                                st.markdown(f"**{col}**")
                                st.write(res[j+1])
                        
                        st.divider()
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")
            logger.error(f"Error performing search: {str(e)}")
    else:
        st.info('Please enter a query to search')


def main():
    """
    Main entry point for the application.
    """
    # Configure page
    st.set_page_config(
        page_title="Semantic Search Engine",
        page_icon="🔍",
        layout="wide"
    )
    
    # Load configuration
    config = load_configuration()
    
    # Select dataset
    dataset_options = list(config.get("datasets", {}).keys())
    if not dataset_options:
        st.error("No datasets configured. Please check your configuration file.")
        return
        
    dataset_name = config.get("search_engine.type", dataset_options[0])
    
    # Allow user to select a different dataset
    if len(dataset_options) > 1:
        dataset_name = st.sidebar.selectbox(
            "Select Dataset",
            dataset_options,
            index=dataset_options.index(dataset_name) if dataset_name in dataset_options else 0
        )
    
    # Create search engine
    with st.spinner('Initializing search engine...'):
        @st.cache_resource
        def get_cached_engine(name):
            """Cache the search engine for better performance"""
            logger.info(f"Creating search engine for dataset: {name}")
            return create_search_engine(config, name)
        
        engine = get_cached_engine(dataset_name)
    
    # Render UI
    render_search_ui(engine, dataset_name, config)


if __name__ == "__main__":
    main()