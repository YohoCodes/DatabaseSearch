"""
Main module for the SearchEngine application.

This module provides a Streamlit web interface for the semantic search engine.
It handles loading configuration, datasets, and presenting the user interface
for searching and viewing results.
"""
import os
import sys
import pandas as pd
import streamlit as st

# Import local modules
from operations import Search  # Core search functionality
from utils import Config, logger  # Configuration and logging utilities

# Constants for application defaults
DEFAULT_CONFIG_PATH = "config.yaml"  # Default configuration file path
DEFAULT_RESULTS = 5  # Default number of search results to display


def load_configuration(config_path: str = DEFAULT_CONFIG_PATH) -> Config:
    """
    Load the application configuration.
    
    This function attempts to load the configuration from the specified YAML file.
    If loading fails, it displays an error message and exits the application.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object with loaded configuration
        
    Raises:
        FileNotFoundError: If the configuration file is not found
        yaml.YAMLError: If the YAML file is malformed
    """
    try:
        # Attempt to load the configuration
        return Config(config_path)
    except Exception as e:
        # Display and log any errors, then exit
        st.error(f"Error loading configuration: {str(e)}")
        logger.error(f"Error loading configuration: {str(e)}")
        sys.exit(1)


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    
    This function attempts to load a dataset from the specified CSV file.
    If loading fails, it displays an error message and exits the application.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame containing the dataset
        
    Raises:
        FileNotFoundError: If the file is not found
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    try:
        # Check if the file exists
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        # Load the CSV file into a pandas DataFrame
        return pd.read_csv(file_path)
    except Exception as e:
        # Display and log any errors, then exit
        st.error(f"Error loading dataset: {str(e)}")
        logger.error(f"Error loading dataset: {str(e)}")
        sys.exit(1)


def create_search_engine(config: Config, dataset_name: str) -> Search:
    """
    Create a Search engine instance based on configuration.
    
    This function creates a Search engine instance configured for the specified dataset.
    It loads the dataset, extracts configuration parameters, and initializes the search engine.
    
    Args:
        config: Configuration object containing settings
        dataset_name: Name of the dataset to use (must be defined in config)
        
    Returns:
        Search engine instance ready for searching
        
    Raises:
        KeyError: If the dataset is not found in configuration
        ValueError: If the by_column or res_column is not valid
        FileNotFoundError: If the dataset file is not found
    """
    try:
        # Get dataset-specific configuration from the config file
        dataset_config = config.get_dataset_config(dataset_name)
        
        # Load the dataset from its CSV file
        data = load_dataset(dataset_config["file"])
        
        # Get search configuration parameters, with defaults if not specified
        by_column = config.get("search_engine.by_column", dataset_config["columns"][0])
        res_columns = dataset_config.get("default_result_columns", [by_column])
        
        # Check if result column is same as by_column for the Search configuration
        by_equal_res = len(res_columns) == 1 and res_columns[0] == by_column
        
        # Create and return the Search engine instance
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
        # Display and log any errors, then exit
        st.error(f"Error creating search engine: {str(e)}")
        logger.error(f"Error creating search engine: {str(e)}")
        sys.exit(1)


def render_search_ui(engine: Search, dataset_name: str, config: Config) -> None:
    """
    Render the Streamlit UI for the search engine.
    
    This function builds the user interface for the search application.
    It creates input controls, displays search results, and provides
    information about the current configuration.
    
    Args:
        engine: Search engine instance to use for searches
        dataset_name: Name of the dataset being searched (for display)
        config: Configuration object for accessing settings
    """
    # Set page title with the dataset name
    st.title(f"{dataset_name} Semantic Search")
    
    # Add an informative description of the application
    st.markdown("""
    This application uses semantic search to find the most relevant entries
    in the dataset based on your query. Enter a natural language query below
    to get started.
    """)
    
    # Create a search input field for the user's query
    query = st.text_input("Enter your search query:", "")
    
    # Create a slider in the sidebar to control number of results
    num_results = st.sidebar.slider(
        "Number of results", 
        min_value=1,  # Minimum allowed value
        max_value=20,  # Maximum allowed value
        value=config.get("search_engine.num_results", DEFAULT_RESULTS)  # Default value from config
    )
    
    # Show model information in a collapsible section
    with st.sidebar.expander("Model Information"):
        # Display the model name being used
        st.write(f"Model: {config.get_model_name()}")
        
        # Display the column being searched
        st.write(f"Search by column: {engine.search_by}")
        
        # Display the result columns with appropriate formatting
        if isinstance(engine.search_res, list):
            st.write(f"Result columns: {', '.join(engine.search_res)}")
        else:
            st.write(f"Result column: {engine.search_res}")
    
    # Process search if a query is provided
    if query:
        try:
            # Show a spinner while searching
            with st.spinner('Searching...'):
                # Perform the search with the specified number of results
                results = engine.top_results(query, n_results=num_results)
                
            # Display results based on the configuration
            if engine.res_is_by:
                # If search and result columns are the same
                for i, res in enumerate(results):
                    # Create a container for each result
                    with st.container():
                        st.subheader(f"Result {i+1}")  # Result number
                        st.write(res)  # Result content
                        st.divider()  # Divider between results
            else:
                # If returning multiple columns
                for i, res in enumerate(results):
                    # Create a container for each result
                    with st.container():
                        st.subheader(f"Result {i+1}")  # Result number
                        
                        # Display the search column value as the main result
                        st.markdown(f"**{engine.search_by}**")  # Column name
                        st.write(res[0])  # Value from search column
                        st.markdown("---")  # Separator
                        
                        # Display values from other result columns
                        if engine.res_is_str:
                            # If there's only one result column
                            st.markdown(f"**{engine.search_res}**")  # Column name
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