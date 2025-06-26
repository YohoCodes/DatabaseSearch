"""
Configuration utilities for the SearchEngine application.

This module provides functionality for loading and accessing configuration
from YAML files. It handles configuration validation, provides default values,
and offers helper methods for accessing specific configuration sections.
"""
import os
import yaml
from typing import Dict, List, Any, Optional, Union


class Config:
    """
    Configuration manager for the SearchEngine application.
    Handles loading and accessing configuration from YAML file.
    
    The configuration file is expected to have the following structure:
    
    ```yaml
    search_engine:
      type: "dataset_name"
      by_column: "column_name"
      num_results: 5
    
    datasets:
      dataset_name:
        file: "filename.csv"
        columns: [...]
        default_result_columns: [...]
    
    model:
      name: "model_name"
      cache_embeddings: true/false
      cache_dir: ".cache"
    ```
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Store the path to the configuration file
        self.config_path = config_path
        
        # Load the configuration data
        self.config_data = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict containing configuration data
        
        Raises:
            FileNotFoundError: If the configuration file is not found
            yaml.YAMLError: If the YAML file is malformed
        """
        try:
            # Check if the configuration file exists
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            # Open and parse the configuration file
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated path to the configuration value (e.g., "model.name")
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get("model.name", "default-model")
            "all-MiniLM-L6-v2"
        """
        # Split the key into parts (e.g., "model.name" -> ["model", "name"])
        keys = key.split('.')
        value = self.config_data
        
        # Navigate through the nested dictionaries
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (lowercase)
            
        Returns:
            Dict containing dataset configuration
            
        Raises:
            KeyError: If the dataset is not found in the configuration
            
        Example:
            >>> config.get_dataset_config("amazon")
            {"file": "Amazon.csv", "columns": [...], "default_result_columns": [...]}
        """
        # Get the dataset configuration section
        dataset_config = self.get(f"datasets.{dataset_name.lower()}")
        
        # Check if the dataset exists
        if not dataset_config:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration")
            
        return dataset_config
        
    def get_model_name(self) -> str:
        """
        Get the configured model name.
        
        Returns:
            Model name string
            
        Example:
            >>> config.get_model_name()
            "all-MiniLM-L6-v2"
        """
        # Return the model name with a default
        return self.get("model.name", "all-MiniLM-L6-v2")
    
    def get_cache_settings(self) -> Dict[str, Any]:
        """
        Get model caching settings.
        
        Returns:
            Dict with cache_embeddings (bool) and cache_dir (str)
            
        Example:
            >>> config.get_cache_settings()
            {"cache_embeddings": True, "cache_dir": ".cache"}
        """
        # Return a dictionary with cache settings
        return {
            "cache_embeddings": self.get("model.cache_embeddings", False),
            "cache_dir": self.get("model.cache_dir", ".cache")
        }
