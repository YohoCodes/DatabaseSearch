"""
Configuration utilities for the SearchEngine application.
"""
import os
import yaml
from typing import Dict, List, Any, Optional, Union


class Config:
    """
    Configuration manager for the SearchEngine application.
    Handles loading and accessing configuration from YAML file.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = config_path
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
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {str(e)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: Dot-separated path to the configuration value
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config_data
        
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
        """
        dataset_config = self.get(f"datasets.{dataset_name.lower()}")
        if not dataset_config:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration")
        return dataset_config
        
    def get_model_name(self) -> str:
        """
        Get the configured model name.
        
        Returns:
            Model name string
        """
        return self.get("model.name", "all-MiniLM-L6-v2")
    
    def get_cache_settings(self) -> Dict[str, Any]:
        """
        Get model caching settings.
        
        Returns:
            Dict with cache_embeddings (bool) and cache_dir (str)
        """
        return {
            "cache_embeddings": self.get("model.cache_embeddings", False),
            "cache_dir": self.get("model.cache_dir", ".cache")
        }
