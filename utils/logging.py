"""
Logging utilities for the SearchEngine application.

This module provides a configurable logging system for the application,
supporting both file and console logging with different formatting options.
It creates a default logger instance that can be imported and used throughout
the application.
"""
import os
import logging
from typing import Optional


def setup_logger(
    name: str = "searchengine",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    This function creates and configures a logger with the specified settings.
    It supports logging to both a file and the console with different formatting
    for each output destination.
    
    Args:
        name: Logger name used to identify the logger instance
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to log file (if None, file logging is disabled)
        console: Whether to log to console (stdout)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger(name="search_module", level=logging.DEBUG, 
        ...                       log_file="logs/search.log")
        >>> logger.info("Search initialized")
    """
    # Create or get the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    # Detailed formatter for file logging with timestamp, logger name, and level
    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Simple formatter for console output
    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")
    
    # Remove existing handlers to avoid duplicate logging
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler for persistent logging
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create and configure file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    # Console handler for immediate feedback
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    return logger


# Create default logger for application-wide use
# This can be imported and used directly in other modules
logger = setup_logger()
