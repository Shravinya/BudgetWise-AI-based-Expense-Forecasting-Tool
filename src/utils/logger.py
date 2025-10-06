"""
Logger utility for the Budget Tool project.
Uses Python's logging module with configurable levels and file output.
Integrates with config.yaml for settings.
"""

import logging
import os
from datetime import datetime
from typing import Optional
import yaml

# Default config if config.yaml not found
DEFAULT_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "artifacts/logs/budget_tool.log"  # Updated to save in artifacts/logs
    }
}

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    Falls back to default if file not found.
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('logging', DEFAULT_CONFIG['logging'])
    return DEFAULT_CONFIG['logging']

class BudgetToolLogger:
    """
    Custom logger class for the Budget Tool.
    Initializes logging with file and console handlers.
    """
    
    def __init__(self, name: str = "budget_tool", config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config['level'].upper()))
        
        # Create artifacts/logs directory if not exists
        log_dir = os.path.dirname(self.config['file'])
        os.makedirs(log_dir, exist_ok=True)
        
        # Formatter
        formatter = logging.Formatter(self.config['format'])
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.config['file'])
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        self.logger.error(message, exc_info=exc_info)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def critical(self, message: str):
        self.logger.critical(message)

# Global logger instance
logger = None

def get_logger(name: str = "budget_tool", config_path: Optional[str] = None) -> BudgetToolLogger:
    """
    Get a logger instance.
    """
    global logger
    if logger is None:
        logger = BudgetToolLogger(name, config_path)
    return logger

# Example usage:
# logger = get_logger()
# logger.info("Project started")
# logger.error("An error occurred", exc_info=True)