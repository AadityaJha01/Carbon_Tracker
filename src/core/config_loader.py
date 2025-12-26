"""
Configuration loading utilities
"""

import yaml
import os
import argparse
from typing import Dict, Optional


def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Dict:
    """
    Load configuration from YAML file and override with CLI args.
    
    Args:
        config_path: Path to YAML config file
        args: Command line arguments namespace
        
    Returns:
        Dictionary with configuration
    """
    config = {}
    
    # Load base config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Override with CLI args
    if args:
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    
    return config

