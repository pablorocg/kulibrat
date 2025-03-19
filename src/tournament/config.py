"""
Tournament Configuration Management.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional


class TournamentConfig:
    """
    Manages tournament configuration with robust loading and validation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize tournament configuration.
        
        Args:
            config_path: Path to the tournament configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self._default_config = {
            'tournament': {
                'matches_per_pairing': 3,
                'target_score': 5,
                'max_turns': 300
            },
            'players': [],
            'output': {
                'results_dir': 'tournament_results',
                'save_csv': True,
                'save_plots': True
            }
        }
        
        # Load configuration
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
        
        Returns:
            Loaded configuration dictionary
        """
        # Use default configuration if no path provided
        if not config_path:
            self.logger.warning("No config path provided. Using default configuration.")
            return self._default_config
        
        # Validate file exists
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found: {config_path}. Using default configuration.")
            return self._default_config
        
        try:
            # Load YAML configuration
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Deep merge with default configuration
            merged_config = self._deep_merge(self._default_config, file_config)
            
            # Validate configuration
            self._validate_config(merged_config)
            
            return merged_config
        
        except (yaml.YAMLError, IOError) as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        Recursively merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary with updates
        
        Returns:
            Merged dictionary
        """
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = self._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    def _validate_config(self, config: Dict[str, Any]):
        """
        Validate tournament configuration.
        
        Args:
            config: Configuration dictionary to validate
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate tournament settings
        tournament_settings = config.get('tournament', {})
        
        if not isinstance(tournament_settings.get('matches_per_pairing'), int) or \
           tournament_settings['matches_per_pairing'] < 1:
            self.logger.warning("Invalid matches_per_pairing. Using default.")
            tournament_settings['matches_per_pairing'] = 3
        
        if not isinstance(tournament_settings.get('target_score'), int) or \
           tournament_settings['target_score'] < 1:
            self.logger.warning("Invalid target_score. Using default.")
            tournament_settings['target_score'] = 5
        
        if not isinstance(tournament_settings.get('max_turns'), int) or \
           tournament_settings['max_turns'] < 10:
            self.logger.warning("Invalid max_turns. Using default.")
            tournament_settings['max_turns'] = 300
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Dot-separated configuration key
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        try:
            # Traverse nested dictionary
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to configuration.
        
        Args:
            key: Configuration key
        
        Returns:
            Configuration value
        """
        return self.get(key)