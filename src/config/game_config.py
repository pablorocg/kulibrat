"""
Centralized configuration management for the Kulibrat game.
Supports multiple configuration sources with priority.
"""


import os
from typing import Any, Dict, Optional

import yaml


class GameConfig:
    """
    Centralized configuration management for the Kulibrat game.
    Supports multiple configuration sources with priority.
    """

    # Singleton instance
    _instance = None


    def __new__(cls):
        """
        Implement singleton pattern for configuration.

        Returns:
            Singleton instance of GameConfig
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._config = cls._load_configuration()
        return cls._instance

    @classmethod
    def _load_configuration(cls) -> Dict[str, Any]:
        """
        Load game configuration from multiple potential sources.

        Returns:
            Dictionary of configuration settings
        """
        # Potential configuration file paths
        config_paths = [
            # Local project config
            os.path.join(os.path.dirname(__file__), "kulibrat_config.yaml"),
            # Project root config
            os.path.join(os.path.dirname(__file__), "..", "..", "kulibrat_config.yaml"),
            # User's home directory config
            os.path.expanduser("~/.kulibrat/config.yaml"),
            # System-wide config
            "/etc/kulibrat/config.yaml",
        ]

        # Get screen width and height from environment variables
        env_screen_width = os.getenv("KULIBRAT_SCREEN_WIDTH")
        env_screen_height = os.getenv("KULIBRAT_SCREEN_HEIGHT")
        screen_width = (int(env_screen_width) * 3) // 4 if env_screen_width else 1024
        screen_height = (int(env_screen_height) * 3) // 4 if env_screen_height else 768

        # Default configuration
        config = {
            "game": {
                "board_rows": 4,
                "board_cols": 3,
                "initial_pieces": 4,
                "target_score": 5,
                "ai_delay": 1.0,
            },
            "strategies": {
                "default_ai": "minimax",
                "minimax": {"max_depth": 6, "use_alpha_beta": True},
                "mcts": {
                    "simulation_time": 2.0,
                    "max_iterations": 15000,
                    "exploration_weight": 1.41,
                    "num_threads": 4,
                },
                "random": {},
            },
            "ui": {
                "interface": "console",
                "screen_width": screen_width,
                "screen_height": screen_height,
            },
        }

        # Try to load configuration from files
        for path in config_paths:
            try:
                with open(path, "r") as config_file:
                    file_config = yaml.safe_load(config_file)

                    # Deep merge configurations
                    cls._deep_merge(config, file_config)

                    break
            except FileNotFoundError:
                continue
            except Exception as e:
                cls._logger.warning(f"Error loading config from {path}: {e}")

        return config

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], update: Dict[str, Any]):
        """
        Recursively merge two dictionaries.

        Args:
            base: Base configuration dictionary
            update: Dictionary with updates to apply
        """
        for key, value in update.items():
            if isinstance(value, dict):
                # Recursively merge nested dictionaries
                base[key] = cls._deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a configuration value by dot-separated key.

        Args:
            key: Dot-separated configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            # Split the key and traverse the config dictionary
            keys = key.split(".")
            value = self._config
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

        Raises:
            KeyError if key not found
        """
        return self.get(key)
