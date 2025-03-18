# src/players/player_factory.py
from typing import Dict, Type, Optional, Any

import importlib
import logging

from src.core.player_color import PlayerColor
from src.players.player import Player
from src.players.human_player import HumanPlayer
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.ai.ai_strategy import AIStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.mcts_strategy import MCTSStrategy
from src.players.ai.random_strategy import RandomStrategy
from src.ui.game_interface import GameInterface
from src.config.game_config import GameConfig


class PlayerFactory:
    """
    Centralized factory for creating players with extensible registration.
    """
    
    # Logger for tracking player creation
    _logger = logging.getLogger(__name__)
    
    # Registry for additional player types
    _player_types: Dict[str, Type[Player]] = {}
    
    # Predefined AI strategies
    _ai_strategies: Dict[str, Type[AIStrategy]] = {
        'minimax': MinimaxStrategy,
        'mcts': MCTSStrategy,
        'random': RandomStrategy
    }
    
    @classmethod
    def register_player_type(cls, name: str, player_class: Type[Player]):
        """
        Register a new player type for dynamic creation.
        
        Args:
            name: Unique identifier for the player type
            player_class: Player class to register
        """
        cls._logger.info(f"Registering player type: {name}")
        cls._player_types[name] = player_class
    
    @classmethod
    def register_ai_strategy(cls, name: str, strategy_class: Type[AIStrategy]):
        """
        Register a new AI strategy.
        
        Args:
            name: Unique identifier for the strategy
            strategy_class: AI strategy class to register
        """
        cls._logger.info(f"Registering AI strategy: {name}")
        cls._ai_strategies[name] = strategy_class
    
    @classmethod
    def create_player(
        cls, 
        player_type: str, 
        color: PlayerColor, 
        interface: GameInterface, 
        **kwargs
    ) -> Player:
        """
        Create a player based on type and configuration.
        
        Args:
            player_type: Type of player to create
            color: Player's color
            interface: Game interface
            **kwargs: Additional configuration parameters
        
        Returns:
            Instantiated player
        
        Raises:
            ValueError: If player type is unknown
        """
        # Normalize player type to lowercase
        player_type = player_type.lower()
        
        # Handle human players
        if player_type == 'human':
            return HumanPlayer(color, interface)
        
        # Handle AI players
        if player_type in cls._ai_strategies:
            # Get configuration for the strategy
            config = GameConfig()
            strategy_config = config.get(f'strategies.{player_type}', {})
            
            # Create strategy with configuration
            strategy_class = cls._ai_strategies[player_type]
            
            # Special handling for minimax to support heuristic configuration
            if player_type == 'minimax':
                strategy_kwargs = {
                    'max_depth': strategy_config.get('max_depth', 4),
                    'use_alpha_beta': strategy_config.get('use_alpha_beta', True),
                    'heuristic': strategy_config.get('heuristic', 'strategic')
                }
            else:
                # For other strategies
                strategy_kwargs = {
                    k: v for k, v in strategy_config.items() 
                    if k in strategy_class.__init__.__code__.co_varnames
                }
                
            strategy = strategy_class(**strategy_kwargs)
            
            return SimpleAIPlayer(color, strategy)
        
        # Check registered player types
        if player_type in cls._player_types:
            player_class = cls._player_types[player_type]
            return player_class(color, interface, **kwargs)
        
        # Dynamic import for unknown player types
        try:
            # Attempt to dynamically import the player class
            module_name, class_name = player_type.rsplit('.', 1)
            module = importlib.import_module(module_name)
            player_class = getattr(module, class_name)
            return player_class(color, interface, **kwargs)
        except (ValueError, ImportError, AttributeError) as e:
            cls._logger.error(f"Failed to create player of type {player_type}: {e}")
            raise ValueError(f"Unknown player type: {player_type}")