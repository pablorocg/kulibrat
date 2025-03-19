import importlib
import logging
from typing import Dict, Type, Optional, Any

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
    """Centralized factory for creating players with extensible registration."""
    
    _logger = logging.getLogger(__name__)
    
    # Predefined AI strategies with simplified configuration
    _AI_STRATEGIES: Dict[str, Type[AIStrategy]] = {
        'minimax': MinimaxStrategy,
        'mcts': MCTSStrategy,
        'random': RandomStrategy
    }
    
    _PLAYER_TYPES: Dict[str, Type[Player]] = {}
    
    @classmethod
    def register_player_type(cls, name: str, player_class: Type[Player]):
        """Register a new player type."""
        cls._logger.info(f"Registering player type: {name}")
        cls._PLAYER_TYPES[name] = player_class
    
    @classmethod
    def register_ai_strategy(cls, name: str, strategy_class: Type[AIStrategy]):
        """Register a new AI strategy."""
        cls._logger.info(f"Registering AI strategy: {name}")
        cls._AI_STRATEGIES[name] = strategy_class
    
    @classmethod
    def create_player(
        cls, 
        player_type: str, 
        color: PlayerColor, 
        interface: GameInterface, 
        **kwargs
    ) -> Player:
        """Create a player based on type and configuration."""
        player_type = player_type.lower()
        
        # Handle human players
        if player_type == 'human':
            return HumanPlayer(color, interface)
        
        # Handle AI players
        if player_type in cls._AI_STRATEGIES:
            config = GameConfig()
            strategy_config = config.get(f'strategies.{player_type}', {})
            
            strategy_class = cls._AI_STRATEGIES[player_type]
            
            # Simplified strategy configuration
            strategy_kwargs = {
                k: v for k, v in strategy_config.items() 
                if k in strategy_class.__init__.__code__.co_varnames
            }
            
            # Special case for minimax to support more complex configuration
            if player_type == 'minimax':
                strategy_kwargs.update({
                    'max_depth': strategy_config.get('max_depth', 4),
                    'use_alpha_beta': strategy_config.get('use_alpha_beta', True),
                    'heuristic': strategy_config.get('heuristic', 'strategic')
                })
            
            strategy = strategy_class(**strategy_kwargs)
            return SimpleAIPlayer(color, strategy)
        
        # Check registered player types
        if player_type in cls._PLAYER_TYPES:
            player_class = cls._PLAYER_TYPES[player_type]
            return player_class(color, interface, **kwargs)
        
        # Dynamic import for unknown player types
        try:
            module_name, class_name = player_type.rsplit('.', 1)
            module = importlib.import_module(module_name)
            player_class = getattr(module, class_name)
            return player_class(color, interface, **kwargs)
        except Exception as e:
            cls._logger.error(f"Failed to create player of type {player_type}: {e}")
            raise ValueError(f"Unknown player type: {player_type}")