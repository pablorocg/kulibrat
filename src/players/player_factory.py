import importlib
import logging
from typing import Dict, Type

from src.config.game_config import GameConfig
from src.core.player_color import PlayerColor
from src.players.human_player import HumanPlayer
from src.players.mcts_player import MCTSPlayer
from src.players.minimax_player import MinimaxPlayer
from src.players.player import Player
from src.players.random_player import RandomPlayer
from src.ui.game_interface import GameInterface


class PlayerFactory:
    """Centralized factory for creating players with extensible registration."""

    _logger = logging.getLogger(__name__)

    # Predefined AI strategies with simplified configuration
    _AI_STRATEGIES: Dict[str, Type[Player]] = {
        "minimax": MinimaxPlayer,
        "mcts": MCTSPlayer,
        "random": RandomPlayer,
    }

    _PLAYER_TYPES: Dict[str, Type[Player]] = {}

    @classmethod
    def register_player_type(cls, name: str, player_class: Type[Player]):
        """Register a new player type."""
        cls._logger.info(f"Registering player type: {name}")
        cls._PLAYER_TYPES[name.lower()] = player_class

    @classmethod
    def register_ai_strategy(cls, name: str, strategy_class: Type[Player]):
        """Register a new AI strategy."""
        cls._logger.info(f"Registering AI strategy: {name}")
        cls._AI_STRATEGIES[name.lower()] = strategy_class

    @classmethod
    def create_player(
        cls, player_type: str, color: PlayerColor, interface: GameInterface, **kwargs
    ) -> Player:
        """Create a player based on type and configuration."""
        player_type = player_type.lower()

        # Handle human players
        if player_type == "human":
            return HumanPlayer(color, interface)

        # Handle AI players (minimax, mcts, random)
        if player_type in cls._AI_STRATEGIES:
            config = GameConfig()
            strategy_config = config.get(f"strategies.{player_type}", {})
            strategy_class = cls._AI_STRATEGIES[player_type]

            # Extract init parameters for the strategy class (skip "self" and "color")
            init_params = strategy_class.__init__.__code__.co_varnames
            strategy_kwargs = {
                k: v for k, v in strategy_config.items() if k in init_params
            }

            # Special case for minimax to support more complex configuration
            if player_type == "minimax":
                strategy_kwargs.setdefault("max_depth", 4)
                strategy_kwargs.setdefault("use_alpha_beta", True)
                strategy_kwargs.setdefault("heuristic", "strategic")

            return strategy_class(color, **strategy_kwargs)

        # Check registered player types
        if player_type in cls._PLAYER_TYPES:
            player_class = cls._PLAYER_TYPES[player_type]
            return player_class(color, interface, **kwargs)

        # Dynamic import for unknown player types
        try:
            module_name, class_name = player_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
            player_class = getattr(module, class_name)
            return player_class(color, interface, **kwargs)
        except Exception as e:
            cls._logger.error(f"Failed to create player of type {player_type}: {e}")
            raise ValueError(f"Unknown player type: {player_type}")
