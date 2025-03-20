import importlib
import logging
from typing import Dict, Type

from src.config.game_config import GameConfig
from src.core.player_color import PlayerColor
from src.players.human_player.human_player import HumanPlayer
from src.players.mcts_player.mcts_player import MCTSPlayer
from src.players.minimax_player.minimax_player import MinimaxPlayer
from src.players.player import Player
from src.players.random_player.random_player import RandomPlayer
from src.ui.game_interface import GameInterface





class PlayerFactory:
    """Centralized factory for creating players with extensible registration."""

    _logger = logging.getLogger(__name__)

    # Player type registry
    _PLAYER_TYPES: Dict[str, Type[Player]] = {
        "minimax": MinimaxPlayer,
        "mcts": MCTSPlayer,
        "random": RandomPlayer,
        "human": HumanPlayer
    }

    @classmethod
    def register_player_type(cls, name: str, player_class: Type[Player]):
        """Register a new player type."""
        cls._logger.info(f"Registering player type: {name}")
        cls._PLAYER_TYPES[name.lower()] = player_class

    @classmethod
    def create_player(
        cls, player_type: str, color: PlayerColor, interface: GameInterface, **kwargs
    ) -> Player:
        """Create a player based on type and configuration."""
        player_type = player_type.lower()

        # Quick check for human players
        if player_type == "human":
            return HumanPlayer(color, interface)

        # Check the player type registry
        if player_type in cls._PLAYER_TYPES:
            config = GameConfig()
            player_config = config.get(f"strategies.{player_type}", {})
            player_class = cls._PLAYER_TYPES[player_type]

            # Extract init parameters for the player class
            init_params = player_class.__init__.__code__.co_varnames
            player_kwargs = {
                k: v for k, v in player_config.items() if k in init_params
            }

            # Special case for minimax to support more complex configuration
            if player_type == "minimax":
                player_kwargs.setdefault("max_depth", 4)
                player_kwargs.setdefault("use_alpha_beta", True)
                player_kwargs.setdefault("heuristic", "strategic")

            return player_class(color, **player_kwargs)

        # Dynamic import for unknown player types
        try:
            module_name, class_name = player_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
            player_class = getattr(module, class_name)
            return player_class(color, interface, **kwargs)
        except Exception as e:
            cls._logger.error(f"Failed to create player of type {player_type}: {e}")
            raise ValueError(f"Unknown player type: {player_type}")