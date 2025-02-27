"""
Abstract base class for all player types in Kulibrat.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor


class Player(ABC):
    """Abstract base class for all player types (human and AI)."""

    def __init__(self, color: PlayerColor):
        """
        Initialize a player.

        Args:
            color: The player's color (BLACK or RED)
        """
        self.color = color
        self.name = f"{color.name} Player"

    @abstractmethod
    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Get the next move from this player.

        Args:
            game_state: Current state of the game

        Returns:
            A valid move or None if no move is possible
        """
        pass

    def notify_move(self, move: Move, resulting_state: GameState) -> None:
        """
        Notify player about a move that was made (by self or opponent).

        Args:
            move: The move that was made
            resulting_state: The state after the move was applied
        """
        # Default implementation does nothing, but subclasses can override
        pass

    def setup(self, game_state: GameState) -> None:
        """
        Called when a new game starts.

        Args:
            game_state: The initial game state
        """
        # Default implementation does nothing
        pass

    def game_over(self, game_state: GameState) -> None:
        """
        Called when the game ends.

        Args:
            game_state: The final game state
        """
        # Default implementation does nothing
        pass
