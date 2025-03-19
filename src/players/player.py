"""
Abstract base class for all player types in Kulibrat.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.core import GameState, Move, PlayerColor


class Player(ABC):
    """Abstract base class for all player types."""

    def __init__(self, color: PlayerColor, name: str = None):
        """
        Initialize a player.

        Args:
            color: The player's color (BLACK or RED)
            name: Optional custom name for the player
        """
        self.color = color
        self.name = name or f"{color.name} Player"

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




