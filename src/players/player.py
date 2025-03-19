"""
Abstract base class for all player types in Kulibrat.
"""

from abc import ABC, abstractmethod
from typing import Optional

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor


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
    
    def setup(self, game_state: GameState) -> None:
        """
        Initialize the player with the game state.
        
        Args:
            game_state: Initial game state
        """
        # Default implementation does nothing
        pass
    
    def notify_move(self, move: Move, game_state: GameState) -> None:
        """
        Notify the player about a move that has been made.
        
        Args:
            move: The move that was made
            game_state: Updated game state after the move
        """
        # Default implementation does nothing
        pass
    
    def game_over(self, game_state: GameState) -> None:
        """
        Notify the player that the game is over.
        
        Args:
            game_state: Final game state
        """
        # Default implementation does nothing
        pass
    




