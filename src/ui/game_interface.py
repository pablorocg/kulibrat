"""
Abstract base class for game interfaces in Kulibrat.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

# Import from core module
from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor


class GameInterface(ABC):
    """Abstract base class for game interfaces (Console, GUI, etc.)."""
    
    @abstractmethod
    def display_state(self, game_state: GameState) -> None:
        """
        Display the current game state.
        
        Args:
            game_state: Current state of the game
        """
        pass
    
    @abstractmethod
    def get_human_move(self, game_state: GameState, player_color: PlayerColor, 
                       valid_moves: List[Move]) -> Move:
        """
        Get a move from a human player.
        
        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move
            valid_moves: List of valid moves
            
        Returns:
            A valid move selected by the human player
        """
        pass
    
    @abstractmethod
    def show_winner(self, winner: Optional[PlayerColor], game_state: GameState) -> None:
        """
        Display the winner of the game.
        
        Args:
            winner: The player who won, or None for a draw
            game_state: Final state of the game
        """
        pass
    
    @abstractmethod
    def show_message(self, message: str) -> None:
        """
        Display a message to the user.
        
        Args:
            message: The message to display
        """
        pass

    def set_players(self, players):
        """
        Set player references for the interface.
        
        Args:
            players: Dictionary mapping player colors to player objects
        """
        pass  # Default implementation does nothing