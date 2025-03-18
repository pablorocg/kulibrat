"""
Abstract base class for AI strategies in Kulibrat.
"""

from abc import ABC, abstractmethod
from typing import Optional


from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor


class AIStrategy(ABC):
    """Abstract base class for AI strategies."""
    
    @abstractmethod
    def select_move(self, game_state: GameState, player_color: PlayerColor) -> Optional[Move]:
        """
        Select the best move according to this strategy.
        
        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move
            
        Returns:
            The selected move or None if no valid moves
        """
        pass