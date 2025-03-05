"""
Abstract base class for all AI players in Kulibrat.
"""

from abc import ABC, abstractmethod
from typing import Optional

# Import from other modules
from src.core.game_state_cy import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.player import Player
from src.players.ai.ai_strategy import AIStrategy


class AIPlayer(Player, ABC):
    """Base class for all AI players."""
    
    def __init__(self, color: PlayerColor, strategy: AIStrategy = None, name: str = None):
        """
        Initialize an AI player.
        
        Args:
            color: The player's color (BLACK or RED)
            strategy: Strategy to use for selecting moves
            name: Optional custom name for the player
        """
        super().__init__(color)
        self.strategy = strategy
        
        if name:
            self.name = name
        else:
            strategy_name = self.strategy.__class__.__name__ if self.strategy else "Unknown"
            self.name = f"AI {color.name} ({strategy_name})"
    
    @abstractmethod
    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Get move from AI strategy.
        
        Args:
            game_state: Current state of the game
            
        Returns:
            The selected move or None if no valid moves
        """
        pass