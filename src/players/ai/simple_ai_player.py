"""
Simple AI player implementation that uses a strategy to select moves.
"""

from typing import Optional

# Import from other modules
from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_player import AIPlayer
from src.players.ai.ai_strategy import AIStrategy
from src.players.ai.random_strategy import RandomStrategy


class SimpleAIPlayer(AIPlayer):
    """AI player that uses a strategy to select moves."""
    
    def __init__(self, color: PlayerColor, strategy: AIStrategy = None, name: str = None):
        """
        Initialize a simple AI player.
        
        Args:
            color: The player's color (BLACK or RED)
            strategy: Strategy to use for selecting moves (defaults to RandomStrategy)
            name: Optional custom name for the player
        """
        # Default to RandomStrategy if none provided
        if strategy is None:
            strategy = RandomStrategy()
            
        super().__init__(color, strategy, name)
    
    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Get move from the AI strategy.
        
        Args:
            game_state: Current state of the game
            
        Returns:
            The selected move or None if no valid moves
        """
        if game_state.current_player != self.color:
            return None
        
        # Let the strategy select the move
        return self.strategy.select_move(game_state, self.color)