"""
Random move selection strategy for Kulibrat AI.
"""

import random
from typing import Optional

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.player import Player

class RandomPlayer(Player):
    """Simple strategy that selects a random valid move."""
    def __init__(self, color: PlayerColor, name: str = None):
        """
        Initialize a random player.
        Args:
            color: The player's color (BLACK or RED)
            name: Optional custom name for the player
        """
        super().__init__(color, name or "Random Player")
    
    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select a random valid move.
        Args:
            game_state: Current state of the game
        Returns:
            A randomly selected valid move or None if no valid moves
        """
        valid_moves = game_state.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None