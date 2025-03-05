"""
Random move selection strategy for Kulibrat AI.
"""

import random
from typing import Optional

# Import from other modules
from src.core.game_state_cy import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy


class RandomStrategy(AIStrategy):
    """Simple strategy that selects a random valid move."""

    def select_move(
        self, game_state: GameState, player_color: PlayerColor
    ) -> Optional[Move]:
        """
        Select a random valid move.

        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move

        Returns:
            A randomly selected valid move or None if no valid moves
        """
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)
