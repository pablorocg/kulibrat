"""
Human player implementation for the Kulibrat game.
"""

from typing import Optional

from src.core.game_state_cy import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.player import Player
from src.ui.game_interface import GameInterface


class HumanPlayer(Player):
    """Human player that gets moves via a game interface."""

    def __init__(
        self, color: PlayerColor, interface: GameInterface, name: str = None
    ):
        """
        Initialize a human player.

        Args:
            color: The player's color (BLACK or RED)
            interface: UI interface to get moves from the player
            name: Optional custom name for the player
        """
        super().__init__(color)
        self.interface = interface
        if name:
            self.name = name
        else:
            self.name = f"Human {color.name}"

    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Get move from human via the interface.

        Args:
            game_state: Current state of the game

        Returns:
            The selected move or None if no valid moves
        """
        valid_moves = game_state.get_valid_moves()

        if not valid_moves:
            return None

        return self.interface.get_human_move(game_state, self.color, valid_moves)
