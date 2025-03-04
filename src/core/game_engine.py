"""
Debugged version of the game engine that fixes the player turn issues.
"""

import time
from typing import List, Optional
import logging

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.player import Player
from src.ui.console_interface import GameStatistics
from src.ui.game_interface import GameInterface
from src.core.move_type import MoveType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameEngine:
    """Fixed game engine that ensures proper player turns and tracks game statistics."""

    def __init__(
        self,
        black_player: Player,
        red_player: Player,
        interface: GameInterface,
        target_score: int = 5,
        ai_delay: float = 0.5,
    ):
        """Initialize the game engine."""
        self.state = GameState(target_score)
        self.players = {PlayerColor.BLACK: black_player, PlayerColor.RED: red_player}
        self.interface = interface
        self.moves_history: List[Move] = []
        self.ai_delay = ai_delay

        self._current_player_color = PlayerColor.BLACK

        # Provide player references to the interface if it supports it
        if hasattr(interface, "set_players"):
            interface.set_players(self.players)

    def start_game(self) -> Optional[PlayerColor]:
        """Start and play the game until completion."""

        logger.info("Starting a new game")

        # Setup players
        for player in self.players.values():
            player.setup(self.state)

        logger.info(f"Initial state: {self.state}")

        # Set initial player
        self._current_player_color = PlayerColor.BLACK
        self.state.current_player = self._current_player_color

        # Display initial state
        self.interface.display_state(self.state)

        # Main game loop
        while not self.state.is_game_over():
            logger.info(f"Starting turn for player {self._current_player_color}")

            self.state.current_player = self._current_player_color

            # Play turn for the current player
            self.play_one_turn()

            # Log state after turn   # board, pieces_off_board, scores, current_player, target_score
            logger.info(
                f"""
                State after {self._current_player_color}'s turn:
                Board: {self.state.board.flatten()}
                Pieces off board: {self.state.pieces_off_board}
                Scores: {self.state.scores}
                Current player: {self.state.current_player}
                Target score: {self.state.target_score}
                """
            )

            # Switch to the other player after the turn
            self._current_player_color = self._current_player_color.opposite()

            # Force the GameState to use our player tracking
            self.state.current_player = self._current_player_color

            # Check if game is over after the turn
            if self.state.is_game_over():
                break

        # Show winner
        winner = self.state.get_winner()
        self.interface.show_winner(winner, self.state)

        # Notify players of game over
        for player in self.players.values():
            player.game_over(self.state)

        logger.info(f"Game over. Winner: {winner}")

        return winner

    def play_one_turn(self) -> bool:
        """Play a single turn for the current player."""
        # CRITICAL: Use our tracked player, not the one in GameState
        current_player = self.players[self._current_player_color]

        # Force GameState to use our tracked player
        self.state.current_player = self._current_player_color

        # Start turn timer if interface supports statistics
        if hasattr(self.interface, "statistics"):
            self.interface.statistics.start_turn_timer(self._current_player_color)

        # Get valid moves for current player
        valid_moves = self.state.get_valid_moves()

        if not valid_moves:
            self.interface.show_message(
                f"{current_player.name} has no valid moves. Skipping turn."
            )

            # End turn timer if interface supports statistics
            if hasattr(self.interface, "statistics"):
                self.interface.statistics.end_turn_timer(self._current_player_color)
                self.interface.statistics.record_turn()

            return False

        # Get move from current player
        move = current_player.get_move(self.state)

        if move:
            # Apply the move - but DON'T rely on the state to track the player after
            success = self.handle_move(move)

            # If it's an AI player, add a delay for better visualization
            if success and "AI" in current_player.name:
                time.sleep(self.ai_delay)

            # End turn timer if interface supports statistics
            if hasattr(self.interface, "statistics"):
                self.interface.statistics.end_turn_timer(self._current_player_color)
                self.interface.statistics.record_turn()

            return success
        else:
            # End turn timer if interface supports statistics
            if hasattr(self.interface, "statistics"):
                self.interface.statistics.end_turn_timer(self._current_player_color)

            self.interface.show_message(f"{current_player.name} couldn't make a move.")
            return False

    def handle_move(self, move: Move) -> bool:
        """Handle a move from a player."""
        # Apply move to the game state
        success = self.state.apply_move(move)

        if success:
            # Add to history
            self.moves_history.append(move)

            # Notify both players about the move
            for player in self.players.values():
                player.notify_move(move, self.state)

            # CRITICAL: Force GameState to use our tracked player
            self.state.current_player = self._current_player_color

            # Display new state
            self.interface.display_state(self.state)

            # Optional: Record move statistics if interface supports it
            if hasattr(self.interface, "statistics"):
                # Record the move for the current player
                self.interface.statistics.record_move(self._current_player_color, move)

                # Check if this is an attack move
                if move.move_type == MoveType.ATTACK:
                    self.interface.statistics.record_capture(self._current_player_color)

            return True
        else:
            self.interface.show_message("Invalid move!")
            return False

    def reset_game(self, target_score: Optional[int] = None) -> None:
        """Reset the game to the initial state."""
        if target_score is not None:
            self.state = GameState(target_score)
        else:
            self.state = GameState(self.state.target_score)

        self.moves_history = []
        self._current_player_color = PlayerColor.BLACK
        self.state.current_player = self._current_player_color

        # Reset statistics if possible
        if hasattr(self.interface, "statistics"):
            self.interface.statistics = GameStatistics()

        # Setup players for new game
        for player in self.players.values():
            player.setup(self.state)
