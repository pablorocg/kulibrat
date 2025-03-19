# src/core/turn_manager.py
import logging
from typing import Optional

from src.core.game_state import GameState
from src.core.move import Move
from src.core.game_rules import GameRules
from src.players.player import Player


class TurnManager:
    """
    Manages the progression of turns and move application in the game.
    """

    def __init__(self, rules_engine: Optional[GameRules] = None):
        """
        Initialize the turn manager.

        Args:
            rules_engine: Optional custom rules engine
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Use provided rules engine or default
        self.rules_engine = rules_engine or GameRules()

    def process_turn(self, game_state: GameState, player: Player, move: Optional[Move] = None) -> GameState:
        """
        Process a player's turn, including move validation and application.

        Args:
            game_state: Current game state
            player: Player taking the turn
            move: Move to be applied (optional)

        Returns:
            Updated game state after processing the turn

        Raises:
            ValueError: If the move is invalid
        """
        # Log turn start
        self.logger.info(f"Processing turn for {player.color}")
        
        # Create a copy of the game state to modify
        new_state = game_state.copy()
        
        # Ensure the current_player is set correctly
        if new_state.current_player != player.color:
            self.logger.warning(f"Current player mismatch. Expected {player.color}, got {new_state.current_player}. Correcting.")
            new_state.current_player = player.color

        # If no moves are available
        valid_moves = new_state.get_valid_moves()
        if not valid_moves:
            self.logger.info(f"No valid moves for {player.color}")
            return self._handle_no_moves(new_state)

        # If no move provided but moves are available
        if move is None:
            self.logger.warning(f"No move selected for {player.color}")
            return new_state

        # Validate the move
        if not self.rules_engine.validate_move(new_state, move):
            error_msg = f"Invalid move for {player.color}: {move}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Apply the move
        success = new_state.apply_move(move)
        if not success:
            error_msg = f"Failed to apply move: {move}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Explicitly switch to the next player after a successful move
        new_state.current_player = new_state.current_player.opposite()

        # Notify player about the move
        player.notify_move(move, new_state)

        # Log move details
        self.logger.info(f"Move applied: {move}")

        return new_state

    def _handle_no_moves(self, game_state: GameState) -> GameState:
        """
        Handle situation when a player has no valid moves.

        Args:
            game_state: Current game state

        Returns:
            Updated game state with player switched
        """
        # Create a copy of the game state
        new_state = game_state.copy()

        # Switch to the other player
        new_state.current_player = new_state.current_player.opposite()

        self.logger.info(f"No moves for {game_state.current_player}. Switching to {new_state.current_player}.")

        return new_state