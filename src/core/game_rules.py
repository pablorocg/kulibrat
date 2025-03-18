# src/core/game_rules.py
import logging
from typing import Optional

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.core.move_type import MoveType


class GameRules:
    """
    Validates moves and enforces game rules for Kulibrat.
    """

    def __init__(self):
        """
        Initialize game rules with logging.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def validate_move(self, game_state: GameState, move: Move) -> bool:
        """
        Validate a move according to Kulibrat game rules.

        Args:
            game_state: Current game state
            move: Proposed move

        Returns:
            Boolean indicating if the move is valid
        """
        # Validate current player
        if move.move_type != MoveType.INSERT and game_state.board[move.start_pos[0], move.start_pos[1]] != game_state.current_player.value:
            self.logger.warning(f"Move validation failed: Piece at start position does not belong to current player")
            return False

        # Move type specific validation
        try:
            if move.move_type == MoveType.INSERT:
                return self._validate_insert_move(game_state, move)
            elif move.move_type == MoveType.DIAGONAL:
                return self._validate_diagonal_move(game_state, move)
            elif move.move_type == MoveType.ATTACK:
                return self._validate_attack_move(game_state, move)
            elif move.move_type == MoveType.JUMP:
                return self._validate_jump_move(game_state, move)
            
            self.logger.warning(f"Unknown move type: {move.move_type}")
            return False

        except Exception as e:
            self.logger.error(f"Error validating move {move}: {e}")
            return False

    def _validate_insert_move(self, game_state: GameState, move: Move) -> bool:
        """
        Validate an insert move.

        Args:
            game_state: Current game state
            move: Proposed move

        Returns:
            Boolean indicating if the move is valid
        """
        player = game_state.current_player

        # Check if player has pieces available to insert
        if game_state.pieces_off_board[player] <= 0:
            self.logger.warning(f"{player} has no pieces left to insert")
            return False

        # Verify insert is on the player's start row
        row, col = move.end_pos
        if row != player.start_row:
            self.logger.warning(f"Insert move must be on {player}'s start row")
            return False

        # Check if the target square is empty
        if game_state.board[row, col] != 0:
            self.logger.warning("Target square is not empty")
            return False

        return True

    def _validate_diagonal_move(self, game_state: GameState, move: Move) -> bool:
        """
        Validate a diagonal move.

        Args:
            game_state: Current game state
            move: Proposed move

        Returns:
            Boolean indicating if the move is valid
        """
        player = game_state.current_player
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Check diagonal move direction
        direction = player.direction
        if end_row != start_row + direction:
            self.logger.warning("Invalid diagonal move direction")
            return False

        # Check column movement (must be one column left or right)
        if abs(end_col - start_col) != 1:
            self.logger.warning("Diagonal move must be to an adjacent column")
            return False

        # Check destination is empty or off the board
        if 0 <= end_row < game_state.BOARD_ROWS and 0 <= end_col < game_state.BOARD_COLS:
            if game_state.board[end_row, end_col] != 0:
                self.logger.warning("Destination square is not empty")
                return False

        return True

    def _validate_attack_move(self, game_state: GameState, move: Move) -> bool:
        """
        Validate an attack move.

        Args:
            game_state: Current game state
            move: Proposed move

        Returns:
            Boolean indicating if the move is valid
        """
        player = game_state.current_player
        opponent = player.opposite()
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Check move direction
        direction = player.direction
        if end_row != start_row + direction:
            self.logger.warning("Invalid attack move direction")
            return False

        # Check column remains the same
        if start_col != end_col:
            self.logger.warning("Attack move must be in the same column")
            return False

        # Check destination has an opponent's piece
        if game_state.board[end_row, end_col] != opponent.value:
            self.logger.warning("Attack move must target an opponent's piece")
            return False

        return True

    def _validate_jump_move(self, game_state: GameState, move: Move) -> bool:
        """
        Validate a jump move.

        Args:
            game_state: Current game state
            move: Proposed move

        Returns:
            Boolean indicating if the move is valid
        """
        player = game_state.current_player
        opponent = player.opposite()
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Check column remains the same
        if start_col != end_col:
            self.logger.warning("Jump move must be in the same column")
            return False

        # Calculate jump direction and line of pieces
        direction = player.direction
        line_length = 0
        current_row = start_row + direction

        # Count opponent pieces in the line
        while (0 <= current_row < game_state.BOARD_ROWS and 
               game_state.board[current_row, start_col] == opponent.value):
            line_length += 1
            current_row += direction

        # Check jump conditions
        if line_length == 0 or line_length > 3:
            self.logger.warning(f"Invalid jump line length: {line_length}")
            return False

        # Calculate landing position
        landing_row = start_row + direction * (line_length + 1)

        # Check landing position
        if 0 <= landing_row < game_state.BOARD_ROWS:
            # On-board landing must be to an empty square
            if game_state.board[landing_row, start_col] != 0:
                self.logger.warning("Jump landing square is not empty")
                return False

        return True