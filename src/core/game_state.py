"""
Game state representation for Kulibrat.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from src.core.player_color import PlayerColor
from src.core.move import Move
from src.core.move_type import MoveType


class GameState:
    """Represents the complete state of a Kulibrat game."""

    def __init__(self, target_score: int = 5):
        """
        Initialize a new game state.

        Args:
            target_score: Score needed to win the game
        """
        # Board is 3x4 (rows x columns)
        # 0 = empty, 1 = black piece, -1 = red piece
        self.board = np.zeros((4, 3), dtype=np.int8)

        # Number of pieces available to be inserted (not on board)
        self.pieces_off_board = {PlayerColor.BLACK: 4, PlayerColor.RED: 4}

        # Current scores
        self.scores = {PlayerColor.BLACK: 0, PlayerColor.RED: 0}

        # Who's turn is it
        self.current_player = PlayerColor.BLACK

        # How many points to win
        self.target_score = target_score

    def copy(self) -> "GameState":
        """Create a deep copy of the current game state."""
        new_state = GameState(self.target_score)
        new_state.board = self.board.copy()
        new_state.pieces_off_board = self.pieces_off_board.copy()
        new_state.scores = self.scores.copy()
        new_state.current_player = self.current_player
        return new_state

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is on the board."""
        row, col = pos
        return 0 <= row < 4 and 0 <= col < 3

    def is_empty(self, pos: Tuple[int, int]) -> bool:
        """Check if a board position is empty."""
        row, col = pos
        return self.board[row, col] == 0

    def get_piece_at(self, pos: Tuple[int, int]) -> int:
        """Get the piece at a specific position."""
        if not self.is_valid_position(pos):
            return 0
        row, col = pos
        return self.board[row, col]

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        # Check if any player reached the target score
        if any(score >= self.target_score for score in self.scores.values()):
            return True

        # Check if both players are locked (no valid moves)
        # Save the current player to restore it later
        original_player = self.current_player

        # Check if BLACK has valid moves
        self.current_player = PlayerColor.BLACK
        black_has_moves = len(self._get_valid_moves()) > 0

        # Check if RED has valid moves
        self.current_player = PlayerColor.RED
        red_has_moves = len(self._get_valid_moves()) > 0

        # Restore original player
        self.current_player = original_player

        # Game is over if neither player can move
        return not (black_has_moves or red_has_moves)

    def get_winner(self) -> Optional[PlayerColor]:
        """Get the winner if the game is over."""
        if not self.is_game_over():
            return None

        # If a player reached the target score
        for player, score in self.scores.items():
            if score >= self.target_score:
                return player

        # If both players are locked, the last player to move loses
        # Save the current player
        original_player = self.current_player

        # Check if RED has valid moves
        self.current_player = PlayerColor.RED
        red_has_moves = len(self._get_valid_moves()) > 0

        # Check if BLACK has valid moves
        self.current_player = PlayerColor.BLACK
        black_has_moves = len(self._get_valid_moves()) > 0

        # Restore the original player
        self.current_player = original_player

        # If neither player can move, the last player to move loses
        if not (red_has_moves or black_has_moves):
            return self.current_player.opposite()

        return None

    def apply_move(self, move: Move) -> bool:
        """
        Apply a move to the game state and return whether it was successful.

        Args:
            move: The move to apply

        Returns:
            True if the move was successful, False otherwise
        """
        # Make sure we're working with a copy to avoid modifying the original state
        # until we know the move is valid
        new_state = self.copy()

        # Dispatch to the appropriate method based on move type
        success = False

        if move.move_type == MoveType.INSERT:
            success = new_state._apply_insert(move)
        elif move.move_type == MoveType.DIAGONAL:
            success = new_state._apply_diagonal(move)
        elif move.move_type == MoveType.ATTACK:
            success = new_state._apply_attack(move)
        elif move.move_type == MoveType.JUMP:
            success = new_state._apply_jump(move)

        if success:
            # Update our state from the new state
            self.board = new_state.board
            self.pieces_off_board = new_state.pieces_off_board
            self.scores = new_state.scores
            return True

        return False

    def _apply_insert(self, move: Move) -> bool:
        """Apply an INSERT move."""
        player = self.current_player
        row, col = move.end_pos

        # Check if the move is valid
        if row != player.start_row:
            return False

        if not self.is_empty((row, col)):
            return False

        if self.pieces_off_board[player] <= 0:
            return False

        # Apply the move
        self.board[row, col] = player.value
        self.pieces_off_board[player] -= 1
        return True

    def _apply_diagonal(self, move: Move) -> bool:
        """Apply a DIAGONAL move."""
        player = self.current_player
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Check if the start position has the player's piece
        if self.board[start_row, start_col] != player.value:
            return False

        # Check if this is a valid diagonal move
        if player == PlayerColor.BLACK:
            if end_row != start_row + 1 or abs(end_col - start_col) != 1:
                return False
        else:  # RED
            if end_row != start_row - 1 or abs(end_col - start_col) != 1:
                return False

        # Check if the end position is on the board
        if self.is_valid_position(move.end_pos):
            # Check if the end position is empty
            if not self.is_empty(move.end_pos):
                return False

            # Apply the move on the board
            self.board[start_row, start_col] = 0
            self.board[end_row, end_col] = player.value
        else:
            # The piece moves off the board (scores a point)
            self.board[start_row, start_col] = 0
            self.pieces_off_board[player] += 1
            self.scores[player] += 1

        return True

    def _apply_attack(self, move: Move) -> bool:
        """Apply an ATTACK move."""
        player = self.current_player
        opponent = player.opposite()
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Check if the start position has the player's piece
        if self.board[start_row, start_col] != player.value:
            return False

        # Check if the end position has the opponent's piece
        if self.board[end_row, end_col] != opponent.value:
            return False

        # Check if this is a valid attack move (piece directly in front)
        if player == PlayerColor.BLACK:
            if end_row != start_row + 1 or end_col != start_col:
                return False
        else:  # RED
            if end_row != start_row - 1 or end_col != start_col:
                return False

        # Apply the attack
        self.board[start_row, start_col] = 0
        self.board[end_row, end_col] = player.value
        self.pieces_off_board[opponent] += 1

        return True

    def _apply_jump(self, move: Move) -> bool:
        """Apply a JUMP move."""
        player = self.current_player
        opponent = player.opposite()
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Check if the start position has the player's piece
        if self.board[start_row, start_col] != player.value:
            return False

        # Check if this is a valid jump move (in same column)
        if start_col != end_col:
            return False

        # Determine the direction of the jump
        direction = player.direction

        # Find the line of opponent pieces
        line_length = 0
        row = start_row + direction

        while 0 <= row < 4 and self.board[row, start_col] == opponent.value:
            line_length += 1
            row += direction

        # Check if there's a line of opponent pieces to jump over
        if line_length == 0 or line_length > 3:
            return False

        # Check if the landing position is valid
        landing_row = start_row + direction * (line_length + 1)

        if self.is_valid_position((landing_row, start_col)):
            # Landing on the board
            if not self.is_empty((landing_row, start_col)):
                return False

            # Apply the jump
            self.board[start_row, start_col] = 0
            self.board[landing_row, start_col] = player.value
        else:
            # Jumping off the board (scores a point)
            self.board[start_row, start_col] = 0
            self.pieces_off_board[player] += 1
            self.scores[player] += 1

        return True

    def get_valid_moves(self) -> List[Move]:
        """Get all valid moves for the current player."""
        return self._get_valid_moves()

    def _get_valid_moves(self) -> List[Move]:
        """Internal method to get all valid moves for the current player."""
        moves = []
        player = self.current_player

        # 1. Insert moves
        if self.pieces_off_board[player] > 0:
            start_row = player.start_row
            for col in range(3):
                if self.is_empty((start_row, col)):
                    moves.append(Move(MoveType.INSERT, end_pos=(start_row, col)))

        # 2. Diagonal moves
        player_value = player.value
        direction = player.direction

        for row in range(4):
            for col in range(3):
                if self.board[row, col] == player_value:
                    # Try both diagonal directions
                    for col_offset in [-1, 1]:
                        new_row = row + direction
                        new_col = col + col_offset

                        # Check if the destination is on the board
                        if 0 <= new_col < 3:
                            if 0 <= new_row < 4:
                                # Destination is on the board
                                if self.is_empty((new_row, new_col)):
                                    moves.append(
                                        Move(
                                            MoveType.DIAGONAL,
                                            start_pos=(row, col),
                                            end_pos=(new_row, new_col),
                                        )
                                    )
                            else:
                                # Moving off the board (scoring)
                                moves.append(
                                    Move(
                                        MoveType.DIAGONAL,
                                        start_pos=(row, col),
                                        end_pos=(new_row, new_col),
                                    )
                                )

        # 3. Attack moves
        opponent_value = player.opposite().value

        for row in range(4):
            for col in range(3):
                if self.board[row, col] == player_value:
                    # Try to attack directly in front
                    new_row = row + direction

                    if 0 <= new_row < 4:
                        if self.board[new_row, col] == opponent_value:
                            moves.append(
                                Move(
                                    MoveType.ATTACK,
                                    start_pos=(row, col),
                                    end_pos=(new_row, col),
                                )
                            )

        # 4. Jump moves
        for row in range(4):
            for col in range(3):
                if self.board[row, col] == player_value:
                    # Look for a line of opponent pieces
                    line_length = 0
                    curr_row = row + direction

                    while (
                        0 <= curr_row < 4
                        and self.board[curr_row, col] == opponent_value
                    ):
                        line_length += 1
                        curr_row += direction

                    if 1 <= line_length <= 3:
                        landing_row = row + (line_length + 1) * direction

                        # Check if landing is on board
                        if 0 <= landing_row < 4:
                            if self.is_empty((landing_row, col)):
                                moves.append(
                                    Move(
                                        MoveType.JUMP,
                                        start_pos=(row, col),
                                        end_pos=(landing_row, col),
                                    )
                                )
                        else:
                            # Jumping off the board (scoring)
                            moves.append(
                                Move(
                                    MoveType.JUMP,
                                    start_pos=(row, col),
                                    end_pos=(landing_row, col),
                                )
                            )

        return moves
