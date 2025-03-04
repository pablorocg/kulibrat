"""
Game state representation for Kulibrat, optimized for performance.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set, FrozenSet, Any
from functools import lru_cache

from src.core.player_color import PlayerColor
from src.core.move import Move
from src.core.move_type import MoveType


class GameState:
    """Represents the complete state of a Kulibrat game."""

    # Reusable constants
    BOARD_ROWS = 4
    BOARD_COLS = 3
    
    # Pre-calculate all valid board positions for faster lookup
    VALID_POSITIONS = frozenset((r, c) for r in range(4) for c in range(3))
    
    # Pre-calculate diagonal offsets by player
    DIAGONAL_OFFSETS = {
        PlayerColor.BLACK: [(1, -1), (1, 1)],  # Down-left, Down-right
        PlayerColor.RED: [(-1, -1), (-1, 1)]   # Up-left, Up-right
    }

    def __init__(self, target_score: int = 5):
        """
        Initialize a new game state.

        Args:
            target_score: Score needed to win the game
        """
        # Board is 4x3 (rows x columns)
        # 0 = empty, 1 = black piece, -1 = red piece
        # Use int8 for memory efficiency
        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS), dtype=np.int8)

        # Number of pieces available to be inserted (not on board)
        self.pieces_off_board = {PlayerColor.BLACK: 4, PlayerColor.RED: 4}

        # Current scores
        self.scores = {PlayerColor.BLACK: 0, PlayerColor.RED: 0}

        # Who's turn is it
        self.current_player = PlayerColor.BLACK

        # How many points to win
        self.target_score = target_score
        
        # Cache for valid moves
        self._valid_moves_cache = {}

    def copy(self) -> "GameState":
        """Create a deep copy of the current game state."""
        new_state = GameState(self.target_score)
        # NumPy copy is faster for small arrays than deepcopy
        new_state.board = self.board.copy()
        # Dict copy is faster than deepcopy for small dicts
        new_state.pieces_off_board = self.pieces_off_board.copy()
        new_state.scores = self.scores.copy()
        new_state.current_player = self.current_player
        return new_state

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is on the board using pre-calculated set."""
        return pos in self.VALID_POSITIONS

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
        """Check if the game is over with optimized checks."""
        # Fast path: check scores first (most common game over condition)
        if any(score >= self.target_score for score in self.scores.values()):
            return True

        # Only check for locked players if score condition isn't met
        original_player = self.current_player
        
        # Check if BLACK has valid moves
        self.current_player = PlayerColor.BLACK
        black_has_moves = bool(self._get_valid_moves())

        # If BLACK has moves, game is not over
        if black_has_moves:
            self.current_player = original_player
            return False
            
        # Check if RED has valid moves
        self.current_player = PlayerColor.RED
        red_has_moves = bool(self._get_valid_moves())
        
        # Restore original player
        self.current_player = original_player

        # Game is over if neither player can move
        return not red_has_moves

    def get_winner(self) -> Optional[PlayerColor]:
        """Get the winner if the game is over."""
        if not self.is_game_over():
            return None

        # Check score-based win first (most common)
        for player, score in self.scores.items():
            if score >= self.target_score:
                return player

        # If both players are locked, the last player to move loses
        original_player = self.current_player

        # Check if either player has valid moves
        self.current_player = PlayerColor.RED
        red_has_moves = bool(self._get_valid_moves())
        
        if red_has_moves:
            self.current_player = original_player
            return None
            
        self.current_player = PlayerColor.BLACK
        black_has_moves = bool(self._get_valid_moves())

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
        # Directly dispatch based on move type for speed
        success = False
        
        if move.move_type == MoveType.INSERT:
            success = self._apply_insert(move)
        elif move.move_type == MoveType.DIAGONAL:
            success = self._apply_diagonal(move)
        elif move.move_type == MoveType.ATTACK:
            success = self._apply_attack(move)
        elif move.move_type == MoveType.JUMP:
            success = self._apply_jump(move)

        if success:
            # Clear the move cache since the board state changed
            self._valid_moves_cache = {}
            
        return success

    def _apply_insert(self, move: Move) -> bool:
        """Apply an INSERT move with faster checks."""
        player = self.current_player
        if self.pieces_off_board[player] <= 0:
            return False
            
        row, col = move.end_pos
        
        # Fast check: verify row is the start row and position is empty
        if row != player.start_row or self.board[row, col] != 0:
            return False

        # Apply the move
        self.board[row, col] = player.value
        self.pieces_off_board[player] -= 1
        return True

    def _apply_diagonal(self, move: Move) -> bool:
        """Apply a DIAGONAL move with optimized checks."""
        player = self.current_player
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Fast fail if start position doesn't have player's piece
        if self.board[start_row, start_col] != player.value:
            return False

        # Check if this is a valid diagonal move
        direction = player.direction
        if end_row != start_row + direction or abs(end_col - start_col) != 1:
            return False

        # Check if end position is on board
        if 0 <= end_row < self.BOARD_ROWS and 0 <= end_col < self.BOARD_COLS:
            # Check if end position is empty
            if self.board[end_row, end_col] != 0:
                return False

            # Move on the board
            self.board[start_row, start_col] = 0
            self.board[end_row, end_col] = player.value
        else:
            # Move off the board (scoring)
            self.board[start_row, start_col] = 0
            self.pieces_off_board[player] += 1
            self.scores[player] += 1

        return True

    def _apply_attack(self, move: Move) -> bool:
        """Apply an ATTACK move with optimized checks."""
        player = self.current_player
        opponent = player.opposite()
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Fast checks: start position has player's piece, end has opponent's piece
        if (self.board[start_row, start_col] != player.value or 
            self.board[end_row, end_col] != opponent.value):
            return False

        # Check if this is a valid attack move (directly in front)
        direction = player.direction
        if end_row != start_row + direction or end_col != start_col:
            return False

        # Apply the attack
        self.board[start_row, start_col] = 0
        self.board[end_row, end_col] = player.value
        self.pieces_off_board[opponent] += 1

        return True

    def _apply_jump(self, move: Move) -> bool:
        """Apply a JUMP move with optimized checks."""
        player = self.current_player
        opponent = player.opposite()
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos

        # Fast checks
        if (self.board[start_row, start_col] != player.value or start_col != end_col):
            return False

        # Calculate jump parameters
        direction = player.direction
        
        # Find line of opponent pieces (optimized)
        line_length = 0
        row = start_row + direction
        
        # Use direct array access instead of bounds checking in loop
        while 0 <= row < self.BOARD_ROWS and self.board[row, start_col] == opponent.value:
            line_length += 1
            row += direction

        # Check if there's a valid line to jump
        if line_length == 0 or line_length > 3:
            return False

        # Calculate landing position
        landing_row = start_row + direction * (line_length + 1)

        # Handle on-board landing
        if 0 <= landing_row < self.BOARD_ROWS:
            # Check if landing position is empty
            if self.board[landing_row, start_col] != 0:
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
        """Get all valid moves for the current player with caching."""
        return self._get_valid_moves()

    def _get_valid_moves(self) -> List[Move]:
        """Internal method to get all valid moves for the current player, optimized with caching."""
        # Check the cache first - create a cache key based on the board state and current player
        # We use the board array's data buffer and the current player as the key
        player = self.current_player
        
        # Use the board's data buffer, which is a more direct representation
        cache_key = (self.board.tobytes(), player, self.pieces_off_board[player])
        
        if cache_key in self._valid_moves_cache:
            return self._valid_moves_cache[cache_key]
        
        moves = []
        player_value = player.value
        opponent_value = -player_value  # Faster than calling opposite()
        direction = player.direction
        
        # 1. Insert moves - only if player has pieces off board
        if self.pieces_off_board[player] > 0:
            start_row = player.start_row
            # Fast iteration over columns
            for col in range(self.BOARD_COLS):
                if self.board[start_row, col] == 0:
                    moves.append(Move(MoveType.INSERT, end_pos=(start_row, col)))

        # Collect positions of player's pieces for faster processing
        player_positions = []
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                if self.board[row, col] == player_value:
                    player_positions.append((row, col))
        
        # Process all moves from each player piece position
        for row, col in player_positions:
            # 2. Diagonal moves - use pre-calculated offsets
            for row_offset, col_offset in self.DIAGONAL_OFFSETS[player]:
                new_row = row + row_offset
                new_col = col + col_offset
                
                if 0 <= new_col < self.BOARD_COLS:
                    if 0 <= new_row < self.BOARD_ROWS:
                        # On-board diagonal move
                        if self.board[new_row, new_col] == 0:
                            moves.append(
                                Move(
                                    MoveType.DIAGONAL,
                                    start_pos=(row, col),
                                    end_pos=(new_row, new_col),
                                )
                            )
                    else:
                        # Off-board diagonal move (scoring)
                        moves.append(
                            Move(
                                MoveType.DIAGONAL,
                                start_pos=(row, col),
                                end_pos=(new_row, new_col),
                            )
                        )
            
            # 3. Attack moves - directly in front
            attack_row = row + direction
            if 0 <= attack_row < self.BOARD_ROWS and self.board[attack_row, col] == opponent_value:
                moves.append(
                    Move(
                        MoveType.ATTACK,
                        start_pos=(row, col),
                        end_pos=(attack_row, col),
                    )
                )
            
            # 4. Jump moves - optimized to avoid redundant checks
            # Find line of opponent pieces
            line_length = 0
            curr_row = row + direction
            
            # Count opponent pieces in a line
            while 0 <= curr_row < self.BOARD_ROWS and self.board[curr_row, col] == opponent_value:
                line_length += 1
                curr_row += direction
            
            # Only process if there's a valid line to jump
            if 1 <= line_length <= 3:
                landing_row = row + (line_length + 1) * direction
                
                # Check landing position
                if 0 <= landing_row < self.BOARD_ROWS:
                    # Landing on board
                    if self.board[landing_row, col] == 0:
                        moves.append(
                            Move(
                                MoveType.JUMP,
                                start_pos=(row, col),
                                end_pos=(landing_row, col),
                            )
                        )
                else:
                    # Jumping off board (scoring)
                    moves.append(
                        Move(
                            MoveType.JUMP,
                            start_pos=(row, col),
                            end_pos=(landing_row, col),
                        )
                    )
        
        # Cache the result
        self._valid_moves_cache[cache_key] = moves
        return moves