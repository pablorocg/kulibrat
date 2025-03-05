"""
Cython implementation of the GameState class for Kulibrat.
Compile with:
    python setup.py build_ext --inplace
"""

# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from typing import List, Optional, Tuple, Dict, Set, Any

# Import related modules
from src.core.player_color import PlayerColor
from src.core.move import Move
from src.core.move_type import MoveType

# Define numpy types
DTYPE = np.int8
ctypedef np.int8_t DTYPE_t

cdef class GameState:
    """High-performance Cython implementation of the Kulibrat game state."""
    
    # Class constants
    cdef public int BOARD_ROWS
    cdef public int BOARD_COLS
    
    # State data
    cdef public np.ndarray board
    cdef public dict pieces_off_board
    cdef public dict scores
    cdef public object current_player
    cdef public int target_score
    cdef dict _valid_moves_cache
    
    # Pre-calculated data for performance
    cdef list BLACK_DIAGONAL_OFFSETS
    cdef list RED_DIAGONAL_OFFSETS
    
    def __init__(self, int target_score=5):
        """Initialize a new game state."""
        # Set constants
        self.BOARD_ROWS = 4
        self.BOARD_COLS = 3
        
        # Initialize board (0=empty, 1=black, -1=red)
        self.board = np.zeros((self.BOARD_ROWS, self.BOARD_COLS), dtype=DTYPE)
        
        # Initialize game state
        self.pieces_off_board = {PlayerColor.BLACK: 4, PlayerColor.RED: 4}
        self.scores = {PlayerColor.BLACK: 0, PlayerColor.RED: 0}
        self.current_player = PlayerColor.BLACK
        self.target_score = target_score
        
        # Initialize cache
        self._valid_moves_cache = {}
        
        # Pre-calculate diagonal offsets for faster move generation
        self.BLACK_DIAGONAL_OFFSETS = [(1, -1), (1, 1)]  # Down-left, Down-right
        self.RED_DIAGONAL_OFFSETS = [(-1, -1), (-1, 1)]   # Up-left, Up-right
    
    cpdef GameState copy(self):
        """Create a deep copy of the current game state."""
        cdef GameState new_state = GameState(self.target_score)
        new_state.board = self.board.copy()
        new_state.pieces_off_board = self.pieces_off_board.copy()
        new_state.scores = self.scores.copy()
        new_state.current_player = self.current_player
        return new_state
    
    cdef inline bint is_valid_position(self, int row, int col):
        """Check if a position is on the board."""
        return 0 <= row < self.BOARD_ROWS and 0 <= col < self.BOARD_COLS
    
    cdef inline bint is_empty(self, int row, int col):
        """Check if a board position is empty."""
        return self.board[row, col] == 0
    
    cdef inline int get_piece_at(self, int row, int col):
        """Get the piece at a specific position."""
        if not self.is_valid_position(row, col):
            return 0
        return self.board[row, col]
    
    cpdef bint is_game_over(self):
        """Check if the game is over."""
        # Check if any player reached the target score
        if self.scores[PlayerColor.BLACK] >= self.target_score or self.scores[PlayerColor.RED] >= self.target_score:
            return True
        
        # Check if both players are locked (no valid moves)
        cdef object original_player = self.current_player
        cdef bint black_has_moves, red_has_moves
        
        # Check if BLACK has valid moves
        self.current_player = PlayerColor.BLACK
        black_has_moves = len(self._get_valid_moves()) > 0
        
        # Early return if BLACK has moves
        if black_has_moves:
            self.current_player = original_player
            return False
        
        # Check if RED has valid moves
        self.current_player = PlayerColor.RED
        red_has_moves = len(self._get_valid_moves()) > 0
        
        # Restore original player
        self.current_player = original_player
        
        # Game is over if neither player can move
        return not red_has_moves
    
    cpdef object get_winner(self):
        """Get the winner if the game is over."""
        if not self.is_game_over():
            return None
        
        # Check if a player reached the target score
        if self.scores[PlayerColor.BLACK] >= self.target_score:
            return PlayerColor.BLACK
        if self.scores[PlayerColor.RED] >= self.target_score:
            return PlayerColor.RED
        
        # If both players are locked, the last player to move loses
        cdef object original_player = self.current_player
        cdef bint red_has_moves, black_has_moves
        
        # Check if RED has valid moves
        self.current_player = PlayerColor.RED
        red_has_moves = len(self._get_valid_moves()) > 0
        
        if red_has_moves:
            self.current_player = original_player
            return None
        
        # Check if BLACK has valid moves
        self.current_player = PlayerColor.BLACK
        black_has_moves = len(self._get_valid_moves()) > 0
        
        # Restore the original player
        self.current_player = original_player
        
        # If neither player can move, the last player to move loses
        if not (red_has_moves or black_has_moves):
            return self.current_player.opposite()
        
        return None
    
    cpdef bint apply_move(self, object move):
        """Apply a move to the game state and return whether it was successful."""
        cdef bint success = False
        
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
            self._valid_moves_cache.clear()
        
        return success
    
    cpdef bint _apply_insert(self, object move):
        """Apply an INSERT move."""
        cdef object player = self.current_player
        cdef int row, col
        
        if self.pieces_off_board[player] <= 0:
            return False
        
        row, col = move.end_pos
        
        # Check if the move is valid
        if row != player.start_row or not self.is_empty(row, col):
            return False
        
        # Apply the move
        self.board[row, col] = player.value
        self.pieces_off_board[player] -= 1
        return True
    
    cpdef bint _apply_diagonal(self, object move):
        """Apply a DIAGONAL move."""
        cdef object player = self.current_player
        cdef int start_row, start_col, end_row, end_col, direction
        
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos
        
        # Check if the start position has the player's piece
        if self.board[start_row, start_col] != player.value:
            return False
        
        # Check if this is a valid diagonal move
        direction = player.direction
        if end_row != start_row + direction or abs(end_col - start_col) != 1:
            return False
        
        # Check if the end position is on the board
        if self.is_valid_position(end_row, end_col):
            # Check if the end position is empty
            if not self.is_empty(end_row, end_col):
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
    
    cpdef bint _apply_attack(self, object move):
        """Apply an ATTACK move."""
        cdef object player = self.current_player
        cdef object opponent = player.opposite()
        cdef int start_row, start_col, end_row, end_col, direction
        
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos
        
        # Fast checks
        if self.board[start_row, start_col] != player.value or self.board[end_row, end_col] != opponent.value:
            return False
        
        # Check valid attack direction
        direction = player.direction
        if end_row != start_row + direction or end_col != start_col:
            return False
        
        # Apply the attack
        self.board[start_row, start_col] = 0
        self.board[end_row, end_col] = player.value
        self.pieces_off_board[opponent] += 1
        
        return True
    
    cpdef bint _apply_jump(self, object move):
        """Apply a JUMP move."""
        cdef object player = self.current_player
        cdef object opponent = player.opposite()
        cdef int start_row, start_col, end_row, end_col, direction
        cdef int line_length, row, landing_row
        
        start_row, start_col = move.start_pos
        end_row, end_col = move.end_pos
        
        # Fast checks
        if self.board[start_row, start_col] != player.value or start_col != end_col:
            return False
        
        # Calculate jump parameters
        direction = player.direction
        
        # Find line of opponent pieces (optimized)
        line_length = 0
        row = start_row + direction
        
        # Count opponent pieces in a line
        while self.is_valid_position(row, start_col) and self.board[row, start_col] == opponent.value:
            line_length += 1
            row += direction
        
        # Check if there's a valid line to jump
        if line_length == 0 or line_length > 3:
            return False
        
        # Calculate landing position
        landing_row = start_row + direction * (line_length + 1)
        
        # Handle on-board landing
        if self.is_valid_position(landing_row, start_col):
            # Check if landing position is empty
            if not self.is_empty(landing_row, start_col):
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
    
    cpdef list get_valid_moves(self):
        """Get all valid moves for the current player with caching."""
        return self._get_valid_moves()
    
    cpdef list _get_valid_moves(self):
        """Internal method to get all valid moves for the current player with caching."""
        cdef object player = self.current_player
        cdef bytes cache_key = self.board.tobytes() + bytes([player.value, self.pieces_off_board[player]])
        
        # Check cache
        if cache_key in self._valid_moves_cache:
            return self._valid_moves_cache[cache_key]
        
        cdef list moves = []
        cdef int player_value = player.value
        cdef int opponent_value = -player_value
        cdef int direction = player.direction
        cdef int row, col, start_row, new_row, new_col, attack_row, curr_row, line_length, landing_row
        cdef list player_positions = []
        cdef list diagonal_offsets
        cdef tuple offset
        
        # 1. Insert moves - only if player has pieces off board
        if self.pieces_off_board[player] > 0:
            start_row = player.start_row
            for col in range(self.BOARD_COLS):
                if self.is_empty(start_row, col):
                    moves.append(Move(MoveType.INSERT, end_pos=(start_row, col)))
        
        # Collect positions of player's pieces
        for row in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
                if self.board[row, col] == player_value:
                    player_positions.append((row, col))
        
        # Set diagonal offsets based on player
        diagonal_offsets = self.BLACK_DIAGONAL_OFFSETS if player == PlayerColor.BLACK else self.RED_DIAGONAL_OFFSETS
        
        # Process all moves from each player piece
        for row, col in player_positions:
            # 2. Diagonal moves
            for row_offset, col_offset in diagonal_offsets:
                new_row = row + row_offset
                new_col = col + col_offset
                
                if 0 <= new_col < self.BOARD_COLS:
                    if 0 <= new_row < self.BOARD_ROWS:
                        # On-board diagonal move
                        if self.is_empty(new_row, new_col):
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
            if self.is_valid_position(attack_row, col) and self.board[attack_row, col] == opponent_value:
                moves.append(
                    Move(
                        MoveType.ATTACK,
                        start_pos=(row, col),
                        end_pos=(attack_row, col),
                    )
                )
            
            # 4. Jump moves
            line_length = 0
            curr_row = row + direction
            
            # Count opponent pieces in a line
            while self.is_valid_position(curr_row, col) and self.board[curr_row, col] == opponent_value:
                line_length += 1
                curr_row += direction
            
            # Only process if there's a valid line to jump
            if 1 <= line_length <= 3:
                landing_row = row + (line_length + 1) * direction
                
                # Check landing position
                if self.is_valid_position(landing_row, col):
                    # Landing on board
                    if self.is_empty(landing_row, col):
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