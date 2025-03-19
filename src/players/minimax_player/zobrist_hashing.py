from enum import Enum
from typing import Tuple
import numpy as np

from src.core import GameState, PlayerColor

class ZobristHashing:
    """
    Implements Zobrist hashing for game states.
    """
    
    def __init__(self, rows: int = 4, cols: int = 3, piece_types: int = 3):
        """
        Initialize Zobrist hashing with random bitstrings.
        
        Args:
            rows: Number of rows on the board
            cols: Number of columns on the board
            piece_types: Number of different piece types (including empty)
        """
        # Initialize random bitstrings for each piece at each position
        np.random.seed(42)  # For reproducibility
        self.piece_position = np.random.randint(
            0, 2**64 - 1, 
            size=(rows, cols, piece_types), 
            dtype=np.uint64
        )
        
        # Hash for player to move (BLACK=1, RED=-1, we'll use index 0 for BLACK, 1 for RED)
        self.player_to_move = np.random.randint(0, 2**64 - 1, size=2, dtype=np.uint64)
        
        # Hash for pieces in hand (0-4 pieces for each player)
        self.pieces_in_hand = {
            PlayerColor.BLACK: np.random.randint(0, 2**64 - 1, size=5, dtype=np.uint64),
            PlayerColor.RED: np.random.randint(0, 2**64 - 1, size=5, dtype=np.uint64)
        }
        
        # Hash for scores (0-10 for each player)
        self.scores = {
            PlayerColor.BLACK: np.random.randint(0, 2**64 - 1, size=11, dtype=np.uint64),
            PlayerColor.RED: np.random.randint(0, 2**64 - 1, size=11, dtype=np.uint64)
        }
        
    def compute_hash(self, state: GameState) -> int:
        """
        Compute the Zobrist hash for a game state.
        
        Args:
            state: Game state to hash
            
        Returns:
            64-bit Zobrist hash
        """
        h = 0
        
        # Hash the board position
        for row in range(state.BOARD_ROWS):
            for col in range(state.BOARD_COLS):
                piece = state.board[row, col]
                # Map piece values to indices (0: empty, 1: BLACK, 2: RED)
                piece_idx = 0  # Default empty
                if piece == PlayerColor.BLACK.value:
                    piece_idx = 1
                elif piece == PlayerColor.RED.value:
                    piece_idx = 2
                    
                h ^= self.piece_position[row, col, piece_idx]
        
        # Hash the player to move
        player_idx = 0 if state.current_player == PlayerColor.BLACK else 1
        h ^= self.player_to_move[player_idx]
        
        # Hash pieces in hand
        for player in [PlayerColor.BLACK, PlayerColor.RED]:
            pieces = min(state.pieces_off_board[player], 4)  # Clamp to 0-4
            h ^= self.pieces_in_hand[player][pieces]
        
        # Hash scores
        for player in [PlayerColor.BLACK, PlayerColor.RED]:
            score = min(state.scores[player], 10)  # Clamp to 0-10
            h ^= self.scores[player][score]
            
        return h