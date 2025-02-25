# src/board.py
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class Piece:
    """Represents a game piece with a color (red or black)"""
    color: str  # "red" or "black"
    
    def __repr__(self):
        return f"{self.color[0].upper()}"  # 'R' for red, 'B' for black


class Board:
    """Represents the 3x4 Kulibrat game board"""
    
    def __init__(self):
        # Create a 4x3 board (4 rows, 3 columns)
        self.grid = [[None for _ in range(3)] for _ in range(4)]
        
        # Track pieces not yet on the board
        self.available_pieces = {
            "red": 4,
            "black": 4
        }
        
        # Debug flag
        self.debug = True
    
    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Get the piece at a specific position, or None if empty"""
        if 0 <= row < 4 and 0 <= col < 3:
            return self.grid[row][col]
        return None
    
    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> None:
        """Place a piece at a specific position, or clear if piece is None"""
        if 0 <= row < 4 and 0 <= col < 3:
            self.grid[row][col] = piece
    
    def insert_piece(self, col: int, color: str) -> bool:
        """Insert a piece on the start row for the given color"""
        if self.available_pieces[color] <= 0:
            return False  # No more pieces available
            
        row = 0 if color == "black" else 3  # Start row depends on color
        
        if self.get_piece(row, col) is not None:
            return False  # Square is occupied
        
        if self.debug:
            print(f"DEBUG: Inserting {color} piece at ({row}, {col})")
            print(f"DEBUG: Available before: {self.available_pieces[color]}")
            
        self.set_piece(row, col, Piece(color))
        self.available_pieces[color] -= 1
        
        if self.debug:
            print(f"DEBUG: Available after: {self.available_pieces[color]}")
            print(f"DEBUG: Board state: {self.grid}")
            
        return True
    
    def remove_piece(self, row: int, col: int) -> Optional[Piece]:
        """Remove a piece from the board and return it"""
        piece = self.get_piece(row, col)
        if piece:
            self.set_piece(row, col, None)
            return piece
        return None
    
    def return_piece_to_player(self, color: str) -> None:
        """Return a piece to the player (when scored or removed)"""
        self.available_pieces[color] += 1
    
    def is_square_free(self, row: int, col: int) -> bool:
        """Check if a square is free (empty)"""
        return self.get_piece(row, col) is None
        
    def __str__(self) -> str:
        """String representation of the board"""
        result = []
        for row in self.grid:
            row_str = "|"
            for cell in row:
                if cell is None:
                    row_str += " |"
                else:
                    row_str += f"{cell}|"
            result.append(row_str)
            
        # Add available pieces info
        result.append(f"Available - Red: {self.available_pieces['red']}, " +
                     f"Black: {self.available_pieces['black']}")
                     
        return "\n".join(result)