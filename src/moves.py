# src/moves.py
from typing import List, Tuple, Optional
from .board import Board, Piece

class MoveValidator:
    """Validates and executes moves for Kulibrat"""
    
    @staticmethod
    def get_diagonal_moves(board: Board, row: int, col: int, color: str) -> List[Tuple[int, int]]:
        """Get all possible diagonal moves for a piece"""
        moves = []
        
        # Determine forward direction based on color
        forward = -1 if color == "black" else 1
        
        # Check diagonal left
        if 0 <= col - 1 < 3:
            new_row = row + forward
            new_col = col - 1
            if 0 <= new_row < 4 and board.is_square_free(new_row, new_col):
                moves.append((new_row, new_col))
        
        # Check diagonal right
        if 0 <= col + 1 < 3:
            new_row = row + forward
            new_col = col + 1
            if 0 <= new_row < 4 and board.is_square_free(new_row, new_col):
                moves.append((new_row, new_col))
        
        return moves
    
    @staticmethod
    def get_attack_moves(board: Board, row: int, col: int, color: str) -> List[Tuple[int, int]]:
        """Get all possible attack moves for a piece"""
        moves = []
        
        # Determine forward direction and opponent color based on color
        forward = -1 if color == "black" else 1
        opponent = "red" if color == "black" else "black"
        
        # Check attack (directly forward)
        new_row = row + forward
        if 0 <= new_row < 4:
            piece = board.get_piece(new_row, col)
            if piece and piece.color == opponent:
                moves.append((new_row, col))
        
        return moves
    
    @staticmethod
    def get_jump_moves(board: Board, row: int, col: int, color: str) -> List[Tuple[int, int, int]]:
        """Get all possible jump moves for a piece"""
        moves = []
        
        # Determine forward direction and opponent color based on color
        forward = -1 if color == "black" else 1
        opponent = "red" if color == "black" else "black"
        
        # Check for lines of opponent pieces to jump over (1, 2, or 3 pieces)
        for jump_length in range(1, 4):
            # Check if all positions in the jump contain opponent pieces
            valid_jump = True
            for i in range(1, jump_length + 1):
                check_row = row + forward * i
                if not (0 <= check_row < 4):
                    valid_jump = False
                    break
                
                piece = board.get_piece(check_row, col)
                if not piece or piece.color != opponent:
                    valid_jump = False
                    break
            
            if valid_jump:
                # The landing position is one step beyond the line
                landing_row = row + forward * (jump_length + 1)
                
                # Check if landing position is valid (either off-board or empty)
                if landing_row < 0 or landing_row >= 4:
                    # Jumping off the board (scoring)
                    moves.append((landing_row, col, jump_length))
                elif board.is_square_free(landing_row, col):
                    # Landing on an empty square
                    moves.append((landing_row, col, jump_length))
        
        return moves
    
    @staticmethod
    def get_insert_moves(board: Board, color: str) -> List[Tuple[int, int]]:
        """Get all possible insert moves for a player"""
        moves = []
        
        if board.available_pieces[color] <= 0:
            return moves  # No pieces available to insert
            
        # Determine the start row based on color
        row = 0 if color == "black" else 3
        
        # Check all three columns for empty spaces
        for col in range(3):
            if board.is_square_free(row, col):
                moves.append((row, col))
        
        return moves
    
    @staticmethod
    def get_all_moves(board: Board, color: str) -> List[Tuple[str, int, int, Optional[int], Optional[int]]]:
        """
        Get all possible moves for a player
        Returns a list of tuples: (move_type, start_row, start_col, end_row, end_col)
        For insert moves, start_row and start_col are None
        """
        all_moves = []
        
        # Get insert moves
        for row, col in MoveValidator.get_insert_moves(board, color):
            all_moves.append(("insert", None, None, row, col))
        
        # Get moves for each piece on the board
        for row in range(4):
            for col in range(3):
                piece = board.get_piece(row, col)
                if piece and piece.color == color:
                    # Add diagonal moves
                    for end_row, end_col in MoveValidator.get_diagonal_moves(board, row, col, color):
                        all_moves.append(("diagonal", row, col, end_row, end_col))
                    
                    # Add attack moves
                    for end_row, end_col in MoveValidator.get_attack_moves(board, row, col, color):
                        all_moves.append(("attack", row, col, end_row, end_col))
                    
                    # Add jump moves
                    for end_row, end_col, _ in MoveValidator.get_jump_moves(board, row, col, color):
                        all_moves.append(("jump", row, col, end_row, end_col))
        
        return all_moves
    
    @staticmethod
    def execute_move(board: Board, move_type: str, start_row: Optional[int], start_col: Optional[int], 
                    end_row: int, end_col: int, color: str) -> bool:
        """
        Execute a move on the board
        Returns True if the move scores a point, False otherwise
        """
        print(f"DEBUG: Executing move: {move_type}, from ({start_row}, {start_col}) to ({end_row}, {end_col}) for {color}")
        
        if move_type == "insert":
            # Insert moves are handled differently
            success = board.insert_piece(end_col, color)
            print(f"DEBUG: Insert result: {success}")
            return False  # Insert moves don't score points
        
        if move_type == "diagonal":
            piece = board.remove_piece(start_row, start_col)
            
            # Check if the move scores (piece moves off the board)
            opponent_start_row = 3 if color == "black" else 0
            if end_row == opponent_start_row and (end_col < 0 or end_col >= 3):
                board.return_piece_to_player(color)  # Piece is reusable
                return True  # Score a point
                
            board.set_piece(end_row, end_col, piece)
            return False
        
        if move_type == "attack":
            piece = board.remove_piece(start_row, start_col)
            opponent_piece = board.remove_piece(end_row, end_col)
            
            # Return opponent piece to their available pieces
            board.return_piece_to_player(opponent_piece.color)
            
            # Place attacking piece in the target position
            board.set_piece(end_row, end_col, piece)
            return False
        
        if move_type == "jump":
            piece = board.remove_piece(start_row, start_col)
            
            # Check if the move scores (piece jumps off the board)
            if end_row < 0 or end_row >= 4:
                board.return_piece_to_player(color)  # Piece is reusable
                return True  # Score a point
            
            # Place the piece in its landing position
            board.set_piece(end_row, end_col, piece)
            return False
        
        return False