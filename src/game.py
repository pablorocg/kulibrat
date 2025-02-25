# src/game.py
from typing import List, Tuple, Optional
from .board import Board
from .moves import MoveValidator

class Game:
    """Manages the game state and rules for Kulibrat"""
    
    def __init__(self, win_score: int = 5):
        self.board = Board()
        self.current_player = "black"  # Black goes first
        self.black_score = 0
        self.red_score = 0
        self.win_score = win_score
        self.game_over = False
    
    def switch_player(self) -> None:
        """Switch to the other player"""
        self.current_player = "red" if self.current_player == "black" else "black"
    
    def get_legal_moves(self, row: Optional[int] = None, col: Optional[int] = None) -> List[Tuple]:
        """
        Get legal moves for a specific piece or all legal moves for current player
        If row and col are None, returns all legal moves for the current player
        If row and col are specified, returns only moves for that piece
        """
        if self.game_over:
            return []
            
        all_moves = MoveValidator.get_all_moves(self.board, self.current_player)
        
        if row is None or col is None:
            return all_moves
            
        # Filter to only include moves for the specified piece
        piece_moves = []
        for move in all_moves:
            move_type, start_row, start_col, end_row, end_col = move
            if start_row == row and start_col == col:
                piece_moves.append(move)
        
        return piece_moves
    
    def has_legal_moves(self, color: str) -> bool:
        """Check if a player has any legal moves"""
        return len(MoveValidator.get_all_moves(self.board, color)) > 0
    
    def make_move(self, start_row: Optional[int], start_col: Optional[int], 
                 end_row: int, end_col: int) -> bool:
        """
        Make a move from (start_row, start_col) to (end_row, end_col)
        If start_row and start_col are None, it's an insert move
        Returns True if the move was successful, False otherwise
        """
        if self.game_over:
            return False
            
        # Get all legal moves
        legal_moves = self.get_legal_moves()
        
        # Find the matching move
        matching_move = None
        for move in legal_moves:
            move_type, move_start_row, move_start_col, move_end_row, move_end_col = move
            if move_start_row == start_row and move_start_col == start_col and \
               move_end_row == end_row and move_end_col == end_col:
                matching_move = move
                break
        
        if not matching_move:
            return False  # No matching legal move found
            
        move_type, _, _, _, _ = matching_move
        current_player = self.current_player  # Store the current player before switching
        
        # Execute the move
        scored = MoveValidator.execute_move(
            self.board, move_type, start_row, start_col, end_row, end_col, current_player
        )
        
        # Update score if a point was scored
        if scored:
            if current_player == "black":
                self.black_score += 1
            else:
                self.red_score += 1
                
            # Check if the game is over
            if self.black_score >= self.win_score or self.red_score >= self.win_score:
                self.game_over = True
        
        # Switch to the next player
        self.switch_player()
        
        # Handle the case where the next player has no legal moves
        if not self.has_legal_moves(self.current_player):
            opponent = "red" if self.current_player == "black" else "black"
            
            # Check if both players are locked
            if not self.has_legal_moves(opponent):
                # Game is locked for both players, current player loses
                if self.current_player == "black":
                    self.red_score = self.win_score  # Red wins
                else:
                    self.black_score = self.win_score  # Black wins
                self.game_over = True
            else:
                # Only current player is locked, switch back to the other player
                self.switch_player()
        
        return True
    
    def make_insert_move(self, col: int) -> bool:
        """Convenience method for making an insert move"""
        row = 0 if self.current_player == "black" else 3
        return self.make_move(None, None, row, col)
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.game_over
    
    def get_winner(self) -> Optional[str]:
        """Get the winner of the game, or None if the game is not over"""
        if not self.game_over:
            return None
            
        if self.black_score >= self.win_score:
            return "black"
        elif self.red_score >= self.win_score:
            return "red"
        
        return None
    
    def __str__(self) -> str:
        """String representation of the game state"""
        result = [
            f"Current player: {self.current_player}",
            f"Score - Black: {self.black_score}, Red: {self.red_score}",
            f"Game over: {self.game_over}",
            "",
            str(self.board)
        ]
        return "\n".join(result)