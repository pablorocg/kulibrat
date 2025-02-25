# src/ai.py
from typing import Tuple, Optional, List
import time
import random
from copy import deepcopy
from .player import Player
from .game import Game

class MinimaxAIPlayer(Player):
    """
    AI player that uses the Minimax algorithm to make decisions
    """
    
    def __init__(self, color: str, max_depth: int = 3, use_alpha_beta: bool = True):
        """
        Initialize the Minimax AI player
        
        Args:
            color: The player's color ('red' or 'black')
            max_depth: Maximum search depth for minimax
            use_alpha_beta: Whether to use alpha-beta pruning
        """
        super().__init__(color)
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.opponent_color = "red" if color == "black" else "black"
        
    def get_move(self, game: Game) -> Tuple[Optional[int], Optional[int], int, int]:
        """Get the best move according to minimax algorithm"""
        print(f"\nAI ({self.color}) is thinking...")
        start_time = time.time()
        
        # Get all legal moves
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None, None, -1, -1
        
        # If there's only one move, just take it
        if len(legal_moves) == 1:
            move_type, start_row, start_col, end_row, end_col = legal_moves[0]
            return start_row, start_col, end_row, end_col
        
        # Clone the game state to perform simulations
        game_clone = deepcopy(game)
        
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Try each move and pick the one with the highest minimax value
        for move in legal_moves:
            move_type, start_row, start_col, end_row, end_col = move
            
            # Make the move on the cloned game
            game_copy = deepcopy(game_clone)
            game_copy.make_move(start_row, start_col, end_row, end_col)
            
            # Evaluate the move using minimax
            if self.use_alpha_beta:
                score = self._minimax(game_copy, self.max_depth - 1, False, alpha, beta)
            else:
                score = self._minimax(game_copy, self.max_depth - 1, False)
            
            if score > best_score:
                best_score = score
                best_move = move
            
            if self.use_alpha_beta:
                alpha = max(alpha, best_score)
        
        # Add a small random factor to make gameplay less predictable when multiple moves are equally good
        if random.random() < 0.1 and len(legal_moves) > 1:
            best_move = random.choice([m for m in legal_moves if m != best_move])
        
        end_time = time.time()
        print(f"AI decided in {end_time - start_time:.2f} seconds")
        
        move_type, start_row, start_col, end_row, end_col = best_move
        return start_row, start_col, end_row, end_col
    
    def _minimax(self, game: Game, depth: int, is_maximizing: bool, 
                alpha: float = float('-inf'), beta: float = float('inf')) -> float:
        """
        Minimax algorithm implementation
        
        Args:
            game: Current game state
            depth: Current depth in the search tree
            is_maximizing: Whether the current player is maximizing
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            The best score for the current player
        """
        # Terminal conditions
        if game.is_game_over():
            if game.get_winner() == self.color:
                return 1000  # Win
            else:
                return -1000  # Loss
        
        if depth == 0:
            return self._evaluate_board(game)
        
        # Get current player
        current_player = game.current_player
        is_current_player_me = current_player == self.color
        
        # Get legal moves
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            # If no legal moves, score is 0 (neutral)
            return 0
        
        if is_maximizing:
            # Maximizing player (us)
            best_score = float('-inf')
            
            for move in legal_moves:
                move_type, start_row, start_col, end_row, end_col = move
                
                # Make the move on a cloned game
                game_copy = deepcopy(game)
                game_copy.make_move(start_row, start_col, end_row, end_col)
                
                # Recursive call
                score = self._minimax(game_copy, depth - 1, False, alpha, beta)
                best_score = max(best_score, score)
                
                if self.use_alpha_beta:
                    alpha = max(alpha, best_score)
                    if beta <= alpha:
                        break  # Beta cutoff
            
            return best_score
        else:
            # Minimizing player (opponent)
            best_score = float('inf')
            
            for move in legal_moves:
                move_type, start_row, start_col, end_row, end_col = move
                
                # Make the move on a cloned game
                game_copy = deepcopy(game)
                game_copy.make_move(start_row, start_col, end_row, end_col)
                
                # Recursive call
                score = self._minimax(game_copy, depth - 1, True, alpha, beta)
                best_score = min(best_score, score)
                
                if self.use_alpha_beta:
                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
            
            return best_score
    
    def _evaluate_board(self, game: Game) -> float:
        """
        Evaluate the current board state
        Positive score favors the AI, negative favors opponent
        
        The evaluation considers:
        1. Score difference
        2. Piece position (advancing pieces are good)
        3. Available pieces
        4. Potential to score
        """
        # Score difference
        my_score = game.black_score if self.color == "black" else game.red_score
        opponent_score = game.red_score if self.color == "black" else game.black_score
        score_diff = (my_score - opponent_score) * 10
        
        # Board evaluation
        board_value = 0
        
        # Weight factor for piece advancement
        my_forward_direction = -1 if self.color == "black" else 1
        
        # Evaluate piece positions
        for row in range(4):
            for col in range(3):
                piece = game.board.get_piece(row, col)
                if piece is None:
                    continue
                
                # Relative row value (how far the piece has advanced)
                if self.color == "black":
                    relative_row = 3 - row  # 0 at top, 3 at bottom
                else:
                    relative_row = row      # 0 at top, 3 at bottom
                
                # Value increases as piece advances
                position_value = relative_row * 2
                
                # Add for our pieces, subtract for opponent pieces
                if piece.color == self.color:
                    board_value += position_value
                    
                    # Extra points for pieces close to scoring
                    if (self.color == "black" and row == 3) or \
                       (self.color == "red" and row == 0):
                        board_value += 5
                else:
                    board_value -= position_value
                    
                    # Penalty for opponent pieces close to scoring
                    if (self.color == "red" and row == 3) or \
                       (self.color == "black" and row == 0):
                        board_value -= 5
        
        # Available pieces
        my_available = game.board.available_pieces[self.color]
        opponent_available = game.board.available_pieces[self.opponent_color]
        available_diff = (my_available - opponent_available) * 0.5
        
        # Combine all factors
        total_score = score_diff + board_value + available_diff
        
        return total_score