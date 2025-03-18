# src/players/ai/heuristics.py
"""
Heuristic functions for the Minimax AI strategy.
This module contains different evaluation functions that can be used with the Minimax algorithm.
"""

from typing import Callable

from src.core.game_state import GameState
from src.core.player_color import PlayerColor


class HeuristicRegistry:
    """
    Registry of available heuristic functions for evaluating game states.
    """
    
    # Dictionary of registered heuristic functions
    _heuristics = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """
        Decorator to register a heuristic function.
        
        Args:
            name: Unique name for the heuristic function
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            cls._heuristics[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Callable:
        """
        Get a registered heuristic function by name.
        
        Args:
            name: Name of the heuristic function
            
        Returns:
            The heuristic function
            
        Raises:
            ValueError: If the heuristic function is not found
        """
        if name not in cls._heuristics:
            raise ValueError(f"Heuristic function '{name}' not found")
        return cls._heuristics[name]
    
    @classmethod
    def get_names(cls) -> list:
        """
        Get a list of all registered heuristic names.
        
        Returns:
            List of heuristic names
        """
        return list(cls._heuristics.keys())


# Simple score-based heuristic
@HeuristicRegistry.register("score_diff")
def score_diff_heuristic(state: GameState, player_color: PlayerColor) -> float:
    """
    Basic heuristic that only considers score difference.
    
    Args:
        state: Game state to evaluate
        player_color: Player to evaluate for
        
    Returns:
        Evaluation score
    """
    opponent_color = player_color.opposite()
    score_diff = state.scores[player_color] - state.scores[opponent_color]
    
    # Check for terminal states
    if state.is_game_over():
        winner = state.get_winner()
        if winner == player_color:
            return 1000.0  # Win
        elif winner is None:
            return 0.0  # Draw
        else:
            return -1000.0  # Loss
            
    return score_diff * 100.0


# Advanced heuristic considering piece positioning
@HeuristicRegistry.register("advanced")
def advanced_heuristic(state: GameState, player_color: PlayerColor) -> float:
    """
    Advanced heuristic that considers piece positioning, mobility, and score.
    
    Args:
        state: Game state to evaluate
        player_color: Player to evaluate for
        
    Returns:
        Evaluation score
    """
    # Check for terminal state
    if state.is_game_over():
        winner = state.get_winner()
        if winner == player_color:
            return 1000.0  # Win
        elif winner is None:
            return 0.0  # Draw
        else:
            return -1000.0  # Loss

    opponent_color = player_color.opposite()
    
    # Score difference (weighted heavily)
    score_diff = 100.0 * (state.scores[player_color] - state.scores[opponent_color])
    evaluation = score_diff
    
    # Piece advantage
    pieces_on_board_player = 4 - state.pieces_off_board[player_color] + state.scores[player_color]
    pieces_on_board_opponent = 4 - state.pieces_off_board[opponent_color] + state.scores[opponent_color]
    evaluation += 10.0 * (pieces_on_board_player - pieces_on_board_opponent)
    
    # Piece advancement (pieces closer to scoring)
    player_advancement = 0
    opponent_advancement = 0
    
    # Calculate advancement for both players
    for row in range(4):
        for col in range(3):
            piece = state.board[row, col]
            if piece == player_color.value:
                # For BLACK, advancement is measured by how far down the board they are
                # For RED, advancement is measured by how far up the board they are
                if player_color == PlayerColor.BLACK:
                    player_advancement += row  # 0 at top, 3 at bottom
                else:
                    player_advancement += 3 - row  # 3 at top, 0 at bottom
            elif piece == opponent_color.value:
                if opponent_color == PlayerColor.BLACK:
                    opponent_advancement += row
                else:
                    opponent_advancement += 3 - row
    
    evaluation += 5.0 * (player_advancement - opponent_advancement)
    
    # Number of valid moves (mobility)
    current_player = state.current_player
    state.current_player = player_color
    player_moves = len(state.get_valid_moves())
    state.current_player = opponent_color
    opponent_moves = len(state.get_valid_moves())
    state.current_player = current_player  # Restore original current player
    
    evaluation += 2.0 * (player_moves - opponent_moves)
    
    return evaluation


# Aggressive heuristic prioritizing attacks
@HeuristicRegistry.register("aggressive")
def aggressive_heuristic(state: GameState, player_color: PlayerColor) -> float:
    """
    Aggressive heuristic that prioritizes attacking opponent pieces.
    
    Args:
        state: Game state to evaluate
        player_color: Player to evaluate for
        
    Returns:
        Evaluation score
    """
    # Get basic evaluation
    eval_score = advanced_heuristic(state, player_color)
    
    # Count possible attacks
    opponent_color = player_color.opposite()
    attack_opportunities = 0
    
    # Get player and opponent piece positions
    player_pieces = []
    opponent_pieces = []
    
    for row in range(4):
        for col in range(3):
            if state.board[row, col] == player_color.value:
                player_pieces.append((row, col))
            elif state.board[row, col] == opponent_color.value:
                opponent_pieces.append((row, col))
    
    # Check for potential attacks
    direction = player_color.direction
    for p_row, p_col in player_pieces:
        next_row = p_row + direction
        if 0 <= next_row < 4:  # Check board bounds
            if state.board[next_row, p_col] == opponent_color.value:
                attack_opportunities += 1
    
    # Bonus for attack opportunities
    eval_score += 15.0 * attack_opportunities
    
    return eval_score


# Defensive heuristic prioritizing safe piece positioning
@HeuristicRegistry.register("defensive")
def defensive_heuristic(state: GameState, player_color: PlayerColor) -> float:
    """
    Defensive heuristic that prioritizes safe piece positioning.
    
    Args:
        state: Game state to evaluate
        player_color: Player to evaluate for
        
    Returns:
        Evaluation score
    """
    # Get basic evaluation
    eval_score = advanced_heuristic(state, player_color)
    
    opponent_color = player_color.opposite()
    
    # Count opponent attack opportunities
    vulnerable_pieces = 0
    
    # Get player piece positions
    player_pieces = []
    
    for row in range(4):
        for col in range(3):
            if state.board[row, col] == player_color.value:
                player_pieces.append((row, col))
    
    # Check for pieces vulnerable to attack
    opponent_direction = opponent_color.direction
    for p_row, p_col in player_pieces:
        prev_row = p_row - opponent_direction  # Position an opponent piece would be to attack
        
        if 0 <= prev_row < 4:  # Check board bounds
            if state.board[prev_row, p_col] == opponent_color.value:
                vulnerable_pieces += 1
    
    # Penalty for vulnerable pieces
    eval_score -= 15.0 * vulnerable_pieces
    
    return eval_score


# Complex strategic heuristic
@HeuristicRegistry.register("strategic")
def strategic_heuristic(state: GameState, player_color: PlayerColor) -> float:
    """
    Complex strategic heuristic that balances multiple factors.
    This is the default heuristic with the best overall performance.
    
    Args:
        state: Game state to evaluate
        player_color: Player to evaluate for
        
    Returns:
        Evaluation score
    """
    # Determine the color of the opponent
    opponent_color = player_color.opposite()
    
    # Board dimensions
    ROWS, COLS = 4, 3
    
    # Starting rows for each player
    player_start = 0 if player_color == PlayerColor.BLACK else ROWS - 1
    opponent_start = ROWS - 1 if player_color == PlayerColor.BLACK else 0
    
    # Movement direction (1 for BLACK moving down, -1 for RED moving up)
    direction = player_color.direction
    
    # 1. Score difference (most important factor)
    score_diff = state.scores[player_color] - state.scores[opponent_color]
    evaluation = score_diff * 10.0  # High weight for score difference
    
    # Variables for analysis
    player_pieces = []
    opponent_pieces = []
    player_count = 0
    opponent_count = 0
    advancement_score = 0
    scoring_positions = 0
    
    # Analyze the board
    for row in range(ROWS):
        for col in range(COLS):
            piece = state.board[row, col]
            
            if piece == player_color.value:
                player_pieces.append((row, col))
                player_count += 1
                
                # Piece progression (value for distance from starting row)
                progress = abs(row - player_start)
                advancement_score += progress * 1.5
                
                # Pieces about to score (on opponent's starting row)
                if row == opponent_start:
                    scoring_positions += 1
                    advancement_score += 2.0
                
                # Bonus for controlling the center column (more flexibility)
                if col == 1:
                    advancement_score += 0.7
            
            elif piece == opponent_color.value:
                opponent_pieces.append((row, col))
                opponent_count += 1
                
                # Penalize opponent pieces about to score
                if row == player_start:
                    advancement_score -= 2.0
    
    # Add scoring for advancement and positioning
    evaluation += advancement_score
    evaluation += scoring_positions * 3.0  # High value for scoring positions
    
    # 2. Piece advantage
    piece_advantage = player_count - opponent_count
    evaluation += piece_advantage * 0.8
    
    # 3. Mobility and special situations
    try:
        # Save current player
        current_player = state.current_player
        
        # Check player mobility
        state.current_player = player_color
        player_moves = state.get_valid_moves()
        
        # Check opponent mobility
        state.current_player = opponent_color
        opponent_moves = state.get_valid_moves()
        
        # Restore current player
        state.current_player = current_player
        
        # Mobility difference
        mobility_diff = len(player_moves) - len(opponent_moves)
        evaluation += mobility_diff * 0.5
        
        # Locking situations (very important)
        if len(player_moves) == 0 and len(opponent_moves) > 0:
            evaluation -= 8.0  # Big penalty for being locked
        elif len(opponent_moves) == 0 and len(player_moves) > 0:
            evaluation += 8.0  # Big bonus for locking opponent
    except:
        # If errors occur getting moves, continue without this part
        pass
    
    # 4. Analyze specific opportunities
    attack_opportunities = 0
    jump_opportunities = 0
    scoring_jump_opportunities = 0
    
    # Count potential attack opportunities
    for p_row, p_col in player_pieces:
        next_row = p_row + direction
        if 0 <= next_row < ROWS:
            if state.board[next_row, p_col] == opponent_color.value:
                attack_opportunities += 1
    
    # Count potential jump opportunities
    for p_row, p_col in player_pieces:
        for jump_len in range(1, 4):  # Can jump over 1-3 pieces
            # Verify there's a line of opponent pieces
            line_valid = True
            for i in range(1, jump_len + 1):
                check_row = p_row + direction * i
                if not (0 <= check_row < ROWS) or state.board[check_row, p_col] != opponent_color.value:
                    line_valid = False
                    break
            
            if line_valid:
                landing_row = p_row + direction * (jump_len + 1)
                # Check if landing is off board (scoring) or on empty square
                if landing_row < 0 or landing_row >= ROWS:
                    scoring_jump_opportunities += 1
                elif 0 <= landing_row < ROWS and state.board[landing_row, p_col] == 0:
                    jump_opportunities += 1
    
    # Add scores for tactical opportunities
    evaluation += attack_opportunities * 1.2
    evaluation += jump_opportunities * 1.0
    evaluation += scoring_jump_opportunities * 2.5  # Higher value for scoring jumps
    
    # 5. Detect strategic patterns for winning cycles
    player_on_opponent_side = sum(1 for r, c in player_pieces if (r - opponent_start) * direction >= 0)
    opponent_on_player_side = sum(1 for r, c in opponent_pieces if (r - player_start) * direction <= 0)
    
    # Potential advantage for cycles if we have more pieces on enemy side
    if player_on_opponent_side >= 2 and player_on_opponent_side > opponent_on_player_side:
        evaluation += 3.0
    
    # 6. Defensive strategy - value blocking
    for p_row, p_col in player_pieces:
        for o_row, o_col in opponent_pieces:
            # If our piece is diagonally blocking an opponent piece
            if abs(p_col - o_col) == 1 and (p_row - o_row) * direction == 1:
                evaluation += 0.8
    
    return evaluation