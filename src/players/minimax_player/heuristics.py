# src/players/ai/heuristics.py
"""
Heuristic functions for the Minimax AI strategy.
This module contains different evaluation functions that can be used with the Minimax algorithm.
"""

from typing import Callable

from src.core.game_state import GameState
from src.core.move import MoveType
from src.core.player_color import PlayerColor
from src.core.move import Move
import numpy as np

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



# Heuristic functions for evaluating game states
# ------------------------------------------------

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
    # copy the state
    local_state = state.copy()
    opponent_color = player_color.opposite()
    score_diff = local_state.scores[player_color] - local_state.scores[opponent_color]

    # Check for terminal states
    if local_state.is_game_over():
        winner = local_state.get_winner()
        if winner == player_color:
            return np.inf  #10000.0 Win
        elif winner is None:
            return 0.0  # Draw
        else:
            return -np.inf #10000.0 Loss
    
    if score_diff == 0:
        return 0.0
    elif score_diff > 0:
        return score_diff ** 2
    else:
        return -score_diff ** 2

# Heuristic that considers score difference and mobility
@HeuristicRegistry.register("score_diff_with_deadlock")
def safe_score_diff(state: GameState, player_color: PlayerColor) -> float:
    # Create a local copy to avoid side effects
    local_state = state.copy()
    
    # Terminal state check
    if local_state.is_game_over():
        winner = local_state.get_winner()
        if winner == player_color:
            return 1000.0
        elif winner is None:
            return 0.0
        else:
            return -1000.0
    
    opponent_color = player_color.opposite()
    score_diff = local_state.scores[player_color] - local_state.scores[opponent_color]
    
    # Store original player
    original_player = local_state.current_player
    
    # Calculate mobility safely
    local_state.current_player = player_color
    player_moves = len(local_state.get_valid_moves())
    
    local_state.current_player = opponent_color
    opponent_moves = len(local_state.get_valid_moves())
    
    # Restore original player
    local_state.current_player = original_player
    
    # Basic evaluation
    evaluation = score_diff * 10.0
    
    # Add tiny mobility factor regardless of score
    if player_moves + opponent_moves > 0:
        mobility_ratio = (player_moves - opponent_moves) / (player_moves + opponent_moves)
        evaluation += mobility_ratio * 0.25
    
    return evaluation

# Heuristic that considers score difference, mobility, and special moves
@HeuristicRegistry.register("score_mobility_special")
# @HeuristicRegistry.register("normalized_heuristic")
def normalized_heuristic(state: GameState, player_color: PlayerColor, weights=None) -> float:
    """
    Normalized heuristic with optimizable weights.
    Each component is normalized to roughly [-1, 1] range.
    
    Args:
        state: Current game state
        player_color: Player to evaluate for
        weights: Optional list of weights for components
    """
    # Default weights if none provided
    if weights is None:
        weights = [1.0, 0.8, 0.6, 0.7, 0.5, 0.4, 0.9, 0.6, 0.7]
    
    opponent_color = player_color.opposite()
    
    # Terminal state check
    if state.is_game_over():
        winner = state.get_winner()
        if winner == player_color:
            return 1000.0  # Win
        elif winner is None:
            return 0.0     # Draw
        else:
            return -1000.0 # Loss
    
    # Store components for weighted sum
    components = []
    
    # 1. Score difference (normalized by target score)
    target_score = state.target_score
    score_diff = state.scores[player_color] - state.scores[opponent_color]
    norm_score_diff = score_diff / target_score  # Normalize to [-1, 1] range
    components.append(norm_score_diff)
    
    # 2. Piece advancement (normalized by board size)
    advancement_sum = 0
    opponent_advancement_sum = 0
    max_advancement = state.BOARD_ROWS - 1  # Maximum possible advancement (3)
    
    for row in range(state.BOARD_ROWS):
        for col in range(state.BOARD_COLS):
            piece = state.board[row, col]
            
            if piece == player_color.value:
                # Calculate normalized advancement (0 to 1)
                if player_color == PlayerColor.BLACK:
                    progress = row / max_advancement
                else:
                    progress = (max_advancement - row) / max_advancement
                advancement_sum += progress
                
            elif piece == opponent_color.value:
                if opponent_color == PlayerColor.BLACK:
                    progress = row / max_advancement
                else:
                    progress = (max_advancement - row) / max_advancement
                opponent_advancement_sum += progress
    
    # Normalize by maximum possible pieces (4)
    norm_advancement = advancement_sum / 4
    norm_opponent_advancement = opponent_advancement_sum / 4
    
    # Add both as separate components
    components.append(norm_advancement)
    components.append(-norm_opponent_advancement)  # Negate for opponent
    
    # 3. Tactical considerations
    original_player = state.current_player
    
    # Calculate mobility
    state.current_player = player_color
    player_moves = state.get_valid_moves()
    
    state.current_player = opponent_color
    opponent_moves = state.get_valid_moves()
    
    # Restore original player
    state.current_player = original_player
    
    # 3a. Mobility ratio (already normalized to [-1, 1])
    mobility_ratio = 0
    if len(player_moves) + len(opponent_moves) > 0:
        mobility_ratio = (len(player_moves) - len(opponent_moves)) / (len(player_moves) + len(opponent_moves))
    components.append(mobility_ratio)
    
    # 3b. Special move advantage (normalized)
    max_special_moves = 6  # Theoretical maximum
    player_special_moves = sum(1 for m in player_moves if m.move_type in [MoveType.JUMP, MoveType.ATTACK])
    opponent_special_moves = sum(1 for m in opponent_moves if m.move_type in [MoveType.JUMP, MoveType.ATTACK])
    
    norm_special_advantage = (player_special_moves - opponent_special_moves) / max(1, max_special_moves)
    components.append(norm_special_advantage)
    
    # 3c. Deadlock advantage (normalized to [-1, 1])
    deadlock_advantage = 0
    if len(player_moves) == 0 and len(opponent_moves) > 0:
        deadlock_advantage = -1  # Bad position
    elif len(opponent_moves) == 0 and len(player_moves) > 0:
        deadlock_advantage = 1   # Good position
    components.append(deadlock_advantage)
    
    # 4. Scoring opportunities (normalized)
    player_scoring_moves = sum(1 for m in player_moves if _is_scoring_move(m, state, player_color))
    opponent_scoring_moves = sum(1 for m in opponent_moves if _is_scoring_move(m, state, opponent_color))
    
    # Normalize by theoretical maximum (4)
    norm_scoring_diff = (player_scoring_moves - opponent_scoring_moves) / 4
    components.append(norm_scoring_diff)
    
    # 5. Endgame awareness
    endgame_factor = 0
    # Player close to winning
    if state.scores[player_color] >= target_score - 1:
        endgame_factor = 0.5
    # Opponent close to winning
    if state.scores[opponent_color] >= target_score - 1:
        endgame_factor -= 0.5
    components.append(endgame_factor)
    
    # 6. Blocking opponent's scoring row
    scoring_row = 0 if player_color == PlayerColor.BLACK else 3
    blocking_count = sum(1 for col in range(state.BOARD_COLS) 
                         if state.board[scoring_row, col] == player_color.value)
    norm_blocking = blocking_count / state.BOARD_COLS  # Normalize by board width
    components.append(norm_blocking)
    
    # Calculate weighted sum
    assert len(components) == len(weights), f"Mismatch between components ({len(components)}) and weights ({len(weights)})"
    evaluation = sum(c * w for c, w in zip(components, weights))
    
    return evaluation

def _is_scoring_move(move: Move, state: GameState, player_color: PlayerColor) -> bool:
    """Helper to identify if a move results in scoring."""
    if not move.end_pos:
        return False
        
    # Check if move ends off the board in the scoring direction
    if player_color == PlayerColor.BLACK and move.end_pos[0] >= state.BOARD_ROWS:
        return True
    if player_color == PlayerColor.RED and move.end_pos[0] < 0:
        return True
        
    return False