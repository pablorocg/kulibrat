"""
Enhanced Minimax strategy implementation for Kulibrat AI with
advanced optimizations including transposition tables and move ordering.
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core import GameState, Move, MoveType, PlayerColor
from src.players.player import Player
from players.minimax_player.heuristics import HeuristicRegistry
from src.players.minimax_player.transposition_table import TranspositionTable
from src.players.minimax_player.move_ordering import MoveOrdering
from src.players.minimax_player.zobrist_hashing import ZobristHashing


class MinimaxPlayer(Player):
    """
    Enhanced AI strategy using minimax algorithm with alpha-beta pruning, 
    transposition tables, and move ordering.
    """

    def __init__(
        self, 
        max_depth: int = 5, 
        use_alpha_beta: bool = True,
        heuristic: str = "strategic",
        tt_size: int = 1000000, color, name
    ):
        """
        Initialize the minimax strategy.

        Args:
            max_depth: Maximum search depth
            use_alpha_beta: Whether to use alpha-beta pruning
            heuristic: Name of the heuristic function to use
            tt_size: Size of the transposition table
        """
        super().__init__(color, name)

        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.nodes_evaluated = 0
        
        # Set the heuristic function
        try:
            self.heuristic_name = heuristic
            self.heuristic_func = HeuristicRegistry.get(heuristic)
        except ValueError:
            # Fall back to strategic heuristic if specified one is not found
            print(f"Warning: Heuristic '{heuristic}' not found, using 'strategic' instead")
            self.heuristic_name = "strategic"
            self.heuristic_func = HeuristicRegistry.get("strategic")
        
        # Initialize transposition table
        self.tt = TranspositionTable(max_size=tt_size)
        
        # Initialize Zobrist hashing
        self.zobrist = ZobristHashing()
        
        # Initialize move ordering
        self.move_ordering = MoveOrdering()
        
        # Game phase detection
        self.early_game_threshold = 15  # Turns
        self.endgame_threshold = 3      # Points to target
        
        # Statistics
        self.stats = {
            'nodes_evaluated': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'cutoffs': 0,
            'max_depth_reached': 0,
            'time_taken': 0,
            'pattern_detections': 0,
            'adaptive_depth_adjustments': 0,
        }
        
        # Game state tracking
        self.turn_counter = 0
        self.previous_scores = {PlayerColor.BLACK: 0, PlayerColor.RED: 0}

    def select_move(
        self, game_state: GameState, player_color: PlayerColor
    ) -> Optional[Move]:
        """
        Select the best move using minimax with alpha-beta pruning and
        transposition tables.

        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move

        Returns:
            The best move according to minimax
        """
        start_time = time.time()
        
        # Reset statistics
        self.stats = {
            'nodes_evaluated': 0,
            'tt_hits': 0,
            'tt_stores': 0,
            'cutoffs': 0,
            'max_depth_reached': 0,
            'time_taken': 0,
            'pattern_detections': 0,
            'adaptive_depth_adjustments': 0,
        }

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None

        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Calculate Zobrist hash for current position
        current_hash = self.zobrist.compute_hash(game_state)
        
        # Check transposition table for best move suggestion
        tt_entry = self.tt.lookup(current_hash)
        tt_move = tt_entry.get('best_move') if tt_entry else None
        
        # Track turn counter and update score history
        self.turn_counter += 1
        # Detect if scores changed
        score_changed = (game_state.scores[PlayerColor.BLACK] != self.previous_scores[PlayerColor.BLACK] or
                         game_state.scores[PlayerColor.RED] != self.previous_scores[PlayerColor.RED])
        self.previous_scores = game_state.scores.copy()
        
        # Determine game phase and adjust max depth accordingly (adaptive depth)
        base_depth = self.max_depth
        adjusted_depth = self._calculate_adaptive_depth(game_state, player_color, score_changed)
        if adjusted_depth != base_depth:
            self.stats['adaptive_depth_adjustments'] += 1
            
        # Order moves (tt_move will be tried first)
        ordered_moves = self.move_ordering.order_moves(valid_moves, tt_move, state=game_state)

        best_score = float("-inf")
        best_moves = []

        # Iterative deepening search
        final_depth = adjusted_depth
        for current_depth in range(2, adjusted_depth + 1):
            current_best_score = float("-inf")
            current_best_moves = []
            
            # Conservative strategy: Favor moves that lead to draws when ahead
            conservatism = 0
            if game_state.scores[player_color] > game_state.scores[player_color.opposite()]:
                conservatism = 5 * (game_state.scores[player_color] - game_state.scores[player_color.opposite()])
            
            # Aspiration windows (start with a narrow window around previous best)
            alpha = best_score - 50 if best_score != float("-inf") else float("-inf")
            beta = best_score + 50 if best_score != float("-inf") else float("inf")
            window_failed = False
            
            while not window_failed:
                window_failed = True  # Assume window will be sufficient
                
                for move in ordered_moves:
                    # Apply the move to a copy of the state
                    next_state = game_state.copy()
                    if not next_state.apply_move(move):
                        continue  # Skip invalid moves
                        
                    # Calculate hash for this position
                    next_hash = self.zobrist.compute_hash(next_state)

                    # Run alpha-beta search
                    score = self._alpha_beta_with_memory(
                        state=next_state,
                        hash_value=next_hash,
                        depth=current_depth - 1,
                        alpha=-beta,  # Negamax formulation
                        beta=-alpha,
                        maximizing_player=False,
                        original_player=player_color,
                    )[0]
                    score = -score  # Negate for Negamax formulation
                    
                    # Apply conservatism - slightly prefer draws when ahead
                    if score == 0 and conservatism > 0:
                        score += conservatism
                    
                    # Check if we need to re-search due to window failure
                    if score <= alpha or score >= beta:
                        # Re-search with wider window
                        if score <= alpha:
                            alpha = float("-inf")
                        if score >= beta:
                            beta = float("inf")
                        window_failed = False
                        break  # Break out and try with new window
                    
                    # Track best moves
                    if score > current_best_score:
                        current_best_score = score
                        current_best_moves = [move]
                    elif score == current_best_score:
                        current_best_moves.append(move)
            
            # If search completed successfully, update best moves
            if window_failed:
                best_score = current_best_score
                best_moves = current_best_moves
                
                # Update move ordering for next iteration
                ordered_moves = self.move_ordering.order_moves(
                    valid_moves, 
                    best_moves[0] if best_moves else None,
                    state=game_state
                )
                
                # Update final depth
                final_depth = current_depth
                
                # Early termination if we found a winning move
                if best_score > 900:
                    break
                    
                # Time check - don't start next iteration if we're running out of time
                elapsed = time.time() - start_time
                if elapsed > 0.9:  # 0.9 second time limit
                    break

        # Choose the best move with some strategy-specific logic
        selected_move = None
        if best_moves:
            # Prioritize certain types of moves when scores are equal
            scoring_moves = [m for m in best_moves if self._is_scoring_move(m, game_state)]
            if scoring_moves:
                # Always take a scoring move if available
                selected_move = random.choice(scoring_moves)
            elif len(best_moves) > 1:
                # Prioritize jump and attack moves over other types
                jump_moves = [m for m in best_moves if m.move_type == MoveType.JUMP]
                attack_moves = [m for m in best_moves if m.move_type == MoveType.ATTACK]
                
                if jump_moves and random.random() < 0.7:  # 70% chance to choose a jump
                    selected_move = random.choice(jump_moves)
                elif attack_moves and random.random() < 0.6:  # 60% chance to choose an attack
                    selected_move = random.choice(attack_moves)
                else:
                    selected_move = random.choice(best_moves)
            else:
                selected_move = best_moves[0]
        else:
            # Fallback if no best moves were found
            selected_move = random.choice(valid_moves) if valid_moves else None

        # Store result in transposition table
        if selected_move:
            self.tt.store(
                current_hash, 
                final_depth, 
                best_score, 
                TranspositionTable.EXACT,
                selected_move
            )

        end_time = time.time()
        elapsed = end_time - start_time
        self.stats['time_taken'] = elapsed
        
        # Update statistics from transposition table
        self.stats['tt_hits'] = self.tt.hits
        self.stats['tt_stores'] = self.tt.stores
        
        print(f"Enhanced Minimax depth {final_depth} evaluated {self.stats['nodes_evaluated']} nodes in {elapsed:.2f}s")
        print(f"TT hits: {self.stats['tt_hits']}, TT stores: {self.stats['tt_stores']}")
        print(f"Cutoffs: {self.stats['cutoffs']}, Pattern detections: {self.stats['pattern_detections']}")
        print(f"Adaptive depth adjustments: {self.stats['adaptive_depth_adjustments']}")
        
        if selected_move:
            print(f"Selected move: {selected_move} with score: {best_score}")
        else:
            print("No move selected!")

        return selected_move

    def _calculate_adaptive_depth(self, state: GameState, player_color: PlayerColor, score_changed: bool) -> int:
        """
        Calculate adaptive search depth based on game state.
        
        Args:
            state: Current game state
            player_color: Player making the move
            score_changed: Whether the score has changed since the last move
            
        Returns:
            Adjusted search depth
        """
        return self.max_depth
        
        
        
    def _is_critical_position(self, state: GameState, player_color: PlayerColor) -> bool:
        """
        Check if current position is critical (requires deeper search).
        
        Args:
            state: Current game state
            player_color: Player to check for
            
        Returns:
            True if position is critical
        """
        opponent_color = player_color.opposite()
        
        # Player or opponent has a piece about to score
        scoring_row_player = 0 if player_color == PlayerColor.RED else state.BOARD_ROWS - 1
        scoring_row_opponent = 0 if opponent_color == PlayerColor.RED else state.BOARD_ROWS - 1
        
        pieces_about_to_score = 0
        opponent_pieces_about_to_score = 0
        
        for col in range(state.BOARD_COLS):
            # Check if player has pieces about to score
            if state.board[scoring_row_opponent, col] == player_color.value:
                pieces_about_to_score += 1
                
            # Check if opponent has pieces about to score
            if state.board[scoring_row_player, col] == opponent_color.value:
                opponent_pieces_about_to_score += 1
                
        if pieces_about_to_score > 0 or opponent_pieces_about_to_score > 0:
            return True
            
        # Close score with high scores
        if (abs(state.scores[player_color] - state.scores[opponent_color]) <= 1 and 
            max(state.scores[player_color], state.scores[opponent_color]) >= 3):
            return True
            
        # Many pieces on the board (complex position)
        player_pieces = sum(1 for row in range(state.BOARD_ROWS) for col in range(state.BOARD_COLS) 
                            if state.board[row, col] == player_color.value)
        opponent_pieces = sum(1 for row in range(state.BOARD_ROWS) for col in range(state.BOARD_COLS)
                              if state.board[row, col] == opponent_color.value)
                              
        if player_pieces + opponent_pieces >= 6:
            return True
            
        return False

    def _alpha_beta_with_memory(
        self,
        state: GameState,
        hash_value: int,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        original_player: PlayerColor,
    ) -> Tuple[float, Optional[Move]]:
        """
        Alpha-Beta pruning with transposition table and move ordering.

        Args:
            state: Current game state
            hash_value: Zobrist hash of the state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: Whether this is a maximizing node
            original_player: The player we're finding the best move for

        Returns:
            Tuple of (score, move)
        """
        self.stats['nodes_evaluated'] += 1
        
        # Check if this is a draw-like position with minimal progress
        if self._is_drawish_position(state) and depth < 3:
            return 0.0, None
        
        original_alpha = alpha
        original_beta = beta
        
        # Check for transposition table hit
        tt_entry = self.tt.lookup(hash_value)
        if tt_entry and tt_entry['depth'] >= depth:
            tt_value = tt_entry['value']
            tt_type = tt_entry['type']
            
            if tt_type == TranspositionTable.EXACT:
                return tt_value, tt_entry['best_move']
            elif tt_type == TranspositionTable.ALPHA and tt_value <= alpha:
                return alpha, tt_entry['best_move']
            elif tt_type == TranspositionTable.BETA and tt_value >= beta:
                return beta, tt_entry['best_move']

        # Terminal conditions
        if state.is_game_over() or depth == 0:
            eval_value = self._evaluate_state(state, original_player)
            self.tt.store(hash_value, depth, eval_value, TranspositionTable.EXACT)
            return eval_value, None
            
        # Track max depth reached for statistics
        if self.max_depth - depth > self.stats['max_depth_reached']:
            self.stats['max_depth_reached'] = self.max_depth - depth
            
        current_player = state.current_player
        valid_moves = state.get_valid_moves()

        # No valid moves means the other player gets an extra turn
        if not valid_moves:
            state.current_player = state.current_player.opposite()
            # Recalculate hash for the new state
            new_hash = self.zobrist.compute_hash(state)
            return self._alpha_beta_with_memory(
                state, new_hash, depth - 1, alpha, beta, not maximizing_player, original_player
            )

        # Detect potential cycles and winning patterns
        if self._detect_winning_pattern(state, original_player if maximizing_player else original_player.opposite()):
            self.stats['pattern_detections'] += 1
            return 900.0 if maximizing_player else -900.0, None  # Large value for winning patterns
        
        best_move = None
        
        # Get best move from transposition table if available
        tt_move = tt_entry.get('best_move') if tt_entry else None
        
        # Order moves to potentially improve cutoffs
        ordered_moves = self.move_ordering.order_moves(valid_moves, tt_move, depth, state)
        
        if maximizing_player:
            best_value = float("-inf")

            for move in ordered_moves:
                # Apply the move to a copy of the state
                next_state = state.copy()
                if not next_state.apply_move(move):
                    continue  # Skip invalid moves
                    
                # Calculate hash for the new state
                next_hash = self.zobrist.compute_hash(next_state)
                
                # Recursively evaluate
                value, _ = self._alpha_beta_with_memory(
                    next_state, next_hash, depth - 1, alpha, beta, False, original_player
                )

                # Update best value and move
                if value > best_value:
                    best_value = value
                    best_move = move

                # Update alpha
                alpha = max(alpha, best_value)

                # Alpha-beta pruning
                if beta <= alpha:
                    self.stats['cutoffs'] += 1
                    self.move_ordering.update_history(move, depth)
                    self.move_ordering.update_killer_move(move, depth)
                    break
        else:
            best_value = float("inf")

            for move in ordered_moves:
                # Apply the move to a copy of the state
                next_state = state.copy()
                if not next_state.apply_move(move):
                    continue  # Skip invalid moves
                    
                # Calculate hash for the new state
                next_hash = self.zobrist.compute_hash(next_state)
                
                # Recursively evaluate
                value, _ = self._alpha_beta_with_memory(
                    next_state, next_hash, depth - 1, alpha, beta, True, original_player
                )

                # Update best value and move
                if value < best_value:
                    best_value = value
                    best_move = move

                # Update beta
                beta = min(beta, best_value)

                # Alpha-beta pruning
                if beta <= alpha:
                    self.stats['cutoffs'] += 1
                    self.move_ordering.update_history(move, depth)
                    self.move_ordering.update_killer_move(move, depth)
                    break

        # Determine node type for transposition table
        tt_flag = TranspositionTable.EXACT
        if best_value <= original_alpha:
            tt_flag = TranspositionTable.ALPHA
        elif best_value >= original_beta:
            tt_flag = TranspositionTable.BETA

        # Store result in transposition table
        self.tt.store(
            hash_value, 
            depth, 
            best_value, 
            tt_flag,
            best_move
        )

        return best_value, best_move
    
    def _evaluate_state(self, state: GameState, player_color: PlayerColor) -> float:
        """
        Evaluate the state using the configured heuristic function.
        
        Args:
            state: Game state to evaluate
            player_color: Player to evaluate for
            
        Returns:
            Evaluation score
        """
        if state.is_game_over():
            winner = state.get_winner()
            if winner == player_color:
                return 1000.0  # Win
            elif winner is None:
                return 0.0  # Draw
            else:
                return -1000.0  # Loss
                
        return self.heuristic_func(state, player_color)
    
    def _detect_winning_pattern(self, state: GameState, player_color: PlayerColor) -> bool:
        """
        Detect potential winning patterns for the given player.
        
        Args:
            state: Current game state
            player_color: Player to check for
            
        Returns:
            True if a winning pattern is detected
        """
        opponent_color = player_color.opposite()
        
        # Calculate player's score advantage
        score_advantage = state.scores[player_color] - state.scores[opponent_color]
        if score_advantage >= 2:
            # Game may already be won with significant score lead
            return True
            
        # Count pieces strategically positioned
        player_on_opponent_half = 0
        opponent_on_player_half = 0
        player_in_scoring_position = 0
        
        # Determine scoring row for each player
        player_scoring_row = state.BOARD_ROWS - 1 if player_color == PlayerColor.BLACK else 0
        opponent_scoring_row = state.BOARD_ROWS - 1 if opponent_color == PlayerColor.BLACK else 0
        
        for row in range(state.BOARD_ROWS):
            for col in range(state.BOARD_COLS):
                piece = state.board[row, col]
                
                # Check player piece positions
                if piece == player_color.value:
                    # Check if piece is in opponent's half
                    if (player_color == PlayerColor.BLACK and row >= state.BOARD_ROWS // 2) or \
                       (player_color == PlayerColor.RED and row < state.BOARD_ROWS // 2):
                        player_on_opponent_half += 1
                    
                    # Check for pieces about to score
                    if row == opponent_scoring_row:
                        player_in_scoring_position += 1
                
                # Check opponent piece positions
                elif piece == opponent_color.value:
                    # Check if opponent piece is in player's half
                    if (opponent_color == PlayerColor.BLACK and row >= state.BOARD_ROWS // 2) or \
                       (opponent_color == PlayerColor.RED and row < state.BOARD_ROWS // 2):
                        opponent_on_player_half += 1
                        
        # Classic winning pattern: multiple pieces on opponent's half with at least one in scoring position
        if player_on_opponent_half >= 2 and player_in_scoring_position > 0:
            return True
            
        # Strong positioning advantage
        if player_on_opponent_half >= 3 and player_on_opponent_half >= opponent_on_player_half + 2:
            return True
            
        # One move from winning by score
        if state.scores[player_color] == state.target_score - 1 and player_in_scoring_position > 0:
            return True
            
        return False
    
    def _is_drawish_position(self, state: GameState) -> bool:
        """
        Check if the position is likely to lead to a draw.
        
        Args:
            state: Current game state
            
        Returns:
            True if position is likely a draw
        """
        # Check if pieces are mostly stuck and not making progress
        black_immobile = 0
        red_immobile = 0
        
        for row in range(state.BOARD_ROWS):
            for col in range(state.BOARD_COLS):
                if state.board[row, col] == PlayerColor.BLACK.value:
                    # Check if BLACK pieces can move
                    if row < state.BOARD_ROWS - 1:
                        if (col > 0 and state.board[row+1, col-1] != 0) and \
                           (col < state.BOARD_COLS - 1 and state.board[row+1, col+1] != 0) and \
                           state.board[row+1, col] != 0:
                            black_immobile += 1
                            
                elif state.board[row, col] == PlayerColor.RED.value:
                    # Check if RED pieces can move
                    if row > 0:
                        if (col > 0 and state.board[row-1, col-1] != 0) and \
                           (col < state.BOARD_COLS - 1 and state.board[row-1, col+1] != 0) and \
                           state.board[row-1, col] != 0:
                            red_immobile += 1
                            
        # Count total pieces
        black_pieces = 4 - state.pieces_off_board[PlayerColor.BLACK]
        red_pieces = 4 - state.pieces_off_board[PlayerColor.RED]
        
        # Consider draw-like if many pieces are stuck
        black_stuck_ratio = black_immobile / black_pieces if black_pieces > 0 else 0
        red_stuck_ratio = red_immobile / red_pieces if red_pieces > 0 else 0
        
        if black_stuck_ratio >= 0.5 and red_stuck_ratio >= 0.5 and black_pieces + red_pieces >= 6:
            return True
            
        return False
        
    def _is_scoring_move(self, move: Move, state: GameState) -> bool:
        """
        Check if a move results in scoring a point.
        
        Args:
            move: Move to check
            state: Current game state
            
        Returns:
            True if the move scores a point
        """
        # Simulate the move
        next_state = state.copy()
        next_state.apply_move(move)
        
        # Check if score changed
        return next_state.scores[state.current_player] > state.scores[state.current_player]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the minimax strategy.

        Returns:
            Dictionary containing strategy statistics
        """
        return {
            "algorithm": "Enhanced Minimax with Alpha-Beta",
            "depth": self.max_depth,
            "heuristic": self.heuristic_name,
            "nodes_evaluated": self.stats['nodes_evaluated'],
            "tt_hits": self.stats['tt_hits'],
            "tt_stores": self.stats['tt_stores'],
            "cutoffs": self.stats['cutoffs'],
            "max_depth": self.stats['max_depth_reached'],
            "time_taken": self.stats['time_taken'],
            "pattern_detections": self.stats['pattern_detections'],
            "adaptive_depth_adjustments": self.stats['adaptive_depth_adjustments']
        }