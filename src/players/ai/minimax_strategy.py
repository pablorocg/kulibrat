"""
Minimax strategy implementation for Kulibrat AI.
"""

import random
import time
from typing import Optional, Tuple, Dict, Any

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy
from src.players.ai.heuristics import HeuristicRegistry


class MinimaxStrategy(AIStrategy):
    """AI strategy using the minimax algorithm with optional alpha-beta pruning."""

    def __init__(
        self, 
        max_depth: int = 5, 
        use_alpha_beta: bool = True,
        heuristic: str = "strategic"
    ):
        """
        Initialize the minimax strategy.

        Args:
            max_depth: Maximum search depth
            use_alpha_beta: Whether to use alpha-beta pruning
            heuristic: Name of the heuristic function to use
        """
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

    def select_move(
        self, game_state: GameState, player_color: PlayerColor
    ) -> Optional[Move]:
        """
        Select the best move using minimax.

        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move

        Returns:
            The best move according to minimax
        """
        start_time = time.time()
        self.nodes_evaluated = 0

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None

        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]

        best_score = float("-inf")
        best_moves = []

        for move in valid_moves:
            # Apply the move to a copy of the state
            next_state = game_state.copy()
            next_state.apply_move(move)

            # Run minimax
            if self.use_alpha_beta:
                score, _ = self._alpha_beta(
                    state=next_state,
                    depth=self.max_depth - 1,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    current_player=player_color.opposite(),
                    maximizing_player=player_color,
                )
            else:
                score, _ = self._minimax(
                    state=next_state,
                    depth=self.max_depth - 1,
                    current_player=player_color.opposite(),
                    maximizing_player=player_color,
                )

            # Track best moves
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        # Choose randomly from equally good moves for variety
        selected_move = random.choice(best_moves)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Minimax evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s using {self.heuristic_name} heuristic")
        print(f"Selected move: {selected_move} with score: {best_score}")

        return selected_move

    def _minimax(
        self,
        state: GameState,
        depth: int,
        current_player: PlayerColor,
        maximizing_player: PlayerColor,
    ) -> Tuple[float, Optional[Move]]:
        """
        Minimax algorithm without alpha-beta pruning.

        Args:
            state: Current game state
            depth: Remaining search depth
            current_player: Player making the move at this level
            maximizing_player: The AI player we're finding the best move for

        Returns:
            Tuple of (score, move)
        """
        self.nodes_evaluated += 1

        # Terminal conditions
        if state.is_game_over() or depth == 0:
            return self._evaluate(state, maximizing_player), None

        valid_moves = state.get_valid_moves()

        # No valid moves means the other player gets an extra turn
        if not valid_moves:
            return self._minimax(
                state, depth - 1, current_player.opposite(), maximizing_player
            )

        is_maximizing = current_player == maximizing_player
        best_score = float("-inf") if is_maximizing else float("inf")
        best_move = None

        for move in valid_moves:
            # Apply the move to a copy of the state
            next_state = state.copy()
            next_state.apply_move(move)

            # Recursively evaluate
            score, _ = self._minimax(
                next_state, depth - 1, next_state.current_player, maximizing_player
            )

            # Update best score
            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_score, best_move

    def _alpha_beta(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        current_player: PlayerColor,
        maximizing_player: PlayerColor,
    ) -> Tuple[float, Optional[Move]]:
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            current_player: Player making the move at this level
            maximizing_player: The AI player we're finding the best move for

        Returns:
            Tuple of (score, move)
        """
        self.nodes_evaluated += 1

        # Terminal conditions
        if state.is_game_over() or depth == 0:
            return self._evaluate(state, maximizing_player), None

        valid_moves = state.get_valid_moves()

        # No valid moves means the other player gets an extra turn
        if not valid_moves:
            return self._alpha_beta(
                state,
                depth - 1,
                alpha,
                beta,
                current_player.opposite(),
                maximizing_player,
            )

        is_maximizing = current_player == maximizing_player
        best_move = None

        if is_maximizing:
            best_score = float("-inf")

            for move in valid_moves:
                # Apply the move to a copy of the state
                next_state = state.copy()
                next_state.apply_move(move)

                # Recursively evaluate
                score, _ = self._alpha_beta(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    next_state.current_player,
                    maximizing_player,
                )

                # Update best score
                if score > best_score:
                    best_score = score
                    best_move = move

                # Update alpha
                alpha = max(alpha, best_score)

                # Alpha-beta pruning
                if beta <= alpha:
                    break

            return best_score, best_move
        else:
            best_score = float("inf")

            for move in valid_moves:
                # Apply the move to a copy of the state
                next_state = state.copy()
                next_state.apply_move(move)

                # Recursively evaluate
                score, _ = self._alpha_beta(
                    next_state,
                    depth - 1,
                    alpha,
                    beta,
                    next_state.current_player,
                    maximizing_player,
                )

                # Update best score
                if score < best_score:
                    best_score = score
                    best_move = move

                # Update beta
                beta = min(beta, best_score)

                # Alpha-beta pruning
                if beta <= alpha:
                    break

            return best_score, best_move

    def _evaluate(self, state: GameState, player_color: PlayerColor) -> float:
        """
        Evaluate the state using the configured heuristic function.

        Args:
            state: Game state to evaluate
            player_color: Player to evaluate the state for

        Returns:
            Score representing how good the state is for the player
        """
        return self.heuristic_func(state, player_color)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the minimax strategy.

        Returns:
            Dictionary containing strategy statistics
        """
        return {
            "algorithm": "Minimax" + (" with alpha-beta" if self.use_alpha_beta else ""),
            "depth": self.max_depth,
            "heuristic": self.heuristic_name,
            "nodes_evaluated": self.nodes_evaluated
        }