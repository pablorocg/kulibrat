"""
Minimax strategy implementation for Kulibrat AI.
"""

import random
import time
from typing import Optional, Tuple

from src.core.game_state_cy import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy


class MinimaxStrategy(AIStrategy):
    """AI strategy using the minimax algorithm with optional alpha-beta pruning."""

    def __init__(self, max_depth: int = 15, use_alpha_beta: bool = True):
        """
        Initialize the minimax strategy.

        Args:
            max_depth: Maximum search depth
            use_alpha_beta: Whether to use alpha-beta pruning
        """
        self.max_depth = max_depth
        self.use_alpha_beta = use_alpha_beta
        self.nodes_evaluated = 0

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
        print(f"Minimax evaluated {self.nodes_evaluated} nodes in {elapsed:.2f}s")
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
        Heuristic: Evaluate the state for the given player.

        Args:
            state: Game state to evaluate
            player_color: Player to evaluate the state for

        Returns:
            Score representing how good the state is for the player
        """
        # Check if the game is over
        if state.is_game_over():
            winner = state.get_winner()
            if winner == player_color:
                return 1000.0  # Win
            elif winner is None:
                return 0.0  # Draw
            else:
                return -1000.0  # Loss

        opponent_color = player_color.opposite()

        # Calculate score based on piece positions and scores
        score = 0.0

        # Score difference
        score += 100.0 * (state.scores[player_color] - state.scores[opponent_color])

        # Piece advantage
        pieces_on_board_player = (
            4 - state.pieces_off_board[player_color] + state.scores[player_color]
        )
        pieces_on_board_opponent = (
            4 - state.pieces_off_board[opponent_color] + state.scores[opponent_color]
        )
        score += 10.0 * (pieces_on_board_player - pieces_on_board_opponent)

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

        score += 5.0 * (player_advancement - opponent_advancement)

        # Number of valid moves (mobility)
        current_player = state.current_player
        state.current_player = player_color
        player_moves = len(state.get_valid_moves())
        state.current_player = opponent_color
        opponent_moves = len(state.get_valid_moves())
        state.current_player = current_player  # Restore original current player

        score += 2.0 * (player_moves - opponent_moves)

        return score
