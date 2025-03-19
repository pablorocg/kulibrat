"""
Enhanced Minimax strategy implementation for Kulibrat AI with
advanced optimizations including transposition tables and move ordering.
"""

import random
import time
from typing import Any, Dict, Optional, Tuple

from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.core.player_color import PlayerColor
from src.players.minimax_player.heuristics import HeuristicRegistry
from src.players.minimax_player.move_ordering import MoveOrdering
from src.players.minimax_player.transposition_table import TranspositionTable
from src.players.minimax_player.zobrist_hashing import ZobristHashing
from src.players.player import Player


class MinimaxPlayer(Player):
    """
    Enhanced AI strategy using minimax algorithm with alpha-beta pruning,
    transposition tables, and move ordering.
    """

    def __init__(
        self,
        color: PlayerColor,
        name: str = "Minimax Player",
        max_depth: int = 5,
        use_alpha_beta: bool = True,
        heuristic: str = "strategic",
        tt_size: int = 1000000,
    ):
        """
        Initialize the minimax strategy.

        Args:
            color: The player's color.
            name: The player's name.
            max_depth: Maximum search depth.
            use_alpha_beta: Whether to use alpha-beta pruning.
            heuristic: Name of the heuristic function to use.
            tt_size: Size of the transposition table.
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
            print(
                f"Warning: Heuristic '{heuristic}' not found, using 'strategic' instead"
            )
            self.heuristic_name = "strategic"
            self.heuristic_func = HeuristicRegistry.get("strategic")

        # Initialize transposition table, Zobrist hashing and move ordering
        self.tt = TranspositionTable(max_size=tt_size)
        self.zobrist = ZobristHashing()
        self.move_ordering = MoveOrdering()

        # Game phase detection parameters
        self.early_game_threshold = 15  # Turns
        self.endgame_threshold = 3  # Points to target

        # Statistics and game state tracking
        self.stats = {
            "nodes_evaluated": 0,
            "tt_hits": 0,
            "tt_stores": 0,
            "cutoffs": 0,
            "max_depth_reached": 0,
            "time_taken": 0,
            "pattern_detections": 0,
            "adaptive_depth_adjustments": 0,
        }
        self.turn_counter = 0
        self.previous_scores = {PlayerColor.BLACK: 0, PlayerColor.RED: 0}

    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Select the best move using minimax with alpha-beta pruning and
        transposition tables.

        Args:
            game_state: Current state of the game

        Returns:
            The best move according to minimax, or None if no move is possible.
        """
        start_time = time.time()

        # Reset statistics for this move
        self.stats = {
            "nodes_evaluated": 0,
            "tt_hits": 0,
            "tt_stores": 0,
            "cutoffs": 0,
            "max_depth_reached": 0,
            "time_taken": 0,
            "pattern_detections": 0,
            "adaptive_depth_adjustments": 0,
        }

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None

        if len(valid_moves) == 1:
            return valid_moves[0]

        current_hash = self.zobrist.compute_hash(game_state)
        tt_entry = self.tt.lookup(current_hash)
        tt_move = tt_entry.get("best_move") if tt_entry else None

        self.turn_counter += 1
        
        self.previous_scores = game_state.scores.copy()

        base_depth = self.max_depth
        adjusted_depth = self.max_depth
        if adjusted_depth != base_depth:
            self.stats["adaptive_depth_adjustments"] += 1

        ordered_moves = self.move_ordering.order_moves(
            valid_moves, tt_move, state=game_state
        )
        best_score = float("-inf")
        best_moves = []
        final_depth = adjusted_depth

        # Iterative deepening search
        for current_depth in range(2, adjusted_depth + 1):
            current_best_score = float("-inf")
            current_best_moves = []
            conservatism = 0
            if game_state.scores[self.color] > game_state.scores[self.color.opposite()]:
                conservatism = 5 * (
                    game_state.scores[self.color]
                    - game_state.scores[self.color.opposite()]
                )
            alpha = best_score - 50 if best_score != float("-inf") else float("-inf")
            beta = best_score + 50 if best_score != float("-inf") else float("inf")
            window_failed = False

            while not window_failed:
                window_failed = True
                for move in ordered_moves:
                    next_state = game_state.copy()
                    if not next_state.apply_move(move):
                        continue
                    next_hash = self.zobrist.compute_hash(next_state)
                    score, _ = self._alpha_beta_with_memory(
                        state=next_state,
                        hash_value=next_hash,
                        depth=current_depth - 1,
                        alpha=-beta,
                        beta=-alpha,
                        maximizing_player=False,
                        original_player=self.color,
                    )
                    score = -score
                    if score == 0 and conservatism > 0:
                        score += conservatism
                    if score <= alpha or score >= beta:
                        if score <= alpha:
                            alpha = float("-inf")
                        if score >= beta:
                            beta = float("inf")
                        window_failed = False
                        break
                    if score > current_best_score:
                        current_best_score = score
                        current_best_moves = [move]
                    elif score == current_best_score:
                        current_best_moves.append(move)

            if window_failed:
                best_score = current_best_score
                best_moves = current_best_moves
                ordered_moves = self.move_ordering.order_moves(
                    valid_moves, best_moves[0] if best_moves else None, state=game_state
                )
                final_depth = current_depth
                if best_score > 900:
                    break
                if time.time() - start_time > 0.9:
                    break

        # Select move using strategy-specific logic
        selected_move = None
        if best_moves:
            scoring_moves = [
                m for m in best_moves if self._is_scoring_move(m, game_state)
            ]
            if scoring_moves:
                selected_move = random.choice(scoring_moves)
            elif len(best_moves) > 1:
                jump_moves = [m for m in best_moves if m.move_type == MoveType.JUMP]
                attack_moves = [m for m in best_moves if m.move_type == MoveType.ATTACK]
                if jump_moves and random.random() < 0.7:
                    selected_move = random.choice(jump_moves)
                elif attack_moves and random.random() < 0.6:
                    selected_move = random.choice(attack_moves)
                else:
                    selected_move = random.choice(best_moves)
            else:
                selected_move = best_moves[0]
        else:
            selected_move = random.choice(valid_moves) if valid_moves else None

        if selected_move:
            self.tt.store(
                current_hash,
                final_depth,
                best_score,
                TranspositionTable.EXACT,
                selected_move,
            )

        elapsed = time.time() - start_time
        self.stats["time_taken"] = elapsed
        self.stats["tt_hits"] = self.tt.hits
        self.stats["tt_stores"] = self.tt.stores

        print(
            f"Enhanced Minimax depth {final_depth} evaluated {self.stats['nodes_evaluated']} nodes in {elapsed:.2f}s"
        )
        print(f"TT hits: {self.stats['tt_hits']}, TT stores: {self.stats['tt_stores']}")
        print(
            f"Cutoffs: {self.stats['cutoffs']}, Pattern detections: {self.stats['pattern_detections']}"
        )
        print(f"Adaptive depth adjustments: {self.stats['adaptive_depth_adjustments']}")
        if selected_move:
            print(f"Selected move: {selected_move} with score: {best_score}")
        else:
            print("No move selected!")

        return selected_move

    def _is_critical_position(
        self, state: GameState, player_color: PlayerColor
    ) -> bool:
        """
        Check if the current position is critical (requires deeper search).

        Args:
            state: Current game state.
            player_color: Player to check for.

        Returns:
            True if the position is critical.
        """
        opponent_color = player_color.opposite()
        scoring_row_player = (
            0 if player_color == PlayerColor.RED else state.BOARD_ROWS - 1
        )
        scoring_row_opponent = (
            0 if opponent_color == PlayerColor.RED else state.BOARD_ROWS - 1
        )

        pieces_about_to_score = 0
        opponent_pieces_about_to_score = 0

        for col in range(state.BOARD_COLS):
            if state.board[scoring_row_opponent, col] == player_color.value:
                pieces_about_to_score += 1
            if state.board[scoring_row_player, col] == opponent_color.value:
                opponent_pieces_about_to_score += 1

        if pieces_about_to_score > 0 or opponent_pieces_about_to_score > 0:
            return True

        if (
            abs(state.scores[player_color] - state.scores[opponent_color]) <= 1
            and max(state.scores[player_color], state.scores[opponent_color]) >= 3
        ):
            return True

        player_pieces = sum(
            1
            for row in range(state.BOARD_ROWS)
            for col in range(state.BOARD_COLS)
            if state.board[row, col] == player_color.value
        )
        opponent_pieces = sum(
            1
            for row in range(state.BOARD_ROWS)
            for col in range(state.BOARD_COLS)
            if state.board[row, col] == opponent_color.value
        )
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
            state: Current game state.
            hash_value: Zobrist hash of the state.
            depth: Remaining search depth.
            alpha: Alpha value for pruning.
            beta: Beta value for pruning.
            maximizing_player: Whether this is a maximizing node.
            original_player: The player for whom the best move is sought.

        Returns:
            Tuple of (score, move).
        """
        self.stats["nodes_evaluated"] += 1

        if self._is_drawish_position(state) and depth < 3:
            return 0.0, None

        original_alpha = alpha
        original_beta = beta

        tt_entry = self.tt.lookup(hash_value)
        if tt_entry and tt_entry["depth"] >= depth:
            tt_value = tt_entry["value"]
            tt_type = tt_entry["type"]
            if tt_type == TranspositionTable.EXACT:
                return tt_value, tt_entry["best_move"]
            elif tt_type == TranspositionTable.ALPHA and tt_value <= alpha:
                return alpha, tt_entry["best_move"]
            elif tt_type == TranspositionTable.BETA and tt_value >= beta:
                return beta, tt_entry["best_move"]

        if state.is_game_over() or depth == 0:
            eval_value = self._evaluate_state(state, original_player)
            self.tt.store(hash_value, depth, eval_value, TranspositionTable.EXACT)
            return eval_value, None

        if self.max_depth - depth > self.stats["max_depth_reached"]:
            self.stats["max_depth_reached"] = self.max_depth - depth

        valid_moves = state.get_valid_moves()
        if not valid_moves:
            state.current_player = state.current_player.opposite()
            new_hash = self.zobrist.compute_hash(state)
            return self._alpha_beta_with_memory(
                state,
                new_hash,
                depth - 1,
                alpha,
                beta,
                not maximizing_player,
                original_player,
            )

        if self._detect_winning_pattern(
            state, original_player if maximizing_player else original_player.opposite()
        ):
            self.stats["pattern_detections"] += 1
            return (900.0 if maximizing_player else -900.0), None

        best_move = None
        tt_move = tt_entry.get("best_move") if tt_entry else None
        ordered_moves = self.move_ordering.order_moves(
            valid_moves, tt_move, depth, state
        )

        if maximizing_player:
            best_value = float("-inf")
            for move in ordered_moves:
                next_state = state.copy()
                if not next_state.apply_move(move):
                    continue
                next_hash = self.zobrist.compute_hash(next_state)
                value, _ = self._alpha_beta_with_memory(
                    next_state,
                    next_hash,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    original_player,
                )
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    self.stats["cutoffs"] += 1
                    self.move_ordering.update_history(move, depth)
                    self.move_ordering.update_killer_move(move, depth)
                    break
        else:
            best_value = float("inf")
            for move in ordered_moves:
                next_state = state.copy()
                if not next_state.apply_move(move):
                    continue
                next_hash = self.zobrist.compute_hash(next_state)
                value, _ = self._alpha_beta_with_memory(
                    next_state, next_hash, depth - 1, alpha, beta, True, original_player
                )
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if beta <= alpha:
                    self.stats["cutoffs"] += 1
                    self.move_ordering.update_history(move, depth)
                    self.move_ordering.update_killer_move(move, depth)
                    break

        tt_flag = TranspositionTable.EXACT
        if best_value <= original_alpha:
            tt_flag = TranspositionTable.ALPHA
        elif best_value >= original_beta:
            tt_flag = TranspositionTable.BETA

        self.tt.store(hash_value, depth, best_value, tt_flag, best_move)
        return best_value, best_move

    def _evaluate_state(self, state: GameState, player_color: PlayerColor) -> float:
        """
        Evaluate the state using the configured heuristic function.

        Args:
            state: Game state to evaluate.
            player_color: Player to evaluate for.

        Returns:
            Evaluation score.
        """
        if state.is_game_over():
            winner = state.get_winner()
            if winner == player_color:
                return 1000.0
            elif winner is None:
                return 0.0
            else:
                return -1000.0
        return self.heuristic_func(state, player_color)

    def _detect_winning_pattern(
        self, state: GameState, player_color: PlayerColor
    ) -> bool:
        """
        Detect potential winning patterns for the given player.

        Args:
            state: Current game state.
            player_color: Player to check for.

        Returns:
            True if a winning pattern is detected.
        """
        opponent_color = player_color.opposite()
        score_advantage = state.scores[player_color] - state.scores[opponent_color]
        if score_advantage >= 2:
            return True

        player_on_opponent_half = 0
        opponent_on_player_half = 0
        player_in_scoring_position = 0

        player_scoring_row = (
            state.BOARD_ROWS - 1 if player_color == PlayerColor.BLACK else 0
        )
        opponent_scoring_row = (
            state.BOARD_ROWS - 1 if opponent_color == PlayerColor.BLACK else 0
        )

        for row in range(state.BOARD_ROWS):
            for col in range(state.BOARD_COLS):
                piece = state.board[row, col]
                if piece == player_color.value:
                    if (
                        player_color == PlayerColor.BLACK
                        and row >= state.BOARD_ROWS // 2
                    ) or (
                        player_color == PlayerColor.RED and row < state.BOARD_ROWS // 2
                    ):
                        player_on_opponent_half += 1
                    if row == opponent_scoring_row:
                        player_in_scoring_position += 1
                elif piece == opponent_color.value:
                    if (
                        opponent_color == PlayerColor.BLACK
                        and row >= state.BOARD_ROWS // 2
                    ) or (
                        opponent_color == PlayerColor.RED
                        and row < state.BOARD_ROWS // 2
                    ):
                        opponent_on_player_half += 1

        if player_on_opponent_half >= 2 and player_in_scoring_position > 0:
            return True
        if (
            player_on_opponent_half >= 3
            and player_on_opponent_half >= opponent_on_player_half + 2
        ):
            return True
        if (
            state.scores[player_color] == state.target_score - 1
            and player_in_scoring_position > 0
        ):
            return True
        return False

    def _is_drawish_position(self, state: GameState) -> bool:
        """
        Check if the position is likely to lead to a draw.

        Args:
            state: Current game state.

        Returns:
            True if the position is likely a draw.
        """
        black_immobile = 0
        red_immobile = 0
        for row in range(state.BOARD_ROWS):
            for col in range(state.BOARD_COLS):
                if state.board[row, col] == PlayerColor.BLACK.value:
                    if row < state.BOARD_ROWS - 1:
                        if (
                            (col > 0 and state.board[row + 1, col - 1] != 0)
                            and (
                                col < state.BOARD_COLS - 1
                                and state.board[row + 1, col + 1] != 0
                            )
                            and state.board[row + 1, col] != 0
                        ):
                            black_immobile += 1
                elif state.board[row, col] == PlayerColor.RED.value:
                    if row > 0:
                        if (
                            (col > 0 and state.board[row - 1, col - 1] != 0)
                            and (
                                col < state.BOARD_COLS - 1
                                and state.board[row - 1, col + 1] != 0
                            )
                            and state.board[row - 1, col] != 0
                        ):
                            red_immobile += 1
        black_pieces = 4 - state.pieces_off_board[PlayerColor.BLACK]
        red_pieces = 4 - state.pieces_off_board[PlayerColor.RED]
        black_stuck_ratio = black_immobile / black_pieces if black_pieces > 0 else 0
        red_stuck_ratio = red_immobile / red_pieces if red_pieces > 0 else 0
        if (
            black_stuck_ratio >= 0.5
            and red_stuck_ratio >= 0.5
            and black_pieces + red_pieces >= 6
        ):
            return True
        return False

    def _is_scoring_move(self, move: Move, state: GameState) -> bool:
        """
        Check if a move results in scoring a point.

        Args:
            move: Move to check.
            state: Current game state.

        Returns:
            True if the move scores a point.
        """
        next_state = state.copy()
        next_state.apply_move(move)
        return (
            next_state.scores[state.current_player] > state.scores[state.current_player]
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the minimax strategy.

        Returns:
            Dictionary containing strategy statistics.
        """
        return {
            "algorithm": "Enhanced Minimax with Alpha-Beta",
            "depth": self.max_depth,
            "heuristic": self.heuristic_name,
            "nodes_evaluated": self.stats["nodes_evaluated"],
            "tt_hits": self.stats["tt_hits"],
            "tt_stores": self.stats["tt_stores"],
            "cutoffs": self.stats["cutoffs"],
            "max_depth": self.stats["max_depth_reached"],
            "time_taken": self.stats["time_taken"],
            "pattern_detections": self.stats["pattern_detections"],
            "adaptive_depth_adjustments": self.stats["adaptive_depth_adjustments"],
        }
