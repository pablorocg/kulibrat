""" """

import logging
from datetime import datetime
from typing import Any, Dict

from src.core.game_state import GameState
from src.core.move_type import MoveType
from src.core.player_color import PlayerColor


class TournamentMatch:
    """Manages a single match between two players."""

    def __init__(
        self,
        player1_name: str,
        player1,
        player2_name: str,
        player2,
        target_score: int = 5,
        max_turns: int = 300,
    ):
        """
        Initialize tournament match configuration.

        Args:
            player1_name: Name of first player
            player1: First player object
            player2_name: Name of second player
            player2: Second player object
            target_score: Score needed to win
            max_turns: Maximum turns before draw
        """
        self.player1_name = player1_name
        self.player1 = player1
        self.player2_name = player2_name
        self.player2 = player2
        self.target_score = target_score
        self.max_turns = max_turns

        # Match result tracking - initialize move_types as dictionaries
        self.match_results = {
            "player1": player1_name,
            "player2": player2_name,
            "winner": None,
            "score_p1": 0,
            "score_p2": 0,
            "turns": 0,
            "total_time": 0,
            "player1_move_types": {m.name: 0 for m in MoveType},
            "player2_move_types": {m.name: 0 for m in MoveType},
            "player1_color": None,
            "player2_color": None,
        }

    def run_match(self, swap_colors: bool = False) -> Dict[str, Any]:
        """
        Execute a match between two players.

        Args:
            swap_colors: Whether to swap default colors

        Returns:
            Detailed match results
        """
        # Determine player colors
        if not swap_colors:
            black_player = self.player1
            red_player = self.player2
            black_player_name = self.player1_name
            red_player_name = self.player2_name
            self.match_results["player1_color"] = PlayerColor.BLACK.name
            self.match_results["player2_color"] = PlayerColor.RED.name
        else:
            black_player = self.player2
            red_player = self.player1
            black_player_name = self.player2_name
            red_player_name = self.player1_name
            self.match_results["player1_color"] = PlayerColor.RED.name
            self.match_results["player2_color"] = PlayerColor.BLACK.name

        # Set player colors
        black_player.color = PlayerColor.BLACK
        red_player.color = PlayerColor.RED

        # Initialize game state
        game_state = GameState(target_score=self.target_score)

        # Track match progress
        turns = 0
        start_time = datetime.now()

        # Record all moves for detailed analysis
        all_moves = []

        while not game_state.is_game_over():  # and turns < self.max_turns
            # Determine current player
            current_color = game_state.current_player
            current_player = (
                black_player if current_color == PlayerColor.BLACK else red_player
            )
            current_player_name = (
                black_player_name
                if current_color == PlayerColor.BLACK
                else red_player_name
            )

            # Get player move
            move = current_player.get_move(game_state)

            if not move:
                # Skip turn if no move possible
                game_state.current_player = game_state.current_player.opposite()
                continue

            # Apply move and track statistics
            if game_state.apply_move(move):
                # Track move types - ensure we're working with a dictionary
                move_types_key = (
                    "player1_move_types"
                    if current_player_name == self.player1_name
                    else "player2_move_types"
                )
                move_types_dict = self.match_results[move_types_key]

                # Explicitly check that we have a dictionary
                if not isinstance(move_types_dict, dict):
                    logging.warning(
                        f"Expected dictionary for {move_types_key}, found {type(move_types_dict)}. Reinitializing."
                    )
                    move_types_dict = {m.name: 0 for m in MoveType}
                    self.match_results[move_types_key] = move_types_dict

                # Now safely update the move type count
                move_type_name = move.move_type.name
                if move_type_name in move_types_dict:
                    move_types_dict[move_type_name] += 1
                else:
                    move_types_dict[move_type_name] = 1

                # Record the move details
                move_record = {
                    "turn": turns,
                    "player": current_player_name,
                    "color": current_color.name,
                    "move_type": move.move_type.name,
                    "start_pos": move.start_pos,
                    "end_pos": move.end_pos,
                }
                all_moves.append(move_record)

                # Switch players and increment turns
                game_state.current_player = game_state.current_player.opposite()
                turns += 1

        # Calculate match duration
        end_time = datetime.now()
        match_duration = (end_time - start_time).total_seconds()

        # Determine winner
        winner = game_state.get_winner()
        if winner == PlayerColor.BLACK:
            winner_name = black_player_name
        elif winner == PlayerColor.RED:
            winner_name = red_player_name
        else:
            winner_name = None

        # Update match results
        self.match_results.update(
            {
                "winner": winner_name,
                "score_p1": game_state.scores[PlayerColor.BLACK]
                if not swap_colors
                else game_state.scores[PlayerColor.RED],
                "score_p2": game_state.scores[PlayerColor.RED]
                if not swap_colors
                else game_state.scores[PlayerColor.BLACK],
                "turns": turns,
                "total_time": match_duration,
                "all_moves": all_moves,
                "final_state": {
                    "board": game_state.board.copy(),
                    "scores": game_state.scores.copy(),
                },
            }
        )

        return self.match_results
