import os
import time
from typing import List, Optional, Dict, Any



from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.core.player_color import PlayerColor
from src.ui.game_interface import GameInterface



class GameStatistics:
    def __init__(self):
        # Basic tracking
        self.total_turns = 0
        self.moves_by_player = {PlayerColor.BLACK: 0, PlayerColor.RED: 0}
        self.captures_by_player = {PlayerColor.BLACK: 0, PlayerColor.RED: 0}

        # Advanced tracking
        self.move_types_used = {
            PlayerColor.BLACK: {
                MoveType.INSERT: 0,
                MoveType.DIAGONAL: 0,
                MoveType.ATTACK: 0,
                MoveType.JUMP: 0,
            },
            PlayerColor.RED: {
                MoveType.INSERT: 0,
                MoveType.DIAGONAL: 0,
                MoveType.ATTACK: 0,
                MoveType.JUMP: 0,
            },
        }

        # Timing
        self.game_start_time = time.time()
        self.turn_start_times = {PlayerColor.BLACK: [], PlayerColor.RED: []}
        self.turn_end_times = {PlayerColor.BLACK: [], PlayerColor.RED: []}

    def record_move(self, player: PlayerColor, move: Move):
        """Record details of a move"""
        # Increment moves for the specific player
        self.moves_by_player[player] += 1

        # Track move type
        self.move_types_used[player][move.move_type] += 1

    def record_turn(self):
        """Record a complete turn"""
        self.total_turns += 1

    def record_capture(self, player: PlayerColor):
        """Record a piece capture"""
        self.captures_by_player[player] += 1

    def start_turn_timer(self, player: PlayerColor):
        """Start timer for turn"""
        self.turn_start_times[player].append(time.time())

    def end_turn_timer(self, player: PlayerColor):
        """End timer and record turn time"""
        if self.turn_start_times[player]:
            # Record end time
            self.turn_end_times[player].append(time.time())

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive game summary"""
        total_game_time = time.time() - self.game_start_time

        # Calculate turn times
        turn_times = {}
        for player in [PlayerColor.BLACK, PlayerColor.RED]:
            # Ensure we have equal number of start and end times
            start_times = self.turn_start_times[player]
            end_times = self.turn_end_times[player]

            # Calculate turn durations
            turn_durations = []
            for start, end in zip(start_times, end_times):
                turn_durations.append(end - start)

            # Calculate average turn time
            if turn_durations:
                turn_times[player.name] = (
                    f"{(sum(turn_durations) / len(turn_durations)):.2f} seconds"
                )
            else:
                turn_times[player.name] = "No turns"

        return {
            "total_turns": self.total_turns,
            "total_game_time": f"{total_game_time:.2f} seconds",
            "moves_by_player": {
                player.name: moves for player, moves in self.moves_by_player.items()
            },
            "captures_by_player": {
                player.name: captures
                for player, captures in self.captures_by_player.items()
            },
            "move_type_distribution": {
                player.name: {
                    move_type.name: count for move_type, count in type_counts.items()
                }
                for player, type_counts in self.move_types_used.items()
            },
            "average_turn_times": turn_times,
        }