import os
import time
from typing import List, Optional

from src.core.game_state_cy import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.core.player_color import PlayerColor
from src.ui.game_interface import GameInterface
from src.ui.game_statistics import GameStatistics


class ConsoleInterface(GameInterface):
    """Console-based implementation of the game interface."""

    def __init__(self, clear_screen: bool = True):
        """
        Initialize the console interface.

        Args:
            clear_screen: Whether to clear the screen between displays
        """
        super().__init__()
        self.clear_screen = clear_screen
        self._selected_piece = None  # For move selection
        self.statistics = GameStatistics()

    def clear(self):
        """Clear the console screen."""
        if self.clear_screen:
            os.system("cls" if os.name == "nt" else "clear")

    def display_state(self, game_state: GameState) -> None:
        """
        Display the current game state in the console.

        Args:
            game_state: Current state of the game
        """
        self.clear()

        print("\n==== KULIBRAT GAME ====")
        print(f"Target Score: {game_state.target_score}")
        print(f"Current Player: {game_state.current_player.name}")
        print(f"Turn Number: {self.statistics.total_turns + 1}")
        print(
            f"Scores - BLACK: {game_state.scores[PlayerColor.BLACK]}, "
            f"RED: {game_state.scores[PlayerColor.RED]}"
        )

        # Display pieces off board
        black_off = game_state.pieces_off_board[PlayerColor.BLACK]
        red_off = game_state.pieces_off_board[PlayerColor.RED]
        print(f"Pieces Off Board - BLACK: {black_off} (of 4), RED: {red_off} (of 4)")

        print("\nBoard:")
        self._print_board(game_state.board)

        # Add a legend at the bottom
        print("\nLegend: B = Black piece, R = Red piece")
        print("BLACK start row: 0 (top), RED start row: 3 (bottom)")

    def _print_board(self, board):
        """
        Print the board to the console with row markers.

        Args:
            board: NumPy array representing the game board
        """
        # Print column headers
        print("\n    0   1   2  ")
        print("  +---+---+---+  ")

        # Print each row with side markers indicating the start rows
        for row in range(board.shape[0]):
            # Print row number and cells
            row_marker = f"{row} "
            print(f"{row_marker}| ", end="")

            for col in range(board.shape[1]):
                piece = board[row, col]
                if piece == PlayerColor.BLACK.value:
                    symbol = "B"
                elif piece == PlayerColor.RED.value:
                    symbol = "R"
                else:
                    symbol = " "
                print(f"{symbol} | ", end="")

            # Add additional marker for special rows
            if row == 0:
                print("  BLACK start row")
            elif row == 3:
                print("  RED start row")
            else:
                print("")  # Just a newline for other rows

            print("  +---+---+---+  ")

    def get_human_move(
        self, game_state: GameState, player_color: PlayerColor, valid_moves: List[Move]
    ) -> Move:
        """
        Get a move from a human player via the console.

        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move
            valid_moves: List of valid moves

        Returns:
            A valid move selected by the human player
        """
        # Start turn timer
        self.statistics.start_turn_timer(player_color)

        # Display current game state
        self.display_state(game_state)

        # Display valid moves
        print(f"\nValid moves for {player_color.name}:")
        for i, move in enumerate(valid_moves):
            print(f"{i + 1}. {self._format_move(move, player_color)}")

        # Get move selection
        while True:
            try:
                choice = int(input("\nEnter move number: ")) - 1
                if 0 <= choice < len(valid_moves):
                    # Get the selected move
                    move = valid_moves[choice]

                    # Record move statistics
                    self.statistics.record_move(player_color, move)

                    # If move is an attack, record capture
                    if move.move_type == MoveType.ATTACK:
                        self.statistics.record_capture(player_color)

                    # End turn timer
                    self.statistics.end_turn_timer(player_color)

                    return move
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")

    def _format_move(self, move: Move, player_color: PlayerColor) -> str:
        """
        Format a move for display with clear descriptions.

        Args:
            move: The move to format
            player_color: The color of the player making the move

        Returns:
            A string representation of the move
        """
        if move.move_type == MoveType.INSERT:
            row, col = move.end_pos
            return f"INSERT a piece at position ({row}, {col})"

        elif move.move_type == MoveType.DIAGONAL:
            start_row, start_col = move.start_pos
            end_row, end_col = move.end_pos

            # Check if this is a scoring move (moving off the board)
            if (player_color == PlayerColor.BLACK and end_row >= 4) or (
                player_color == PlayerColor.RED and end_row < 0
            ):
                return f"DIAGONAL from ({start_row}, {start_col}) to score a point (off board)"
            else:
                return f"DIAGONAL from ({start_row}, {start_col}) to ({end_row}, {end_col})"

        elif move.move_type == MoveType.ATTACK:
            start_row, start_col = move.start_pos
            end_row, end_col = move.end_pos
            return f"ATTACK from ({start_row}, {start_col}) to ({end_row}, {end_col})"

        elif move.move_type == MoveType.JUMP:
            start_row, start_col = move.start_pos
            end_row, end_col = move.end_pos

            # Check if this is a scoring move
            if (player_color == PlayerColor.BLACK and end_row >= 4) or (
                player_color == PlayerColor.RED and end_row < 0
            ):
                return (
                    f"JUMP from ({start_row}, {start_col}) to score a point (off board)"
                )
            else:
                return f"JUMP from ({start_row}, {start_col}) to ({end_row}, {end_col})"

        return str(move)

    def show_winner(self, winner: Optional[PlayerColor], game_state: GameState) -> None:
        """
        Display the winner of the game.

        Args:
            winner: The player who won, or None for a draw
            game_state: Final state of the game
        """
        # Ensure the last turn is counted
        if not self.statistics.total_turns:
            self.statistics.record_turn()

        # Display game state
        self.display_state(game_state)

        # Show winner
        if winner:
            print(f"\n🏆 Game Over! {winner.name} player wins! 🏆")
            print(
                f"Final Score - BLACK: {game_state.scores[PlayerColor.BLACK]}, "
                f"RED: {game_state.scores[PlayerColor.RED]}"
            )
        else:
            print("\nGame Over! It's a draw!")

        # Display game statistics
        print("\n==== GAME STATISTICS ====")
        stats_summary = self.statistics.get_summary()

        # Custom formatting for detailed statistics
        print(f"Total Turns: {stats_summary['total_turns']}")
        print(f"Total Game Time: {stats_summary['total_game_time']}")

        print("\n--- Moves ---")
        for player, moves in stats_summary["moves_by_player"].items():
            print(f"{player} Moves: {moves}")

        print("\n--- Captures ---")
        for player, captures in stats_summary["captures_by_player"].items():
            print(f"{player} Captures: {captures}")

        print("\n--- Move Type Distribution ---")
        for player, move_types in stats_summary["move_type_distribution"].items():
            print(f"{player} Move Types:")
            for move_type, count in move_types.items():
                print(f"  {move_type}: {count}")

        print("\n--- Turn Times ---")
        for player, avg_time in stats_summary["average_turn_times"].items():
            print(f"{player} Average Turn Time: {avg_time}")

    def show_message(self, message: str) -> None:
        """
        Display a message to the user.

        Args:
            message: The message to display
        """
        print(f"\n{message}")
        time.sleep(1)  # Give user time to read the message
