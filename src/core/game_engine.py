""" """

from typing import Dict, List, Optional

from src.core.game_rules import GameRules
from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.turn_manager import TurnManager
from src.players.player import Player
from src.ui.game_interface import GameInterface


class GameEngine:
    """
    Manages the overall game flow, coordinating between game rules,
    turn management, and player interactions.
    """

    def __init__(
        self,
        rules_engine: GameRules,
        turn_manager: TurnManager,
        interface: Optional[GameInterface] = None,
        target_score: int = 5,
    ):
        """
        Initialize the game engine.

        Args:
            rules_engine: Game rules validation engine
            turn_manager: Manages turn progression
            interface: Optional game interface for display and interaction
            target_score: Score needed to win the game
        """

        # Game components
        self.rules_engine = rules_engine
        self.turn_manager = turn_manager
        self.interface = interface

        # Game state initialization
        self.state = GameState(target_score)

        # Game tracking
        self.moves_history: List[tuple] = []
        self.players: Dict[PlayerColor, Player] = {}
        self.current_player_color = PlayerColor.BLACK

    def start_game(
        self, black_player: Player, red_player: Player
    ) -> Optional[PlayerColor]:
        """
        Start and play the game until completion.

        Args:
            black_player: Player controlling black pieces
            red_player: Player controlling red pieces

        Returns:
            The winning player's color, or None if the game is a draw
        """
        # Setup players
        self.players = {PlayerColor.BLACK: black_player, PlayerColor.RED: red_player}

        # Validate players
        if not self._validate_players():
            return None

        # Initialize players
        for player in self.players.values():
            player.setup(self.state)

        # Main game loop
        self._display_state()

        # Reset current player to BLACK
        self.current_player_color = PlayerColor.BLACK
        self.state.current_player = PlayerColor.BLACK

        while not self.state.is_game_over():
            try:
                # Get current player
                current_player = self.players[self.current_player_color]

                # Get player's move
                move = current_player.get_move(self.state)

                # Process the turn
                new_state = self.turn_manager.process_turn(
                    self.state, current_player, move
                )

                # Update the game state
                self.state = new_state

                # Record move history if a move was made
                if move:
                    self.moves_history.append((self.current_player_color, move))

                # Display updated state
                self._display_state()

                # Switch current player ONLY if the state's current player has changed
                # This ensures we respect turn skipping when a player has no moves
                if self.current_player_color != self.state.current_player:
                    self.current_player_color = self.state.current_player

            except Exception:
                import traceback

                traceback.print_exc()
                break

        # Determine winner
        winner = self.state.get_winner()

        # Notify players of game over
        for player in self.players.values():
            player.game_over(self.state)

        # Show winner
        if self.interface:
            self.interface.show_winner(winner, self.state)

        return winner

    def reset_game(self, target_score: Optional[int] = None):
        """
        Reset the game to initial state.

        Args:
            target_score: Optional new target score
        """
        # Reset game state
        self.state = GameState(target_score or self.state.target_score)

        # Reset tracking variables
        self.moves_history = []
        self.current_player_color = PlayerColor.BLACK

        # Reinitialize players
        for player in self.players.values():
            player.setup(self.state)

    def _validate_players(self) -> bool:
        """
        Validate player configuration.

        Returns:
            Boolean indicating if players are valid
        """
        # Check if both players are present
        if len(self.players) != 2:
            return False

        # Check if players have different colors
        player_colors = list(self.players.keys())
        if player_colors[0] == player_colors[1]:
            return False

        return True

    def _display_state(self):
        """
        Display current game state via interface if available.
        """
        if self.interface:
            self.interface.display_state(self.state)
