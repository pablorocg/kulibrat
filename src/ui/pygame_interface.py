"""
Enhanced Pygame interface for the Kulibrat game.
"""

import os
import sys
import time
import math
from typing import List, Optional, Dict, Tuple

import pygame

from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.core.player_color import PlayerColor
from src.ui.game_interface import GameInterface
from src.ui.game_statistics import GameStatistics


class KulibratGUI(GameInterface):
    """
    A minimalistic graphical interface for the Kulibrat game.
    Focuses on essential gameplay functionality with clean, simple visuals.
    """

    def __init__(self, screen_width=1024, screen_height=768):
        """Initialize the GUI with default settings."""
        pygame.init()
        pygame.display.set_caption("Kulibrat")

        # Screen setup
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # Color palette
        self.colors = {
            # Basic interface colors
            "background": (240, 240, 245),
            "text_dark": (50, 50, 60),
            "text_light": (250, 250, 250),
            # Board colors
            "board_light": (230, 230, 240),
            "board_dark": (200, 200, 220),
            "board_border": (180, 180, 200),
            # Player colors
            "black_piece": (40, 45, 55),
            "black_piece_highlight": (70, 75, 85),
            "red_piece": (200, 60, 70),
            "red_piece_highlight": (230, 90, 100),
            # UI elements
            "highlight_valid": (100, 200, 100, 160),
            "highlight_selected": (100, 150, 240, 160),
            "panel_bg": (250, 250, 255),
            "panel_border": (200, 200, 210),
            "button": (70, 130, 200),
            "button_hover": (90, 150, 220),
            "button_text": (255, 255, 255),
            # Start rows highlight
            "start_row_black": (180, 190, 240, 80),
            "start_row_red": (240, 190, 190, 80),
            # Score zones (new)
            "score_zone_inactive": (245, 245, 180, 80),  # Light yellow with transparency
            "score_zone_active": (255, 215, 0, 150),     # Gold with transparency
            "score_zone_text": (180, 0, 0),              # Dark red for score text
        }

        # Font setup
        pygame.font.init()
        font_size_factor = min(self.screen_width / 1024, self.screen_height / 768)

        self.fonts = {
            "title": pygame.font.SysFont("Arial", int(36 * font_size_factor)),
            "header": pygame.font.SysFont("Arial", int(28 * font_size_factor)),
            "normal": pygame.font.SysFont("Arial", int(20 * font_size_factor)),
            "small": pygame.font.SysFont("Arial", int(16 * font_size_factor)),
        }

        # Board dimensions
        self.board_rows = 4
        self.board_cols = 3

        # Layout calculations - responsive design
        self.sidebar_width = int(min(300, self.screen_width * 0.25))
        self.board_area_width = self.screen_width - self.sidebar_width

        # Calculate cell size to fit in available space
        self.board_margin = int(min(50, self.screen_height * 0.05))
        available_width = self.board_area_width - 2 * self.board_margin
        available_height = self.screen_height - 2 * self.board_margin

        width_based_cell = available_width / self.board_cols
        height_based_cell = available_height / self.board_rows
        self.cell_size = int(min(width_based_cell, height_based_cell) * 0.9)

        # Board positioning
        self.board_width = self.cell_size * self.board_cols
        self.board_height = self.cell_size * self.board_rows
        self.board_x = (self.board_area_width - self.board_width) // 2
        self.board_y = (self.screen_height - self.board_height) // 2

        # Piece properties
        self.piece_radius = int(self.cell_size * 0.4)

        # Game state
        self.current_game_state = None
        self.valid_moves = []
        self.selected_pos = None
        self.selected_move = None
        self.waiting_for_move = False

        # UI elements
        self.buttons = []
        self.message = None
        self.message_timer = 0

        # Game clock
        self.clock = pygame.time.Clock()
        self.fps = 60

        # Statistics tracking
        self.statistics = GameStatistics()

        # Player references (will be set by set_players)
        self.players = {}

    def display_state(self, game_state: GameState) -> None:
        """Display the current game state."""
        self.current_game_state = game_state
        self._draw_screen()

    def get_human_move(
        self, game_state: GameState, player_color: PlayerColor, valid_moves: List[Move]
    ) -> Move:
        """Get a move from a human player via the GUI."""
        self.current_game_state = game_state
        self.valid_moves = valid_moves
        self.waiting_for_move = True
        self.selected_pos = None
        self.selected_move = None

        # Start turn timer for statistics
        self.statistics.start_turn_timer(player_color)

        # Show a prompt to the user
        self.show_message(
            f"{player_color.name} player's turn - Select a piece or position"
        )

        # Main interaction loop
        while self.waiting_for_move:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle mouse clicks
                    mouse_pos = pygame.mouse.get_pos()
                    self._handle_mouse_click(mouse_pos)

                if event.type == pygame.KEYDOWN:
                    # Handle keyboard shortcuts
                    if event.key == pygame.K_ESCAPE:
                        # Deselect if a position is selected
                        if self.selected_pos:
                            self.selected_pos = None
                            self._draw_screen()
                        else:
                            pygame.quit()
                            sys.exit()

            # Update the screen
            self._draw_screen()
            self.clock.tick(self.fps)

        # Record the selected move for statistics
        if self.selected_move:
            self.statistics.record_move(player_color, self.selected_move)

            # If move is an attack, record capture
            if self.selected_move.move_type == MoveType.ATTACK:
                self.statistics.record_capture(player_color)

            # End turn timer
            self.statistics.end_turn_timer(player_color)

        self.waiting_for_move = False
        self.valid_moves = []

        # Reset selection
        self.selected_pos = None

        # Return the selected move
        return self.selected_move

    def show_winner(self, winner: Optional[PlayerColor], game_state: GameState) -> None:
        """Display the winner of the game."""
        self.current_game_state = game_state

        # Ensure the last turn is counted
        if not self.statistics.total_turns:
            self.statistics.record_turn()

        # Draw the final game state with a semi-transparent overlay
        self._draw_screen()

        # Create overlay
        overlay = pygame.Surface(
            (self.screen_width, self.screen_height), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))

        # Create winner panel
        panel_width = int(self.screen_width * 0.6)
        panel_height = int(self.screen_height * 0.5)
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2

        # Draw panel background
        pygame.draw.rect(
            self.screen,
            self.colors["panel_bg"],
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10,
        )

        pygame.draw.rect(
            self.screen,
            self.colors["panel_border"],
            (panel_x, panel_y, panel_width, panel_height),
            width=2,
            border_radius=10,
        )

        # Draw content based on game result
        title_text = "GAME OVER"
        title_color = self.colors["text_dark"]
        if winner:
            winner_color = (
                self.colors["black_piece"]
                if winner == PlayerColor.BLACK
                else self.colors["red_piece"]
            )
            result_text = f"{winner.name} PLAYER WINS!"
            result_color = winner_color
        else:
            result_text = "IT'S A DRAW!"
            result_color = self.colors["text_dark"]

        # Render title
        title = self.fonts["title"].render(title_text, True, title_color)
        title_rect = title.get_rect(centerx=panel_x + panel_width // 2, y=panel_y + 20)
        self.screen.blit(title, title_rect)

        # Render result
        result = self.fonts["header"].render(result_text, True, result_color)
        result_rect = result.get_rect(
            centerx=panel_x + panel_width // 2, y=panel_y + 80
        )
        self.screen.blit(result, result_rect)

        # Display final score
        score_y = result_rect.bottom + 40
        score_text = f"Final Score: BLACK {game_state.scores[PlayerColor.BLACK]} - RED {game_state.scores[PlayerColor.RED]}"
        score = self.fonts["normal"].render(score_text, True, self.colors["text_dark"])
        score_rect = score.get_rect(centerx=panel_x + panel_width // 2, y=score_y)
        self.screen.blit(score, score_rect)

        # Create buttons
        button_width = 160
        button_height = 50
        button_margin = 40
        buttons_y = panel_y + panel_height - button_height - 20

        # Play again button
        play_again_rect = pygame.Rect(
            panel_x + panel_width // 2 - button_width - button_margin // 2,
            buttons_y,
            button_width,
            button_height,
        )

        # Quit button
        quit_rect = pygame.Rect(
            panel_x + panel_width // 2 + button_margin // 2,
            buttons_y,
            button_width,
            button_height,
        )

        # Draw buttons
        pygame.draw.rect(
            self.screen, self.colors["button"], play_again_rect, border_radius=5
        )
        pygame.draw.rect(
            self.screen, self.colors["red_piece"], quit_rect, border_radius=5
        )

        # Button text
        play_text = self.fonts["normal"].render(
            "Play Again", True, self.colors["button_text"]
        )
        play_text_rect = play_text.get_rect(center=play_again_rect.center)
        self.screen.blit(play_text, play_text_rect)

        quit_text = self.fonts["normal"].render(
            "Quit", True, self.colors["button_text"]
        )
        quit_text_rect = quit_text.get_rect(center=quit_rect.center)
        self.screen.blit(quit_text, quit_text_rect)

        # Update display
        pygame.display.flip()

        # Wait for user input
        waiting = True
        play_again = False

        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if play_again_rect.collidepoint(mouse_pos):
                        waiting = False
                        play_again = True
                    elif quit_rect.collidepoint(mouse_pos):
                        pygame.quit()
                        sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            self.clock.tick(30)

        return

    def show_message(self, message: str) -> None:
        """Display a message to the user."""
        self.message = message
        self.message_timer = 180  # Display for 3 seconds at 60 FPS

        # Force redraw to show message immediately
        self._draw_screen()

    def set_players(self, players):
        """
        Set player references for the interface.

        Args:
            players: Dictionary mapping player colors to player objects
        """
        self.players = players

    def _draw_screen(self):
        """Render the full game screen."""
        # Start with the background
        self.screen.fill(self.colors["background"])

        # Draw the board
        self._draw_board()

        # Draw the sidebar
        self._draw_sidebar()

        # Draw any active messages
        if self.message and self.message_timer > 0:
            self._draw_message()
            self.message_timer -= 1

        # Update the display
        pygame.display.flip()

    def _draw_board(self):
        """Draw the game board and pieces."""
        # Draw board background
        pygame.draw.rect(
            self.screen,
            self.colors["board_border"],
            (
                self.board_x - 5,
                self.board_y - 5,
                self.board_width + 10,
                self.board_height + 10,
            ),
            border_radius=3,
        )

        # Highlight start rows
        black_start_row = pygame.Surface(
            (self.board_width, self.cell_size), pygame.SRCALPHA
        )
        black_start_row.fill(self.colors["start_row_black"])
        self.screen.blit(black_start_row, (self.board_x, self.board_y))

        red_start_row = pygame.Surface(
            (self.board_width, self.cell_size), pygame.SRCALPHA
        )
        red_start_row.fill(self.colors["start_row_red"])
        self.screen.blit(
            red_start_row,
            (self.board_x, self.board_y + self.cell_size * (self.board_rows - 1)),
        )

        # Draw score zones (Always visible, even when not active)
        # Draw BLACK's scoring zone (RED's start row)
        black_score_zone = pygame.Surface(
            (self.board_width, self.cell_size // 4), pygame.SRCALPHA
        )
        black_score_zone.fill(self.colors["score_zone_inactive"])
        self.screen.blit(
            black_score_zone,
            (self.board_x, self.board_y + self.cell_size * self.board_rows),
        )
        
        # Add "SCORE ZONE" text for BLACK
        score_text = self.fonts["small"].render("BLACK SCORES HERE", True, self.colors["score_zone_text"])
        text_rect = score_text.get_rect(
            center=(
                self.board_x + self.board_width // 2,
                self.board_y + self.cell_size * self.board_rows + self.cell_size // 8,
            )
        )
        self.screen.blit(score_text, text_rect)

        # Draw RED's scoring zone (BLACK's start row)
        red_score_zone = pygame.Surface(
            (self.board_width, self.cell_size // 4), pygame.SRCALPHA
        )
        red_score_zone.fill(self.colors["score_zone_inactive"])
        self.screen.blit(
            red_score_zone,
            (self.board_x, self.board_y - self.cell_size // 4),
        )
        
        # Add "SCORE ZONE" text for RED
        score_text = self.fonts["small"].render("RED SCORES HERE", True, self.colors["score_zone_text"])
        text_rect = score_text.get_rect(
            center=(
                self.board_x + self.board_width // 2,
                self.board_y - self.cell_size // 8,
            )
        )
        self.screen.blit(score_text, text_rect)

        # Draw grid cells
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                # Calculate cell position
                cell_x = self.board_x + col * self.cell_size
                cell_y = self.board_y + row * self.cell_size

                # Draw cell with alternating colors
                cell_color = (
                    self.colors["board_light"]
                    if (row + col) % 2 == 0
                    else self.colors["board_dark"]
                )
                pygame.draw.rect(
                    self.screen,
                    cell_color,
                    (cell_x, cell_y, self.cell_size, self.cell_size),
                )

                # Draw cell border
                pygame.draw.rect(
                    self.screen,
                    self.colors["board_border"],
                    (cell_x, cell_y, self.cell_size, self.cell_size),
                    width=1,
                )

        # Highlight scoring opportunities if a piece is selected
        self._highlight_scoring_opportunities()

        # Draw valid move highlights
        if self.waiting_for_move and self.selected_pos:
            valid_end_positions = [
                m.end_pos
                for m in self.valid_moves
                if m.start_pos == self.selected_pos
                and m.end_pos
                and 0 <= m.end_pos[0] < self.board_rows
                and 0 <= m.end_pos[1] < self.board_cols
            ]

            for row, col in valid_end_positions:
                cell_x = self.board_x + col * self.cell_size
                cell_y = self.board_y + row * self.cell_size

                # Draw highlight
                highlight = pygame.Surface(
                    (self.cell_size, self.cell_size), pygame.SRCALPHA
                )
                highlight.fill(self.colors["highlight_valid"])
                self.screen.blit(highlight, (cell_x, cell_y))

        # Check for insert moves when no position is selected
        elif self.waiting_for_move and not self.selected_pos:
            for move in self.valid_moves:
                if move.move_type == MoveType.INSERT and move.end_pos:
                    row, col = move.end_pos
                    cell_x = self.board_x + col * self.cell_size
                    cell_y = self.board_y + row * self.cell_size

                    # Draw highlight
                    highlight = pygame.Surface(
                        (self.cell_size, self.cell_size), pygame.SRCALPHA
                    )
                    highlight.fill(self.colors["highlight_valid"])
                    self.screen.blit(highlight, (cell_x, cell_y))

        # Highlight selected position
        if self.selected_pos:
            row, col = self.selected_pos
            cell_x = self.board_x + col * self.cell_size
            cell_y = self.board_y + row * self.cell_size

            highlight = pygame.Surface(
                (self.cell_size, self.cell_size), pygame.SRCALPHA
            )
            highlight.fill(self.colors["highlight_selected"])
            self.screen.blit(highlight, (cell_x, cell_y))

        # Draw pieces
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                if self.current_game_state:
                    piece_value = self.current_game_state.board[row, col]
                    if piece_value != 0:
                        # Calculate position
                        cell_x = self.board_x + col * self.cell_size
                        cell_y = self.board_y + row * self.cell_size

                        # Determine piece color
                        if piece_value == PlayerColor.BLACK.value:
                            piece_color = self.colors["black_piece"]
                            highlight_color = self.colors["black_piece_highlight"]
                        else:  # RED
                            piece_color = self.colors["red_piece"]
                            highlight_color = self.colors["red_piece_highlight"]

                        # Draw piece with shadow
                        shadow_offset = 2
                        pygame.draw.circle(
                            self.screen,
                            (20, 20, 20, 100),
                            (
                                cell_x + self.cell_size // 2 + shadow_offset,
                                cell_y + self.cell_size // 2 + shadow_offset,
                            ),
                            self.piece_radius,
                        )

                        # Draw piece
                        pygame.draw.circle(
                            self.screen,
                            piece_color,
                            (
                                cell_x + self.cell_size // 2,
                                cell_y + self.cell_size // 2,
                            ),
                            self.piece_radius,
                        )

                        # Add highlight for 3D effect
                        pygame.draw.circle(
                            self.screen,
                            highlight_color,
                            (
                                cell_x + self.cell_size // 2 - self.piece_radius // 3,
                                cell_y + self.cell_size // 2 - self.piece_radius // 3,
                            ),
                            self.piece_radius // 2,
                        )

    def _highlight_scoring_opportunities(self):
        """Highlight potential scoring moves in a clear, distinctive way."""
        if not self.waiting_for_move or not self.selected_pos or not self.current_game_state:
            return

        piece_row, piece_col = self.selected_pos
        current_player = self.current_game_state.current_player
        scoring_moves = []

        # Find all scoring moves for the selected piece
        for move in self.valid_moves:
            if move.start_pos == self.selected_pos:
                # Check if this is a scoring move
                if (current_player == PlayerColor.BLACK and move.end_pos and move.end_pos[0] >= self.board_rows) or \
                   (current_player == PlayerColor.RED and move.end_pos and move.end_pos[0] < 0):
                    scoring_moves.append(move)

        if not scoring_moves:
            return

        # Define the scoring row based on current player
        scoring_row = self.board_rows - 1 if current_player == PlayerColor.BLACK else 0
        
        # Highlight scoring zones more prominently when scoring is possible
        if current_player == PlayerColor.BLACK:
            # BLACK player can score - highlight bottom zone
            score_zone = pygame.Surface(
                (self.board_width, self.cell_size // 3), pygame.SRCALPHA
            )
            score_zone.fill(self.colors["score_zone_active"])
            self.screen.blit(
                score_zone,
                (self.board_x, self.board_y + self.cell_size * self.board_rows),
            )
            
            # Add flashing "SCORE!" text
            flash_alpha = 128 + int(127 * math.sin(pygame.time.get_ticks() / 200))
            score_text = self.fonts["normal"].render("SCORE!", True, (255, 0, 0, flash_alpha))
            text_rect = score_text.get_rect(
                center=(
                    self.board_x + self.board_width // 2,
                    self.board_y + self.cell_size * self.board_rows + self.cell_size // 6,
                )
            )
            self.screen.blit(score_text, text_rect)
            
        elif current_player == PlayerColor.RED:
            # RED player can score - highlight top zone
            score_zone = pygame.Surface(
                (self.board_width, self.cell_size // 3), pygame.SRCALPHA
            )
            score_zone.fill(self.colors["score_zone_active"])
            self.screen.blit(
                score_zone,
                (self.board_x, self.board_y - self.cell_size // 3),
            )
            
            # Add flashing "SCORE!" text
            flash_alpha = 128 + int(127 * math.sin(pygame.time.get_ticks() / 200))
            score_text = self.fonts["normal"].render("SCORE!", True, (255, 0, 0, flash_alpha))
            text_rect = score_text.get_rect(
                center=(
                    self.board_x + self.board_width // 2,
                    self.board_y - self.cell_size // 6,
                )
            )
            self.screen.blit(score_text, text_rect)

        # Highlight individual scoring options
        for move in scoring_moves:
            # Calculate where the move would end (even if off board)
            target_col = piece_col
            if move.move_type == MoveType.DIAGONAL:
                # For diagonal moves, determine the column offset
                target_col = move.end_pos[1]  # Use the actual target column from the move
            
            # Only highlight if the column is valid
            if 0 <= target_col < self.board_cols:
                # Get screen coordinates for the scoring edge cell
                cell_x = self.board_x + target_col * self.cell_size
                cell_y = self.board_y + scoring_row * self.cell_size
                
                # Draw special pulsing highlight for scoring move
                pulse_factor = 0.7 + 0.3 * math.sin(pygame.time.get_ticks() / 200)
                pulse_color = (
                    int(255 * pulse_factor),
                    int(215 * pulse_factor),
                    int(0 * pulse_factor),
                    180
                )
                
                highlight = pygame.Surface(
                    (self.cell_size, self.cell_size), pygame.SRCALPHA
                )
                highlight.fill(pulse_color)
                self.screen.blit(highlight, (cell_x, cell_y))
                
                # Add an arrow pointing to the scoring zone
                arrow_color = (255, 0, 0)  # Red arrow
                if current_player == PlayerColor.BLACK:
                    # Draw arrow pointing down
                    points = [
                        (cell_x + self.cell_size // 2, cell_y + self.cell_size),
                        (cell_x + self.cell_size // 4, cell_y + self.cell_size * 2 // 3),
                        (cell_x + self.cell_size * 3 // 4, cell_y + self.cell_size * 2 // 3)
                    ]
                else:  # RED
                    # Draw arrow pointing up
                    points = [
                        (cell_x + self.cell_size // 2, cell_y),
                        (cell_x + self.cell_size // 4, cell_y + self.cell_size // 3),
                        (cell_x + self.cell_size * 3 // 4, cell_y + self.cell_size // 3)
                    ]
                
                pygame.draw.polygon(self.screen, arrow_color, points)
                
                # Add move type label
                move_type_text = "DIAGONAL" if move.move_type == MoveType.DIAGONAL else "JUMP"
                text = self.fonts["small"].render(move_type_text, True, (0, 0, 0))
                text_rect = text.get_rect(
                    center=(cell_x + self.cell_size // 2, cell_y + self.cell_size // 2)
                )
                self.screen.blit(text, text_rect)

    def _draw_sidebar(self):
        """Draw the game information sidebar."""
        if not self.current_game_state:
            return

        # Sidebar background
        sidebar_x = self.board_area_width
        sidebar_rect = pygame.Rect(sidebar_x, 0, self.sidebar_width, self.screen_height)
        pygame.draw.rect(self.screen, self.colors["panel_bg"], sidebar_rect)

        # Game title
        title = self.fonts["title"].render("KULIBRAT", True, self.colors["text_dark"])
        title_rect = title.get_rect(centerx=sidebar_x + self.sidebar_width // 2, y=20)
        self.screen.blit(title, title_rect)

        # Current player indicator
        current_player = self.current_game_state.current_player
        player_color = (
            self.colors["black_piece"]
            if current_player == PlayerColor.BLACK
            else self.colors["red_piece"]
        )

        player_text = self.fonts["header"].render(
            f"{current_player.name}'s Turn", True, player_color
        )
        player_rect = player_text.get_rect(
            centerx=sidebar_x + self.sidebar_width // 2, y=title_rect.bottom + 30
        )
        self.screen.blit(player_text, player_rect)

        # Score display
        score_y = player_rect.bottom + 40
        score_title = self.fonts["normal"].render(
            "SCORE", True, self.colors["text_dark"]
        )
        score_rect = score_title.get_rect(
            centerx=sidebar_x + self.sidebar_width // 2, y=score_y
        )
        self.screen.blit(score_title, score_rect)

        # Score bars
        bar_width = int(self.sidebar_width * 0.8)
        bar_height = 24
        bar_x = sidebar_x + (self.sidebar_width - bar_width) // 2

        # Display target score
        target_score = self.current_game_state.target_score
        target_text = self.fonts["small"].render(
            f"Target: {target_score}", True, self.colors["text_dark"]
        )
        target_rect = target_text.get_rect(
            centerx=sidebar_x + self.sidebar_width // 2, y=score_rect.bottom + 10
        )
        self.screen.blit(target_text, target_rect)

        # Black score
        black_score = self.current_game_state.scores[PlayerColor.BLACK]
        black_bar_y = target_rect.bottom + 15
        black_progress = min(black_score / target_score, 1.0)

        # Background bar
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            pygame.Rect(bar_x, black_bar_y, bar_width, bar_height),
            border_radius=3,
        )

        # Progress bar
        if black_progress > 0:
            pygame.draw.rect(
                self.screen,
                self.colors["black_piece"],
                pygame.Rect(
                    bar_x, black_bar_y, int(bar_width * black_progress), bar_height
                ),
                border_radius=3,
            )

        # Black score label
        black_label = self.fonts["normal"].render(
            f"BLACK: {black_score}", True, self.colors["black_piece"]
        )
        self.screen.blit(black_label, (bar_x, black_bar_y + bar_height + 5))

        # Red score
        red_score = self.current_game_state.scores[PlayerColor.RED]
        red_bar_y = black_bar_y + bar_height + 30
        red_progress = min(red_score / target_score, 1.0)

        # Background bar
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            pygame.Rect(bar_x, red_bar_y, bar_width, bar_height),
            border_radius=3,
        )

        # Progress bar
        if red_progress > 0:
            pygame.draw.rect(
                self.screen,
                self.colors["red_piece"],
                pygame.Rect(
                    bar_x, red_bar_y, int(bar_width * red_progress), bar_height
                ),
                border_radius=3,
            )

        # Red score label
        red_label = self.fonts["normal"].render(
            f"RED: {red_score}", True, self.colors["red_piece"]
        )
        self.screen.blit(red_label, (bar_x, red_bar_y + bar_height + 5))

        # Pieces available
        pieces_y = red_bar_y + bar_height + 50
        pieces_title = self.fonts["normal"].render(
            "PIECES AVAILABLE", True, self.colors["text_dark"]
        )
        pieces_rect = pieces_title.get_rect(
            centerx=sidebar_x + self.sidebar_width // 2, y=pieces_y
        )
        self.screen.blit(pieces_title, pieces_rect)

        # Available pieces
        black_pieces = self.current_game_state.pieces_off_board[PlayerColor.BLACK]
        red_pieces = self.current_game_state.pieces_off_board[PlayerColor.RED]

        # Display pieces as circles
        piece_radius = 10
        piece_spacing = 25
        piece_y_offset = 35

        # Black pieces
        black_pieces_y = pieces_rect.bottom + 15
        for i in range(4):
            piece_x = (
                sidebar_x
                + self.sidebar_width // 2
                - 3 * piece_spacing // 2
                + i * piece_spacing
            )
            piece_y = black_pieces_y

            if i < black_pieces:
                # Available piece
                pygame.draw.circle(
                    self.screen,
                    self.colors["black_piece"],
                    (piece_x, piece_y),
                    piece_radius,
                )
            else:
                # Used piece (outline only)
                pygame.draw.circle(
                    self.screen,
                    self.colors["black_piece"],
                    (piece_x, piece_y),
                    piece_radius,
                    2,
                )

        # Label for black pieces
        black_pieces_label = self.fonts["small"].render(
            f"BLACK: {black_pieces}/4", True, self.colors["black_piece"]
        )
        black_pieces_label_rect = black_pieces_label.get_rect(
            centerx=sidebar_x + self.sidebar_width // 2,
            y=black_pieces_y + piece_y_offset,
        )
        self.screen.blit(black_pieces_label, black_pieces_label_rect)

        # Red pieces
        red_pieces_y = black_pieces_y + piece_y_offset + 30
        for i in range(4):
            piece_x = (
                sidebar_x
                + self.sidebar_width // 2
                - 3 * piece_spacing // 2
                + i * piece_spacing
            )
            piece_y = red_pieces_y

            if i < red_pieces:
                # Available piece
                pygame.draw.circle(
                    self.screen,
                    self.colors["red_piece"],
                    (piece_x, piece_y),
                    piece_radius,
                )
            else:
                # Used piece (outline only)
                pygame.draw.circle(
                    self.screen,
                    self.colors["red_piece"],
                    (piece_x, piece_y),
                    piece_radius,
                    2,
                )

        # Label for red pieces
        red_pieces_label = self.fonts["small"].render(
            f"RED: {red_pieces}/4", True, self.colors["red_piece"]
        )
        red_pieces_label_rect = red_pieces_label.get_rect(
            centerx=sidebar_x + self.sidebar_width // 2, y=red_pieces_y + piece_y_offset
        )
        self.screen.blit(red_pieces_label, red_pieces_label_rect)

        # Move type explanation box
        move_box_y = red_pieces_y + piece_y_offset + 50
        move_box_title = self.fonts["normal"].render(
            "MOVE TYPES", True, self.colors["text_dark"]
        )
        move_box_rect = move_box_title.get_rect(
            centerx=sidebar_x + self.sidebar_width // 2, y=move_box_y
        )
        self.screen.blit(move_box_title, move_box_rect)
        
        # Move type explanations
        move_types = [
            "INSERT: Place new piece on your start row",
            "DIAGONAL: Move diagonally forward one space",
            "ATTACK: Capture opponent's piece in front",
            "JUMP: Leap over opponent's piece(s)"
        ]
        
        move_y = move_box_rect.bottom + 15
        for move_type in move_types:
            move_text = self.fonts["small"].render(move_type, True, self.colors["text_dark"])
            self.screen.blit(move_text, (bar_x, move_y))
            move_y += 20

        # Game rules button at the bottom
        button_height = 40
        button_width = int(self.sidebar_width * 0.7)
        button_x = sidebar_x + (self.sidebar_width - button_width) // 2
        button_y = self.screen_height - button_height - 20

        rules_button = pygame.Rect(button_x, button_y, button_width, button_height)
        pygame.draw.rect(
            self.screen, self.colors["button"], rules_button, border_radius=5
        )

        rules_text = self.fonts["normal"].render(
            "Game Rules", True, self.colors["button_text"]
        )
        rules_rect = rules_text.get_rect(center=rules_button.center)
        self.screen.blit(rules_text, rules_rect)

        # Add the button to the list of UI buttons
        self.buttons = [("rules", rules_button)]

    def _draw_message(self):
        """Display temporary messages to the user."""
        if not self.message:
            return

        # Create semi-transparent overlay at the bottom of the screen
        message_height = 50
        message_surface = pygame.Surface(
            (self.screen_width, message_height), pygame.SRCALPHA
        )
        message_surface.fill((50, 50, 50, 200))

        # Render message text
        message_text = self.fonts["normal"].render(self.message, True, (255, 255, 255))

        # Center the text in the message bar
        text_rect = message_text.get_rect(
            center=(message_surface.get_width() // 2, message_surface.get_height() // 2)
        )
        message_surface.blit(message_text, text_rect)

        # Position at bottom of screen
        self.screen.blit(message_surface, (0, self.screen_height - message_height))

    def _handle_mouse_click(self, mouse_pos):
        """Handle mouse clicks during move selection."""
        # Check if a UI button was clicked
        for button_id, button_rect in self.buttons:
            if button_rect.collidepoint(mouse_pos):
                if button_id == "rules":
                    self._show_rules()
                return

        # Check if click is on the board
        if not (
            self.board_x <= mouse_pos[0] <= self.board_x + self.board_width
            and self.board_y <= mouse_pos[1] <= self.board_y + self.board_height
        ):
            return

        # Convert mouse position to board coordinates
        board_x = mouse_pos[0] - self.board_x
        board_y = mouse_pos[1] - self.board_y

        col = board_x // self.cell_size
        row = board_y // self.cell_size

        # Validate coordinates
        if not (0 <= row < self.board_rows and 0 <= col < self.board_cols):
            return

        # Debug print to help troubleshoot
        print(f"Click at board position: ({row}, {col})")

        # Check the clicked position
        pos = (row, col)

        # If no position is selected yet
        if self.selected_pos is None:
            # Check if there's a piece at the clicked position or if there are insert moves
            piece_value = self.current_game_state.board[row, col]
            # Filter insert moves if clicking on start row
            insert_moves = [
                m
                for m in self.valid_moves
                if m.move_type == MoveType.INSERT and m.end_pos == pos
            ]

            # Filter moves that have this position as start
            piece_moves = [m for m in self.valid_moves if m.start_pos == pos]

            if (
                piece_value == self.current_game_state.current_player.value
                and piece_moves
            ):
                # Select this piece for moving
                self.selected_pos = pos
                self.show_message(
                    f"Piece selected at {pos}. Click a highlighted square to move."
                )
            elif insert_moves:
                # Directly make an insert move
                self.selected_move = insert_moves[0]
                self.waiting_for_move = False
                self.show_message(f"Inserting piece at {pos}.")

        else:
            # A position was already selected, check if the new click is a valid move
            piece_row, piece_col = self.selected_pos
            current_player = self.current_game_state.current_player

            # Enhanced logic for ALL valid move types
            possible_moves = []
            
            # Check for standard moves (within board)
            standard_moves = [
                m for m in self.valid_moves 
                if m.start_pos == self.selected_pos and m.end_pos == pos
            ]
            possible_moves.extend(standard_moves)
            
            # Check for scoring diagonal moves
            if (current_player == PlayerColor.BLACK and row == self.board_rows - 1) or \
               (current_player == PlayerColor.RED and row == 0):
                # Find diagonal scoring moves that would match this edge position
                diagonal_scoring_moves = [
                    m for m in self.valid_moves
                    if m.start_pos == self.selected_pos and 
                    m.move_type == MoveType.DIAGONAL and
                    abs(m.end_pos[1] - piece_col) == 1 and
                    ((current_player == PlayerColor.BLACK and m.end_pos[0] >= self.board_rows) or
                     (current_player == PlayerColor.RED and m.end_pos[0] < 0)) and
                    m.end_pos[1] == col  # Make sure column matches
                ]
                possible_moves.extend(diagonal_scoring_moves)
                
            # Check for scoring jump moves
            if ((current_player == PlayerColor.BLACK and row == self.board_rows - 1 and col == piece_col) or 
                (current_player == PlayerColor.RED and row == 0 and col == piece_col)):
                # Find jump scoring moves in the same column
                jump_scoring_moves = [
                    m for m in self.valid_moves
                    if m.start_pos == self.selected_pos and
                    m.move_type == MoveType.JUMP and
                    ((current_player == PlayerColor.BLACK and m.end_pos[0] >= self.board_rows) or
                     (current_player == PlayerColor.RED and m.end_pos[0] < 0)) and
                    m.end_pos[1] == col  # Make sure column matches
                ]
                possible_moves.extend(jump_scoring_moves)

            if possible_moves:
                # Make the move (take the first valid move - should only be one)
                self.selected_move = possible_moves[0]
                self.waiting_for_move = False

                # Show appropriate message based on move type
                move_type = self.selected_move.move_type
                is_scoring = (
                    (current_player == PlayerColor.BLACK and 
                     self.selected_move.end_pos and 
                     self.selected_move.end_pos[0] >= self.board_rows) or
                    (current_player == PlayerColor.RED and 
                     self.selected_move.end_pos and 
                     self.selected_move.end_pos[0] < 0)
                )
                
                if move_type == MoveType.DIAGONAL:
                    if is_scoring:
                        self.show_message(f"Diagonal move scores a point for {current_player.name}!")
                    else:
                        self.show_message(f"Diagonal move from {self.selected_pos} to {pos}.")
                elif move_type == MoveType.ATTACK:
                    self.show_message(f"Attack! {current_player.name} captures opponent's piece.")
                elif move_type == MoveType.JUMP:
                    if is_scoring:
                        self.show_message(f"Jump scores a point for {current_player.name}!")
                    else:
                        self.show_message(f"Jump from {self.selected_pos} to {pos}.")
            else:
                # Check if clicked on another one of player's pieces
                piece_value = self.current_game_state.board[row, col]
                piece_moves = [m for m in self.valid_moves if m.start_pos == pos]

                if (
                    piece_value == self.current_game_state.current_player.value
                    and piece_moves
                ):
                    self.selected_pos = pos
                    self.show_message(f"Changed selection to piece at {pos}.")
                else:
                    # Invalid move, deselect
                    self.selected_pos = None
                    self.show_message(
                        "Invalid move. Please select a piece or valid move."
                    )

    def _show_rules(self):
        """Display the game rules."""
        # Create a semi-transparent overlay
        overlay = pygame.Surface(
            (self.screen_width, self.screen_height), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))

        # Create rules panel
        panel_width = int(self.screen_width * 0.7)
        panel_height = int(self.screen_height * 0.8)
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2

        panel = pygame.Surface((panel_width, panel_height))
        panel.fill(self.colors["panel_bg"])

        # Rules title
        title = self.fonts["header"].render(
            "KULIBRAT RULES", True, self.colors["text_dark"]
        )
        title_rect = title.get_rect(centerx=panel_width // 2, y=20)
        panel.blit(title, title_rect)

        # Rules content
        rules_text = [
            "Kulibrat is a strategic board game played on a 3x4 grid.",
            "Each player has 4 pieces of their color.",
            "",
            "OBJECTIVE:",
            "Be the first to score points by moving pieces across the board.",
            "Default target score is 5 points.",
            "",
            "MOVES:",
            "1. INSERT: Place a piece on your start row if a space is available.",
            "2. DIAGONAL: Move diagonally forward to an empty square.",
            "3. ATTACK: Capture an opponent's piece directly in front.",
            "4. JUMP: Jump over a line of 1-3 opponent pieces to an empty square.",
            "",
            "SCORING:",
            "Score a point when your piece moves off the opponent's edge,",
            "either by a diagonal move or jump.",
            "",
            "SPECIAL RULES:",
            "- If a player has no valid moves, their turn is skipped.",
            "- If neither player can move, the last player to move loses.",
            "- BLACK moves pieces from top to bottom, RED from bottom to top.",
            "- When a piece scores, it's removed and can be reused."
        ]

        y_offset = 80
        for line in rules_text:
            if (
                line.startswith("OBJECTIVE:")
                or line.startswith("MOVES:")
                or line.startswith("SCORING:")
                or line.startswith("SPECIAL RULES:")
            ):
                text = self.fonts["normal"].render(line, True, self.colors["text_dark"])
                y_offset += 10
            else:
                text = self.fonts["small"].render(line, True, self.colors["text_dark"])

            panel.blit(text, (40, y_offset))
            y_offset += 30

        # Close button
        close_button = pygame.Rect(panel_width // 2 - 50, panel_height - 60, 100, 40)
        pygame.draw.rect(panel, self.colors["button"], close_button, border_radius=5)

        close_text = self.fonts["normal"].render(
            "Close", True, self.colors["button_text"]
        )
        close_rect = close_text.get_rect(center=close_button.center)
        panel.blit(close_text, close_rect)

        # Display the panel
        self.screen.blit(panel, (panel_x, panel_y))
        pygame.display.flip()

        # Wait for close button click
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    adjusted_pos = (mouse_pos[0] - panel_x, mouse_pos[1] - panel_y)

                    if close_button.collidepoint(adjusted_pos):
                        waiting = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False

            self.clock.tick(30)

        # Redraw the screen
        self._draw_screen()