import os
import sys
from typing import List, Optional

import pygame

from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.core.player_color import PlayerColor
from src.ui.asset_manager import AssetManager
from src.ui.console_interface import GameStatistics
from src.ui.enhanced_board_renderer import EnhancedBoardRenderer
from src.ui.game_config_screen import GameConfigScreen
from src.ui.game_interface import GameInterface


class KulibratGUI(GameInterface):
    """Enhanced graphical interface for the Kulibrat game with smooth animations."""

    def __init__(self, screen_width=1024, screen_height=768):
        """Initialize the GUI with responsive sizing."""
        pygame.init()
        pygame.display.set_caption("Kulibrat")

        # Screen configuration
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height

        # Check if fullscreen is requested in a config file or env variable
        self.fullscreen = False
        self.screen = pygame.display.set_mode(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        # Color palette - modern and accessible
        self.COLORS = {
            "BACKGROUND": (245, 245, 250),
            "BOARD_DARK": (210, 215, 230),
            "BOARD_LIGHT": (235, 240, 250),
            "GRID_LINE": (180, 190, 210),
            "BLACK_PIECE": (40, 45, 55),
            "BLACK_PIECE_HIGHLIGHT": (80, 85, 95),
            "RED_PIECE": (200, 60, 70),
            "RED_PIECE_HIGHLIGHT": (230, 90, 100),
            "HIGHLIGHT_VALID": (100, 200, 100, 160),
            "HIGHLIGHT_SELECTED": (100, 150, 240, 160),
            "TEXT_DARK": (50, 55, 70),
            "TEXT_LIGHT": (245, 245, 250),
            "TEXT_ACCENT": (70, 130, 200),
            "PANEL_BACKGROUND": (250, 250, 255),
            "PANEL_HEADER": (70, 100, 170),
            "BLACK_SCORE": (40, 45, 55),
            "RED_SCORE": (200, 60, 70),
            "BUTTON": (70, 130, 200),
            "BUTTON_HOVER": (90, 150, 220),
            "BUTTON_TEXT": (255, 255, 255),
            "START_ROW_BLACK": (180, 200, 240, 100),
            "START_ROW_RED": (240, 200, 200, 100),
        }

        # Initialize asset manager
        self.asset_manager = AssetManager()

        # Font setup with responsive sizing
        pygame.font.init()
        
        # Calculate font sizes based on screen dimensions
        # This ensures text scales appropriately on different screen sizes
        font_size_factor = min(self.SCREEN_WIDTH / 1024, self.SCREEN_HEIGHT / 768)
        
        # Ensure minimum readable sizes and maximum sizes for very large screens
        def responsive_size(base_size):
            size = int(base_size * font_size_factor)
            return max(10, min(size, base_size * 2))  # Min 10px, max 2x original size
        
        # Apply responsive font sizes
        title_size = responsive_size(48)
        header_size = responsive_size(32)
        normal_size = responsive_size(24)
        small_size = responsive_size(18)
        tiny_size = responsive_size(14)
        
        try:
            font_path = os.path.join(
                "src", "ui", "assets", "fonts", "Roboto-Regular.ttf"
            )
            if os.path.exists(font_path):
                self.FONTS = {
                    "TITLE": pygame.font.Font(font_path, title_size),
                    "HEADER": pygame.font.Font(font_path, header_size),
                    "NORMAL": pygame.font.Font(font_path, normal_size),
                    "SMALL": pygame.font.Font(font_path, small_size),
                    "TINY": pygame.font.Font(font_path, tiny_size),
                }
            else:
                self.FONTS = {
                    "TITLE": pygame.font.SysFont("Arial", title_size),
                    "HEADER": pygame.font.SysFont("Arial", header_size),
                    "NORMAL": pygame.font.SysFont("Arial", normal_size),
                    "SMALL": pygame.font.SysFont("Arial", small_size),
                    "TINY": pygame.font.SysFont("Arial", tiny_size),
                }
        except:
            # Fallback to system fonts if custom font fails
            self.FONTS = {
                "TITLE": pygame.font.SysFont("Arial", title_size),
                "HEADER": pygame.font.SysFont("Arial", header_size),
                "NORMAL": pygame.font.SysFont("Arial", normal_size),
                "SMALL": pygame.font.SysFont("Arial", small_size),
                "TINY": pygame.font.SysFont("Arial", tiny_size),
            }

        # Game configuration
        self.game_config = {
            "black_player": {"name": "Player 1", "type": "Human", "color": "Black"},
            "red_player": {"name": "Player 2", "type": "AI (Random)", "color": "Red"},
            "target_score": 5,
            "ai_delay": 0.5,
            "fullscreen": False,
            "rl_model_path": "models/kulibrat_rl_model_best.pt",
        }

        # Show configuration screen first
        self.config_screen = GameConfigScreen(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.game_config = self.config_screen.show(self.screen)

        # Apply fullscreen setting if requested
        if self.game_config["fullscreen"] and not self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.screen.get_size()
            self.fullscreen = True
        elif not self.game_config["fullscreen"] and self.fullscreen:
            self.screen = pygame.display.set_mode(
                (1024, 768), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.screen.get_size()
            self.fullscreen = False

        # Layout calculations - improved responsive design
        
        # Dynamic margins that scale with screen size
        self.BOARD_MARGIN_X = int(self.SCREEN_WIDTH * 0.03)  # Dynamic margins
        self.BOARD_MARGIN_Y = int(self.SCREEN_HEIGHT * 0.05)
        
        # Check screen aspect ratio to determine optimal layout
        aspect_ratio = self.SCREEN_WIDTH / self.SCREEN_HEIGHT
        
        # Responsive sidebar width based on screen dimensions and aspect ratio
        # Narrower sidebar on portrait-like screens, wider on landscape
        if aspect_ratio < 1.0:  # Portrait orientation
            sidebar_ratio = 0.15  # Much smaller sidebar on portrait screens
            min_sidebar_width = int(self.SCREEN_WIDTH * 0.15)
            max_sidebar_width = int(self.SCREEN_WIDTH * 0.3)
        elif aspect_ratio < 1.33:  # Standard 4:3 and similar
            sidebar_ratio = 0.25
            min_sidebar_width = 180
            max_sidebar_width = int(self.SCREEN_WIDTH * 0.3)
        else:  # Widescreen and ultrawide
            sidebar_ratio = 0.2 if self.SCREEN_WIDTH > 1200 else 0.25
            min_sidebar_width = 200
            max_sidebar_width = 500
        
        # Calculate sidebar width with minimum and maximum constraints
        raw_sidebar_width = int(self.SCREEN_WIDTH * sidebar_ratio)
        self.SIDEBAR_WIDTH = max(min_sidebar_width, min(raw_sidebar_width, max_sidebar_width))
        
        # Ensure sidebar width doesn't take too much space on small screens
        if self.SCREEN_WIDTH < 600:
            self.SIDEBAR_WIDTH = min(self.SIDEBAR_WIDTH, int(self.SCREEN_WIDTH * 0.25))
        
        self.BOARD_AREA_WIDTH = self.SCREEN_WIDTH - self.SIDEBAR_WIDTH

        # Board dimensions
        self.BOARD_ROWS = 4
        self.BOARD_COLS = 3

        # Calculate available space for the board
        available_width = self.BOARD_AREA_WIDTH - 2 * self.BOARD_MARGIN_X
        available_height = self.SCREEN_HEIGHT - 2 * self.BOARD_MARGIN_Y
        
        # Determine optimal cell size to fit either width or height
        width_based_cell = available_width / self.BOARD_COLS
        height_based_cell = available_height / self.BOARD_ROWS
        
        # Choose smaller value to ensure board fits on screen
        # Adding a padding factor to prevent board from touching edges
        self.CELL_SIZE = int(min(width_based_cell, height_based_cell) * 0.95)
        
        # Ensure minimum cell size for playability
        min_cell_size = 40
        self.CELL_SIZE = max(min_cell_size, self.CELL_SIZE)
        
        # Recalculate board dimensions based on cell size
        self.BOARD_WIDTH = self.CELL_SIZE * self.BOARD_COLS
        self.BOARD_HEIGHT = self.CELL_SIZE * self.BOARD_ROWS
        
        # Center the board in the available area
        self.BOARD_X = int((self.BOARD_AREA_WIDTH - self.BOARD_WIDTH) / 2)
        self.BOARD_Y = int((self.SCREEN_HEIGHT - self.BOARD_HEIGHT) / 2)

        # Piece properties - proportional to cell size
        self.PIECE_RADIUS = int(self.CELL_SIZE * 0.4)

        # Initialize enhanced board renderer
        self.board_renderer = EnhancedBoardRenderer(
            board_x=self.BOARD_X,
            board_y=self.BOARD_Y,
            cell_size=self.CELL_SIZE,
            board_rows=self.BOARD_ROWS,
            board_cols=self.BOARD_COLS,
        )

        # Animation properties
        self.ANIMATION_SPEED = 15  # pixels per frame
        self.animation_in_progress = False
        self.animation_frames = []
        self.animation_timer = 0

        # Game state
        self.current_game_state = None
        self.valid_moves = []
        self.selected_pos = None
        self.selected_move = None
        self.waiting_for_move = False

        # Game clock and timing
        self.clock = pygame.time.Clock()
        self.FPS = 25

        # Statistics tracking
        self.statistics = GameStatistics()

        # UI elements
        self.buttons = []
        self.tooltips = {}
        self.message = None
        self.message_timer = 0

        # Animations
        self.animations = []

        # Load assets
        self._load_assets()

        # Create background
        self._create_background()

    def _load_assets(self):
        """Load images for the game."""
        # Load piece images
        black_piece = self.asset_manager.load_image(
            "black_piece.png", (self.PIECE_RADIUS * 2, self.PIECE_RADIUS * 2)
        )
        red_piece = self.asset_manager.load_image(
            "red_piece.png", (self.PIECE_RADIUS * 2, self.PIECE_RADIUS * 2)
        )

    def _create_background(self):
        """Create a static background with subtle patterns."""
        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.background.fill(self.COLORS["BACKGROUND"])

        # Add subtle pattern
        for i in range(0, self.SCREEN_WIDTH, 20):
            for j in range(0, self.SCREEN_HEIGHT, 20):
                if (i + j) % 40 == 0:
                    pygame.draw.circle(self.background, (230, 235, 245), (i, j), 2)

    def display_state(self, game_state: GameState) -> None:
        """Display the current game state."""
        self.current_game_state = game_state
        self._draw_screen()

    def _draw_screen(self):
        """Render the full game screen."""
        # Start with the background
        self.screen.blit(self.background, (0, 0))

        # Draw the board
        self._draw_board()

        # Draw the sidebar
        self._draw_sidebar()

        # Draw animations if any
        if self.animations:
            self._process_animations()

        # Draw any active messages
        if self.message and self.message_timer > 0:
            self._draw_message()
            self.message_timer -= 1

        # Update the display
        pygame.display.flip()
        self.clock.tick(self.FPS)

    def _draw_board(self):
        """Draw the game board and pieces."""
        # Use the enhanced renderer to draw the board
        valid_move_positions = [
            (m.end_pos[0], m.end_pos[1])
            for m in self.valid_moves
            if m.end_pos
            and 0 <= m.end_pos[0] < self.BOARD_ROWS
            and 0 <= m.end_pos[1] < self.BOARD_COLS
        ]

        self.board_renderer.render(
            screen=self.screen,
            board=self.current_game_state.board,
            selected_pos=self.selected_pos,
            valid_moves=valid_move_positions,
        )

    def _draw_sidebar(self):
        """Draw the game information sidebar with adaptive layout."""
        if not self.current_game_state:
            return

        # Sidebar background
        sidebar_x = self.BOARD_AREA_WIDTH
        sidebar_rect = pygame.Rect(sidebar_x, 0, self.SIDEBAR_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLORS["PANEL_BACKGROUND"], sidebar_rect)
        
        # Get sidebar center for alignment
        sidebar_center_x = sidebar_x + self.SIDEBAR_WIDTH // 2
        
        # Calculate responsive spacing based on screen height
        # This ensures elements are properly spaced regardless of screen size
        spacing_factor = self.SCREEN_HEIGHT / 768  # Base on standard height
        spacing_unit = int(25 * spacing_factor)
        
        # Calculate vertical positions relative to screen height
        # Use relative percentages rather than fixed pixel values
        title_y = int(self.SCREEN_HEIGHT * 0.05)  # 5% from top
        player_y = title_y + spacing_unit * 3
        score_section_y = player_y + spacing_unit * 3
        
        # Game title with scaled positioning
        title = self.FONTS["TITLE"].render(
            "KULIBRAT", True, self.COLORS["PANEL_HEADER"]
        )
        title_rect = title.get_rect(centerx=sidebar_center_x, y=title_y)
        self.screen.blit(title, title_rect)

        # Current player indicator
        current_player = self.current_game_state.current_player
        player_color = (
            self.COLORS["BLACK_PIECE"]
            if current_player == PlayerColor.BLACK
            else self.COLORS["RED_PIECE"]
        )

        player_text = self.FONTS["HEADER"].render(
            f"{current_player.name}'s Turn", True, player_color
        )
        player_rect = player_text.get_rect(
            centerx=sidebar_center_x, y=player_y
        )
        self.screen.blit(player_text, player_rect)

        # Score display section
        score_title = self.FONTS["NORMAL"].render(
            "SCORE", True, self.COLORS["TEXT_DARK"]
        )
        score_rect = score_title.get_rect(
            centerx=sidebar_center_x, y=score_section_y
        )
        self.screen.blit(score_title, score_rect)

        # Adaptively size UI elements based on sidebar width
        # Ensure minimum sizes for playability
        bar_width = max(100, int(self.SIDEBAR_WIDTH * 0.75))
        bar_height = max(20, int(spacing_unit * 1.2))
        
        # Calculate margins to center elements
        bar_x = sidebar_x + (self.SIDEBAR_WIDTH - bar_width) // 2
        
        # Score values with visual indicator of target
        target_score = self.current_game_state.target_score
        black_score = self.current_game_state.scores[PlayerColor.BLACK]
        red_score = self.current_game_state.scores[PlayerColor.RED]

        # Black score bar (position based on score title)
        black_bar_y = score_rect.bottom + spacing_unit
        black_progress = min(black_score / target_score, 1.0)
        
        # Background bar
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            pygame.Rect(bar_x, black_bar_y, bar_width, bar_height),
            border_radius=3
        )
        
        # Progress bar
        pygame.draw.rect(
            self.screen,
            self.COLORS["BLACK_SCORE"],
            pygame.Rect(
                bar_x, black_bar_y, int(bar_width * black_progress), bar_height
            ),
            border_radius=3
        )

        # Target marker on black bar
        target_x = bar_x + bar_width
        pygame.draw.line(
            self.screen,
            self.COLORS["TEXT_ACCENT"],
            (target_x, black_bar_y - 3),
            (target_x, black_bar_y + bar_height + 3),
            2,
        )

        # Black score label - responsive sizing for small screens
        # Scale text to fit available width
        black_score_text = f"BLACK: {black_score}/{target_score}"
        black_label = self.FONTS["SMALL"].render(
            black_score_text, True, self.COLORS["BLACK_SCORE"]
        )
        
        # Check if label is too wide for the bar
        if black_label.get_width() > bar_width - 10:
            # Use tiny font or abbreviate text on small screens
            black_label = self.FONTS["TINY"].render(
                f"BLK: {black_score}/{target_score}", True, self.COLORS["BLACK_SCORE"]
            )
        
        # Position label within the bar
        label_y = black_bar_y + (bar_height - black_label.get_height()) // 2
        self.screen.blit(black_label, (bar_x + 5, label_y))

        # Red score bar with spacing relative to black bar
        red_bar_y = black_bar_y + bar_height + spacing_unit
        red_progress = min(red_score / target_score, 1.0)
        
        # Background bar
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            pygame.Rect(bar_x, red_bar_y, bar_width, bar_height),
            border_radius=3
        )
        
        # Progress bar
        pygame.draw.rect(
            self.screen,
            self.COLORS["RED_SCORE"],
            pygame.Rect(bar_x, red_bar_y, int(bar_width * red_progress), bar_height),
            border_radius=3
        )

        # Target marker on red bar
        pygame.draw.line(
            self.screen,
            self.COLORS["TEXT_ACCENT"],
            (target_x, red_bar_y - 3),
            (target_x, red_bar_y + bar_height + 3),
            2,
        )

        # Red score label - responsive sizing
        red_score_text = f"RED: {red_score}/{target_score}"
        red_label = self.FONTS["SMALL"].render(
            red_score_text, True, self.COLORS["RED_SCORE"]
        )
        
        # Check if label is too wide for the bar
        if red_label.get_width() > bar_width - 10:
            # Use tiny font or abbreviate text on small screens
            red_label = self.FONTS["TINY"].render(
                f"RED: {red_score}/{target_score}", True, self.COLORS["RED_SCORE"]
            )
        
        # Position label within the bar
        label_y = red_bar_y + (bar_height - red_label.get_height()) // 2
        self.screen.blit(red_label, (bar_x + 5, label_y))

        # Calculate remaining space for the rest of the UI
        remaining_height = self.SCREEN_HEIGHT - (red_bar_y + bar_height + spacing_unit)
        
        # Determine if we need to compress the layout for small screens
        is_compact_layout = remaining_height < 300 or self.SCREEN_HEIGHT < 600
        
        # Pieces off board information section
        if is_compact_layout:
            # For very small screens, show a simplified view
            pieces_y = red_bar_y + bar_height + spacing_unit
            pieces_title = self.FONTS["SMALL"].render(
                "AVAILABLE PIECES", True, self.COLORS["TEXT_DARK"]
            )
        else:
            pieces_y = red_bar_y + bar_height + spacing_unit * 2
            pieces_title = self.FONTS["NORMAL"].render(
                "PIECES AVAILABLE", True, self.COLORS["TEXT_DARK"]
            )
            
        pieces_rect = pieces_title.get_rect(centerx=sidebar_center_x, y=pieces_y)
        self.screen.blit(pieces_title, pieces_rect)

        # Get piece counts
        black_pieces = self.current_game_state.pieces_off_board[PlayerColor.BLACK]
        red_pieces = self.current_game_state.pieces_off_board[PlayerColor.RED]

        # Adjust piece radius based on sidebar width for small screens
        display_radius = min(self.PIECE_RADIUS, int(self.SIDEBAR_WIDTH * 0.1))
        display_radius = max(display_radius, 10)  # Ensure minimum visibility
        
        # Calculate spacing between pieces
        piece_spacing = max(5, min(10, int(self.SIDEBAR_WIDTH * 0.025)))
        
        # Black pieces - centered in sidebar
        black_pieces_y = pieces_rect.bottom + spacing_unit
        
        # Calculate total width needed for pieces
        pieces_total_width = 4 * (display_radius * 2 + piece_spacing)
        pieces_start_x = sidebar_center_x - pieces_total_width // 2 + display_radius
        
        for i in range(4):
            piece_x = pieces_start_x + i * (display_radius * 2 + piece_spacing)
            piece_y = black_pieces_y

            if i < black_pieces:
                # Available piece
                pygame.draw.circle(
                    self.screen,
                    self.COLORS["BLACK_PIECE"],
                    (piece_x, piece_y),
                    display_radius,
                )
            else:
                # Used piece (outline only)
                pygame.draw.circle(
                    self.screen,
                    self.COLORS["BLACK_PIECE"],
                    (piece_x, piece_y),
                    display_radius,
                    2,
                )

        # Only show labels if there's enough space
        if not is_compact_layout:
            black_pieces_label = self.FONTS["SMALL"].render(
                f"BLACK: {black_pieces}", True, self.COLORS["BLACK_SCORE"]
            )
            label_rect = black_pieces_label.get_rect(centerx=sidebar_center_x, y=black_pieces_y + display_radius + 5)
            self.screen.blit(black_pieces_label, label_rect)

        # Red pieces with adaptive spacing
        red_pieces_y = black_pieces_y + display_radius * 2 + spacing_unit
        
        for i in range(4):
            piece_x = pieces_start_x + i * (display_radius * 2 + piece_spacing)
            piece_y = red_pieces_y

            if i < red_pieces:
                # Available piece
                pygame.draw.circle(
                    self.screen,
                    self.COLORS["RED_PIECE"],
                    (piece_x, piece_y),
                    display_radius,
                )
            else:
                # Used piece (outline only)
                pygame.draw.circle(
                    self.screen,
                    self.COLORS["RED_PIECE"],
                    (piece_x, piece_y),
                    display_radius,
                    2,
                )

        if not is_compact_layout:
            red_pieces_label = self.FONTS["SMALL"].render(
                f"RED: {red_pieces}", True, self.COLORS["RED_SCORE"]
            )
            label_rect = red_pieces_label.get_rect(centerx=sidebar_center_x, y=red_pieces_y + display_radius + 5)
            self.screen.blit(red_pieces_label, label_rect)

        # Game statistics section - only show if there's space
        # For very small screens, skip statistics to save space
        if not is_compact_layout:
            stats_y = red_pieces_y + display_radius * 2 + spacing_unit * 2
            stats_title = self.FONTS["NORMAL"].render(
                "GAME STATISTICS", True, self.COLORS["TEXT_DARK"]
            )
            stats_rect = stats_title.get_rect(centerx=sidebar_center_x, y=stats_y)
            self.screen.blit(stats_title, stats_rect)

            # Counters for players
            black_moves = self.statistics.moves_by_player[PlayerColor.BLACK]
            red_moves = self.statistics.moves_by_player[PlayerColor.RED]
            
            # Compact display for statistics
            stats_y = stats_rect.bottom + spacing_unit
            stats_text = f"Turn: {self.statistics.total_turns + 1}  •  Moves: B:{black_moves} R:{red_moves}"
            stats_label = self.FONTS["SMALL"].render(stats_text, True, self.COLORS["TEXT_DARK"])
            stats_rect = stats_label.get_rect(centerx=sidebar_center_x, y=stats_y)
            self.screen.blit(stats_label, stats_rect)
        
        # Help button at the bottom of sidebar - always show
        # Calculate position to ensure it's always visible at the bottom
        button_height = max(30, int(spacing_unit * 1.5))
        help_y = self.SCREEN_HEIGHT - button_height - spacing_unit
        
        # Size button based on available width
        button_width = min(bar_width, self.SIDEBAR_WIDTH - 20)
        button_x = sidebar_center_x - button_width // 2
        
        help_button = pygame.Rect(button_x, help_y, button_width, button_height)
        pygame.draw.rect(
            self.screen, self.COLORS["BUTTON"], help_button, border_radius=5
        )

        # Scale button text to fit
        help_text = self.FONTS["NORMAL"].render(
            "Game Rules", True, self.COLORS["BUTTON_TEXT"]
        )
        
        # If text is too wide, use smaller font
        if help_text.get_width() > button_width - 10:
            help_text = self.FONTS["SMALL"].render(
                "Game Rules", True, self.COLORS["BUTTON_TEXT"]
            )
            
        help_rect = help_text.get_rect(center=help_button.center)
        self.screen.blit(help_text, help_rect)

        # Add the button to the list of UI buttons
        self.buttons = [("help", help_button)]

    def _draw_message(self):
        """Display temporary messages to the user with responsive sizing."""
        if not self.message:
            return

        # Calculate message bar height based on screen dimensions
        # Ensure it's visible but not too large on small screens
        message_height = min(int(self.SCREEN_HEIGHT * 0.1), 80)
        message_height = max(message_height, 40)  # Ensure minimum height

        # Create semi-transparent overlay
        message_surface = pygame.Surface((self.SCREEN_WIDTH, message_height), pygame.SRCALPHA)
        message_surface.fill((50, 50, 50, 200))

        # Render message text with appropriate font size
        # Choose font size based on message bar size
        if message_height < 50:
            font = self.FONTS["SMALL"]
        else:
            font = self.FONTS["NORMAL"]
            
        # Render message text
        message_text = font.render(self.message, True, (255, 255, 255))
        
        # Check if message is too wide for the screen
        if message_text.get_width() > self.SCREEN_WIDTH - 20:
            # Use smaller font or truncate on very small screens
            if message_height < 40:
                # Severely constrained space - abbreviate message
                words = self.message.split()
                if len(words) > 4:
                    abbreviated = " ".join(words[:3]) + "..."
                    message_text = self.FONTS["TINY"].render(abbreviated, True, (255, 255, 255))
                else:
                    # Just use smallest font
                    message_text = self.FONTS["TINY"].render(self.message, True, (255, 255, 255))
            else:
                # Try with smaller font
                message_text = self.FONTS["SMALL"].render(self.message, True, (255, 255, 255))
                
                # If still too wide, abbreviate
                if message_text.get_width() > self.SCREEN_WIDTH - 20:
                    message_text = self.FONTS["TINY"].render(self.message, True, (255, 255, 255))
        
        # Center the text in the message bar
        text_rect = message_text.get_rect(
            center=(message_surface.get_width() // 2, message_surface.get_height() // 2)
        )
        message_surface.blit(message_text, text_rect)

        # Position at bottom of screen
        self.screen.blit(message_surface, (0, self.SCREEN_HEIGHT - message_height))

    def _process_animations(self):
        """Process and render any active animations."""
        completed = []
        for i, anim in enumerate(self.animations):
            # Check if animation is complete
            if anim["current_frame"] >= anim["total_frames"]:
                completed.append(i)
                continue

            # Process next frame
            anim["current_frame"] += 1
            progress = anim["current_frame"] / anim["total_frames"]

            # Calculate current position based on animation type
            if anim["type"] == "move":
                start_x, start_y = anim["start_pos"]
                end_x, end_y = anim["end_pos"]
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (end_y - start_y) * progress

                # Draw the animated piece
                piece_color = (
                    self.COLORS["BLACK_PIECE"]
                    if anim["piece_color"] == PlayerColor.BLACK.value
                    else self.COLORS["RED_PIECE"]
                )
                pygame.draw.circle(
                    self.screen,
                    piece_color,
                    (int(current_x), int(current_y)),
                    self.PIECE_RADIUS,
                )

            elif anim["type"] == "fade":
                alpha = int(255 * (1 - progress))
                fade_surface = pygame.Surface(
                    (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA
                )
                fade_surface.fill(
                    (anim["color"][0], anim["color"][1], anim["color"][2], alpha)
                )
                self.screen.blit(fade_surface, (0, 0))

        # Remove completed animations
        for i in sorted(completed, reverse=True):
            self.animations.pop(i)

    def _add_move_animation(self, start_pos, end_pos, piece_color, frames=15):
        """Add a piece movement animation."""
        # Convert board coordinates to screen coordinates
        if isinstance(start_pos, tuple) and len(start_pos) == 2:
            start_row, start_col = start_pos
            start_x = self.BOARD_X + start_col * self.CELL_SIZE + self.CELL_SIZE // 2
            start_y = self.BOARD_Y + start_row * self.CELL_SIZE + self.CELL_SIZE // 2
        else:
            # If not board coordinates, assume direct screen coordinates
            start_x, start_y = start_pos

        # If end position is on the board, convert to screen coordinates
        if isinstance(end_pos, tuple) and len(end_pos) == 2:
            end_row, end_col = end_pos

            # Check if the end position is off the board (scoring move)
            if 0 <= end_row < self.BOARD_ROWS and 0 <= end_col < self.BOARD_COLS:
                end_x = self.BOARD_X + end_col * self.CELL_SIZE + self.CELL_SIZE // 2
                end_y = self.BOARD_Y + end_row * self.CELL_SIZE + self.CELL_SIZE // 2
            else:
                # For scoring moves, animate off the board
                direction = 1 if piece_color == PlayerColor.BLACK.value else -1
                end_x = self.BOARD_X + end_col * self.CELL_SIZE + self.CELL_SIZE // 2
                end_y = (
                    self.BOARD_Y + self.BOARD_HEIGHT + self.CELL_SIZE
                    if direction > 0
                    else self.BOARD_Y - self.CELL_SIZE
                )
        else:
            # If not board coordinates, assume direct screen coordinates
            end_x, end_y = end_pos

        # Add the animation to the queue
        self.animations.append(
            {
                "type": "move",
                "start_pos": (start_x, start_y),
                "end_pos": (end_x, end_y),
                "piece_color": piece_color,
                "current_frame": 0,
                "total_frames": frames,
            }
        )

    def _add_fade_animation(self, color, frames=30):
        """Add a fade animation (for transitions)."""
        self.animations.append(
            {"type": "fade", "color": color, "current_frame": 0, "total_frames": frames}
        )

    def show_message(self, message: str) -> None:
        """Display a message to the user."""
        self.message = message
        self.message_timer = 120  # Display for 2 seconds at 60 FPS

        # Force redraw to show message immediately
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

            # Update the screen
            self._draw_screen()
            self.clock.tick(self.FPS)

        # Record the selected move for statistics
        if self.selected_move:
            self.statistics.record_move(player_color, self.selected_move)

            # If it's a move with start and end positions
            if self.selected_move.start_pos and self.selected_move.end_pos:
                # Determine piece value
                piece_value = self.current_game_state.board[
                    self.selected_move.start_pos[0]
                ][self.selected_move.start_pos[1]]

                # Animate the move
                self.board_renderer.animate_move(
                    screen=self.screen,
                    board=self.current_game_state.board,
                    start_pos=self.selected_move.start_pos,
                    end_pos=self.selected_move.end_pos,
                    piece_value=piece_value,
                )

            # If move is an attack, record capture
            if self.selected_move.move_type == MoveType.ATTACK:
                self.statistics.record_capture(player_color)

                # Add capture particles
                self.board_renderer.add_capture_particles(
                    row=self.selected_move.end_pos[0],
                    col=self.selected_move.end_pos[1],
                    is_black_capturing=(player_color == PlayerColor.BLACK),
                )

            # For scoring moves
            if self.selected_move.move_type in [MoveType.DIAGONAL, MoveType.JUMP] and (
                self.selected_move.end_pos[0] < 0
                or self.selected_move.end_pos[0] >= self.BOARD_ROWS
            ):
                # Add score particles
                self.board_renderer.add_score_particles(
                    row=self.selected_move.start_pos[0],
                    col=self.selected_move.start_pos[1],
                    is_black=(player_color == PlayerColor.BLACK),
                )

            # End turn timer
            self.statistics.end_turn_timer(player_color)

        self.waiting_for_move = False
        self.valid_moves = []

        # Reset selection
        self.selected_pos = None

        # Return the selected move
        return self.selected_move

    def _handle_mouse_click(self, mouse_pos):
        """Handle mouse clicks during move selection."""
        # Check if a UI button was clicked
        for button_id, button_rect in self.buttons:
            if button_rect.collidepoint(mouse_pos):
                if button_id == "help":
                    self._show_rules()
                return

        # Check if click is on the board
        if not (
            self.BOARD_X <= mouse_pos[0] <= self.BOARD_X + self.BOARD_WIDTH
            and self.BOARD_Y <= mouse_pos[1] <= self.BOARD_Y + self.BOARD_HEIGHT
        ):
            return

        # Convert mouse position to board coordinates
        board_x = mouse_pos[0] - self.BOARD_X
        board_y = mouse_pos[1] - self.BOARD_Y

        col = board_x // self.CELL_SIZE
        row = board_y // self.CELL_SIZE

        # Validate coordinates
        if not (0 <= row < self.BOARD_ROWS and 0 <= col < self.BOARD_COLS):
            return

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
            valid_end_moves = [
                m
                for m in self.valid_moves
                if m.start_pos == self.selected_pos
                and (
                    m.end_pos == pos
                    or (
                        m.move_type in [MoveType.DIAGONAL, MoveType.JUMP]
                        and (
                            (
                                self.current_game_state.current_player
                                == PlayerColor.BLACK
                                and pos[0] == 3
                                and abs(pos[1] - self.selected_pos[1]) == 1
                            )
                            or (
                                self.current_game_state.current_player
                                == PlayerColor.RED
                                and pos[0] == 0
                                and abs(pos[1] - self.selected_pos[1]) == 1
                            )
                        )
                    )
                )
            ]

            if valid_end_moves:
                # Make the move
                self.selected_move = valid_end_moves[0]
                self.waiting_for_move = False

                # Show appropriate message based on move type
                if self.selected_move.move_type == MoveType.DIAGONAL:
                    if (
                        self.selected_move.end_pos[0] < 0
                        or self.selected_move.end_pos[0] >= self.BOARD_ROWS
                    ):
                        self.show_message("Diagonal move to score a point!")
                    else:
                        self.show_message(
                            f"Diagonal move from {self.selected_pos} to {pos}."
                        )
                elif self.selected_move.move_type == MoveType.ATTACK:
                    self.show_message(f"Attack move from {self.selected_pos} to {pos}.")
                elif self.selected_move.move_type == MoveType.JUMP:
                    if (
                        self.selected_move.end_pos[0] < 0
                        or self.selected_move.end_pos[0] >= self.BOARD_ROWS
                    ):
                        self.show_message("Jump to score a point!")
                    else:
                        self.show_message(f"Jump from {self.selected_pos} to {pos}.")
            else:
                # If clicked on own piece, change selection
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
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))

        # Create rules panel
        panel_width = int(self.SCREEN_WIDTH * 0.8)
        panel_height = int(self.SCREEN_HEIGHT * 0.8)
        panel_x = (self.SCREEN_WIDTH - panel_width) // 2
        panel_y = (self.SCREEN_HEIGHT - panel_height) // 2

        panel = pygame.Surface((panel_width, panel_height))
        panel.fill(self.COLORS["PANEL_BACKGROUND"])

        # Rules title
        title = self.FONTS["HEADER"].render(
            "KULIBRAT RULES", True, self.COLORS["PANEL_HEADER"]
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
        ]

        y_offset = 80
        for line in rules_text:
            if (
                line.startswith("OBJECTIVE:")
                or line.startswith("MOVES:")
                or line.startswith("SCORING:")
                or line.startswith("SPECIAL RULES:")
            ):
                text = self.FONTS["NORMAL"].render(
                    line, True, self.COLORS["TEXT_ACCENT"]
                )
                y_offset += 10
            else:
                text = self.FONTS["SMALL"].render(line, True, self.COLORS["TEXT_DARK"])

            panel.blit(text, (40, y_offset))
            y_offset += 30

        # Close button
        close_button = pygame.Rect(panel_width // 2 - 50, panel_height - 60, 100, 40)
        pygame.draw.rect(panel, self.COLORS["BUTTON"], close_button, border_radius=5)

        close_text = self.FONTS["NORMAL"].render(
            "Close", True, self.COLORS["BUTTON_TEXT"]
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

    def show_winner(self, winner: Optional[PlayerColor], game_state: GameState) -> None:
        """Display the winner of the game with animation."""
        self.current_game_state = game_state

        # Ensure the last turn is counted
        if not self.statistics.total_turns:
            self.statistics.record_turn()

        # Add fade-in animation
        self._add_fade_animation((0, 0, 0), frames=20)

        # Update the display with the animation
        for _ in range(25):  # Allow animation to complete
            self._draw_screen()
            self.clock.tick(60)

        # Draw the final game state
        self._draw_screen()

        # Create winner announcement overlay
        overlay = pygame.Surface(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 150))

        # Create winner panel
        panel_width = int(self.SCREEN_WIDTH * 0.7)
        panel_height = int(self.SCREEN_HEIGHT * 0.6)
        panel_x = (self.SCREEN_WIDTH - panel_width) // 2
        panel_y = (self.SCREEN_HEIGHT - panel_height) // 2

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((250, 250, 255, 240))

        # Draw winner information
        if winner:
            winner_color = (
                self.COLORS["BLACK_PIECE"]
                if winner == PlayerColor.BLACK
                else self.COLORS["RED_PIECE"]
            )

            # Title with glow effect
            for offset in range(5, 0, -1):
                title = self.FONTS["TITLE"].render(
                    "GAME OVER",
                    True,
                    (winner_color[0], winner_color[1], winner_color[2], 50 // offset),
                )
                title_rect = title.get_rect(centerx=panel_width // 2, y=20)
                panel.blit(title, (title_rect.x + offset, title_rect.y + offset))

            title = self.FONTS["TITLE"].render("GAME OVER", True, winner_color)
            title_rect = title.get_rect(centerx=panel_width // 2, y=20)
            panel.blit(title, title_rect)

            # Winner announcement
            winner_text = self.FONTS["HEADER"].render(
                f"{winner.name} PLAYER WINS!", True, winner_color
            )
            winner_rect = winner_text.get_rect(centerx=panel_width // 2, y=100)
            panel.blit(winner_text, winner_rect)
        else:
            # Draw case
            title = self.FONTS["TITLE"].render(
                "GAME OVER", True, self.COLORS["TEXT_DARK"]
            )
            title_rect = title.get_rect(centerx=panel_width // 2, y=20)
            panel.blit(title, title_rect)

            draw_text = self.FONTS["HEADER"].render(
                "IT'S A DRAW!", True, self.COLORS["TEXT_DARK"]
            )
            draw_rect = draw_text.get_rect(centerx=panel_width // 2, y=100)
            panel.blit(draw_text, draw_rect)

        # Score summary
        score_y = 170
        score_text = self.FONTS["NORMAL"].render(
            "FINAL SCORE", True, self.COLORS["TEXT_DARK"]
        )
        score_rect = score_text.get_rect(centerx=panel_width // 2, y=score_y)
        panel.blit(score_text, score_rect)

        black_score = game_state.scores[PlayerColor.BLACK]
        red_score = game_state.scores[PlayerColor.RED]

        score_detail = self.FONTS["NORMAL"].render(
            f"BLACK: {black_score}  -  RED: {red_score}", True, self.COLORS["TEXT_DARK"]
        )
        score_detail_rect = score_detail.get_rect(
            centerx=panel_width // 2, y=score_y + 40
        )
        panel.blit(score_detail, score_detail_rect)

        # Game statistics
        stats_y = score_y + 100
        stats_text = self.FONTS["NORMAL"].render(
            "GAME STATISTICS", True, self.COLORS["TEXT_DARK"]
        )
        stats_rect = stats_text.get_rect(centerx=panel_width // 2, y=stats_y)
        panel.blit(stats_text, stats_rect)

        # Gather statistics
        stats_summary = self.statistics.get_summary()

        stats_lines = [
            f"Total Turns: {stats_summary['total_turns']}",
            f"Total Game Time: {stats_summary['total_game_time']}",
            f"BLACK Moves: {stats_summary['moves_by_player']['BLACK']}",
            f"RED Moves: {stats_summary['moves_by_player']['RED']}",
            f"BLACK Captures: {stats_summary['captures_by_player']['BLACK']}",
            f"RED Captures: {stats_summary['captures_by_player']['RED']}",
        ]

        stats_line_y = stats_y + 40
        for line in stats_lines:
            line_text = self.FONTS["SMALL"].render(line, True, self.COLORS["TEXT_DARK"])
            line_rect = line_text.get_rect(centerx=panel_width // 2, y=stats_line_y)
            panel.blit(line_text, line_rect)
            stats_line_y += 30

        # Play again button
        play_again_button = pygame.Rect(
            panel_width // 4 - 75, panel_height - 70, 150, 50
        )
        pygame.draw.rect(
            panel, self.COLORS["BUTTON"], play_again_button, border_radius=5
        )

        play_again_text = self.FONTS["NORMAL"].render(
            "Play Again", True, self.COLORS["BUTTON_TEXT"]
        )
        play_again_rect = play_again_text.get_rect(center=play_again_button.center)
        panel.blit(play_again_text, play_again_rect)

        # Quit button
        quit_button = pygame.Rect(panel_width * 3 // 4 - 75, panel_height - 70, 150, 50)
        pygame.draw.rect(panel, self.COLORS["RED_PIECE"], quit_button, border_radius=5)

        quit_text = self.FONTS["NORMAL"].render(
            "Quit", True, self.COLORS["BUTTON_TEXT"]
        )
        quit_rect = quit_text.get_rect(center=quit_button.center)
        panel.blit(quit_text, quit_rect)

        # Display the results
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(panel, (panel_x, panel_y))
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
                    adjusted_pos = (mouse_pos[0] - panel_x, mouse_pos[1] - panel_y)

                    if play_again_button.collidepoint(adjusted_pos):
                        waiting = False
                        play_again = True
                    elif quit_button.collidepoint(adjusted_pos):
                        pygame.quit()
                        sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            self.clock.tick(30)

        # User wants to play again
        if play_again:
            # Add a fade out animation
            self._add_fade_animation((255, 255, 255), frames=20)

            # Update the display with the animation
            for _ in range(25):  # Allow animation to complete
                self._draw_screen()
                self.clock.tick(60)

            return
