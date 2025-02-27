import pygame
import sys
import os
import time
import math
from typing import List, Optional, Tuple, Dict, Any

# Import game core components
from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.move import Move
from src.core.move_type import MoveType
from src.ui.game_interface import GameInterface
from src.ui.console_interface import GameStatistics

"""
Enhanced PyGame GUI for the Kulibrat game.
"""

import pygame
import sys
import os
import time
import math
from typing import List, Optional, Tuple, Dict, Any

# Import game core components
from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.move import Move
from src.core.move_type import MoveType
from src.ui.game_interface import GameInterface
from src.ui.console_interface import GameStatistics
from src.ui.enhanced_board_renderer import EnhancedBoardRenderer
from src.ui.asset_manager import AssetManager
from src.ui.game_config_screen import GameConfigScreen


class KulibratGUI(GameInterface):
    """Enhanced graphical interface for the Kulibrat game with smooth animations."""
    
    def __init__(self, screen_width=1024, screen_height=768):
        """Initialize the GUI with responsive sizing."""
        pygame.init()
        pygame.display.set_caption('Kulibrat')
        
        # Screen configuration
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height

        # Check if fullscreen is requested in a config file or env variable
        self.fullscreen = False
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        # Color palette - modern and accessible
        self.COLORS = {
            'BACKGROUND': (245, 245, 250),
            'BOARD_DARK': (210, 215, 230),
            'BOARD_LIGHT': (235, 240, 250),
            'GRID_LINE': (180, 190, 210),
            'BLACK_PIECE': (40, 45, 55),
            'BLACK_PIECE_HIGHLIGHT': (80, 85, 95),
            'RED_PIECE': (200, 60, 70),
            'RED_PIECE_HIGHLIGHT': (230, 90, 100),
            'HIGHLIGHT_VALID': (100, 200, 100, 160),
            'HIGHLIGHT_SELECTED': (100, 150, 240, 160),
            'TEXT_DARK': (50, 55, 70),
            'TEXT_LIGHT': (245, 245, 250),
            'TEXT_ACCENT': (70, 130, 200),
            'PANEL_BACKGROUND': (250, 250, 255),
            'PANEL_HEADER': (70, 100, 170),
            'BLACK_SCORE': (40, 45, 55),
            'RED_SCORE': (200, 60, 70),
            'BUTTON': (70, 130, 200),
            'BUTTON_HOVER': (90, 150, 220),
            'BUTTON_TEXT': (255, 255, 255),
            'START_ROW_BLACK': (180, 200, 240, 100),
            'START_ROW_RED': (240, 200, 200, 100),
        }
        
        # Initialize asset manager
        self.asset_manager = AssetManager()
        
        # Font setup
        pygame.font.init()
        try:
            font_path = os.path.join("src", "ui", "assets", "fonts", "Roboto-Regular.ttf")
            if os.path.exists(font_path):
                self.FONTS = {
                    'TITLE': pygame.font.Font(font_path, 48),
                    'HEADER': pygame.font.Font(font_path, 32),
                    'NORMAL': pygame.font.Font(font_path, 24),
                    'SMALL': pygame.font.Font(font_path, 18),
                    'TINY': pygame.font.Font(font_path, 14)
                }
            else:
                self.FONTS = {
                    'TITLE': pygame.font.SysFont('Arial', 48),
                    'HEADER': pygame.font.SysFont('Arial', 32),
                    'NORMAL': pygame.font.SysFont('Arial', 24),
                    'SMALL': pygame.font.SysFont('Arial', 18),
                    'TINY': pygame.font.SysFont('Arial', 14)
                }
        except:
            # Fallback to system fonts if custom font fails
            self.FONTS = {
                'TITLE': pygame.font.SysFont('Arial', 48),
                'HEADER': pygame.font.SysFont('Arial', 32),
                'NORMAL': pygame.font.SysFont('Arial', 24),
                'SMALL': pygame.font.SysFont('Arial', 18),
                'TINY': pygame.font.SysFont('Arial', 14)
            }
        
        # Game configuration
        self.game_config = {
            "black_player": {
                "name": "Player 1",
                "type": "Human",
                "color": "Black"
            },
            "red_player": {
                "name": "Player 2", 
                "type": "AI (Random)",
                "color": "Red"
            },
            "target_score": 5,
            "ai_delay": 0.5,
            "fullscreen": False,
            "rl_model_path": "models/kulibrat_rl_model_best.pt"
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
            self.screen = pygame.display.set_mode((1024, 768), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.SCREEN_WIDTH, self.SCREEN_HEIGHT = self.screen.get_size()
            self.fullscreen = False
        
        # Layout calculations - updated for responsive design
        self.BOARD_MARGIN_X = int(self.SCREEN_WIDTH * 0.03)  # Dynamic margins
        self.BOARD_MARGIN_Y = int(self.SCREEN_HEIGHT * 0.05)
        
        # Responsive sidebar width based on screen dimensions
        sidebar_ratio = 0.25 if self.SCREEN_WIDTH > 1200 else 0.3
        self.SIDEBAR_WIDTH = int(self.SCREEN_WIDTH * sidebar_ratio)
        self.BOARD_AREA_WIDTH = self.SCREEN_WIDTH - self.SIDEBAR_WIDTH
        
        # Board dimensions
        self.BOARD_ROWS = 4
        self.BOARD_COLS = 3
        
        # Calculate cell size based on available space - larger on bigger screens
        max_area_width = min(
            self.BOARD_AREA_WIDTH - 2 * self.BOARD_MARGIN_X,
            self.SCREEN_HEIGHT - 2 * self.BOARD_MARGIN_Y * 1.2  # Slightly more vertical space
        )
        
        # Ensure board fits proportionally to the screen
        self.CELL_SIZE = int(max_area_width / max(self.BOARD_COLS, self.BOARD_ROWS * 1.05))
        
        # Board position - centered in the available area
        self.BOARD_WIDTH = self.CELL_SIZE * self.BOARD_COLS
        self.BOARD_HEIGHT = self.CELL_SIZE * self.BOARD_ROWS
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
            board_cols=self.BOARD_COLS
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
        """Load images and sounds for the game."""
        # Load piece images
        black_piece = self.asset_manager.load_image("black_piece.png", (self.PIECE_RADIUS * 2, self.PIECE_RADIUS * 2))
        red_piece = self.asset_manager.load_image("red_piece.png", (self.PIECE_RADIUS * 2, self.PIECE_RADIUS * 2))
        
        # Load sound effects if available
        self.sounds = {
            'move': self.asset_manager.load_sound("move.wav"),
            'capture': self.asset_manager.load_sound("capture.wav"),
            'score': self.asset_manager.load_sound("score.wav")
        }
    
    def _create_background(self):
        """Create a static background with subtle patterns."""
        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.background.fill(self.COLORS['BACKGROUND'])
        
        # Add subtle pattern
        for i in range(0, self.SCREEN_WIDTH, 20):
            for j in range(0, self.SCREEN_HEIGHT, 20):
                if (i + j) % 40 == 0:
                    pygame.draw.circle(
                        self.background, 
                        (230, 235, 245), 
                        (i, j), 
                        2
                    )
    
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
        valid_move_positions = [(m.end_pos[0], m.end_pos[1]) for m in self.valid_moves 
                              if m.end_pos and 0 <= m.end_pos[0] < self.BOARD_ROWS 
                              and 0 <= m.end_pos[1] < self.BOARD_COLS]
        
        self.board_renderer.render(
            screen=self.screen,
            board=self.current_game_state.board,
            selected_pos=self.selected_pos,
            valid_moves=valid_move_positions
        )
    
    def _draw_sidebar(self):
        """Draw the game information sidebar."""
        if not self.current_game_state:
            return
        
        # Sidebar background
        sidebar_x = self.BOARD_AREA_WIDTH
        sidebar_rect = pygame.Rect(sidebar_x, 0, self.SIDEBAR_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLORS['PANEL_BACKGROUND'], sidebar_rect)
        
        # Game title
        title = self.FONTS['TITLE'].render("KULIBRAT", True, self.COLORS['PANEL_HEADER'])
        title_rect = title.get_rect(centerx=sidebar_x + self.SIDEBAR_WIDTH // 2, y=30)
        self.screen.blit(title, title_rect)
        
        # Current player indicator
        current_player = self.current_game_state.current_player
        player_color = self.COLORS['BLACK_PIECE'] if current_player == PlayerColor.BLACK else self.COLORS['RED_PIECE']
        
        player_text = self.FONTS['HEADER'].render(f"{current_player.name}'s Turn", True, player_color)
        player_rect = player_text.get_rect(centerx=sidebar_x + self.SIDEBAR_WIDTH // 2, y=100)
        self.screen.blit(player_text, player_rect)
        
        # Score display
        score_y = 170
        score_title = self.FONTS['NORMAL'].render("SCORE", True, self.COLORS['TEXT_DARK'])
        score_rect = score_title.get_rect(centerx=sidebar_x + self.SIDEBAR_WIDTH // 2, y=score_y)
        self.screen.blit(score_title, score_rect)
        
        # Score values with visual indicator of target
        target_score = self.current_game_state.target_score
        black_score = self.current_game_state.scores[PlayerColor.BLACK]
        red_score = self.current_game_state.scores[PlayerColor.RED]
        
        # Score bar background
        bar_width = int(self.SIDEBAR_WIDTH * 0.7)
        bar_height = 30
        bar_x = sidebar_x + (self.SIDEBAR_WIDTH - bar_width) // 2
        black_bar_y = score_y + 40
        red_bar_y = black_bar_y + bar_height + 20
        
        # Black score bar
        black_progress = min(black_score / target_score, 1.0)
        pygame.draw.rect(self.screen, (220, 220, 220), pygame.Rect(bar_x, black_bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLORS['BLACK_SCORE'], pygame.Rect(bar_x, black_bar_y, int(bar_width * black_progress), bar_height))
        
        # Target marker on black bar
        target_x = bar_x + bar_width
        pygame.draw.line(self.screen, self.COLORS['TEXT_ACCENT'], (target_x, black_bar_y - 5), (target_x, black_bar_y + bar_height + 5), 2)
        
        # Black score label
        black_label = self.FONTS['NORMAL'].render(f"BLACK: {black_score}/{target_score}", True, self.COLORS['BLACK_SCORE'])
        black_label_rect = black_label.get_rect(centerx=sidebar_x + self.SIDEBAR_WIDTH // 2, y=black_bar_y + bar_height + 5)
        self.screen.blit(black_label, (bar_x, black_bar_y + 5))
        
        # Red score bar
        red_progress = min(red_score / target_score, 1.0)
        pygame.draw.rect(self.screen, (220, 220, 220), pygame.Rect(bar_x, red_bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLORS['RED_SCORE'], pygame.Rect(bar_x, red_bar_y, int(bar_width * red_progress), bar_height))
        
        # Target marker on red bar
        pygame.draw.line(self.screen, self.COLORS['TEXT_ACCENT'], (target_x, red_bar_y - 5), (target_x, red_bar_y + bar_height + 5), 2)
        
        # Red score label
        red_label = self.FONTS['NORMAL'].render(f"RED: {red_score}/{target_score}", True, self.COLORS['RED_SCORE'])
        red_label_rect = red_label.get_rect(centerx=sidebar_x + self.SIDEBAR_WIDTH // 2, y=red_bar_y + bar_height + 5)
        self.screen.blit(red_label, (bar_x, red_bar_y + 5))
        
        # Pieces off board information
        pieces_y = red_bar_y + bar_height + 50
        pieces_title = self.FONTS['NORMAL'].render("PIECES AVAILABLE", True, self.COLORS['TEXT_DARK'])
        pieces_rect = pieces_title.get_rect(centerx=sidebar_x + self.SIDEBAR_WIDTH // 2, y=pieces_y)
        self.screen.blit(pieces_title, pieces_rect)
        
        # Draw piece counts with visual indicators
        black_pieces = self.current_game_state.pieces_off_board[PlayerColor.BLACK]
        red_pieces = self.current_game_state.pieces_off_board[PlayerColor.RED]
        
        # Black pieces
        black_pieces_y = pieces_y + 40
        for i in range(4):
            piece_x = bar_x + i * (self.PIECE_RADIUS * 2 + 10)
            piece_y = black_pieces_y
            
            if i < black_pieces:
                # Available piece
                pygame.draw.circle(self.screen, self.COLORS['BLACK_PIECE'], (piece_x + self.PIECE_RADIUS, piece_y + self.PIECE_RADIUS), self.PIECE_RADIUS)
            else:
                # Used piece (outline only)
                pygame.draw.circle(self.screen, self.COLORS['BLACK_PIECE'], (piece_x + self.PIECE_RADIUS, piece_y + self.PIECE_RADIUS), self.PIECE_RADIUS, 2)
        
        black_pieces_label = self.FONTS['SMALL'].render(f"BLACK: {black_pieces} available", True, self.COLORS['BLACK_SCORE'])
        self.screen.blit(black_pieces_label, (bar_x, black_pieces_y + self.PIECE_RADIUS * 2 + 5))
        
        # Red pieces
        red_pieces_y = black_pieces_y + self.PIECE_RADIUS * 2 + 30
        for i in range(4):
            piece_x = bar_x + i * (self.PIECE_RADIUS * 2 + 10)
            piece_y = red_pieces_y
            
            if i < red_pieces:
                # Available piece
                pygame.draw.circle(self.screen, self.COLORS['RED_PIECE'], (piece_x + self.PIECE_RADIUS, piece_y + self.PIECE_RADIUS), self.PIECE_RADIUS)
            else:
                # Used piece (outline only)
                pygame.draw.circle(self.screen, self.COLORS['RED_PIECE'], (piece_x + self.PIECE_RADIUS, piece_y + self.PIECE_RADIUS), self.PIECE_RADIUS, 2)
        
        red_pieces_label = self.FONTS['SMALL'].render(f"RED: {red_pieces} available", True, self.COLORS['RED_SCORE'])
        self.screen.blit(red_pieces_label, (bar_x, red_pieces_y + self.PIECE_RADIUS * 2 + 5))
        
        # Game statistics section
        stats_y = red_pieces_y + self.PIECE_RADIUS * 2 + 50
        stats_title = self.FONTS['NORMAL'].render("GAME STATISTICS", True, self.COLORS['TEXT_DARK'])
        stats_rect = stats_title.get_rect(centerx=sidebar_x + self.SIDEBAR_WIDTH // 2, y=stats_y)
        self.screen.blit(stats_title, stats_rect)
        
        # Display turn count and move types
        turn_label = self.FONTS['SMALL'].render(f"Turn: {self.statistics.total_turns + 1}", True, self.COLORS['TEXT_DARK'])
        self.screen.blit(turn_label, (bar_x, stats_y + 40))
        
        # Display move counts for each player
        black_moves = self.statistics.moves_by_player[PlayerColor.BLACK]
        red_moves = self.statistics.moves_by_player[PlayerColor.RED]
        
        move_label_black = self.FONTS['SMALL'].render(f"Black Moves: {black_moves}", True, self.COLORS['BLACK_SCORE'])
        move_label_red = self.FONTS['SMALL'].render(f"Red Moves: {red_moves}", True, self.COLORS['RED_SCORE'])
        
        self.screen.blit(move_label_black, (bar_x, stats_y + 70))
        self.screen.blit(move_label_red, (bar_x, stats_y + 100))
        
        # Rules and help button at the bottom of sidebar
        help_y = self.SCREEN_HEIGHT - 100
        help_button = pygame.Rect(bar_x, help_y, bar_width, 40)
        pygame.draw.rect(self.screen, self.COLORS['BUTTON'], help_button, border_radius=5)
        
        help_text = self.FONTS['NORMAL'].render("Game Rules", True, self.COLORS['BUTTON_TEXT'])
        help_rect = help_text.get_rect(center=help_button.center)
        self.screen.blit(help_text, help_rect)
        
        # Add the button to the list of UI buttons
        self.buttons = [("help", help_button)]
    
    def _draw_message(self):
        """Display temporary messages to the user."""
        if not self.message:
            return
            
        # Create semi-transparent overlay
        message_surface = pygame.Surface((self.SCREEN_WIDTH, 80), pygame.SRCALPHA)
        message_surface.fill((50, 50, 50, 200))
        
        # Render message text
        message_text = self.FONTS['NORMAL'].render(self.message, True, (255, 255, 255))
        text_rect = message_text.get_rect(center=(message_surface.get_width() // 2, message_surface.get_height() // 2))
        message_surface.blit(message_text, text_rect)
        
        # Position at bottom of screen
        self.screen.blit(message_surface, (0, self.SCREEN_HEIGHT - 80))
    
    def _process_animations(self):
        """Process and render any active animations."""
        completed = []
        for i, anim in enumerate(self.animations):
            # Check if animation is complete
            if anim['current_frame'] >= anim['total_frames']:
                completed.append(i)
                continue
                
            # Process next frame
            anim['current_frame'] += 1
            progress = anim['current_frame'] / anim['total_frames']
            
            # Calculate current position based on animation type
            if anim['type'] == 'move':
                start_x, start_y = anim['start_pos']
                end_x, end_y = anim['end_pos']
                current_x = start_x + (end_x - start_x) * progress
                current_y = start_y + (end_y - start_y) * progress
                
                # Draw the animated piece
                piece_color = self.COLORS['BLACK_PIECE'] if anim['piece_color'] == PlayerColor.BLACK.value else self.COLORS['RED_PIECE']
                pygame.draw.circle(self.screen, piece_color, (int(current_x), int(current_y)), self.PIECE_RADIUS)
            
            elif anim['type'] == 'fade':
                alpha = int(255 * (1 - progress))
                fade_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                fade_surface.fill((anim['color'][0], anim['color'][1], anim['color'][2], alpha))
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
                end_y = self.BOARD_Y + self.BOARD_HEIGHT + self.CELL_SIZE if direction > 0 else self.BOARD_Y - self.CELL_SIZE
        else:
            # If not board coordinates, assume direct screen coordinates
            end_x, end_y = end_pos
        
        # Add the animation to the queue
        self.animations.append({
            'type': 'move',
            'start_pos': (start_x, start_y),
            'end_pos': (end_x, end_y),
            'piece_color': piece_color,
            'current_frame': 0,
            'total_frames': frames
        })
    
    def _add_fade_animation(self, color, frames=30):
        """Add a fade animation (for transitions)."""
        self.animations.append({
            'type': 'fade',
            'color': color,
            'current_frame': 0,
            'total_frames': frames
        })
    
    def show_message(self, message: str) -> None:
        """Display a message to the user."""
        self.message = message
        self.message_timer = 120  # Display for 2 seconds at 60 FPS
        
        # Force redraw to show message immediately
        self._draw_screen()
    
    def get_human_move(self, game_state: GameState, player_color: PlayerColor, 
                     valid_moves: List[Move]) -> Move:
        """Get a move from a human player via the GUI."""
        self.current_game_state = game_state
        self.valid_moves = valid_moves
        self.waiting_for_move = True
        self.selected_pos = None
        self.selected_move = None
        
        # Start turn timer for statistics
        self.statistics.start_turn_timer(player_color)
        
        # Show a prompt to the user
        self.show_message(f"{player_color.name} player's turn - Select a piece or position")
        
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
                piece_value = self.current_game_state.board[self.selected_move.start_pos[0]][self.selected_move.start_pos[1]]
                
                # Animate the move
                self.board_renderer.animate_move(
                    screen=self.screen,
                    board=self.current_game_state.board,
                    start_pos=self.selected_move.start_pos,
                    end_pos=self.selected_move.end_pos,
                    piece_value=piece_value
                )
            
            # If move is an attack, record capture
            if self.selected_move.move_type == MoveType.ATTACK:
                self.statistics.record_capture(player_color)
                
                # Add capture particles
                self.board_renderer.add_capture_particles(
                    row=self.selected_move.end_pos[0],
                    col=self.selected_move.end_pos[1],
                    is_black_capturing=(player_color == PlayerColor.BLACK)
                )
                
            # For scoring moves
            if (self.selected_move.move_type in [MoveType.DIAGONAL, MoveType.JUMP] and 
                (self.selected_move.end_pos[0] < 0 or self.selected_move.end_pos[0] >= self.BOARD_ROWS)):
                # Add score particles
                self.board_renderer.add_score_particles(
                    row=self.selected_move.start_pos[0],
                    col=self.selected_move.start_pos[1],
                    is_black=(player_color == PlayerColor.BLACK)
                )
                
                # Play score sound if available
                if 'score' in self.sounds and self.sounds['score']:
                    self.sounds['score'].play()
            
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
        if not (self.BOARD_X <= mouse_pos[0] <= self.BOARD_X + self.BOARD_WIDTH and
                self.BOARD_Y <= mouse_pos[1] <= self.BOARD_Y + self.BOARD_HEIGHT):
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
            insert_moves = [m for m in self.valid_moves if m.move_type == MoveType.INSERT and m.end_pos == pos]
            
            # Filter moves that have this position as start
            piece_moves = [m for m in self.valid_moves if m.start_pos == pos]
            
            if piece_value == self.current_game_state.current_player.value and piece_moves:
                # Select this piece for moving
                self.selected_pos = pos
                
                # Play move selection sound if available
                if 'move' in self.sounds and self.sounds['move']:
                    self.sounds['move'].play()
                    
                self.show_message(f"Piece selected at {pos}. Click a highlighted square to move.")
            elif insert_moves:
                # Directly make an insert move
                self.selected_move = insert_moves[0]
                self.waiting_for_move = False
                self.show_message(f"Inserting piece at {pos}.")
                
                # Play insert sound if available
                if 'move' in self.sounds and self.sounds['move']:
                    self.sounds['move'].play()
        else:
            # A position was already selected, check if the new click is a valid move
            valid_end_moves = [m for m in self.valid_moves if m.start_pos == self.selected_pos and 
                             (m.end_pos == pos or 
                              (m.move_type in [MoveType.DIAGONAL, MoveType.JUMP] and 
                               ((self.current_game_state.current_player == PlayerColor.BLACK and pos[0] == 3 and abs(pos[1] - self.selected_pos[1]) == 1) or
                                (self.current_game_state.current_player == PlayerColor.RED and pos[0] == 0 and abs(pos[1] - self.selected_pos[1]) == 1))))]
            
            if valid_end_moves:
                # Make the move
                self.selected_move = valid_end_moves[0]
                self.waiting_for_move = False
                
                # Play appropriate sound
                if self.selected_move.move_type == MoveType.ATTACK:
                    if 'capture' in self.sounds and self.sounds['capture']:
                        self.sounds['capture'].play()
                else:
                    if 'move' in self.sounds and self.sounds['move']:
                        self.sounds['move'].play()
                
                # Show appropriate message based on move type
                if self.selected_move.move_type == MoveType.DIAGONAL:
                    if self.selected_move.end_pos[0] < 0 or self.selected_move.end_pos[0] >= self.BOARD_ROWS:
                        self.show_message(f"Diagonal move to score a point!")
                    else:
                        self.show_message(f"Diagonal move from {self.selected_pos} to {pos}.")
                elif self.selected_move.move_type == MoveType.ATTACK:
                    self.show_message(f"Attack move from {self.selected_pos} to {pos}.")
                elif self.selected_move.move_type == MoveType.JUMP:
                    if self.selected_move.end_pos[0] < 0 or self.selected_move.end_pos[0] >= self.BOARD_ROWS:
                        self.show_message(f"Jump to score a point!")
                    else:
                        self.show_message(f"Jump from {self.selected_pos} to {pos}.")
            else:
                # If clicked on own piece, change selection
                piece_value = self.current_game_state.board[row, col]
                piece_moves = [m for m in self.valid_moves if m.start_pos == pos]
                
                if piece_value == self.current_game_state.current_player.value and piece_moves:
                    self.selected_pos = pos
                    
                    # Play selection sound
                    if 'move' in self.sounds and self.sounds['move']:
                        self.sounds['move'].play(maxtime=300)  # shorter sound for selection change
                        
                    self.show_message(f"Changed selection to piece at {pos}.")
                else:
                    # Invalid move, deselect
                    self.selected_pos = None
                    self.show_message(f"Invalid move. Please select a piece or valid move.")
    
    def _show_rules(self):
        """Display the game rules."""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Create rules panel
        panel_width = int(self.SCREEN_WIDTH * 0.8)
        panel_height = int(self.SCREEN_HEIGHT * 0.8)
        panel_x = (self.SCREEN_WIDTH - panel_width) // 2
        panel_y = (self.SCREEN_HEIGHT - panel_height) // 2
        
        panel = pygame.Surface((panel_width, panel_height))
        panel.fill(self.COLORS['PANEL_BACKGROUND'])
        
        # Rules title
        title = self.FONTS['HEADER'].render("KULIBRAT RULES", True, self.COLORS['PANEL_HEADER'])
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
            "- If neither player can move, the last player to move loses."
        ]
        
        y_offset = 80
        for line in rules_text:
            if line.startswith("OBJECTIVE:") or line.startswith("MOVES:") or line.startswith("SCORING:") or line.startswith("SPECIAL RULES:"):
                text = self.FONTS['NORMAL'].render(line, True, self.COLORS['TEXT_ACCENT'])
                y_offset += 10
            else:
                text = self.FONTS['SMALL'].render(line, True, self.COLORS['TEXT_DARK'])
            
            panel.blit(text, (40, y_offset))
            y_offset += 30
        
        # Close button
        close_button = pygame.Rect(panel_width // 2 - 50, panel_height - 60, 100, 40)
        pygame.draw.rect(panel, self.COLORS['BUTTON'], close_button, border_radius=5)
        
        close_text = self.FONTS['NORMAL'].render("Close", True, self.COLORS['BUTTON_TEXT'])
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
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
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
            winner_color = self.COLORS['BLACK_PIECE'] if winner == PlayerColor.BLACK else self.COLORS['RED_PIECE']
            
            # Title with glow effect
            for offset in range(5, 0, -1):
                title = self.FONTS['TITLE'].render("GAME OVER", True, (winner_color[0], winner_color[1], winner_color[2], 50//offset))
                title_rect = title.get_rect(centerx=panel_width // 2, y=20)
                panel.blit(title, (title_rect.x + offset, title_rect.y + offset))
            
            title = self.FONTS['TITLE'].render("GAME OVER", True, winner_color)
            title_rect = title.get_rect(centerx=panel_width // 2, y=20)
            panel.blit(title, title_rect)
            
            # Winner announcement
            winner_text = self.FONTS['HEADER'].render(f"{winner.name} PLAYER WINS!", True, winner_color)
            winner_rect = winner_text.get_rect(centerx=panel_width // 2, y=100)
            panel.blit(winner_text, winner_rect)
        else:
            # Draw case
            title = self.FONTS['TITLE'].render("GAME OVER", True, self.COLORS['TEXT_DARK'])
            title_rect = title.get_rect(centerx=panel_width // 2, y=20)
            panel.blit(title, title_rect)
            
            draw_text = self.FONTS['HEADER'].render("IT'S A DRAW!", True, self.COLORS['TEXT_DARK'])
            draw_rect = draw_text.get_rect(centerx=panel_width // 2, y=100)
            panel.blit(draw_text, draw_rect)
        
        # Score summary
        score_y = 170
        score_text = self.FONTS['NORMAL'].render("FINAL SCORE", True, self.COLORS['TEXT_DARK'])
        score_rect = score_text.get_rect(centerx=panel_width // 2, y=score_y)
        panel.blit(score_text, score_rect)
        
        black_score = game_state.scores[PlayerColor.BLACK]
        red_score = game_state.scores[PlayerColor.RED]
        
        score_detail = self.FONTS['NORMAL'].render(f"BLACK: {black_score}  -  RED: {red_score}", True, self.COLORS['TEXT_DARK'])
        score_detail_rect = score_detail.get_rect(centerx=panel_width // 2, y=score_y + 40)
        panel.blit(score_detail, score_detail_rect)
        
        # Game statistics
        stats_y = score_y + 100
        stats_text = self.FONTS['NORMAL'].render("GAME STATISTICS", True, self.COLORS['TEXT_DARK'])
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
            f"RED Captures: {stats_summary['captures_by_player']['RED']}"
        ]
        
        stats_line_y = stats_y + 40
        for line in stats_lines:
            line_text = self.FONTS['SMALL'].render(line, True, self.COLORS['TEXT_DARK'])
            line_rect = line_text.get_rect(centerx=panel_width // 2, y=stats_line_y)
            panel.blit(line_text, line_rect)
            stats_line_y += 30
        
        # Play again button
        play_again_button = pygame.Rect(panel_width // 4 - 75, panel_height - 70, 150, 50)
        pygame.draw.rect(panel, self.COLORS['BUTTON'], play_again_button, border_radius=5)
        
        play_again_text = self.FONTS['NORMAL'].render("Play Again", True, self.COLORS['BUTTON_TEXT'])
        play_again_rect = play_again_text.get_rect(center=play_again_button.center)
        panel.blit(play_again_text, play_again_rect)
        
        # Quit button
        quit_button = pygame.Rect(panel_width * 3 // 4 - 75, panel_height - 70, 150, 50)
        pygame.draw.rect(panel, self.COLORS['RED_PIECE'], quit_button, border_radius=5)
        
        quit_text = self.FONTS['NORMAL'].render("Quit", True, self.COLORS['BUTTON_TEXT'])
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