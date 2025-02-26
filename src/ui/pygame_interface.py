import pygame
import sys
import numpy as np
from typing import List, Optional, Tuple

# Import game core components
from src.core.game_state import GameState
from src.core.player_color import PlayerColor
from src.core.move import Move
from src.core.move_type import MoveType

# Import game interface
from src.ui.game_interface import GameInterface

class KulibratGUI(GameInterface):
    def __init__(self, screen_width=1440, screen_height=900):
        """Ultimate Kulibrat Game Interface"""
        pygame.init()
        
        # Screen and window configuration
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Kulibrat - Strategic Board Game')
        
        # Refined color palette with depth and nuance
        self.COLORS = {
            'BACKGROUND': (250, 250, 255),
            'BOARD_BG': (240, 245, 250),
            'GRID_LINE': (200, 210, 220),
            'BLACK_PIECE': (40, 50, 60),
            'RED_PIECE': (220, 60, 60),
            'HIGHLIGHT_GREEN': (100, 220, 130, 100),
            'TEXT_PRIMARY': (30, 40, 50),
            'TEXT_SECONDARY': (80, 90, 100),
            'MOVE_HIGHLIGHT': (0, 200, 83, 80),
            'SIDEBAR_BG': (245, 247, 250),
            'PIECE_SHADOW': (0, 0, 0, 50)
        }
        
        # Advanced typography with font loading
        pygame.font.init()
        self.FONTS = {
            'TITLE': pygame.font.Font(None, 64),
            'HEADER': pygame.font.Font(None, 42),
            'BODY': pygame.font.Font(None, 32),
            'SMALL': pygame.font.Font(None, 26)
        }
        
        # Precise layout calculations
        self.LAYOUT = {
            'BOARD_MARGIN_X': 80,
            'BOARD_MARGIN_Y': 50,
            'BOARD_ROWS': 4,
            'BOARD_COLS': 3,
            'SIDEBAR_WIDTH': 350
        }
        
        # Dynamic board sizing
        available_board_width = self.SCREEN_WIDTH - self.LAYOUT['SIDEBAR_WIDTH'] - 2 * self.LAYOUT['BOARD_MARGIN_X']
        available_board_height = self.SCREEN_HEIGHT - 2 * self.LAYOUT['BOARD_MARGIN_Y']
        
        # Calculate cell dimensions to maximize board usage
        self.CELL_WIDTH = available_board_width // self.LAYOUT['BOARD_COLS']
        self.CELL_HEIGHT = available_board_height // self.LAYOUT['BOARD_ROWS']
        
        # Piece sizing relative to cell size
        self.PIECE_RADIUS = min(self.CELL_WIDTH, self.CELL_HEIGHT) // 2 - 10
        
        # Game state management
        self.current_game_state = None
        self.valid_moves = []
        self.waiting_for_move = False
        self.selected_move = None
        
        # Performance and animation
        self.clock = pygame.time.Clock()
        
        # Prepare background
        self._prepare_background()
    
    def _prepare_background(self):
        """Create a sophisticated, layered background"""
        self.background = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        # Soft gradient background
        for y in range(self.SCREEN_HEIGHT):
            # Smooth color transition
            r = int(250 - (y / self.SCREEN_HEIGHT) * 20)
            g = int(250 - (y / self.SCREEN_HEIGHT) * 20)
            b = int(255 - (y / self.SCREEN_HEIGHT) * 15)
            color = (r, g, b)
            pygame.draw.line(self.background, color, (0, y), (self.SCREEN_WIDTH, y))
        
        # Subtle noise texture for depth
        noise_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for _ in range(10000):
            x = np.random.randint(0, self.SCREEN_WIDTH)
            y = np.random.randint(0, self.SCREEN_HEIGHT)
            noise_surface.set_at((x, y), (255, 255, 255, 5))
        
        self.background.blit(noise_surface, (0, 0))
    
    def _draw_board(self) -> pygame.Surface:
        """Render the game board with precise 4x3 grid"""
        board_width = self.CELL_WIDTH * self.LAYOUT['BOARD_COLS']
        board_height = self.CELL_HEIGHT * self.LAYOUT['BOARD_ROWS']
        
        board_surface = pygame.Surface((board_width, board_height))
        board_surface.fill(self.COLORS['BOARD_BG'])
        
        # Draw grid lines with anti-aliasing
        for row in range(1, self.LAYOUT['BOARD_ROWS']):
            pygame.draw.line(board_surface, self.COLORS['GRID_LINE'], 
                             (0, row * self.CELL_HEIGHT), 
                             (board_width, row * self.CELL_HEIGHT), 2)
        
        for col in range(1, self.LAYOUT['BOARD_COLS']):
            pygame.draw.line(board_surface, self.COLORS['GRID_LINE'], 
                             (col * self.CELL_WIDTH, 0), 
                             (col * self.CELL_WIDTH, board_height), 2)
        
        # Highlight start rows
        start_rows = [0, 3]  # Black and Red start rows
        highlight_color = (200, 220, 255, 30)
        for row in start_rows:
            highlight_rect = pygame.Rect(0, row * self.CELL_HEIGHT, board_width, self.CELL_HEIGHT)
            highlight_surface = pygame.Surface((board_width, self.CELL_HEIGHT), pygame.SRCALPHA)
            highlight_surface.fill(highlight_color)
            board_surface.blit(highlight_surface, highlight_rect)
        
        # Render pieces with enhanced visual depth
        if self.current_game_state:
            for row in range(self.LAYOUT['BOARD_ROWS']):
                for col in range(self.LAYOUT['BOARD_COLS']):
                    piece_value = self.current_game_state.board[row, col]
                    center = (col * self.CELL_WIDTH + self.CELL_WIDTH // 2, 
                              row * self.CELL_HEIGHT + self.CELL_HEIGHT // 2)
                    
                    # Soft shadow effect
                    shadow_offset = 4
                    pygame.draw.circle(board_surface, self.COLORS['PIECE_SHADOW'], 
                                       (center[0] + shadow_offset, center[1] + shadow_offset), 
                                       self.PIECE_RADIUS + 2)
                    
                    # Piece rendering with anti-aliasing
                    if piece_value == PlayerColor.BLACK.value:
                        pygame.draw.circle(board_surface, self.COLORS['BLACK_PIECE'], 
                                           center, self.PIECE_RADIUS)
                    elif piece_value == PlayerColor.RED.value:
                        pygame.draw.circle(board_surface, self.COLORS['RED_PIECE'], 
                                           center, self.PIECE_RADIUS)
        
        return board_surface
    
    def _draw_sidebar(self) -> pygame.Surface:
        """Create an informative and elegant sidebar"""
        if not self.current_game_state:
            return pygame.Surface((0, 0))
        
        sidebar = pygame.Surface((self.LAYOUT['SIDEBAR_WIDTH'], self.SCREEN_HEIGHT))
        sidebar.fill(self.COLORS['SIDEBAR_BG'])
        
        # Game title with subtle shadow
        title = self.FONTS['TITLE'].render('Kulibrat', True, self.COLORS['TEXT_PRIMARY'])
        title_shadow = self.FONTS['TITLE'].render('Kulibrat', True, (200, 200, 210))
        sidebar.blit(title_shadow, (52, 52))
        sidebar.blit(title, (50, 50))
        
        # Detailed game state information
        info_entries = [
            f"Current Player: {self.current_game_state.current_player.name}",
            f"Target Score: {self.current_game_state.target_score}",
            f"Black Score: {self.current_game_state.scores[PlayerColor.BLACK]}",
            f"Red Score: {self.current_game_state.scores[PlayerColor.RED]}",
            f"Black Pieces: {4 - self.current_game_state.pieces_off_board[PlayerColor.BLACK]}/4",
            f"Red Pieces: {4 - self.current_game_state.pieces_off_board[PlayerColor.RED]}/4",
        ]
        
        y_offset = 200
        for entry in info_entries:
            text = self.FONTS['BODY'].render(entry, True, self.COLORS['TEXT_SECONDARY'])
            sidebar.blit(text, (50, y_offset))
            y_offset += 50
        
        return sidebar
    
    def display_state(self, game_state: GameState) -> None:
        """Modern and precise game state display"""
        self.current_game_state = game_state
        
        # Apply background
        self.screen.blit(self.background, (0, 0))
        
        # Draw board with correct 4x3 layout
        board_surface = self._draw_board()
        board_x = self.LAYOUT['BOARD_MARGIN_X']
        board_y = self.LAYOUT['BOARD_MARGIN_Y']
        self.screen.blit(board_surface, (board_x, board_y))
        
        # Draw sidebar
        sidebar = self._draw_sidebar()
        sidebar_x = board_x + board_surface.get_width() + self.LAYOUT['BOARD_MARGIN_X']
        self.screen.blit(sidebar, (sidebar_x, 0))
        
        # Highlight valid moves
        if self.waiting_for_move and self.valid_moves:
            self._highlight_valid_moves(board_surface, board_x, board_y)
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _highlight_valid_moves(self, board_surface: pygame.Surface, offset_x: int, offset_y: int):
        """Sophisticated move highlighting with precise positioning"""
        highlight_surface = pygame.Surface(board_surface.get_size(), pygame.SRCALPHA)
        
        for move in self.valid_moves:
            if move.end_pos:
                col, row = move.end_pos[1], move.end_pos[0]
                highlight_rect = pygame.Rect(
                    col * self.CELL_WIDTH, 
                    row * self.CELL_HEIGHT, 
                    self.CELL_WIDTH, 
                    self.CELL_HEIGHT
                )
                
                # Pulsing highlight effect
                pulse_colors = [
                    (*self.COLORS['MOVE_HIGHLIGHT'][:3], 80),
                    (*self.COLORS['MOVE_HIGHLIGHT'][:3], 50),
                    (*self.COLORS['MOVE_HIGHLIGHT'][:3], 30)
                ]
                
                for pulse_color in pulse_colors:
                    pulse_surface = pygame.Surface((self.CELL_WIDTH, self.CELL_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.rect(pulse_surface, pulse_color, pulse_surface.get_rect(), border_radius=10)
                    highlight_surface.blit(pulse_surface, highlight_rect)
        
        # Blit highlighted surface onto the board
        board_surface.blit(highlight_surface, (0, 0))
        self.screen.blit(board_surface, (offset_x, offset_y))
        pygame.display.flip()
    
    def get_human_move(self, game_state: GameState, player_color: PlayerColor, 
                   valid_moves: List[Move]) -> Move:
        """Precise and intuitive move selection"""
        self.valid_moves = valid_moves
        self.waiting_for_move = True
        self.selected_move = None
        
        self.display_state(game_state)
        
        while self.waiting_for_move:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    
                    # Adjust for board margin
                    board_x = self.LAYOUT['BOARD_MARGIN_X']
                    board_y = self.LAYOUT['BOARD_MARGIN_Y']
                    
                    # Validate board click
                    if (board_x <= mouse_x < board_x + self.CELL_WIDTH * self.LAYOUT['BOARD_COLS'] and 
                        board_y <= mouse_y < board_y + self.CELL_HEIGHT * self.LAYOUT['BOARD_ROWS']):
                        
                        # Calculate grid position
                        col = (mouse_x - board_x) // self.CELL_WIDTH
                        row = (mouse_y - board_y) // self.CELL_HEIGHT
                        
                        # Find matching move
                        for move in self.valid_moves:
                            if ((move.end_pos and move.end_pos == (row, col)) or 
                                (move.start_pos and move.start_pos == (row, col))):
                                self.selected_move = move
                                self.waiting_for_move = False
                                break
            
            self.clock.tick(30)
        
        return self.selected_move
    
    def show_winner(self, winner: Optional[PlayerColor], game_state: GameState) -> None:
        """Dramatic and elegant game conclusion"""
        self.display_state(game_state)
        
        # Dim overlay
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        # Winner text with sophisticated typography
        if winner:
            title = self.FONTS['TITLE'].render('Game Over', True, (255, 255, 255))
            winner_text = self.FONTS['HEADER'].render(f"{winner.name} Player Wins!", True, (255, 255, 255))
        else:
            title = self.FONTS['TITLE'].render('Game Over', True, (255, 255, 255))
            winner_text = self.FONTS['HEADER'].render("It's a Draw!", True, (255, 255, 255))
        
        # Center texts precisely
        title_rect = title.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 100))
        winner_text_rect = winner_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        
        self.screen.blit(title, title_rect)
        self.screen.blit(winner_text, winner_text_rect)
        
        # Smooth fading animation
        alpha = 0
        while alpha < 255:
            overlay.fill((0, 0, 0, alpha))
            self.screen.blit(overlay, (0, 0))
            pygame.display.flip()
            alpha += 5
            pygame.time.delay(30)
        
        # Wait for user input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
        
        pygame.quit()

if __name__ == "__main__":
    gui = KulibratGUI()
    # Add code here to integrate the GUI with the game logic