# gui/ui.py
import pygame
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from src
from src.board import Board
from src.game import Game

class KulibratGUI:
    # Constants
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    SQUARE_SIZE = 100
    BOARD_MARGIN_X = 250
    BOARD_MARGIN_Y = 150
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BOARD_COLOR = (210, 180, 140)  # Tan
    HIGHLIGHT_COLOR = (255, 255, 0, 128)  # Semi-transparent yellow
    
    def __init__(self, win_score=5):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Kulibrat")
        
        self.clock = pygame.time.Clock()
        self.game = Game(win_score=win_score)
        self.selected_piece = None
        
        # Load assets
        self.assets_dir = Path(__file__).parent / "assets"
        self.load_assets()
        
    def load_assets(self):
        """Load all image assets for the game"""
        try:
            self.red_piece = pygame.image.load(self.assets_dir / "red_piece.png")
            self.black_piece = pygame.image.load(self.assets_dir / "black_piece.png")
            
            # Scale pieces to fit the squares
            piece_size = int(self.SQUARE_SIZE * 0.8)
            self.red_piece = pygame.transform.scale(self.red_piece, (piece_size, piece_size))
            self.black_piece = pygame.transform.scale(self.black_piece, (piece_size, piece_size))
        except FileNotFoundError:
            print("Warning: Could not load piece images. Using colored circles instead.")
            self.red_piece = None
            self.black_piece = None
    
    def draw_board(self):
        """Draw the 3x4 game board"""
        # Draw board background
        board_width = 3 * self.SQUARE_SIZE
        board_height = 4 * self.SQUARE_SIZE
        pygame.draw.rect(
            self.screen, 
            self.BOARD_COLOR, 
            (self.BOARD_MARGIN_X, self.BOARD_MARGIN_Y, board_width, board_height)
        )
        
        # Draw grid lines
        for i in range(5):  # Horizontal lines (5 for 4 rows)
            y = self.BOARD_MARGIN_Y + i * self.SQUARE_SIZE
            pygame.draw.line(
                self.screen, 
                self.BLACK, 
                (self.BOARD_MARGIN_X, y), 
                (self.BOARD_MARGIN_X + board_width, y),
                2
            )
            
        for i in range(4):  # Vertical lines (4 for 3 columns)
            x = self.BOARD_MARGIN_X + i * self.SQUARE_SIZE
            pygame.draw.line(
                self.screen, 
                self.BLACK, 
                (x, self.BOARD_MARGIN_Y), 
                (x, self.BOARD_MARGIN_Y + board_height),
                2
            )
            
        # Highlight the start rows
        black_start_row = pygame.Surface((board_width, self.SQUARE_SIZE), pygame.SRCALPHA)
        black_start_row.fill((0, 0, 0, 30))  # Semi-transparent black
        self.screen.blit(black_start_row, (self.BOARD_MARGIN_X, self.BOARD_MARGIN_Y))
        
        red_start_row = pygame.Surface((board_width, self.SQUARE_SIZE), pygame.SRCALPHA)
        red_start_row.fill((255, 0, 0, 30))  # Semi-transparent red
        self.screen.blit(red_start_row, (self.BOARD_MARGIN_X, self.BOARD_MARGIN_Y + 3 * self.SQUARE_SIZE))
    
    def draw_pieces(self):
        """Draw all pieces on the board based on current game state"""
        for row in range(4):
            for col in range(3):
                piece = self.game.board.get_piece(row, col)
                if piece:
                    x = self.BOARD_MARGIN_X + col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                    y = self.BOARD_MARGIN_Y + row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2
                    
                    if piece.color == "red":
                        if self.red_piece:
                            rect = self.red_piece.get_rect(center=(x, y))
                            self.screen.blit(self.red_piece, rect)
                        else:
                            # Fallback to a colored circle
                            pygame.draw.circle(self.screen, self.RED, (x, y), int(self.SQUARE_SIZE * 0.4))
                    else:
                        if self.black_piece:
                            rect = self.black_piece.get_rect(center=(x, y))
                            self.screen.blit(self.black_piece, rect)
                        else:
                            # Fallback to a colored circle
                            pygame.draw.circle(self.screen, self.BLACK, (x, y), int(self.SQUARE_SIZE * 0.4))
                    
                    # Highlight selected piece
                    if self.selected_piece and self.selected_piece == (row, col):
                        pygame.draw.circle(
                            self.screen, 
                            self.HIGHLIGHT_COLOR, 
                            (x, y), 
                            int(self.SQUARE_SIZE * 0.45),
                            3
                        )
    
    def draw_available_pieces(self):
        """Draw the available pieces for each player"""
        # Black available pieces
        font = pygame.font.SysFont("Arial", 20)
        black_text = font.render(f"Available pieces: {self.game.board.available_pieces['black']}", True, self.BLACK)
        self.screen.blit(black_text, (50, 280))
        
        # Red available pieces
        red_text = font.render(f"Available pieces: {self.game.board.available_pieces['red']}", True, self.RED)
        self.screen.blit(red_text, (50, 330))
    
    def draw_scores(self):
        """Display the current score"""
        font = pygame.font.SysFont("Arial", 28)
        
        # Game info title
        info_title = font.render("Game Info", True, self.BLACK)
        self.screen.blit(info_title, (50, 100))
        
        # Black score
        black_score_text = f"Black: {self.game.black_score}"
        black_score_surface = font.render(black_score_text, True, self.BLACK)
        self.screen.blit(black_score_surface, (50, 150))
        
        # Red score
        red_score_text = f"Red: {self.game.red_score}"
        red_score_surface = font.render(red_score_text, True, self.RED)
        self.screen.blit(red_score_surface, (50, 200))
        
        # Win condition
        win_text = font.render(f"First to {self.game.win_score} points wins", True, self.BLACK)
        self.screen.blit(win_text, (50, 400))
        
        # Current player
        font_big = pygame.font.SysFont("Arial", 36)
        current_player_text = f"Current player: {self.game.current_player.capitalize()}"
        current_player_color = self.RED if self.game.current_player == "red" else self.BLACK
        current_player_surface = font_big.render(current_player_text, True, current_player_color)
        self.screen.blit(current_player_surface, (self.WINDOW_WIDTH // 2 - current_player_surface.get_width() // 2, 50))
    
    def highlight_legal_moves(self):
        """Highlight squares where the selected piece can move"""
        if not self.selected_piece:
            return
            
        row, col = self.selected_piece
        legal_moves = self.game.get_legal_moves(row, col)
        
        highlight_surface = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
        highlight_surface.fill(self.HIGHLIGHT_COLOR)
        
        for move in legal_moves:
            move_type, _, _, dest_row, dest_col = move
            if 0 <= dest_row < 4 and 0 <= dest_col < 3:  # Only highlight moves on the board
                x = self.BOARD_MARGIN_X + dest_col * self.SQUARE_SIZE
                y = self.BOARD_MARGIN_Y + dest_row * self.SQUARE_SIZE
                self.screen.blit(highlight_surface, (x, y))
    
    def highlight_legal_inserts(self):
        """Highlight squares where pieces can be inserted"""
        if self.selected_piece is not None:
            return  # Only show insert options when no piece is selected
            
        if self.game.board.available_pieces[self.game.current_player] <= 0:
            return  # No pieces available to insert
            
        # Get legal insert locations
        insert_row = 0 if self.game.current_player == "black" else 3
        
        highlight_surface = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE), pygame.SRCALPHA)
        highlight_surface.fill(self.HIGHLIGHT_COLOR)
        
        for col in range(3):
            if self.game.board.is_square_free(insert_row, col):
                x = self.BOARD_MARGIN_X + col * self.SQUARE_SIZE
                y = self.BOARD_MARGIN_Y + insert_row * self.SQUARE_SIZE
                self.screen.blit(highlight_surface, (x, y))
    
    def handle_click(self, pos):
        """Handle mouse click on the board"""
        x, y = pos
        
        # Check if click is on the board
        if (self.BOARD_MARGIN_X <= x < self.BOARD_MARGIN_X + 3 * self.SQUARE_SIZE and
            self.BOARD_MARGIN_Y <= y < self.BOARD_MARGIN_Y + 4 * self.SQUARE_SIZE):
            
            # Convert to board coordinates
            col = (x - self.BOARD_MARGIN_X) // self.SQUARE_SIZE
            row = (y - self.BOARD_MARGIN_Y) // self.SQUARE_SIZE
            
            # Case 1: Selecting a piece
            if self.selected_piece is None:
                piece = self.game.board.get_piece(row, col)
                if piece and piece.color == self.game.current_player:
                    self.selected_piece = (row, col)
                elif (row == 0 and self.game.current_player == "black") or \
                     (row == 3 and self.game.current_player == "red"):
                    # Try to insert a piece
                    if self.game.board.is_square_free(row, col) and \
                       self.game.board.available_pieces[self.game.current_player] > 0:
                        self.game.make_move(None, None, row, col)
                return
            
            # Case 2: Moving a selected piece
            selected_row, selected_col = self.selected_piece
            
            # Try to move the piece
            if self.game.make_move(selected_row, selected_col, row, col):
                self.selected_piece = None
            else:
                # If the clicked square has a piece of the current player, select it instead
                piece = self.game.board.get_piece(row, col)
                if piece and piece.color == self.game.current_player:
                    self.selected_piece = (row, col)
                else:
                    # Otherwise, deselect
                    self.selected_piece = None
        else:
            # Clicked outside the board, deselect
            self.selected_piece = None
    
    def show_game_over(self):
        """Display game over message"""
        if not self.game.is_game_over():
            return
            
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        font = pygame.font.SysFont("Arial", 60)
        game_over_text = "Game Over!"
        game_over_surface = font.render(game_over_text, True, self.WHITE)
        
        # Winner text
        winner = "Black" if self.game.black_score > self.game.red_score else "Red"
        winner_color = self.RED if winner == "Red" else self.WHITE
        winner_text = f"{winner} wins!"
        winner_surface = font.render(winner_text, True, winner_color)
        
        # Final score
        score_font = pygame.font.SysFont("Arial", 36)
        score_text = f"Final Score - Black: {self.game.black_score}, Red: {self.game.red_score}"
        score_surface = score_font.render(score_text, True, self.WHITE)
        
        # Draw text centered on screen
        self.screen.blit(game_over_surface, 
                        (self.WINDOW_WIDTH // 2 - game_over_surface.get_width() // 2, 
                         self.WINDOW_HEIGHT // 2 - 100))
        self.screen.blit(winner_surface, 
                        (self.WINDOW_WIDTH // 2 - winner_surface.get_width() // 2, 
                         self.WINDOW_HEIGHT // 2))
        self.screen.blit(score_surface, 
                        (self.WINDOW_WIDTH // 2 - score_surface.get_width() // 2, 
                         self.WINDOW_HEIGHT // 2 + 80))
        
        # Play again instructions
        restart_font = pygame.font.SysFont("Arial", 24)
        restart_text = "Press 'R' to play again or 'Q' to quit"
        restart_surface = restart_font.render(restart_text, True, self.WHITE)
        self.screen.blit(restart_surface, 
                        (self.WINDOW_WIDTH // 2 - restart_surface.get_width() // 2, 
                         self.WINDOW_HEIGHT // 2 + 150))
    
    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        if not self.game.is_game_over():
                            self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if self.game.is_game_over():
                        if event.key == pygame.K_r:  # Restart
                            self.__init__(win_score=self.game.win_score)
                        elif event.key == pygame.K_q:  # Quit
                            running = False
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_board()
            if not self.game.is_game_over():
                self.highlight_legal_inserts()
                self.highlight_legal_moves()
            self.draw_pieces()
            self.draw_scores()
            self.draw_available_pieces()
            
            if self.game.is_game_over():
                self.show_game_over()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = KulibratGUI()
    gui.run()