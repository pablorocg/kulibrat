"""
Enhanced board rendering for Kulibrat GUI with textures, lighting effects,
and beautiful piece animations.
"""

"""
Enhanced board rendering for Kulibrat GUI with textures, lighting effects,
and beautiful piece animations.
"""

import pygame
import math
import random
from typing import Tuple, List, Optional


class EnhancedBoardRenderer:
    """
    Class for rendering a beautiful Kulibrat board with visual effects.
    This can be integrated with the KulibratGUI class.
    """

    def __init__(
        self,
        board_x: int,
        board_y: int,
        cell_size: int,
        board_rows: int = 4,
        board_cols: int = 3,
    ):
        """Initialize the enhanced board renderer."""
        self.board_x = board_x
        self.board_y = board_y
        self.cell_size = cell_size
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.board_width = cell_size * board_cols
        self.board_height = cell_size * board_rows

        # Piece properties
        self.piece_radius = int(cell_size * 0.4)

        # Visual effects settings
        self.use_textures = True
        self.use_lighting = True
        self.use_shadows = True
        self.use_reflections = True

        # Animation properties
        self.animation_frames = 15

        # Particles
        self.particles = []
        self.max_particles = 100

        # Cached resources
        self.textures = {}
        self.glow_surfaces = {}

        # Load resources
        self._load_resources()

    def _load_resources(self):
        """Load and prepare visual resources."""
        # Generate board textures if requested
        if self.use_textures:
            self._generate_wood_texture("board_light", (235, 220, 200), (215, 200, 180))
            self._generate_wood_texture("board_dark", (170, 135, 100), (150, 115, 80))

            # Generate border texture
            self._generate_wood_texture("board_border", (120, 90, 70), (100, 70, 50))

        # Pre-render glows for pieces
        self._generate_glow_surface("red_glow", (255, 100, 100), self.piece_radius * 2)
        self._generate_glow_surface(
            "black_glow", (100, 140, 255), self.piece_radius * 2
        )
        self._generate_glow_surface(
            "select_glow", (255, 255, 150), self.piece_radius * 2
        )

    def _generate_wood_texture(
        self,
        name: str,
        base_color: Tuple[int, int, int],
        grain_color: Tuple[int, int, int],
        size: int = 256,
    ):
        """Generate a procedural wood texture."""
        texture = pygame.Surface((size, size))
        texture.fill(base_color)

        # Add wood grain
        for i in range(size):
            noise = math.sin(i * 0.1) * 10 + random.randint(-5, 5)
            for j in range(size):
                if (i + j + noise) % 20 < 10:
                    # Darken pixel slightly
                    darkness = random.randint(-15, 5)
                    color = tuple(
                        max(0, min(255, base_color[k] + darkness)) for k in range(3)
                    )
                    texture.set_at((i, j), color)

                # Add occasional knots in the wood
                if random.random() < 0.0001:
                    self._add_wood_knot(texture, i, j, grain_color)

        self.textures[name] = texture

    def _add_wood_knot(
        self,
        texture: pygame.Surface,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        radius: int = 10,
    ):
        """Add a knot pattern to the wood texture."""
        for i in range(x - radius, x + radius):
            for j in range(y - radius, y + radius):
                if (
                    (i - x) ** 2 + (j - y) ** 2 < radius**2
                    and 0 <= i < texture.get_width()
                    and 0 <= j < texture.get_height()
                ):
                    # Radial darkening
                    dist = math.sqrt((i - x) ** 2 + (j - y) ** 2)
                    if dist < radius:
                        factor = 1 - dist / radius
                        darkness = int(-30 * factor**2)
                        pixel = texture.get_at((i, j))
                        new_color = tuple(
                            max(0, min(255, pixel[k] + darkness)) for k in range(3)
                        )
                        texture.set_at((i, j), new_color)

    def _generate_glow_surface(self, name: str, color: Tuple[int, int, int], size: int):
        """Generate a glowing effect surface."""
        glow = pygame.Surface((size, size), pygame.SRCALPHA)

        radius = size // 2
        center = (radius, radius)

        # Create a radial gradient
        for x in range(size):
            for y in range(size):
                # Calculate distance from center
                distance = math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

                # Calculate alpha based on distance (fade out from center)
                alpha = int(255 * (1 - min(1, distance / radius)))

                if alpha > 0:
                    # Apply color with calculated alpha
                    glow.set_at((x, y), (*color, alpha))

        self.glow_surfaces[name] = glow

    def render(
        self,
        screen: pygame.Surface,
        board: List[List[int]],
        selected_pos: Optional[Tuple[int, int]] = None,
        valid_moves: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Render the enhanced board with pieces and effects.

        Args:
            screen: Pygame surface to render to
            board: 2D array representing the game board
            selected_pos: Selected position (row, col) if any
            valid_moves: List of valid move end positions
        """
        # Draw board background with border
        self._draw_board_background(screen)

        # Draw grid cells
        self._draw_grid(screen)

        # Highlight valid moves if provided
        if valid_moves:
            self._highlight_valid_moves(screen, valid_moves)

        # Draw pieces with effects
        self._draw_pieces(screen, board, selected_pos)

        # Draw particles
        self._update_and_draw_particles(screen)

    def _draw_board_background(self, screen: pygame.Surface):
        """Draw the board background with decorative border."""
        # Create a larger rectangle for the border
        border_rect = pygame.Rect(
            self.board_x - 20,
            self.board_y - 20,
            self.board_width + 40,
            self.board_height + 40,
        )

        if self.use_textures and "board_border" in self.textures:
            # Draw textured border
            border_texture = self.textures["board_border"]

            # Scale the texture if needed
            if border_texture.get_size() != (border_rect.width, border_rect.height):
                scaled_texture = pygame.transform.scale(
                    border_texture, (border_rect.width, border_rect.height)
                )
            else:
                scaled_texture = border_texture

            screen.blit(scaled_texture, border_rect)
        else:
            pygame.draw.rect(screen, (120, 90, 70), border_rect, border_radius=10)

        # Inner board background
        board_rect = pygame.Rect(
            self.board_x, self.board_y, self.board_width, self.board_height
        )

        pygame.draw.rect(screen, (210, 180, 150), board_rect)

        # Add decorative corner elements
        self._draw_decorative_corners(screen, border_rect)

    def _draw_decorative_corners(
        self, screen: pygame.Surface, border_rect: pygame.Rect
    ):
        """Draw decorative corner elements on the board."""
        corner_size = 15
        corners = [
            (border_rect.left, border_rect.top),  # Top-left
            (border_rect.right - corner_size, border_rect.top),  # Top-right
            (border_rect.left, border_rect.bottom - corner_size),  # Bottom-left
            (
                border_rect.right - corner_size,
                border_rect.bottom - corner_size,
            ),  # Bottom-right
        ]

        for corner in corners:
            pygame.draw.rect(
                screen,
                (80, 60, 40),
                pygame.Rect(corner[0], corner[1], corner_size, corner_size),
                border_radius=5,
            )

    def _draw_grid(self, screen: pygame.Surface):
        """Draw the grid cells with alternating colors/textures."""
        for row in range(self.board_rows):
            for col in range(self.board_cols):
                cell_rect = pygame.Rect(
                    self.board_x + col * self.cell_size,
                    self.board_y + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Alternating pattern
                is_light = (row + col) % 2 == 0
                texture_name = "board_light" if is_light else "board_dark"

                if self.use_textures and texture_name in self.textures:
                    # Use wood texture
                    texture = self.textures[texture_name]

                    # Scale texture to cell size
                    scaled_texture = pygame.transform.scale(
                        texture, (self.cell_size, self.cell_size)
                    )
                    screen.blit(scaled_texture, cell_rect)
                else:
                    # Use solid color
                    color = (235, 220, 200) if is_light else (170, 135, 100)
                    pygame.draw.rect(screen, color, cell_rect)

                # Draw cell border
                pygame.draw.rect(screen, (120, 100, 80), cell_rect, width=1)

                # Mark start rows
                if row == 0 or row == self.board_rows - 1:
                    marker_color = (
                        (100, 140, 255, 70) if row == 0 else (255, 100, 100, 70)
                    )
                    marker = pygame.Surface(
                        (self.cell_size, self.cell_size), pygame.SRCALPHA
                    )
                    marker.fill(marker_color)
                    screen.blit(marker, cell_rect)

    def _highlight_valid_moves(
        self, screen: pygame.Surface, valid_moves: List[Tuple[int, int]]
    ):
        """Highlight valid move positions with a pulsing effect."""
        # Calculate pulse based on time
        current_time = pygame.time.get_ticks()
        pulse_alpha = int(127 + 64 * math.sin(current_time / 300))

        for pos in valid_moves:
            row, col = pos
            if 0 <= row < self.board_rows and 0 <= col < self.board_cols:
                highlight_rect = pygame.Rect(
                    self.board_x + col * self.cell_size,
                    self.board_y + row * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                # Create highlight surface with pulsing alpha
                highlight = pygame.Surface(
                    (self.cell_size, self.cell_size), pygame.SRCALPHA
                )
                highlight.fill((100, 200, 100, pulse_alpha))

                # Draw highlight
                screen.blit(highlight, highlight_rect)

                # Draw border
                pygame.draw.rect(screen, (100, 200, 100), highlight_rect, width=2)

    def _draw_pieces(
        self,
        screen: pygame.Surface,
        board: List[List[int]],
        selected_pos: Optional[Tuple[int, int]] = None,
    ):
        """Draw game pieces with lighting effects."""
        for row in range(len(board)):
            for col in range(len(board[row]) if row < len(board) else 0):
                piece_value = board[row][col]

                if piece_value != 0:  # If there's a piece here
                    # Calculate center position
                    center_x = self.board_x + col * self.cell_size + self.cell_size // 2
                    center_y = self.board_y + row * self.cell_size + self.cell_size // 2

                    # Draw glow effect under selected piece
                    is_selected = selected_pos and selected_pos == (row, col)

                    if is_selected and "select_glow" in self.glow_surfaces:
                        glow_surf = self.glow_surfaces["select_glow"]
                        glow_rect = glow_surf.get_rect(center=(center_x, center_y))
                        screen.blit(glow_surf, glow_rect)

                    # Draw piece shadow
                    if self.use_shadows:
                        shadow_offset = 5
                        pygame.draw.circle(
                            screen,
                            (0, 0, 0, 100),
                            (center_x + shadow_offset, center_y + shadow_offset),
                            self.piece_radius,
                        )

                    # Determine piece color
                    if piece_value == 1:  # Black piece
                        base_color = (40, 45, 55)
                        highlight_color = (70, 75, 85)
                        glow_name = "black_glow"
                    else:  # Red piece
                        base_color = (200, 60, 70)
                        highlight_color = (230, 90, 100)
                        glow_name = "red_glow"

                    # Draw base piece
                    pygame.draw.circle(
                        screen, base_color, (center_x, center_y), self.piece_radius
                    )

                    # Add highlight reflection
                    if self.use_lighting:
                        highlight_offset = -self.piece_radius // 3
                        highlight_radius = self.piece_radius // 2

                        pygame.draw.circle(
                            screen,
                            highlight_color,
                            (center_x + highlight_offset, center_y + highlight_offset),
                            highlight_radius,
                        )

                    # Add subtle rim light
                    pygame.draw.circle(
                        screen,
                        base_color,
                        (center_x, center_y),
                        self.piece_radius,
                        width=2,
                    )

                    # Add glowing effect for selected pieces
                    if is_selected and glow_name in self.glow_surfaces:
                        # Pulse the glow based on time
                        current_time = pygame.time.get_ticks()
                        scale_factor = 1.0 + 0.2 * math.sin(current_time / 200)

                        glow_surf = self.glow_surfaces[glow_name]
                        scaled_size = int(glow_surf.get_width() * scale_factor)

                        scaled_glow = pygame.transform.scale(
                            glow_surf, (scaled_size, scaled_size)
                        )
                        glow_rect = scaled_glow.get_rect(center=(center_x, center_y))

                        # Apply the glow
                        screen.blit(scaled_glow, glow_rect)

    def _update_and_draw_particles(self, screen: pygame.Surface):
        """Update and draw particle effects."""
        # Remove dead particles
        self.particles = [p for p in self.particles if p["lifetime"] > 0]

        # Update and draw each particle
        for particle in self.particles:
            # Update position
            particle["x"] += particle["vx"]
            particle["y"] += particle["vy"]

            # Update velocity (add gravity or other forces)
            particle["vy"] += particle["gravity"]

            # Fade out by reducing alpha
            particle["alpha"] = int(
                255 * (particle["lifetime"] / particle["max_lifetime"])
            )

            # Reduce lifetime
            particle["lifetime"] -= 1

            # Draw the particle
            if particle["lifetime"] > 0:
                particle_color = (*particle["color"], particle["alpha"])

                if particle["type"] == "circle":
                    pygame.draw.circle(
                        screen,
                        particle_color,
                        (int(particle["x"]), int(particle["y"])),
                        particle["size"],
                    )
                elif particle["type"] == "spark":
                    end_x = particle["x"] + particle["vx"] * 2
                    end_y = particle["y"] + particle["vy"] * 2

                    pygame.draw.line(
                        screen,
                        particle_color,
                        (int(particle["x"]), int(particle["y"])),
                        (int(end_x), int(end_y)),
                        particle["size"],
                    )

    def add_particles(
        self,
        x: int,
        y: int,
        particle_type: str,
        color: Tuple[int, int, int],
        count: int = 20,
    ):
        """Add particles at the specified position."""
        for _ in range(min(count, self.max_particles - len(self.particles))):
            # Random velocity
            speed = random.uniform(1, 3)
            angle = random.uniform(0, 2 * math.pi)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)

            # Create new particle
            particle = {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "gravity": 0.05 if particle_type == "circle" else 0,
                "color": color,
                "alpha": 255,
                "size": random.randint(1, 3),
                "lifetime": random.randint(20, 40),
                "max_lifetime": 40,
                "type": particle_type,
            }

            self.particles.append(particle)

    def add_score_particles(self, row: int, col: int, is_black: bool):
        """Add celebratory particles when a player scores."""
        # Calculate position at the edge of the board
        x = self.board_x + col * self.cell_size + self.cell_size // 2
        y = (
            self.board_y
            + (0 if not is_black else self.board_rows - 1) * self.cell_size
            + self.cell_size // 2
        )

        # Particle color based on player
        color = (40, 80, 255) if is_black else (255, 60, 60)

        # Add different types of particles
        self.add_particles(x, y, "circle", color, count=30)
        self.add_particles(x, y, "spark", (255, 255, 200), count=20)

    def add_capture_particles(self, row: int, col: int, is_black_capturing: bool):
        """Add particles when a piece is captured."""
        x = self.board_x + col * self.cell_size + self.cell_size // 2
        y = self.board_y + row * self.cell_size + self.cell_size // 2

        # Use the color of the captured piece
        color = (200, 60, 70) if is_black_capturing else (40, 45, 55)

        # Add particles
        self.add_particles(x, y, "circle", color, count=15)

    def animate_move(
        self,
        screen: pygame.Surface,
        board: List[List[int]],
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        piece_value: int,
        callback=None,
    ):
        """
        Animate a piece moving from start to end position.

        Args:
            screen: Pygame surface to render to
            board: Game board state
            start_pos: Starting position (row, col)
            end_pos: Ending position (row, col) or None if moving off board
            piece_value: Value of the piece being moved
            callback: Function to call when animation completes
        """
        # Calculate start screen position
        start_row, start_col = start_pos
        start_x = self.board_x + start_col * self.cell_size + self.cell_size // 2
        start_y = self.board_y + start_row * self.cell_size + self.cell_size // 2

        # Calculate end screen position
        if (
            end_pos
            and 0 <= end_pos[0] < self.board_rows
            and 0 <= end_pos[1] < self.board_cols
        ):
            end_row, end_col = end_pos
            end_x = self.board_x + end_col * self.cell_size + self.cell_size // 2
            end_y = self.board_y + end_row * self.cell_size + self.cell_size // 2
        else:
            # Moving off board (scoring) - use direction based on piece color
            direction = 1 if piece_value == 1 else -1  # Down for black, up for red
            end_row = self.board_rows if direction > 0 else -1
            end_col = start_col
            end_x = start_x
            end_y = self.board_y + end_row * self.cell_size + self.cell_size // 2

        # Animation properties
        frames = self.animation_frames

        # Temp board without the moving piece
        temp_board = [row.copy() for row in board]
        if 0 <= start_row < len(temp_board) and 0 <= start_col < len(
            temp_board[start_row]
        ):
            temp_board[start_row][start_col] = 0

        # Animation loop
        for frame in range(frames + 1):
            # Calculate current position
            progress = frame / frames
            current_x = start_x + (end_x - start_x) * progress
            current_y = start_y + (end_y - start_y) * progress

            # Redraw board
            self.render(screen, temp_board)

            # Draw moving piece
            if piece_value == 1:  # Black piece
                base_color = (40, 45, 55)
                highlight_color = (70, 75, 85)
            else:  # Red piece
                base_color = (200, 60, 70)
                highlight_color = (230, 90, 100)

            # Shadow
            if self.use_shadows:
                shadow_offset = 5
                pygame.draw.circle(
                    screen,
                    (0, 0, 0, 100),
                    (int(current_x + shadow_offset), int(current_y + shadow_offset)),
                    self.piece_radius,
                )

            # Draw piece
            pygame.draw.circle(
                screen, base_color, (int(current_x), int(current_y)), self.piece_radius
            )

            # Highlight
            if self.use_lighting:
                highlight_offset = -self.piece_radius // 3
                highlight_radius = self.piece_radius // 2

                pygame.draw.circle(
                    screen,
                    highlight_color,
                    (
                        int(current_x + highlight_offset),
                        int(current_y + highlight_offset),
                    ),
                    highlight_radius,
                )

            # Add motion blur
            if frame > 0:
                blur_points = 3
                for i in range(1, blur_points + 1):
                    blur_progress = progress - (i * 0.05)
                    if blur_progress >= 0:
                        blur_x = start_x + (end_x - start_x) * blur_progress
                        blur_y = start_y + (end_y - start_y) * blur_progress

                        blur_alpha = int(100 / (i + 1))
                        blur_color = (*base_color, blur_alpha)

                        blur_surf = pygame.Surface(
                            (self.piece_radius * 2, self.piece_radius * 2),
                            pygame.SRCALPHA,
                        )
                        pygame.draw.circle(
                            blur_surf,
                            blur_color,
                            (self.piece_radius, self.piece_radius),
                            int(self.piece_radius * (1 - i * 0.2)),
                        )

                        blur_rect = blur_surf.get_rect(center=(blur_x, blur_y))
                        screen.blit(blur_surf, blur_rect)

            # Update display
            pygame.display.flip()
            pygame.time.delay(16)  # ~60 FPS

        # Animation complete, call callback if provided
        if callback:
            callback()

        # Add particles at the destination if scoring
        if end_pos is None or end_pos[0] < 0 or end_pos[0] >= self.board_rows:
            self.add_score_particles(start_row, start_col, piece_value == 1)
