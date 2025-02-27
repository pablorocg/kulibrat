import pygame
import sys
import os
from typing import Dict, Any, List, Optional, Tuple


import pygame
import sys
import os
from typing import Dict, Any, List, Optional, Tuple

# This class handles the game configuration screen before the main game starts
class GameConfigScreen:
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize the configuration screen."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Setup colors
        self.COLORS = {
            'BACKGROUND': (245, 245, 250),
            'PANEL_BG': (255, 255, 255),
            'PANEL_BORDER': (200, 200, 210),
            'TEXT_DARK': (50, 55, 70),
            'TEXT_LIGHT': (245, 245, 250),
            'HEADER': (70, 100, 170),
            'BUTTON': (70, 130, 200),
            'BUTTON_HOVER': (90, 150, 220),
            'BUTTON_TEXT': (255, 255, 255),
            'INPUT_BG': (240, 240, 245),
            'INPUT_BORDER': (180, 190, 210),
            'INPUT_ACTIVE': (100, 150, 250),
            'BLACK_PIECE': (40, 45, 55),
            'RED_PIECE': (200, 60, 70),
            'AI_COLOR': (120, 200, 120),
            'HUMAN_COLOR': (100, 150, 250),
            'RL_COLOR': (220, 150, 50),
        }
        
        # Font setup with responsive sizing
        pygame.font.init()
        
        # Calculate font sizes based on screen dimensions
        # This ensures text scales appropriately on different screen sizes
        font_size_factor = min(self.screen_width / 1024, self.screen_height / 768)
        
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
            font_path = os.path.join("src", "ui", "assets", "fonts", "Roboto-Regular.ttf")
            if os.path.exists(font_path):
                self.FONTS = {
                    'TITLE': pygame.font.Font(font_path, title_size),
                    'HEADER': pygame.font.Font(font_path, header_size),
                    'NORMAL': pygame.font.Font(font_path, normal_size),
                    'SMALL': pygame.font.Font(font_path, small_size),
                    'TINY': pygame.font.Font(font_path, tiny_size)
                }
            else:
                self.FONTS = {
                    'TITLE': pygame.font.SysFont('Arial', title_size),
                    'HEADER': pygame.font.SysFont('Arial', header_size),
                    'NORMAL': pygame.font.SysFont('Arial', normal_size),
                    'SMALL': pygame.font.SysFont('Arial', small_size),
                    'TINY': pygame.font.SysFont('Arial', tiny_size)
                }
        except:
            # Fallback to system fonts if custom font fails
            self.FONTS = {
                'TITLE': pygame.font.SysFont('Arial', title_size),
                'HEADER': pygame.font.SysFont('Arial', header_size),
                'NORMAL': pygame.font.SysFont('Arial', normal_size),
                'SMALL': pygame.font.SysFont('Arial', small_size),
                'TINY': pygame.font.SysFont('Arial', tiny_size)
            }
        
        # Player configuration options
        self.player_types = ["Human", "AI (Random)", "AI (Minimax)", "RL Model"]
        self.player_colors = ["Black", "Red"]
        
        # Default configuration
        self.config = {
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
        
        # UI elements
        self.buttons = []
        self.dropdowns = []
        self.input_fields = []
        self.sliders = []
        self.checkboxes = []
        
        # Active UI element
        self.active_element = None
        self.active_dropdown = None
        
        # Create background pattern
        self.background = self._create_background()
    
    def _create_background(self):
        """Create a subtle background pattern."""
        bg = pygame.Surface((self.screen_width, self.screen_height))
        bg.fill(self.COLORS['BACKGROUND'])
        
        # Add subtle pattern
        for i in range(0, self.screen_width, 20):
            for j in range(0, self.screen_height, 20):
                if (i + j) % 40 == 0:
                    pygame.draw.circle(
                        bg, 
                        (230, 235, 245), 
                        (i, j), 
                        2
                    )
        
        return bg
    
    def show(self, screen: pygame.Surface) -> Dict[str, Any]:
        """
        Show the configuration screen and return the game configuration.
        
        Args:
            screen: The pygame surface to render on
            
        Returns:
            Dictionary with game configuration options
        """
        # Setup UI elements
        self._setup_ui_elements()
        
        # Main configuration loop
        running = True
        clock = pygame.time.Clock()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # ESC key exits
                        pygame.quit()
                        sys.exit()
                    elif self.active_element and "input" in self.active_element["type"]:
                        # Handle text input
                        if event.key == pygame.K_BACKSPACE:
                            self.active_element["value"] = self.active_element["value"][:-1]
                        elif event.key == pygame.K_RETURN:
                            # Deactivate on Enter
                            self.active_element = None
                        else:
                            # Add character to input if it's a valid character
                            if event.unicode.isprintable():
                                # Limit input length to prevent overflow
                                if len(self.active_element["value"]) < 20:
                                    self.active_element["value"] += event.unicode
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if event.button == 1:  # Left click
                        # Check button clicks
                        for button in self.buttons:
                            if button["rect"].collidepoint(pos):
                                if button["id"] == "start_game":
                                    # Apply configuration and return
                                    self._apply_configuration()
                                    return self.config
                                elif button["id"] == "toggle_fullscreen":
                                    self.config["fullscreen"] = not self.config["fullscreen"]
                                    if self.config["fullscreen"]:
                                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                                        self.screen_width, self.screen_height = screen.get_size()
                                    else:
                                        screen = pygame.display.set_mode((1024, 768))
                                        self.screen_width, self.screen_height = screen.get_size()
                                    # Recreate UI elements for new screen size
                                    self._setup_ui_elements()
                                    break
                        
                        # Check input field clicks
                        self.active_element = None
                        for input_field in self.input_fields:
                            if input_field["rect"].collidepoint(pos):
                                self.active_element = input_field
                                break
                        
                        # Check dropdown clicks
                        for dropdown in self.dropdowns:
                            if dropdown["rect"].collidepoint(pos):
                                if self.active_dropdown == dropdown:
                                    self.active_dropdown = None
                                else:
                                    self.active_dropdown = dropdown
                                break
                            elif self.active_dropdown and self.active_dropdown["id"] == dropdown["id"]:
                                # Check if clicked on a dropdown option
                                option_height = 30
                                dropdown_options = dropdown["options"]
                                for i, option in enumerate(dropdown_options):
                                    option_rect = pygame.Rect(
                                        dropdown["rect"].x,
                                        dropdown["rect"].y + dropdown["rect"].height + i * option_height,
                                        dropdown["rect"].width,
                                        option_height
                                    )
                                    if option_rect.collidepoint(pos):
                                        dropdown["value"] = option
                                        self.active_dropdown = None
                                        # Update corresponding values
                                        if dropdown["id"] == "black_player_type":
                                            self.config["black_player"]["type"] = option
                                        elif dropdown["id"] == "red_player_type":
                                            self.config["red_player"]["type"] = option
                                        break
                                
                        # Check checkbox clicks
                        for checkbox in self.checkboxes:
                            if checkbox["rect"].collidepoint(pos):
                                checkbox["value"] = not checkbox["value"]
                                # Update config
                                if checkbox["id"] == "fullscreen":
                                    self.config["fullscreen"] = checkbox["value"]
                                break
                        
                        # Check slider clicks
                        for slider in self.sliders:
                            slider_rect = pygame.Rect(
                                slider["rect"].x, 
                                slider["rect"].y, 
                                slider["rect"].width, 
                                slider["rect"].height
                            )
                            if slider_rect.collidepoint(pos):
                                # Calculate value based on position
                                value_range = slider["max"] - slider["min"]
                                pos_ratio = (pos[0] - slider["rect"].x) / slider["rect"].width
                                new_value = slider["min"] + value_range * pos_ratio
                                # Round to steps if needed
                                if slider["step"] > 0:
                                    new_value = round(new_value / slider["step"]) * slider["step"]
                                slider["value"] = min(max(new_value, slider["min"]), slider["max"])
                                # Update config
                                if slider["id"] == "target_score":
                                    self.config["target_score"] = int(slider["value"])
                                elif slider["id"] == "ai_delay":
                                    self.config["ai_delay"] = slider["value"]
                                break
                
                # Check if mouse is held down for sliders
                elif event.type == pygame.MOUSEMOTION and event.buttons[0] == 1:
                    for slider in self.sliders:
                        slider_rect = pygame.Rect(
                            slider["rect"].x - 10, 
                            slider["rect"].y - 10, 
                            slider["rect"].width + 20, 
                            slider["rect"].height + 20
                        )
                        if slider_rect.collidepoint(event.pos):
                            # Calculate value based on position
                            value_range = slider["max"] - slider["min"]
                            pos_ratio = (event.pos[0] - slider["rect"].x) / slider["rect"].width
                            pos_ratio = max(0, min(1, pos_ratio))  # Clamp between 0 and 1
                            new_value = slider["min"] + value_range * pos_ratio
                            # Round to steps if needed
                            if slider["step"] > 0:
                                new_value = round(new_value / slider["step"]) * slider["step"]
                            slider["value"] = min(max(new_value, slider["min"]), slider["max"])
                            # Update config
                            if slider["id"] == "target_score":
                                self.config["target_score"] = int(slider["value"])
                            elif slider["id"] == "ai_delay":
                                self.config["ai_delay"] = slider["value"]
                            break
            
            # Update display
            self._draw(screen)
            pygame.display.flip()
            clock.tick(60)
        
        return self.config
    
    def _setup_ui_elements(self):
        """Set up all UI elements for the configuration screen with enhanced responsiveness."""
        # Reset all UI elements
        self.buttons = []
        self.dropdowns = []
        self.input_fields = []
        self.sliders = []
        self.checkboxes = []
        
        # Get screen aspect ratio to optimize layout
        aspect_ratio = self.screen_width / self.screen_height
        
        # Calculate responsive layout positions based on screen size and aspect ratio
        # Handle different screen dimensions and orientations
        if aspect_ratio < 1.0:  # Portrait orientation (mobile screens)
            # Use more vertical layout for portrait orientation
            panel_width = min(int(self.screen_width * 0.95), 800)
            panel_height = min(int(self.screen_height * 0.85), 800)
        elif self.screen_width < 800 or self.screen_height < 600:  # Small screens
            # For small screens, use larger percentage of available space
            panel_width = min(int(self.screen_width * 0.9), 800)
            panel_height = min(int(self.screen_height * 0.9), 600)
        else:  # Normal and large screens
            # Standard layout with fixed maximum size
            panel_width = min(800, self.screen_width - 80)
            panel_height = min(600, self.screen_height - 80)
        
        # Center the panel on screen
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        
        # Adjust column width for very narrow screens
        if panel_width < 500:
            # For narrow screens, use single column layout by making each column full width
            column_width = panel_width - 40
        else:
            # For wider screens, use two-column layout
            column_width = (panel_width - 60) // 2
        
        # Create the main panel
        self.panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        # Determine if we should use single-column or two-column layout
        # For narrow screens, stack the player configurations vertically
        use_single_column = panel_width < 500
        
        # Determine element height based on screen size for better touch interaction
        element_height = max(40, int(min(self.screen_height, self.screen_width) * 0.06))
        
        # Determine spacing between elements based on panel height
        spacing = max(30, int(panel_height * 0.05))
        
        # Title spacing - adjust for smaller screens
        title_spacing = int(panel_height * 0.12)  # Space for the title area
        
        # Starting Y position after title
        y_pos = panel_y + title_spacing
        
        # Player 1 (BLACK) configuration
        black_header_y = y_pos
        
        # Player 1 name input
        self.input_fields.append({
            "id": "black_player_name",
            "type": "input",
            "label": "Black Player Name",
            "value": self.config["black_player"]["name"],
            "rect": pygame.Rect(panel_x + 30, y_pos, column_width, element_height)
        })
        
        y_pos += spacing
        
        # Player 1 type dropdown
        self.dropdowns.append({
            "id": "black_player_type",
            "label": "Player Type",
            "value": self.config["black_player"]["type"],
            "options": self.player_types,
            "rect": pygame.Rect(panel_x + 30, y_pos, column_width, element_height)
        })
        
        if use_single_column:
            # In single column mode, continue adding elements below
            y_pos += spacing * 1.5  # Add extra spacing between player sections
            red_header_y = y_pos
            
            # Player 2 name input
            self.input_fields.append({
                "id": "red_player_name",
                "type": "input",
                "label": "Red Player Name",
                "value": self.config["red_player"]["name"],
                "rect": pygame.Rect(panel_x + 30, y_pos, column_width, element_height)
            })
            
            y_pos += spacing
            
            # Player 2 type dropdown
            self.dropdowns.append({
                "id": "red_player_type",
                "label": "Player Type",
                "value": self.config["red_player"]["type"],
                "options": self.player_types,
                "rect": pygame.Rect(panel_x + 30, y_pos, column_width, element_height)
            })
        else:
            # In two-column mode, place Player 2 on the right
            # Player 2 (RED) configuration - Right column
            red_header_y = black_header_y  # Same level as black header
            right_column_x = panel_x + panel_width - column_width - 30
            
            # Player 2 name input
            self.input_fields.append({
                "id": "red_player_name",
                "type": "input",
                "label": "Red Player Name",
                "value": self.config["red_player"]["name"],
                "rect": pygame.Rect(right_column_x, black_header_y, column_width, element_height)
            })
            
            # Player 2 type dropdown (align with black dropdown)
            self.dropdowns.append({
                "id": "red_player_type",
                "label": "Player Type",
                "value": self.config["red_player"]["type"],
                "options": self.player_types,
                "rect": pygame.Rect(right_column_x, black_header_y + spacing, column_width, element_height)
            })
        
        # Track the next Y position for game settings
        if use_single_column:
            # In single column, continue from the last element
            settings_y = y_pos + spacing * 1.5  # Add extra spacing after player configs
        else:
            # In two columns, start after the player configuration section
            settings_y = black_header_y + spacing * 2.5
        
        # Game settings section
        # Set slider height based on screen size
        slider_height = max(20, int(element_height * 0.5))
        
        # Target score slider
        self.sliders.append({
            "id": "target_score",
            "label": "Target Score",
            "value": self.config["target_score"],
            "min": 1,
            "max": 10,
            "step": 1,
            "rect": pygame.Rect(panel_x + 30, settings_y, column_width if use_single_column else panel_width - 60, slider_height)
        })
        
        settings_y += spacing
        
        # AI delay slider
        self.sliders.append({
            "id": "ai_delay",
            "label": "AI Move Delay (seconds)",
            "value": self.config["ai_delay"],
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
            "rect": pygame.Rect(panel_x + 30, settings_y, column_width if use_single_column else panel_width - 60, slider_height)
        })
        
        settings_y += spacing
        
        # Make checkbox size responsive
        checkbox_size = max(18, int(element_height * 0.6))
        
        # RL model path input - skip on very small screens to save space
        if self.screen_height >= 500:
            self.input_fields.append({
                "id": "rl_model_path",
                "type": "input",
                "label": "RL Model Path (for RL players)",
                "value": self.config["rl_model_path"],
                "rect": pygame.Rect(panel_x + 30, settings_y, column_width if use_single_column else panel_width - 60, element_height)
            })
            
            settings_y += spacing
        
        # Fullscreen checkbox
        self.checkboxes.append({
            "id": "fullscreen",
            "label": "Fullscreen Mode",
            "value": self.config["fullscreen"],
            "rect": pygame.Rect(panel_x + 30, settings_y, checkbox_size, checkbox_size)
        })
        
        # Calculate button sizes and positions based on screen size
        button_height = max(40, int(element_height * 1.2))
        button_width = max(150, int(column_width * 0.8))
        button_spacing = int(panel_width * 0.05)
        
        # Buttons at the bottom - use responsive positioning
        # Reserve space at bottom based on screen size
        bottom_margin = max(20, int(panel_height * 0.05))
        buttons_y = panel_y + panel_height - button_height - bottom_margin
        
        if use_single_column:
            # For single column layout, stack buttons if panel is too narrow
            if panel_width < 350:
                # Toggle fullscreen button on top
                self.buttons.append({
                    "id": "toggle_fullscreen",
                    "label": "Toggle Fullscreen",
                    "rect": pygame.Rect(
                        panel_x + (panel_width - button_width) // 2,  # Center horizontally
                        buttons_y - button_height - 10,  # Position above start button
                        button_width, 
                        button_height
                    )
                })
                
                # Start game button at bottom
                self.buttons.append({
                    "id": "start_game",
                    "label": "Start Game",
                    "rect": pygame.Rect(
                        panel_x + (panel_width - button_width) // 2,  # Center horizontally
                        buttons_y,
                        button_width,
                        button_height
                    )
                })
            else:
                # Side by side buttons for wider panels in single column mode
                # Toggle fullscreen button on left
                self.buttons.append({
                    "id": "toggle_fullscreen",
                    "label": "Toggle Fullscreen",
                    "rect": pygame.Rect(
                        panel_x + 30,
                        buttons_y,
                        button_width,
                        button_height
                    )
                })
                
                # Start game button on right
                self.buttons.append({
                    "id": "start_game",
                    "label": "Start Game",
                    "rect": pygame.Rect(
                        panel_x + panel_width - button_width - 30,
                        buttons_y,
                        button_width,
                        button_height
                    )
                })
        else:
            # Two column layout - standard button placement
            # Toggle fullscreen button
            self.buttons.append({
                "id": "toggle_fullscreen",
                "label": "Toggle Fullscreen",
                "rect": pygame.Rect(
                    panel_x + 30,
                    buttons_y,
                    button_width,
                    button_height
                )
            })
            
            # Start game button
            self.buttons.append({
                "id": "start_game",
                "label": "Start Game",
                "rect": pygame.Rect(
                    panel_x + panel_width - button_width - 30,
                    buttons_y,
                    button_width,
                    button_height
                )
            })
    
    def _draw(self, screen: pygame.Surface):
        """Draw the configuration screen."""
        # Draw background
        screen.blit(self.background, (0, 0))
        
        # Draw main panel
        pygame.draw.rect(screen, self.COLORS['PANEL_BG'], self.panel_rect, border_radius=10)
        pygame.draw.rect(screen, self.COLORS['PANEL_BORDER'], self.panel_rect, width=2, border_radius=10)
        
        # Draw title
        title = self.FONTS['TITLE'].render("KULIBRAT", True, self.COLORS['HEADER'])
        title_rect = title.get_rect(centerx=self.panel_rect.centerx, y=self.panel_rect.y + 20)
        screen.blit(title, title_rect)
        
        # Draw subtitle
        subtitle = self.FONTS['NORMAL'].render("Game Configuration", True, self.COLORS['TEXT_DARK'])
        subtitle_rect = subtitle.get_rect(centerx=self.panel_rect.centerx, y=title_rect.bottom + 10)
        screen.blit(subtitle, subtitle_rect)
        
        # Draw player sections headers
        black_header = self.FONTS['HEADER'].render("Black Player", True, self.COLORS['BLACK_PIECE'])
        screen.blit(black_header, (self.panel_rect.x + 30, self.panel_rect.y + 70))
        
        red_header = self.FONTS['HEADER'].render("Red Player", True, self.COLORS['RED_PIECE'])
        red_header_rect = red_header.get_rect(right=self.panel_rect.right - 30, y=self.panel_rect.y + 70)
        screen.blit(red_header, red_header_rect)
        
        # Draw game settings header
        settings_header = self.FONTS['HEADER'].render("Game Settings", True, self.COLORS['TEXT_DARK'])
        settings_header_rect = settings_header.get_rect(centerx=self.panel_rect.centerx, y=self.panel_rect.y + 210)
        screen.blit(settings_header, settings_header_rect)
        
        # Draw input fields
        for input_field in self.input_fields:
            self._draw_input_field(screen, input_field)
        
        # Draw dropdowns
        for dropdown in self.dropdowns:
            self._draw_dropdown(screen, dropdown)
        
        # Draw sliders
        for slider in self.sliders:
            self._draw_slider(screen, slider)
        
        # Draw checkboxes
        for checkbox in self.checkboxes:
            self._draw_checkbox(screen, checkbox)
        
        # Draw buttons
        for button in self.buttons:
            self._draw_button(screen, button)
    
    def _draw_input_field(self, screen: pygame.Surface, input_field: Dict[str, Any]):
        """Draw an input field."""
        # Draw label
        label = self.FONTS['SMALL'].render(input_field["label"], True, self.COLORS['TEXT_DARK'])
        screen.blit(label, (input_field["rect"].x, input_field["rect"].y - 25))
        
        # Draw input box
        border_color = self.COLORS['INPUT_ACTIVE'] if self.active_element == input_field else self.COLORS['INPUT_BORDER']
        pygame.draw.rect(screen, self.COLORS['INPUT_BG'], input_field["rect"], border_radius=5)
        pygame.draw.rect(screen, border_color, input_field["rect"], width=2, border_radius=5)
        
        # Draw text
        text = self.FONTS['NORMAL'].render(input_field["value"], True, self.COLORS['TEXT_DARK'])
        # Ensure text stays within the input field
        text_rect = text.get_rect(midleft=(input_field["rect"].x + 10, input_field["rect"].centery))
        screen.blit(text, text_rect)
        
        # Draw cursor if this is the active input
        if self.active_element == input_field:
            cursor_pos = text_rect.right + 2
            pygame.draw.line(
                screen,
                self.COLORS['TEXT_DARK'],
                (cursor_pos, input_field["rect"].y + 10),
                (cursor_pos, input_field["rect"].y + input_field["rect"].height - 10),
                2
            )
    
    def _draw_dropdown(self, screen: pygame.Surface, dropdown: Dict[str, Any]):
        """Draw a dropdown menu."""
        # Draw label
        label = self.FONTS['SMALL'].render(dropdown["label"], True, self.COLORS['TEXT_DARK'])
        screen.blit(label, (dropdown["rect"].x, dropdown["rect"].y - 25))
        
        # Draw dropdown box
        border_color = self.COLORS['INPUT_ACTIVE'] if self.active_dropdown == dropdown else self.COLORS['INPUT_BORDER']
        pygame.draw.rect(screen, self.COLORS['INPUT_BG'], dropdown["rect"], border_radius=5)
        pygame.draw.rect(screen, border_color, dropdown["rect"], width=2, border_radius=5)
        
        # Draw selected value
        text = self.FONTS['NORMAL'].render(dropdown["value"], True, self.COLORS['TEXT_DARK'])
        text_rect = text.get_rect(midleft=(dropdown["rect"].x + 10, dropdown["rect"].centery))
        screen.blit(text, text_rect)
        
        # Draw dropdown arrow
        arrow_points = [
            (dropdown["rect"].right - 20, dropdown["rect"].centery - 5),
            (dropdown["rect"].right - 10, dropdown["rect"].centery + 5),
            (dropdown["rect"].right - 30, dropdown["rect"].centery + 5)
        ]
        pygame.draw.polygon(screen, self.COLORS['TEXT_DARK'], arrow_points)
        
        # Draw dropdown options if this dropdown is active
        if self.active_dropdown == dropdown:
            option_height = 30
            dropdown_height = len(dropdown["options"]) * option_height
            dropdown_rect = pygame.Rect(
                dropdown["rect"].x,
                dropdown["rect"].y + dropdown["rect"].height,
                dropdown["rect"].width,
                dropdown_height
            )
            
            # Draw dropdown background
            pygame.draw.rect(screen, self.COLORS['INPUT_BG'], dropdown_rect, border_radius=5)
            pygame.draw.rect(screen, self.COLORS['INPUT_BORDER'], dropdown_rect, width=2, border_radius=5)
            
            # Draw each option
            for i, option in enumerate(dropdown["options"]):
                option_rect = pygame.Rect(
                    dropdown["rect"].x,
                    dropdown["rect"].y + dropdown["rect"].height + i * option_height,
                    dropdown["rect"].width,
                    option_height
                )
                
                # Highlight if mouse is over this option
                mouse_pos = pygame.mouse.get_pos()
                if option_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.COLORS['INPUT_ACTIVE'], option_rect, border_radius=5)
                
                # Draw option text
                option_text = self.FONTS['NORMAL'].render(option, True, self.COLORS['TEXT_DARK'])
                option_text_rect = option_text.get_rect(midleft=(option_rect.x + 10, option_rect.centery))
                screen.blit(option_text, option_text_rect)
    
    def _draw_slider(self, screen: pygame.Surface, slider: Dict[str, Any]):
        """Draw a slider."""
        # Draw label
        label = self.FONTS['SMALL'].render(slider["label"], True, self.COLORS['TEXT_DARK'])
        screen.blit(label, (slider["rect"].x, slider["rect"].y - 25))
        
        # Draw current value
        if slider["id"] == "target_score":
            value_text = self.FONTS['SMALL'].render(f"{int(slider['value'])}", True, self.COLORS['TEXT_DARK'])
        else:
            value_text = self.FONTS['SMALL'].render(f"{slider['value']:.1f}", True, self.COLORS['TEXT_DARK'])
        value_rect = value_text.get_rect(midright=(slider["rect"].right, slider["rect"].y - 25))
        screen.blit(value_text, value_rect)
        
        # Draw slider track
        pygame.draw.rect(screen, self.COLORS['INPUT_BORDER'], slider["rect"], border_radius=5)
        
        # Calculate fill width based on value
        value_ratio = (slider["value"] - slider["min"]) / (slider["max"] - slider["min"])
        fill_width = int(slider["rect"].width * value_ratio)
        fill_rect = pygame.Rect(slider["rect"].x, slider["rect"].y, fill_width, slider["rect"].height)
        pygame.draw.rect(screen, self.COLORS['BUTTON'], fill_rect, border_radius=5)
        
        # Draw slider handle
        handle_x = slider["rect"].x + fill_width
        handle_rect = pygame.Rect(handle_x - 8, slider["rect"].y - 5, 16, slider["rect"].height + 10)
        pygame.draw.rect(screen, self.COLORS['BUTTON_HOVER'], handle_rect, border_radius=8)
    
    def _draw_checkbox(self, screen: pygame.Surface, checkbox: Dict[str, Any]):
        """Draw a checkbox."""
        # Draw checkbox box
        pygame.draw.rect(screen, self.COLORS['INPUT_BG'], checkbox["rect"], border_radius=3)
        pygame.draw.rect(screen, self.COLORS['INPUT_BORDER'], checkbox["rect"], width=2, border_radius=3)
        
        # Draw check mark if checked
        if checkbox["value"]:
            check_margin = 5
            pygame.draw.line(
                screen,
                self.COLORS['BUTTON'],
                (checkbox["rect"].x + check_margin, checkbox["rect"].centery),
                (checkbox["rect"].centerx - 2, checkbox["rect"].y + checkbox["rect"].height - check_margin),
                3
            )
            pygame.draw.line(
                screen,
                self.COLORS['BUTTON'],
                (checkbox["rect"].centerx - 2, checkbox["rect"].y + checkbox["rect"].height - check_margin),
                (checkbox["rect"].x + checkbox["rect"].width - check_margin, checkbox["rect"].y + check_margin),
                3
            )
        
        # Draw label
        label = self.FONTS['NORMAL'].render(checkbox["label"], True, self.COLORS['TEXT_DARK'])
        screen.blit(label, (checkbox["rect"].x + checkbox["rect"].width + 10, checkbox["rect"].centery - 12))
    
    def _draw_button(self, screen: pygame.Surface, button: Dict[str, Any]):
        """Draw a button."""
        # Check if mouse is hovering over button
        mouse_pos = pygame.mouse.get_pos()
        button_color = self.COLORS['BUTTON_HOVER'] if button["rect"].collidepoint(mouse_pos) else self.COLORS['BUTTON']
        
        # Special coloring for start game button
        if button["id"] == "start_game":
            button_color = (100, 180, 100) if button["rect"].collidepoint(mouse_pos) else (80, 160, 80)
        
        # Draw button
        pygame.draw.rect(screen, button_color, button["rect"], border_radius=5)
        
        # Draw button text
        text = self.FONTS['NORMAL'].render(button["label"], True, self.COLORS['BUTTON_TEXT'])
        text_rect = text.get_rect(center=button["rect"].center)
        screen.blit(text, text_rect)
    
    def _apply_configuration(self):
        """Apply all configuration changes to the config dictionary."""
        # Extract values from UI elements
        for input_field in self.input_fields:
            if input_field["id"] == "black_player_name":
                self.config["black_player"]["name"] = input_field["value"]
            elif input_field["id"] == "red_player_name":
                self.config["red_player"]["name"] = input_field["value"]
            elif input_field["id"] == "rl_model_path":
                self.config["rl_model_path"] = input_field["value"]
        
        for dropdown in self.dropdowns:
            if dropdown["id"] == "black_player_type":
                self.config["black_player"]["type"] = dropdown["value"]
            elif dropdown["id"] == "red_player_type":
                self.config["red_player"]["type"] = dropdown["value"]
        
        for slider in self.sliders:
            if slider["id"] == "target_score":
                self.config["target_score"] = int(slider["value"])
            elif slider["id"] == "ai_delay":
                self.config["ai_delay"] = slider["value"]
        
        for checkbox in self.checkboxes:
            if checkbox["id"] == "fullscreen":
                self.config["fullscreen"] = checkbox["value"]


# Modify the KulibratGUI class to incorporate the configuration screen
class KulibratGUI:
    """Enhanced graphical interface for the Kulibrat game with responsive design and configuration."""
    
    def __init__(self, screen_width=1024, screen_height=768):
        """Initialize the GUI with responsive sizing."""
        pygame.init()
        pygame.display.set_caption('Kulibrat - Strategic Board Game')
        
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
        
        # Initialize font system
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