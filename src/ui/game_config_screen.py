import pygame
import sys
import os
import math
from typing import Dict, Any, List, Optional, Tuple

# This class handles the game configuration screen before the main game starts
class GameConfigScreen:
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize the configuration screen."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Setup enhanced color palette for visual harmony
        self.COLORS = {
            # Background and panels
            'BACKGROUND': (242, 245, 250),
            'PANEL_BG': (252, 252, 255),
            'PANEL_BORDER': (210, 215, 225),
            'PANEL_SHADOW': (230, 235, 245, 120),
            
            # Text colors with improved contrast
            'TEXT_DARK': (40, 45, 60),
            'TEXT_LIGHT': (250, 250, 255),
            'TEXT_MUTED': (120, 130, 150),
            'TEXT_ACCENT': (70, 110, 180),
            
            # Header and title elements
            'HEADER': (60, 90, 160),
            'TITLE_GRADIENT_1': (50, 80, 150),
            'TITLE_GRADIENT_2': (80, 120, 200),
            
            # Interactive elements
            'BUTTON_PRIMARY': (70, 120, 210),
            'BUTTON_PRIMARY_HOVER': (85, 135, 225),
            'BUTTON_SECONDARY': (80, 160, 80),
            'BUTTON_SECONDARY_HOVER': (100, 180, 100),
            'BUTTON_TEXT': (250, 250, 255),
            
            # Form elements
            'INPUT_BG': (245, 247, 252),
            'INPUT_BG_FOCUS': (248, 250, 255),
            'INPUT_BORDER': (200, 210, 225),
            'INPUT_BORDER_FOCUS': (100, 140, 230),
            'INPUT_TEXT': (50, 55, 70),
            'INPUT_PLACEHOLDER': (150, 160, 180),
            
            # Slider elements
            'SLIDER_TRACK': (210, 220, 235),
            'SLIDER_FILL_BLACK': (60, 65, 80),
            'SLIDER_FILL_RED': (190, 70, 80),
            'SLIDER_HANDLE': (255, 255, 255),
            'SLIDER_HANDLE_BORDER': (160, 180, 210),
            
            # Player colors
            'BLACK_PIECE': (40, 45, 55),
            'BLACK_PIECE_LIGHT': (60, 65, 75),
            'RED_PIECE': (190, 60, 70),
            'RED_PIECE_LIGHT': (210, 80, 90),
            
            # Player type indicators
            'AI_COLOR': (100, 180, 100),
            'HUMAN_COLOR': (80, 130, 220),
            'RL_COLOR': (220, 150, 50),
            
            # Dropdown elements
            'DROPDOWN_ARROW': (100, 120, 150),
            'DROPDOWN_OPTION_HOVER': (240, 245, 255),
            
            # Checkbox elements
            'CHECKBOX_BG': (245, 247, 252),
            'CHECKBOX_BORDER': (200, 210, 225),
            'CHECKBOX_CHECK': (70, 120, 210),
        }
        
        # Font setup with responsive sizing
        pygame.font.init()
        
        # Calculate font sizes based on screen dimensions
        # This ensures text scales appropriately on different screen sizes
        font_size_factor = min(self.screen_width / 1024, self.screen_height / 768)
        
        # Ensure minimum readable sizes and maximum sizes for very large screens
        def responsive_size(base_size):
            size = int(base_size * font_size_factor)
            return max(12, min(size, base_size * 2))  # Min 12px, max 2x original size
        
        # Apply responsive font sizes with improved scale ratios
        title_size = responsive_size(52)  # Larger title for more prominence
        header_size = responsive_size(32)
        subheader_size = responsive_size(26)  # New intermediate size
        normal_size = responsive_size(22)  # Slightly smaller for better readability
        small_size = responsive_size(18)
        tiny_size = responsive_size(14)
        
        # Font loading with better error handling
        try:
            font_path = os.path.join("src", "ui", "assets", "fonts", "Roboto-Regular.ttf")
            if os.path.exists(font_path):
                self.FONTS = {
                    'TITLE': pygame.font.Font(font_path, title_size),
                    'HEADER': pygame.font.Font(font_path, header_size),
                    'SUBHEADER': pygame.font.Font(font_path, subheader_size),
                    'NORMAL': pygame.font.Font(font_path, normal_size),
                    'SMALL': pygame.font.Font(font_path, small_size),
                    'TINY': pygame.font.Font(font_path, tiny_size)
                }
            else:
                self.FONTS = {
                    'TITLE': pygame.font.SysFont('Arial', title_size),
                    'HEADER': pygame.font.SysFont('Arial', header_size),
                    'SUBHEADER': pygame.font.SysFont('Arial', subheader_size),
                    'NORMAL': pygame.font.SysFont('Arial', normal_size),
                    'SMALL': pygame.font.SysFont('Arial', small_size),
                    'TINY': pygame.font.SysFont('Arial', tiny_size)
                }
        except Exception as e:
            # Fallback to system fonts if custom font fails
            self.FONTS = {
                'TITLE': pygame.font.SysFont('Arial', title_size),
                'HEADER': pygame.font.SysFont('Arial', header_size),
                'SUBHEADER': pygame.font.SysFont('Arial', subheader_size),
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
        
        # Active UI element tracking
        self.active_element = None
        self.active_dropdown = None
        self.hover_element = None
        
        # Animation variables
        self.animation_counter = 0
        
        # Create background pattern
        self.background = self._create_background()
    
    def _create_background(self):
        """Create an enhanced subtle background pattern."""
        bg = pygame.Surface((self.screen_width, self.screen_height))
        bg.fill(self.COLORS['BACKGROUND'])
        
        # Create a more refined dot pattern with varying sizes
        dot_spacing = max(16, int(min(self.screen_width, self.screen_height) * 0.02))
        
        for i in range(0, self.screen_width, dot_spacing):
            for j in range(0, self.screen_height, dot_spacing):
                # Create visual interest with varied pattern
                if (i + j) % (dot_spacing * 2) == 0:
                    # Larger dots at regular intervals
                    dot_size = 2
                    pygame.draw.circle(
                        bg, 
                        (230, 235, 245), 
                        (i, j), 
                        dot_size
                    )
                elif (i + j) % (dot_spacing) == 0:
                    # Smaller dots for background texture
                    dot_size = 1
                    pygame.draw.circle(
                        bg, 
                        (235, 240, 250), 
                        (i, j), 
                        dot_size
                    )
        
        # Add subtle gradient overlay to create depth
        gradient = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        for y in range(self.screen_height):
            alpha = int(25 * (1 - abs(y / self.screen_height - 0.5) * 2))
            pygame.draw.line(
                gradient,
                (220, 230, 245, alpha),
                (0, y),
                (self.screen_width, y)
            )
        bg.blit(gradient, (0, 0))
        
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
        
        # Track hover elements for UI feedback
        self.hover_element = None
        
        # Main configuration loop
        running = True
        clock = pygame.time.Clock()
        last_time = pygame.time.get_ticks()
        
        while running:
            # Calculate delta time for smooth animations
            current_time = pygame.time.get_ticks()
            delta_time = (current_time - last_time) / 1000.0
            last_time = current_time
            
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
                        
                    # Tab key navigation between input fields
                    elif event.key == pygame.K_TAB:
                        if self.active_element:
                            # Find next input field
                            current_index = -1
                            for i, field in enumerate(self.input_fields):
                                if field == self.active_element:
                                    current_index = i
                                    break
                            
                            # Move to next field or wrap around
                            if current_index >= 0:
                                next_index = (current_index + 1) % len(self.input_fields)
                                self.active_element = self.input_fields[next_index]
                        else:
                            # If no active element, select first input field
                            if self.input_fields:
                                self.active_element = self.input_fields[0]
                    
                    # Handle text input for active input field
                    elif self.active_element and "input" in self.active_element["type"]:
                        if event.key == pygame.K_BACKSPACE:
                            self.active_element["value"] = self.active_element["value"][:-1]
                        elif event.key == pygame.K_RETURN:
                            # Deactivate on Enter
                            self.active_element = None
                        else:
                            # Add character to input if it's a valid character
                            if event.unicode.isprintable():
                                # Use different length limits based on field type
                                max_length = 30 if self.active_element["id"] == "rl_model_path" else 20
                                if len(self.active_element["value"]) < max_length:
                                    self.active_element["value"] += event.unicode
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if event.button == 1:  # Left click
                        # Check button clicks with sound feedback
                        button_clicked = False
                        for button in self.buttons:
                            if button["rect"].collidepoint(pos):
                                button_clicked = True
                                if button["id"] == "start_game":
                                    # Apply configuration and return
                                    self._apply_configuration()
                                    return self.config
                                elif button["id"] == "toggle_fullscreen":
                                    self.config["fullscreen"] = not self.config["fullscreen"]
                                    # Update fullscreen checkbox to match
                                    for checkbox in self.checkboxes:
                                        if checkbox["id"] == "fullscreen":
                                            checkbox["value"] = self.config["fullscreen"]
                                    
                                    # Switch display mode
                                    if self.config["fullscreen"]:
                                        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                                        self.screen_width, self.screen_height = screen.get_size()
                                    else:
                                        screen = pygame.display.set_mode((1024, 768))
                                        self.screen_width, self.screen_height = screen.get_size()
                                    
                                    # Recreate UI elements for new screen size
                                    self._setup_ui_elements()
                                    break
                        
                        # Close any open dropdown if clicking elsewhere
                        if self.active_dropdown and not button_clicked:
                            dropdown_area = pygame.Rect(
                                self.active_dropdown["rect"].x,
                                self.active_dropdown["rect"].y,
                                self.active_dropdown["rect"].width,
                                self.active_dropdown["rect"].height + (len(self.active_dropdown["options"]) * 36)
                            )
                            if not dropdown_area.collidepoint(pos):
                                self.active_dropdown = None
                        
                        # Check input field clicks
                        input_clicked = False
                        for input_field in self.input_fields:
                            # Use slightly larger hit area for better touch response
                            input_hit_rect = input_field["rect"].inflate(4, 4)
                            if input_hit_rect.collidepoint(pos):
                                self.active_element = input_field
                                input_clicked = True
                                break
                        
                        # Clear active element if clicking elsewhere
                        if not input_clicked and self.active_element:
                            self.active_element = None
                        
                        # Check dropdown clicks with improved collision
                        dropdown_clicked = False
                        for dropdown in self.dropdowns:
                            # Expanded hit area for better usability
                            dropdown_hit_rect = dropdown["rect"].inflate(4, 4)
                            if dropdown_hit_rect.collidepoint(pos):
                                dropdown_clicked = True
                                if self.active_dropdown == dropdown:
                                    self.active_dropdown = None
                                else:
                                    self.active_dropdown = dropdown
                                break
                        
                        # Check dropdown option clicks with improved hit boxes
                        if self.active_dropdown and not dropdown_clicked:
                            option_height = 36  # Match the drawing height
                            dropdown_options = self.active_dropdown["options"]
                            
                            # Calculate dropdown options area
                            for i, option in enumerate(dropdown_options):
                                option_rect = pygame.Rect(
                                    self.active_dropdown["rect"].x,
                                    self.active_dropdown["rect"].y + self.active_dropdown["rect"].height + 2 + i * option_height,
                                    self.active_dropdown["rect"].width,
                                    option_height
                                )
                                if option_rect.collidepoint(pos):
                                    # Update dropdown value
                                    self.active_dropdown["value"] = option
                                    
                                    # Update corresponding config values with proper type detection
                                    dropdown_id = self.active_dropdown["id"]
                                    if dropdown_id == "black_player_type":
                                        self.config["black_player"]["type"] = option
                                    elif dropdown_id == "red_player_type":
                                        self.config["red_player"]["type"] = option
                                    
                                    # Close dropdown after selection
                                    self.active_dropdown = None
                                    dropdown_clicked = True
                                    break
                        
                        # Check checkbox clicks with improved hit area
                        for checkbox in self.checkboxes:
                            # Extended hit area to include label
                            checkbox_hit_rect = pygame.Rect(
                                checkbox["rect"].x,
                                checkbox["rect"].y,
                                checkbox["rect"].width + 200,  # Include label
                                checkbox["rect"].height
                            )
                            if checkbox_hit_rect.collidepoint(pos):
                                # Toggle checkbox state
                                checkbox["value"] = not checkbox["value"]
                                
                                # Update config
                                if checkbox["id"] == "fullscreen":
                                    self.config["fullscreen"] = checkbox["value"]
                                break
                        
                        # Check slider clicks with improved hit detection
                        for slider in self.sliders:
                            # Use larger hit area for better usability
                            slider_hit_rect = slider["rect"].inflate(0, 20)
                            if slider_hit_rect.collidepoint(pos):
                                # Calculate value based on click position
                                value_range = slider["max"] - slider["min"]
                                pos_ratio = (pos[0] - slider["rect"].x) / slider["rect"].width
                                pos_ratio = max(0, min(1, pos_ratio))  # Clamp between 0 and 1
                                new_value = slider["min"] + value_range * pos_ratio
                                
                                # Apply step constraints
                                if slider["step"] > 0:
                                    new_value = round(new_value / slider["step"]) * slider["step"]
                                
                                # Update slider value with range constraints
                                slider["value"] = min(max(new_value, slider["min"]), slider["max"])
                                
                                # Update corresponding config value
                                if slider["id"] == "target_score":
                                    self.config["target_score"] = int(slider["value"])
                                elif slider["id"] == "ai_delay":
                                    self.config["ai_delay"] = slider["value"]
                                break
                
                # Handle slider dragging with improved hit detection and range constraints
                elif event.type == pygame.MOUSEMOTION and event.buttons[0] == 1:
                    for slider in self.sliders:
                        # Use larger drag area for better usability
                        slider_drag_rect = pygame.Rect(
                            slider["rect"].x - 15, 
                            slider["rect"].y - 20, 
                            slider["rect"].width + 30, 
                            slider["rect"].height + 40
                        )
                        if slider_drag_rect.collidepoint(event.pos):
                            # Calculate new value based on mouse position
                            value_range = slider["max"] - slider["min"]
                            pos_ratio = (event.pos[0] - slider["rect"].x) / slider["rect"].width
                            pos_ratio = max(0, min(1, pos_ratio))  # Clamp between 0 and 1
                            new_value = slider["min"] + value_range * pos_ratio
                            
                            # Apply step constraints for discrete sliders
                            if slider["step"] > 0:
                                new_value = round(new_value / slider["step"]) * slider["step"]
                            
                            # Update slider value with range constraints
                            slider["value"] = min(max(new_value, slider["min"]), slider["max"])
                            
                            # Update corresponding config value
                            if slider["id"] == "target_score":
                                self.config["target_score"] = int(slider["value"])
                            elif slider["id"] == "ai_delay":
                                self.config["ai_delay"] = slider["value"]
                            break
                
                # Track hover state for mouse cursor
                elif event.type == pygame.MOUSEMOTION:
                    # Update hover state for buttons and UI elements
                    pos = event.pos
                    self.hover_element = None
                    
                    # Check if hovering over buttons
                    for button in self.buttons:
                        if button["rect"].collidepoint(pos):
                            self.hover_element = button
                            break
            
            # Update display with smooth animations
            self._draw(screen)
            pygame.display.flip()
            clock.tick(60)
        
        return self.config
    
    def _setup_ui_elements(self):
        """Set up all UI elements for the configuration screen with improved layout and alignments."""
        # Reset all UI elements
        self.buttons = []
        self.dropdowns = []
        self.input_fields = []
        self.sliders = []
        self.checkboxes = []
        
        # Get screen aspect ratio to optimize layout
        aspect_ratio = self.screen_width / self.screen_height
        
        # Calculate responsive layout dimensions
        # Handle different screen dimensions more elegantly
        if aspect_ratio < 1.0:  # Portrait orientation
            # More vertical layout for portrait orientation
            panel_width = min(int(self.screen_width * 0.95), 800)
            panel_height = min(int(self.screen_height * 0.85), 900)
            panel_radius = 12  # Slightly rounder corners on mobile
        elif self.screen_width < 800 or self.screen_height < 600:  # Small screens
            # For small screens, use more of the available space
            panel_width = min(int(self.screen_width * 0.92), 800)
            panel_height = min(int(self.screen_height * 0.92), 650)
            panel_radius = 10
        else:  # Normal and large screens
            # Standard layout with fixed maximum size and generous margins
            panel_width = min(840, self.screen_width - 100)
            panel_height = min(680, self.screen_height - 100)
            panel_radius = 14  # More pronounced rounded corners on larger screens
        
        # Store panel radius for drawing
        self.panel_radius = panel_radius
        
        # Center the panel on screen with slight vertical offset for visual balance
        panel_x = (self.screen_width - panel_width) // 2
        panel_y = (self.screen_height - panel_height) // 2
        if self.screen_height > 800:
            panel_y = int(self.screen_height * 0.45 - panel_height / 2)  # Slight upward shift on tall screens
        
        # Determine if we should use single or two-column layout
        self.use_single_column = panel_width < 540 or aspect_ratio < 1.0
        
        # Calculate column widths with improved proportions
        if self.use_single_column:
            # For single column layout, set proper margins
            left_margin = right_margin = int(panel_width * 0.08)
            column_width = panel_width - left_margin - right_margin
        else:
            # For two-column layout, create balanced columns with proper spacing
            column_spacing = int(panel_width * 0.06)  # 6% of panel width for spacing
            left_margin = right_margin = int(panel_width * 0.06)  # 6% of panel width for margins
            column_width = (panel_width - left_margin - right_margin - column_spacing) // 2
        
        # Store margins for consistent alignment
        self.left_margin = left_margin
        self.right_margin = right_margin
        
        # Create the main panel
        self.panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        
        # Calculate responsive element sizes
        base_scale = min(panel_width / 800, panel_height / 600)
        
        # Element heights with proper proportional relationships
        element_height = max(40, int(44 * base_scale))
        dropdown_height = max(42, int(46 * base_scale))
        slider_height = max(18, int(20 * base_scale))
        checkbox_size = max(22, int(24 * base_scale))
        button_height = max(48, int(52 * base_scale))
        
        # Create consistent spacing throughout the layout
        base_spacing = max(28, int(32 * base_scale))
        header_spacing = max(32, int(38 * base_scale))
        section_spacing = max(40, int(44 * base_scale))
        
        # Calculate title area spacing
        title_height = int(panel_height * 0.13)
        
        # Content area starts after title
        content_y = panel_y + title_height
        
        # SECTION: PLAYER CONFIGURATION
        # Store section coordinates for headers
        self.black_header_pos = (panel_x + left_margin, content_y)
        
        # BLACK PLAYER (LEFT SIDE OR TOP IN SINGLE COLUMN)
        input_y = content_y + header_spacing
        
        # Black player name input
        self.input_fields.append({
            "id": "black_player_name",
            "type": "input",
            "label": "Player Name",
            "value": self.config["black_player"]["name"],
            "rect": pygame.Rect(panel_x + left_margin, input_y, column_width, element_height),
            "player": "black"
        })
        
        # Black player type dropdown
        dropdown_y = input_y + element_height + base_spacing
        self.dropdowns.append({
            "id": "black_player_type",
            "label": "Player Type",
            "value": self.config["black_player"]["type"],
            "options": self.player_types,
            "rect": pygame.Rect(panel_x + left_margin, dropdown_y, column_width, dropdown_height),
            "player": "black"
        })
        
        # RED PLAYER (RIGHT SIDE OR BELOW IN SINGLE COLUMN)
        if self.use_single_column:
            # In single column mode, position Red player section below Black
            red_section_y = dropdown_y + dropdown_height + section_spacing
            self.red_header_pos = (panel_x + left_margin, red_section_y)
            
            # Red player name input below black player section
            red_input_y = red_section_y + header_spacing
            self.input_fields.append({
                "id": "red_player_name",
                "type": "input",
                "label": "Player Name",
                "value": self.config["red_player"]["name"],
                "rect": pygame.Rect(panel_x + left_margin, red_input_y, column_width, element_height),
                "player": "red"
            })
            
            # Red player type dropdown
            red_dropdown_y = red_input_y + element_height + base_spacing
            self.dropdowns.append({
                "id": "red_player_type",
                "label": "Player Type",
                "value": self.config["red_player"]["type"],
                "options": self.player_types,
                "rect": pygame.Rect(panel_x + left_margin, red_dropdown_y, column_width, dropdown_height),
                "player": "red"
            })
            
            # Set next section starting position
            next_section_y = red_dropdown_y + dropdown_height + section_spacing
        else:
            # In two-column layout, place Red player in right column
            right_column_x = panel_x + panel_width - right_margin - column_width
            self.red_header_pos = (right_column_x, content_y)
            
            # Red player name input (aligned with black player input)
            self.input_fields.append({
                "id": "red_player_name",
                "type": "input",
                "label": "Player Name",
                "value": self.config["red_player"]["name"],
                "rect": pygame.Rect(right_column_x, input_y, column_width, element_height),
                "player": "red"
            })
            
            # Red player type dropdown (aligned with black player dropdown)
            self.dropdowns.append({
                "id": "red_player_type",
                "label": "Player Type",
                "value": self.config["red_player"]["type"],
                "options": self.player_types,
                "rect": pygame.Rect(right_column_x, dropdown_y, column_width, dropdown_height),
                "player": "red"
            })
            
            # Set next section starting position
            next_section_y = dropdown_y + dropdown_height + section_spacing
        
        # SECTION: GAME SETTINGS
        # Store settings header position
        self.settings_header_pos = (panel_x + panel_width // 2, next_section_y)
        settings_content_y = next_section_y + header_spacing
        
        # Calculate slider width based on layout
        if self.use_single_column:
            slider_width = column_width
            slider_x = panel_x + left_margin
        else:
            # For two-column layout, center sliders or make them wider
            slider_width = panel_width - left_margin - right_margin
            slider_x = panel_x + left_margin
        
        # Target score slider
        score_slider_y = settings_content_y + base_spacing
        self.sliders.append({
            "id": "target_score",
            "label": "Target Score",
            "value": self.config["target_score"],
            "min": 1,
            "max": 10,
            "step": 1,
            "color": "BLACK",  # Match with player color
            "rect": pygame.Rect(slider_x, score_slider_y, slider_width, slider_height)
        })
        
        # AI delay slider
        delay_slider_y = score_slider_y + slider_height + base_spacing * 1.2
        self.sliders.append({
            "id": "ai_delay",
            "label": "AI Move Delay (seconds)",
            "value": self.config["ai_delay"],
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
            "color": "RED",  # Match with player color
            "rect": pygame.Rect(slider_x, delay_slider_y, slider_width, slider_height)
        })
        
        # RL model path input
        if self.screen_height >= 500:  # Skip on very small screens
            rl_input_y = delay_slider_y + slider_height + base_spacing * 1.2
            self.input_fields.append({
                "id": "rl_model_path",
                "type": "input",
                "label": "RL Model Path",
                "value": self.config["rl_model_path"],
                "rect": pygame.Rect(slider_x, rl_input_y, slider_width, element_height)
            })
            
            # Position checkbox below RL input
            checkbox_y = rl_input_y + element_height + base_spacing
        else:
            # Position checkbox directly below sliders if RL input is skipped
            checkbox_y = delay_slider_y + slider_height + base_spacing * 1.2
        
        # Fullscreen checkbox
        self.checkboxes.append({
            "id": "fullscreen",
            "label": "Fullscreen Mode",
            "value": self.config["fullscreen"],
            "rect": pygame.Rect(slider_x, checkbox_y, checkbox_size, checkbox_size)
        })
        
        # SECTION: BUTTONS
        # Calculate button sizes with consistent proportions
        button_width = max(160, int(column_width * 0.85))
        
        # Position buttons at bottom with proper spacing
        button_margin = int(panel_width * 0.06)
        buttons_y = panel_y + panel_height - button_height - button_margin
        
        # Decide on button layout based on screen width
        if self.use_single_column and panel_width < 380:
            # For very narrow screens, stack buttons vertically
            fullscreen_button_y = buttons_y - button_height - int(base_spacing * 0.8)
            
            # Create aligned, vertically stacked buttons
            self.buttons.append({
                "id": "toggle_fullscreen",
                "label": "Toggle Fullscreen",
                "type": "secondary",
                "rect": pygame.Rect(
                    panel_x + (panel_width - button_width) // 2,
                    fullscreen_button_y,
                    button_width, 
                    button_height
                )
            })
            
            self.buttons.append({
                "id": "start_game",
                "label": "Start Game",
                "type": "primary",
                "rect": pygame.Rect(
                    panel_x + (panel_width - button_width) // 2,
                    buttons_y,
                    button_width,
                    button_height
                )
            })
        else:
            # For wider screens, place buttons side by side
            # Calculate button positions to ensure they are symmetrically aligned
            left_button_x = panel_x + button_margin
            right_button_x = panel_x + panel_width - button_width - button_margin
            
            # Create side-by-side buttons
            self.buttons.append({
                "id": "toggle_fullscreen",
                "label": "Toggle Fullscreen",
                "type": "secondary",
                "rect": pygame.Rect(
                    left_button_x,
                    buttons_y,
                    button_width,
                    button_height
                )
            })
            
            self.buttons.append({
                "id": "start_game",
                "label": "Start Game",
                "type": "primary",
                "rect": pygame.Rect(
                    right_button_x,
                    buttons_y,
                    button_width,
                    button_height
                )
            })
    
    def _draw(self, screen: pygame.Surface):
        """Draw the configuration screen with enhanced visual styling."""
        # Update animation counter (for subtle animations)
        self.animation_counter = (self.animation_counter + 0.5) % 360
        
        # Draw background pattern
        screen.blit(self.background, (0, 0))
        
        # Draw main panel with shadow effect for depth
        # First draw shadow slightly offset
        shadow_rect = self.panel_rect.copy()
        shadow_rect.x += 6
        shadow_rect.y += 6
        pygame.draw.rect(screen, self.COLORS['PANEL_SHADOW'], shadow_rect, border_radius=self.panel_radius)
        
        # Draw panel background
        pygame.draw.rect(screen, self.COLORS['PANEL_BG'], self.panel_rect, border_radius=self.panel_radius)
        
        # Draw subtle inner highlight on top/left edges for depth
        highlight_points = [
            (self.panel_rect.left + 2, self.panel_rect.bottom - self.panel_radius - 2),
            (self.panel_rect.left + 2, self.panel_rect.top + self.panel_radius + 2),
            (self.panel_rect.left + self.panel_radius + 2, self.panel_rect.top + 2),
            (self.panel_rect.right - self.panel_radius - 2, self.panel_rect.top + 2)
        ]
        pygame.draw.lines(screen, (255, 255, 255, 40), False, highlight_points, 2)
        
        # Draw panel border
        pygame.draw.rect(screen, self.COLORS['PANEL_BORDER'], self.panel_rect, width=2, border_radius=self.panel_radius)
        
        # Draw title with gradient effect
        title_text = "KULIBRAT"
        title_x = self.panel_rect.centerx
        title_y = self.panel_rect.y + 22
        
        # Create gradient effect for title
        title_shadow = self.FONTS['TITLE'].render(title_text, True, (30, 40, 60))
        title_shadow_rect = title_shadow.get_rect(center=(title_x + 2, title_y + 2))
        screen.blit(title_shadow, title_shadow_rect)
        
        title = self.FONTS['TITLE'].render(title_text, True, self.COLORS['HEADER'])
        title_rect = title.get_rect(center=(title_x, title_y))
        screen.blit(title, title_rect)
        
        # Draw subtitle with slight animation
        subtitle_y_offset = math.sin(math.radians(self.animation_counter)) * 1.5
        subtitle = self.FONTS['NORMAL'].render("Game Configuration", True, self.COLORS['TEXT_ACCENT'])
        subtitle_rect = subtitle.get_rect(centerx=self.panel_rect.centerx, centery=title_rect.bottom + 18 + subtitle_y_offset)
        screen.blit(subtitle, subtitle_rect)
        
        # Draw section dividers
        divider_y = self.settings_header_pos[1] - 20
        pygame.draw.line(
            screen,
            self.COLORS['PANEL_BORDER'],
            (self.panel_rect.x + self.left_margin, divider_y),
            (self.panel_rect.x + self.panel_rect.width - self.right_margin, divider_y),
            1
        )
        
        # Draw player section headers with improved styling
        # Black player header
        self._draw_section_header(
            screen,
            "Black Player",
            self.black_header_pos,
            self.COLORS['BLACK_PIECE'],
            self.COLORS['BLACK_PIECE_LIGHT'],
            align="left"
        )
        
        # Red player header
        self._draw_section_header(
            screen,
            "Red Player",
            self.red_header_pos,
            self.COLORS['RED_PIECE'],
            self.COLORS['RED_PIECE_LIGHT'],
            align="left" if hasattr(self, 'use_single_column') and self.use_single_column else "right"
        )
        
        # Draw game settings header
        self._draw_section_header(
            screen,
            "Game Settings",
            self.settings_header_pos,
            self.COLORS['TEXT_ACCENT'],
            self.COLORS['TEXT_MUTED'],
            align="center"
        )
        
        # Draw UI elements in specific order for proper layering
        # First draw backgrounds
        for input_field in self.input_fields:
            self._draw_input_field_background(screen, input_field)
        
        for dropdown in self.dropdowns:
            self._draw_dropdown_background(screen, dropdown)
        
        # Draw interactive elements on top
        for input_field in self.input_fields:
            self._draw_input_field_content(screen, input_field)
        
        for dropdown in self.dropdowns:
            self._draw_dropdown_content(screen, dropdown)
        
        # Draw sliders
        for slider in self.sliders:
            self._draw_slider(screen, slider)
        
        # Draw checkboxes
        for checkbox in self.checkboxes:
            self._draw_checkbox(screen, checkbox)
        
        # Draw buttons
        for button in self.buttons:
            self._draw_button(screen, button)
    
    def _draw_section_header(self, screen, text, position, color, shadow_color, align="left"):
        """Draw a section header with enhanced styling."""
        header_font = self.FONTS['HEADER']
        
        # Create header text
        header_text = header_font.render(text, True, color)
        
        # Set position based on alignment
        if align == "left":
            header_rect = header_text.get_rect(topleft=position)
        elif align == "right":
            header_rect = header_text.get_rect(topright=position)
        else:  # center
            header_rect = header_text.get_rect(midtop=position)
        
        # Draw subtle shadow for depth
        shadow_text = header_font.render(text, True, shadow_color)
        shadow_rect = shadow_text.get_rect(topleft=(header_rect.x + 1, header_rect.y + 1))
        screen.blit(shadow_text, shadow_rect)
        
        # Draw main text
        screen.blit(header_text, header_rect)
        
        # Draw accent underline
        line_y = header_rect.bottom + 4
        line_length = header_text.get_width() * 0.8
        
        if align == "left":
            line_start = (header_rect.x, line_y)
            line_end = (header_rect.x + line_length, line_y)
        elif align == "right":
            line_start = (header_rect.right - line_length, line_y)
            line_end = (header_rect.right, line_y)
        else:  # center
            half_length = line_length / 2
            line_start = (header_rect.centerx - half_length, line_y)
            line_end = (header_rect.centerx + half_length, line_y)
        
        # Draw line with slight gradient effect
        for i in range(2):
            alpha = 180 if i == 0 else 100
            offset = 0 if i == 0 else 2
            pygame.draw.line(
                screen,
                (*color[:3], alpha),
                line_start,
                line_end,
                2 if i == 0 else 1
            )
    
    def _draw_input_field_background(self, screen, input_field):
        """Draw the background of an input field."""
        # Determine if this input is active
        is_active = self.active_element == input_field
        is_hovered = input_field["rect"].collidepoint(pygame.mouse.get_pos())
        
        # Draw input box with proper visual state
        bg_color = self.COLORS['INPUT_BG_FOCUS'] if is_active else self.COLORS['INPUT_BG']
        border_color = self.COLORS['INPUT_BORDER_FOCUS'] if is_active else self.COLORS['INPUT_BORDER']
        
        # Apply special styling for player-specific input fields
        if "player" in input_field:
            if input_field["player"] == "black":
                border_color = self.COLORS['BLACK_PIECE'] if is_active else self.COLORS['INPUT_BORDER']
            elif input_field["player"] == "red":
                border_color = self.COLORS['RED_PIECE'] if is_active else self.COLORS['INPUT_BORDER']
        
        # Draw input field background with slight shadow for depth
        shadow_rect = input_field["rect"].copy()
        shadow_rect.y += 2
        pygame.draw.rect(screen, self.COLORS['PANEL_SHADOW'], shadow_rect, border_radius=6)
        
        # Draw main input background
        pygame.draw.rect(screen, bg_color, input_field["rect"], border_radius=6)
        
        # Draw border with varying thickness based on state
        border_width = 2 if is_active else 1
        pygame.draw.rect(screen, border_color, input_field["rect"], width=border_width, border_radius=6)
        
        # Add subtle highlight when hovered
        if is_hovered and not is_active:
            highlight_rect = input_field["rect"].inflate(-4, -4)
            pygame.draw.rect(screen, (*border_color[:3], 40), highlight_rect, width=1, border_radius=4)
    
    def _draw_input_field_content(self, screen, input_field):
        """Draw the content of an input field."""
        # Draw label with proper positioning
        font_color = self.COLORS['INPUT_TEXT']
        if "player" in input_field:
            if input_field["player"] == "black":
                font_color = self.COLORS['BLACK_PIECE']
            elif input_field["player"] == "red":
                font_color = self.COLORS['RED_PIECE']
        
        label = self.FONTS['SMALL'].render(input_field["label"], True, font_color)
        label_y = input_field["rect"].y - 24
        screen.blit(label, (input_field["rect"].x, label_y))
        
        # Prepare text with clipping if needed
        value = input_field["value"]
        max_width = input_field["rect"].width - 20  # Padding on both sides
        
        # Check if text needs to be clipped
        text = self.FONTS['NORMAL'].render(value, True, self.COLORS['INPUT_TEXT'])
        if text.get_width() > max_width:
            # Show text from the end if it's too long (especially for paths)
            visible_chars = len(value)
            while visible_chars > 0:
                clipped_text = "..." + value[-visible_chars:] if visible_chars < len(value) else value
                text = self.FONTS['NORMAL'].render(clipped_text, True, self.COLORS['INPUT_TEXT'])
                if text.get_width() <= max_width:
                    break
                visible_chars -= 1
        
        # Center text vertically within input field
        text_rect = text.get_rect(midleft=(input_field["rect"].x + 10, input_field["rect"].centery))
        screen.blit(text, text_rect)
        
        # Draw cursor if this is the active input field (with animation)
        if self.active_element == input_field:
            cursor_x = text_rect.right + 2
            cursor_height = input_field["rect"].height * 0.6
            cursor_y = input_field["rect"].centery - cursor_height / 2
            
            # Add subtle pulsing animation to cursor
            alpha = int(180 + 75 * math.sin(math.radians(self.animation_counter * 3)))
            cursor_color = (*self.COLORS['INPUT_TEXT'][:3], alpha)
            
            pygame.draw.line(
                screen,
                cursor_color,
                (cursor_x, cursor_y),
                (cursor_x, cursor_y + cursor_height),
                2
            )
    
    def _draw_dropdown_background(self, screen, dropdown):
        """Draw the background of a dropdown."""
        # Determine if this dropdown is active
        is_active = self.active_dropdown == dropdown
        is_hovered = dropdown["rect"].collidepoint(pygame.mouse.get_pos())
        
        # Choose colors based on state
        bg_color = self.COLORS['INPUT_BG_FOCUS'] if is_active else self.COLORS['INPUT_BG']
        border_color = self.COLORS['INPUT_BORDER_FOCUS'] if is_active else self.COLORS['INPUT_BORDER']
        
        # Apply special styling for player-specific dropdowns
        if "player" in dropdown:
            if dropdown["player"] == "black":
                border_color = self.COLORS['BLACK_PIECE'] if is_active else self.COLORS['INPUT_BORDER']
            elif dropdown["player"] == "red":
                border_color = self.COLORS['RED_PIECE'] if is_active else self.COLORS['INPUT_BORDER']
        
        # Draw dropdown background with slight shadow for depth
        shadow_rect = dropdown["rect"].copy()
        shadow_rect.y += 2
        pygame.draw.rect(screen, self.COLORS['PANEL_SHADOW'], shadow_rect, border_radius=6)
        
        # Draw main dropdown background
        pygame.draw.rect(screen, bg_color, dropdown["rect"], border_radius=6)
        
        # Draw border with varying thickness based on state
        border_width = 2 if is_active else 1
        pygame.draw.rect(screen, border_color, dropdown["rect"], width=border_width, border_radius=6)
        
        # Add subtle highlight when hovered
        if is_hovered and not is_active:
            highlight_rect = dropdown["rect"].inflate(-4, -4)
            pygame.draw.rect(screen, (*border_color[:3], 40), highlight_rect, width=1, border_radius=4)
            
    def _draw_dropdown_content(self, screen, dropdown):
        """Draw the content of a dropdown."""
        # Draw label with proper positioning
        font_color = self.COLORS['TEXT_DARK']
        if "player" in dropdown:
            if dropdown["player"] == "black":
                font_color = self.COLORS['BLACK_PIECE']
            elif dropdown["player"] == "red":
                font_color = self.COLORS['RED_PIECE']
        
        label = self.FONTS['SMALL'].render(dropdown["label"], True, font_color)
        label_y = dropdown["rect"].y - 24
        screen.blit(label, (dropdown["rect"].x, label_y))
        
        # Draw selected value with color based on value type
        value_color = self.COLORS['TEXT_DARK']
        if dropdown["value"].startswith("AI"):
            value_color = self.COLORS['AI_COLOR']
        elif dropdown["value"] == "Human":
            value_color = self.COLORS['HUMAN_COLOR']
        elif dropdown["value"] == "RL Model":
            value_color = self.COLORS['RL_COLOR']
            
        # Render selected text
        text = self.FONTS['NORMAL'].render(dropdown["value"], True, value_color)
        text_rect = text.get_rect(midleft=(dropdown["rect"].x + 10, dropdown["rect"].centery))
        screen.blit(text, text_rect)
        
        # Draw dropdown arrow with slight animation
        arrow_y_offset = math.sin(math.radians(self.animation_counter)) * 1.0 if self.active_dropdown == dropdown else 0
        arrow_points = [
            (dropdown["rect"].right - 20, dropdown["rect"].centery - 5 + arrow_y_offset),
            (dropdown["rect"].right - 10, dropdown["rect"].centery + 5 + arrow_y_offset),
            (dropdown["rect"].right - 30, dropdown["rect"].centery + 5 + arrow_y_offset)
        ]
        pygame.draw.polygon(screen, self.COLORS['DROPDOWN_ARROW'], arrow_points)
        
        # Draw dropdown options if this dropdown is active
        if self.active_dropdown == dropdown:
            option_height = 36  # Increased for better touch targets
            dropdown_height = len(dropdown["options"]) * option_height
            
            # Create dropdown panel
            dropdown_rect = pygame.Rect(
                dropdown["rect"].x,
                dropdown["rect"].y + dropdown["rect"].height + 2,
                dropdown["rect"].width,
                dropdown_height
            )
            
            # Draw dropdown panel background with shadow
            shadow_rect = dropdown_rect.copy()
            shadow_rect.x += 4
            shadow_rect.y += 4
            pygame.draw.rect(screen, self.COLORS['PANEL_SHADOW'], shadow_rect, border_radius=6)
            
            # Draw main dropdown panel
            pygame.draw.rect(screen, self.COLORS['INPUT_BG'], dropdown_rect, border_radius=6)
            pygame.draw.rect(screen, self.COLORS['INPUT_BORDER'], dropdown_rect, width=1, border_radius=6)
            
            # Draw each option
            mouse_pos = pygame.mouse.get_pos()
            for i, option in enumerate(dropdown["options"]):
                option_rect = pygame.Rect(
                    dropdown["rect"].x,
                    dropdown["rect"].y + dropdown["rect"].height + 2 + i * option_height,
                    dropdown["rect"].width,
                    option_height
                )
                
                # Determine option color based on type
                option_color = self.COLORS['TEXT_DARK']
                if option.startswith("AI"):
                    option_color = self.COLORS['AI_COLOR']
                elif option == "Human":
                    option_color = self.COLORS['HUMAN_COLOR']
                elif option == "RL Model":
                    option_color = self.COLORS['RL_COLOR']
                
                # Highlight if mouse is over this option or if it's the selected option
                if option_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, self.COLORS['DROPDOWN_OPTION_HOVER'], option_rect, border_radius=4)
                    # Add subtle highlight outline
                    pygame.draw.rect(screen, self.COLORS['INPUT_BORDER_FOCUS'], option_rect, width=1, border_radius=4)
                
                # Show selected state
                if option == dropdown["value"]:
                    # Draw subtle checkmark for selected item
                    checkmark_x = option_rect.x + 10
                    checkmark_y = option_rect.centery
                    checkmark_points = [
                        (checkmark_x, checkmark_y),
                        (checkmark_x + 5, checkmark_y + 5),
                        (checkmark_x + 10, checkmark_y - 5)
                    ]
                    pygame.draw.lines(screen, option_color, False, checkmark_points, 2)
                
                # Draw option text
                option_text = self.FONTS['NORMAL'].render(option, True, option_color)
                option_text_rect = option_text.get_rect(midleft=(option_rect.x + 26, option_rect.centery))
                screen.blit(option_text, option_text_rect)
    
    def _draw_slider(self, screen: pygame.Surface, slider: Dict[str, Any]):
        """Draw a slider with enhanced visuals."""
        # Determine if mouse is hovering over slider
        mouse_pos = pygame.mouse.get_pos()
        slider_hover_area = slider["rect"].inflate(0, 20)
        is_hovered = slider_hover_area.collidepoint(mouse_pos)
        
        # Draw label with player-specific color
        if slider.get("color") == "BLACK":
            label_color = self.COLORS['BLACK_PIECE']
        elif slider.get("color") == "RED":
            label_color = self.COLORS['RED_PIECE']
        else:
            label_color = self.COLORS['TEXT_DARK']
            
        # Label and value display
        label = self.FONTS['SMALL'].render(slider["label"], True, label_color)
        screen.blit(label, (slider["rect"].x, slider["rect"].y - 24))
        
        # Draw current value with enhanced styling
        if slider["id"] == "target_score":
            value_text = f"{int(slider['value'])}"
        else:
            value_text = f"{slider['value']:.1f}s"
            
        value_display = self.FONTS['SMALL'].render(value_text, True, label_color)
        value_rect = value_display.get_rect(midright=(slider["rect"].right, slider["rect"].y - 24))
        screen.blit(value_display, value_rect)
        
        # Draw slider track with shadow for depth
        track_shadow = slider["rect"].copy()
        track_shadow.y += 2
        pygame.draw.rect(screen, self.COLORS['PANEL_SHADOW'], track_shadow, border_radius=6)
        
        # Draw main track
        pygame.draw.rect(screen, self.COLORS['SLIDER_TRACK'], slider["rect"], border_radius=6)
        
        # Calculate fill width based on value
        value_ratio = (slider["value"] - slider["min"]) / (slider["max"] - slider["min"])
        fill_width = max(0, min(int(slider["rect"].width * value_ratio), slider["rect"].width))
        
        # Determine fill color based on slider type
        if slider.get("color") == "BLACK":
            fill_color = self.COLORS['SLIDER_FILL_BLACK']
        elif slider.get("color") == "RED":
            fill_color = self.COLORS['SLIDER_FILL_RED']
        else:
            fill_color = self.COLORS['BUTTON_PRIMARY']
            
        # Draw fill area
        if fill_width > 0:
            fill_rect = pygame.Rect(slider["rect"].x, slider["rect"].y, fill_width, slider["rect"].height)
            pygame.draw.rect(screen, fill_color, fill_rect, border_radius=6)
            
            # Draw slight gradient effect at right edge of fill
            gradient_width = min(8, fill_width)
            if gradient_width > 0:
                for i in range(gradient_width):
                    alpha = 100 - (i * 100 // gradient_width)
                    grad_x = fill_rect.right - gradient_width + i
                    grad_rect = pygame.Rect(grad_x, fill_rect.y, 1, fill_rect.height)
                    color_with_alpha = (*fill_color[:3], alpha)
                    pygame.draw.rect(screen, color_with_alpha, grad_rect)
        
        # Calculate handle position
        handle_x = slider["rect"].x + fill_width
        handle_y = slider["rect"].centery
        handle_radius = 10 if is_hovered else 8
        
        # Add subtle animation to handle when hovered
        if is_hovered:
            # Subtle pulsing effect
            pulse = math.sin(math.radians(self.animation_counter * 4)) * 2
            handle_radius += pulse
        
        # Draw handle shadow for depth
        shadow_offset = 2
        pygame.draw.circle(
            screen,
            self.COLORS['PANEL_SHADOW'],
            (handle_x + shadow_offset, handle_y + shadow_offset),
            handle_radius
        )
        
        # Draw handle with border
        pygame.draw.circle(
            screen,
            self.COLORS['SLIDER_HANDLE'],
            (handle_x, handle_y),
            handle_radius
        )
        
        pygame.draw.circle(
            screen,
            self.COLORS['SLIDER_HANDLE_BORDER'],
            (handle_x, handle_y),
            handle_radius,
            2
        )
        
        # Draw subtle tick marks for discrete sliders (like target score)
        if slider["step"] >= 1.0:
            tick_count = int(slider["max"] - slider["min"]) + 1
            for i in range(tick_count):
                tick_x = slider["rect"].x + (i * slider["rect"].width / (tick_count - 1))
                tick_top = slider["rect"].y + slider["rect"].height + 4
                tick_bottom = tick_top + 4
                
                # Make current tick more visible
                tick_value = slider["min"] + i
                alpha = 180 if tick_value == slider["value"] else 80
                tick_color = (*label_color[:3], alpha)
                
                pygame.draw.line(
                    screen,
                    tick_color,
                    (tick_x, tick_top),
                    (tick_x, tick_bottom),
                    1
                )
    
    def _draw_checkbox(self, screen: pygame.Surface, checkbox: Dict[str, Any]):
        """Draw a checkbox with enhanced visuals."""
        # Determine if mouse is hovering over checkbox
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = checkbox["rect"].collidepoint(mouse_pos) or pygame.Rect(
            checkbox["rect"].x,
            checkbox["rect"].y,
            checkbox["rect"].width + 200,  # Extend hit area to include label
            checkbox["rect"].height
        ).collidepoint(mouse_pos)
        
        # Draw checkbox background with shadow for depth
        shadow_rect = checkbox["rect"].copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        pygame.draw.rect(screen, self.COLORS['PANEL_SHADOW'], shadow_rect, border_radius=4)
        
        # Draw checkbox box with proper state colors
        bg_color = self.COLORS['INPUT_BG_FOCUS'] if is_hovered else self.COLORS['CHECKBOX_BG']
        border_color = self.COLORS['INPUT_BORDER_FOCUS'] if is_hovered else self.COLORS['CHECKBOX_BORDER']
        
        pygame.draw.rect(screen, bg_color, checkbox["rect"], border_radius=4)
        pygame.draw.rect(screen, border_color, checkbox["rect"], width=2, border_radius=4)
        
        # Draw checkmark with animation if checked
        if checkbox["value"]:
            # Calculate checkmark parameters with subtle animation
            check_margin = 6
            animation_offset = math.sin(math.radians(self.animation_counter * 3)) * 0.5
            
            # Draw check mark with better proportions
            check_points = [
                (checkbox["rect"].x + check_margin, checkbox["rect"].centery + animation_offset),
                (checkbox["rect"].centerx - 2, checkbox["rect"].y + checkbox["rect"].height - check_margin + animation_offset),
                (checkbox["rect"].x + checkbox["rect"].width - check_margin, checkbox["rect"].y + check_margin + animation_offset)
            ]
            
            # Draw shadow for depth
            shadow_points = [(p[0] + 1, p[1] + 1) for p in check_points]
            pygame.draw.lines(screen, (*self.COLORS['CHECKBOX_CHECK'][:3], 60), False, shadow_points, 3)
            
            # Draw main checkmark
            pygame.draw.lines(screen, self.COLORS['CHECKBOX_CHECK'], False, check_points, 3)
        
        # Draw label with hover effect
        label_y_offset = 1 if is_hovered else 0
        label_color = self.COLORS['TEXT_ACCENT'] if is_hovered else self.COLORS['TEXT_DARK']
        label = self.FONTS['NORMAL'].render(checkbox["label"], True, label_color)
        screen.blit(label, (checkbox["rect"].x + checkbox["rect"].width + 12, checkbox["rect"].centery - 10 + label_y_offset))
    
    def _draw_button(self, screen: pygame.Surface, button: Dict[str, Any]):
        """Draw a button with enhanced visuals and animations."""
        # Get button position, check hover state
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = button["rect"].collidepoint(mouse_pos)
        
        # Determine button colors based on type and state
        if button["type"] == "primary":
            bg_color = self.COLORS['BUTTON_SECONDARY_HOVER'] if is_hovered else self.COLORS['BUTTON_SECONDARY']
        else:
            bg_color = self.COLORS['BUTTON_PRIMARY_HOVER'] if is_hovered else self.COLORS['BUTTON_PRIMARY']
        
        # Draw shadow under button for depth
        shadow_rect = button["rect"].copy()
        shadow_rect.x += 3
        shadow_rect.y += 3
        
        # Make shadow responsive to hover state with subtle animation
        if is_hovered:
            # Animate shadow on hover
            shadow_offset = 2 + math.sin(math.radians(self.animation_counter * 4)) * 1.0
            shadow_rect.x = button["rect"].x + shadow_offset
            shadow_rect.y = button["rect"].y + shadow_offset
            
        pygame.draw.rect(screen, (*bg_color[:3], 80), shadow_rect, border_radius=8)
        
        # Draw button with subtle hover translation effect
        button_rect = button["rect"].copy()
        if is_hovered:
            # Move button slightly up when hovered
            button_rect.y -= 1
            
        # Draw main button
        pygame.draw.rect(screen, bg_color, button_rect, border_radius=8)
        
        # Add subtle inner highlight at top for 3D effect
        highlight_rect = button_rect.copy()
        highlight_rect.height = 5
        highlight_gradient = pygame.Surface((button_rect.width, 5), pygame.SRCALPHA)
        for i in range(5):
            alpha = 60 - (i * 12)
            pygame.draw.line(
                highlight_gradient,
                (255, 255, 255, alpha),
                (i + 1, i),
                (button_rect.width - i - 1, i),
                1
            )
        screen.blit(highlight_gradient, highlight_rect)
        
        # Draw button text with subtle animation on hover
        text_offset_y = -1 if is_hovered else 0
        text = self.FONTS['NORMAL'].render(button["label"], True, self.COLORS['BUTTON_TEXT'])
        text_rect = text.get_rect(center=(button_rect.centerx, button_rect.centery + text_offset_y))
        
        # Add slight shadow to text for readability
        text_shadow = self.FONTS['NORMAL'].render(button["label"], True, (0, 0, 0, 40))
        shadow_rect = text_shadow.get_rect(center=(text_rect.centerx + 1, text_rect.centery + 1))
        screen.blit(text_shadow, shadow_rect)
        
        # Draw main text
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