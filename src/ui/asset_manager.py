"""
Asset management for the Kulibrat game GUI.
Handles loading and caching of images, sounds, and fonts.
"""

import os
import pygame
from typing import Dict, Any, Optional


class AssetManager:
    """
    Asset manager for the Kulibrat game.
    Handles loading and caching of images, sounds, and fonts.
    """
    
    def __init__(self):
        """Initialize the asset manager."""
        self.assets = {
            'images': {},
            'sounds': {},
            'fonts': {}
        }
        
        # Define asset directory paths
        self.asset_dir = os.path.join("src", "ui", "assets")
        self.image_dir = os.path.join(self.asset_dir, "images")
        self.sound_dir = os.path.join(self.asset_dir, "sounds")
        self.font_dir = os.path.join(self.asset_dir, "fonts")
        
        # Create asset directories if they don't exist
        self._create_asset_directories()
    
    def _create_asset_directories(self):
        """Create the asset directories if they don't exist."""
        os.makedirs(self.asset_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.sound_dir, exist_ok=True)
        os.makedirs(self.font_dir, exist_ok=True)
    
    def load_image(self, name: str, scale_to: Optional[tuple] = None) -> Optional[pygame.Surface]:
        """
        Load an image from the assets directory.
        
        Args:
            name: Name of the image file (with extension)
            scale_to: Optional tuple (width, height) to scale the image to
            
        Returns:
            The loaded image, or None if the image couldn't be loaded
        """
        # Check if the image is already loaded
        if name in self.assets['images']:
            # If scaling is requested but different from cached version, reload
            if scale_to and self.assets['images'][name].get_size() != scale_to:
                pass  # Continue to loading
            else:
                return self.assets['images'][name]
        
        # Try to load the image
        try:
            path = os.path.join(self.image_dir, name)
            if not os.path.exists(path):
                print(f"Image not found: {path}")
                return None
            
            image = pygame.image.load(path).convert_alpha()
            
            # Scale the image if requested
            if scale_to:
                image = pygame.transform.scale(image, scale_to)
            
            # Cache the loaded image
            self.assets['images'][name] = image
            
            return image
        except Exception as e:
            print(f"Error loading image '{name}': {e}")
            return None
    
    def load_sound(self, name: str) -> Optional[pygame.mixer.Sound]:
        """
        Load a sound from the assets directory.
        
        Args:
            name: Name of the sound file (with extension)
            
        Returns:
            The loaded sound, or None if the sound couldn't be loaded
        """
        # Initialize pygame mixer if not already done
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except pygame.error:
                print("Warning: Pygame mixer initialization failed")
                return None
        
        # Check if sound is already loaded
        if name in self.assets['sounds']:
            return self.assets['sounds'][name]
        
        # Try to load the sound
        try:
            path = os.path.join(self.sound_dir, name)
            if not os.path.exists(path):
                print(f"Sound not found: {path}")
                return None
            
            sound = pygame.mixer.Sound(path)
            
            # Cache the loaded sound
            self.assets['sounds'][name] = sound
            
            return sound
        except Exception as e:
            print(f"Error loading sound '{name}': {e}")
            return None
    
    def load_font(self, name: str, size: int) -> pygame.font.Font:
        """
        Load a font from the assets directory or fall back to system font.
        
        Args:
            name: Name of the font file (with extension) or system font name
            size: Font size in points
            
        Returns:
            The loaded font
        """
        # Create a unique key for the font
        font_key = f"{name}_{size}"
        
        # Check if font is already loaded
        if font_key in self.assets['fonts']:
            return self.assets['fonts'][font_key]
        
        # Try to load a custom font
        try:
            path = os.path.join(self.font_dir, name)
            if os.path.exists(path):
                font = pygame.font.Font(path, size)
            else:
                # Fall back to system font
                font = pygame.font.SysFont(name, size)
            
            # Cache the loaded font
            self.assets['fonts'][font_key] = font
            
            return font
        except Exception as e:
            print(f"Error loading font '{name}' (size {size}): {e}")
            
            # Fall back to default font
            default_font = pygame.font.SysFont("Arial", size)
            self.assets['fonts'][font_key] = default_font
            return default_font