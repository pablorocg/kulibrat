#!/usr/bin/env python3
"""
Utility script to generate piece images for Kulibrat
Creates red_piece.png and black_piece.png in the gui/assets directory
"""
import pygame
import os
import sys

def generate_pieces():
    """Generate red and black piece images"""
    # Create assets directory if it doesn't exist
    os.makedirs("gui/assets", exist_ok=True)

    # Initialize pygame
    pygame.init()

    # Define image size and colors
    SIZE = 100
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    
    # Create surfaces with alpha channel (transparency)
    red_surface = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    black_surface = pygame.Surface((SIZE, SIZE), pygame.SRCALPHA)
    
    # Draw red piece with shine effect
    pygame.draw.circle(red_surface, RED, (SIZE//2, SIZE//2), SIZE//2 - 5)
    # Add highlight for 3D effect
    pygame.draw.circle(red_surface, (255, 150, 150), (SIZE//3, SIZE//3), SIZE//6)
    # Add outline
    pygame.draw.circle(red_surface, (150, 0, 0), (SIZE//2, SIZE//2), SIZE//2 - 5, 3)
    
    # Draw black piece with shine effect
    pygame.draw.circle(black_surface, BLACK, (SIZE//2, SIZE//2), SIZE//2 - 5)
    # Add highlight for 3D effect
    pygame.draw.circle(black_surface, (100, 100, 100), (SIZE//3, SIZE//3), SIZE//6)
    # Add outline
    pygame.draw.circle(black_surface, (50, 50, 50), (SIZE//2, SIZE//2), SIZE//2 - 5, 3)
    
    # Save images
    try:
        pygame.image.save(red_surface, "gui/assets/red_piece.png")
        pygame.image.save(black_surface, "gui/assets/black_piece.png")
        print("Piece images created successfully in gui/assets/ directory!")
    except Exception as e:
        print(f"Error saving images: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if generate_pieces():
        sys.exit(0)
    else:
        sys.exit(1)