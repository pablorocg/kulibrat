#!/usr/bin/env python3
"""
Main entry point for the Kulibrat game.

This script initializes and runs the Kulibrat game with configurations from YAML.

Run this file from the project root:
python main.py
"""

import logging
import time
import os
import sys
from typing import Optional

# Core game imports
from src.core.game_engine import GameEngine
from src.core.game_rules import GameRules
from src.core.turn_manager import TurnManager
from src.core.player_color import PlayerColor

# Configuration and factory imports
from src.config.game_config import GameConfig
from src.players.player_factory import PlayerFactory

# UI imports
from src.ui.console_interface import ConsoleInterface


def create_interface(interface_type: str, screen_width: int = 1024, screen_height: int = 800):
    """
    Create game interface based on type.

    Args:
        interface_type: Type of interface to create
        screen_width: Width of the screen for graphical interfaces
        screen_height: Height of the screen for graphical interfaces

    Returns:
        Game interface instance
    """
    # Default to console interface
    if interface_type.lower() == "pygame":
        try:
            # Lazy import to avoid dependency
            from src.ui.pygame_interface import KulibratGUI
            return KulibratGUI(screen_width, screen_height)
        except ImportError as e:
            logging.warning(f"PyGame not installed or error initializing GUI: {e}")
            logging.warning("Falling back to console interface.")
    
    return ConsoleInterface()


def validate_player_configuration(
    player_1_type: str, 
    player_2_type: str
) -> str:
    """
    Validate the player configuration.

    Args:
        player_1_type: Type of player 1
        player_2_type: Type of player 2

    Returns:
        Game mode description
    """
    if player_1_type == "human" and player_2_type == "human":
        return "human vs human"
    elif player_1_type == "human":
        return "human vs ai"
    elif player_2_type == "human":
        return "ai vs human"
    else:
        return "ai vs ai"


def main():
    """Main entry point for the game."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Create configuration
        config = GameConfig()

        # Retrieve game configuration
        interface_type = config.get('ui.interface', 'console')
        screen_width = config.get('ui.screen_width', 1024)
        screen_height = config.get('ui.screen_height', 800)
        player_1_type = config.get('players.player_1.type', 'human')
        player_2_type = config.get('players.player_2.type', 'minimax')
        player_1_color_name = config.get('players.player_1.color', 'black')
        target_score = config.get('game.target_score', 5)
        ai_delay = config.get('game.ai_delay', 1.0)  # Add AI delay setting

        # Validate color name and set player color
        player_1_color = PlayerColor.BLACK if player_1_color_name.lower() == 'black' else PlayerColor.RED
        player_2_color = PlayerColor.RED if player_1_color == PlayerColor.BLACK else PlayerColor.BLACK

        # Create interface
        interface = create_interface(interface_type, screen_width, screen_height)

        # Validate and get game mode
        game_mode = validate_player_configuration(player_1_type, player_2_type)

        # Create players using PlayerFactory
        try:
            player_1 = PlayerFactory.create_player(
                player_type=player_1_type,
                color=player_1_color,
                interface=interface
            )

            player_2 = PlayerFactory.create_player(
                player_type=player_2_type,
                color=player_2_color,
                interface=interface
            )
        except ValueError as e:
            logging.error(f"Error creating players: {e}")
            return 1
        except Exception as e:
            logging.error(f"Unexpected error creating players: {e}")
            return 1

        # Create game engine components
        rules_engine = GameRules()
        turn_manager = TurnManager(rules_engine)

        # Create game engine
        engine = GameEngine(
            rules_engine=rules_engine,
            turn_manager=turn_manager,
            interface=interface,
            target_score=target_score
        )

        # Console interface welcome message
        if interface_type.lower() == "console":
            print("\n=== WELCOME TO KULIBRAT ===")
            print(f"Game Mode: {game_mode}")
            print(f"Target Score: {target_score}")
            print(f"Player 1 ({player_1_type}): {player_1_color.name}")
            print(f"Player 2 ({player_2_type}): {player_2_color.name}")
            print("\nStarting game...")
            time.sleep(1)  # Shorter pause for better UX

        # Assign players to their correct roles (black or red)
        black_player = player_1 if player_1_color == PlayerColor.BLACK else player_2
        red_player = player_2 if player_2_color == PlayerColor.RED else player_1

        # Start the game
        play_again = True
        while play_again:
            # Start game with black and red players
            winner = engine.start_game(black_player, red_player)

            # For console interface, ask if the player wants to play again
            if interface_type.lower() == "console":
                play_again = (
                    input("\nDo you want to play again? (y/n): ").lower().startswith("y")
                )
            else:
                # For PyGame interface, play_again logic is handled in show_winner method
                play_again = False

            if play_again:
                # Reset the game
                engine.reset_game()
                # Reset players
                if hasattr(black_player, 'setup'):
                    black_player.setup(engine.state)
                if hasattr(red_player, 'setup'):
                    red_player.setup(engine.state)

        if interface_type.lower() == "console":
            print("\nThanks for playing Kulibrat!")
        
        return 0
    
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())