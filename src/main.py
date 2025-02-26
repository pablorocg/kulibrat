#!/usr/bin/env python3
"""
Main entry point for the Kulibrat game.

This script initializes and runs the Kulibrat game with different modes and configurations.

Run this file from the project root:
python main.py

Command line options:
--mode: Game mode (human-vs-human, human-vs-ai, ai-vs-ai)
--target-score: Score needed to win the game (default: 5)
--ai-algorithm: AI algorithm to use (random, minimax)
--delay: Delay between AI moves in seconds (default: 0.5)
"""

import sys
import os
import argparse
import time
from typing import Dict, Any

# Add the project root to the Python path to make imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import core components - using direct imports
from src.core.player_color import PlayerColor
from src.core.game_state import GameState
from src.core.game_engine import GameEngine

# Import player implementations
from src.players.player import Player
from src.players.human_player import HumanPlayer
from src.players.ai.ai_player import AIPlayer
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.ai.ai_strategy import AIStrategy
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy

# Import UI components
from src.ui.game_interface import GameInterface
from src.ui.console_interface import ConsoleInterface

"""
Enhanced PyGame GUI integration with the Kulibrat game.

To use this GUI:
1. Make sure pygame is installed: pip install pygame
2. Run the game with --interface pygame flag:
   python main.py --interface pygame
"""

# Import the completed KulibratGUI class
from src.ui.pygame_interface import KulibratGUI

# Update main.py to support the pygame interface
def setup_game_options() -> Dict[str, Any]:
    """
    Parse command line arguments and setup game options.
    
    Returns:
        Dictionary of game options
    """
    parser = argparse.ArgumentParser(description="Kulibrat Game")
    
    # Add interface selection option
    parser.add_argument(
        "--interface", 
        choices=["console", "pygame"],
        default="console",
        help="Choose game interface (console or pygame)"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["human-vs-human", "human-vs-ai", "ai-vs-ai"], 
        default="human-vs-human",
        help="Game mode (human-vs-human, human-vs-ai, ai-vs-ai)"
    )
    
    parser.add_argument(
        "--ai-algorithm",
        choices=["random", "minimax"],
        default="random",
        help="AI algorithm to use (random, minimax)"
    )
    
    parser.add_argument(
        "--target-score", 
        type=int, 
        default=5,
        help="Score needed to win the game"
    )
    
    parser.add_argument(
        "--ai-delay", 
        type=float, 
        default=0.5,
        help="Delay between AI moves in seconds"
    )
    
    parser.add_argument(
        "--ai-color",
        choices=["black", "red"],
        default="red",
        help="Color for the AI player in human-vs-ai mode"
    )
    
    return vars(parser.parse_args())

def main():
    """Main entry point for the game."""
    # Setup game options
    options = setup_game_options()
    
    # Create interface based on user selection
    if options["interface"] == "pygame":
        try:
            interface = KulibratGUI()
            print("Using PyGame interface")
        except ImportError:
            print("PyGame not installed. Falling back to console interface.")
            interface = ConsoleInterface()
    else:
        interface = ConsoleInterface()
    
    # Set up AI strategy based on selected algorithm
    if options["ai_algorithm"] == "minimax":
        ai_strategy = MinimaxStrategy(max_depth=3, use_alpha_beta=True)
    else:  # random
        ai_strategy = RandomStrategy()
    
    # Create players based on selected mode
    if options["mode"] == "human-vs-human":
        black_player = HumanPlayer(PlayerColor.BLACK, interface)
        red_player = HumanPlayer(PlayerColor.RED, interface)
    elif options["mode"] == "human-vs-ai":
        if options["ai_color"] == "black":
            black_player = SimpleAIPlayer(PlayerColor.BLACK, ai_strategy)
            red_player = HumanPlayer(PlayerColor.RED, interface)
        else:  # AI is red
            black_player = HumanPlayer(PlayerColor.BLACK, interface)
            red_player = SimpleAIPlayer(PlayerColor.RED, ai_strategy)
    else:  # ai-vs-ai
        black_player = SimpleAIPlayer(PlayerColor.BLACK, ai_strategy)
        red_player = SimpleAIPlayer(PlayerColor.RED, ai_strategy if options["ai_algorithm"] != "random" else RandomStrategy())
    
    # Create game engine
    engine = GameEngine(
        black_player=black_player,
        red_player=red_player,
        interface=interface,
        target_score=options["target_score"],
        ai_delay=options["ai_delay"]
    )
    
    # Welcome message (for console interface)
    if options["interface"] == "console":
        print("\n=== WELCOME TO KULIBRAT ===")
        print(f"Game Mode: {options['mode']}")
        print(f"Target Score: {options['target_score']}")
        
        if "ai" in options["mode"]:
            print(f"AI Algorithm: {options['ai_algorithm']}")
            print(f"AI Delay: {options['ai_delay']} seconds")
        
        print("\nStarting game...")
        time.sleep(1)
    
    # Start the game
    play_again = True
    
    while play_again:
        winner = engine.start_game()
        
        # For console interface, ask if the player wants to play again
        if options["interface"] == "console":
            play_again = input("\nDo you want to play again? (y/n): ").lower().startswith('y')
        else:
            # For PyGame interface, the play_again logic is handled in the show_winner method
            play_again = False
            
        if play_again:
            # Reset the game
            engine.reset_game()
    
    if options["interface"] == "console":
        print("\nThanks for playing Kulibrat!")

if __name__ == "__main__":
    main()