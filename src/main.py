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


def setup_game_options() -> Dict[str, Any]:
    """
    Parse command line arguments and setup game options.
    
    Returns:
        Dictionary of game options
    """
    parser = argparse.ArgumentParser(description="Kulibrat Game")
    
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
        "--delay", 
        type=float, 
        default=0.5,
        help="Delay between AI moves in seconds"
    )
    
    return vars(parser.parse_args())


def create_players(mode: str, interface, ai_algorithm: str = "random"):
    """
    Create players based on the selected mode.
    
    Args:
        mode: Game mode (human-vs-human, human-vs-ai, ai-vs-ai)
        interface: UI interface for human players
        ai_algorithm: AI algorithm to use ("random" or "minimax")
        
    Returns:
        Dictionary mapping player colors to player objects
    """
    # Select AI strategy based on algorithm choice
    if ai_algorithm == "minimax":
        ai_strategy = MinimaxStrategy(max_depth=3, use_alpha_beta=True)
    else:  # random
        ai_strategy = RandomStrategy()
    
    if mode == "human-vs-human":
        black_player = HumanPlayer(PlayerColor.BLACK, interface)
        red_player = HumanPlayer(PlayerColor.RED, interface)
    elif mode == "human-vs-ai":
        black_player = HumanPlayer(PlayerColor.BLACK, interface)
        red_player = SimpleAIPlayer(PlayerColor.RED, ai_strategy)
    else:  # ai-vs-ai
        black_player = SimpleAIPlayer(PlayerColor.BLACK, ai_strategy)
        red_player = SimpleAIPlayer(PlayerColor.RED, RandomStrategy())  # Second AI uses random strategy for variety
    
    return {
        PlayerColor.BLACK: black_player,
        PlayerColor.RED: red_player
    }


def main():
    """Main entry point for the game."""
    try:
        # Setup game options
        options = setup_game_options()
        
        # Create interface
        interface = ConsoleInterface()
        
        # Create players
        players = create_players(options["mode"], interface, options["ai_algorithm"])
        
        # Create game engine
        engine = GameEngine(
            players[PlayerColor.BLACK], 
            players[PlayerColor.RED], 
            interface, 
            options["target_score"]
        )
        
        # Set delay for AI players
        if options["mode"] in ["human-vs-ai", "ai-vs-ai"]:
            print(f"AI will pause for {options['delay']} seconds between moves.")
            time.sleep(1)
        
        # Start the game
        print("\n=== Welcome to Kulibrat ===")
        print("Starting game...")
        time.sleep(1)
        
        # Run the game
        winner = engine.start_game()
        
        # Ask if the player wants to play again
        play_again = input("\nDo you want to play again? (y/n): ").lower().startswith('y')
        
        while play_again:
            # Reset the game
            engine.reset_game()
            
            # Start a new game
            winner = engine.start_game()
            
            # Ask if the player wants to play again
            play_again = input("\nDo you want to play again? (y/n): ").lower().startswith('y')
        
        print("\nThanks for playing Kulibrat!")
    
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("\nCheck that your project structure matches the expected imports.")
        print("You may need to create any missing modules or fix import statements.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()