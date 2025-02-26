#!/usr/bin/env python3
"""
Main entry point for the Kulibrat game.

This script allows playing Kulibrat with different game modes:
- Human vs Human: Two players at the same computer
- Human vs AI: Play against an AI opponent
- AI vs AI: Watch two AI players compete

Run this file from the project root:
python main.py

Command line options:
--mode: Game mode (human-vs-human, human-vs-ai, ai-vs-ai)
--target-score: Score needed to win the game (default: 5)
--ai-algorithm: AI algorithm to use (random, minimax)
--ai-delay: Delay between AI moves in seconds (default: 0.5)
"""

import argparse
import time
from typing import Dict, Any

# Import core components
from src.core.player_color import PlayerColor
from src.core.game_state import GameState
from src.core.game_engine import GameEngine

# Import player implementations
from src.players.human_player import HumanPlayer
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy

# Import UI components
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
    
    # Create interface
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
        red_player = SimpleAIPlayer(PlayerColor.RED, RandomStrategy())
    
    # Create game engine
    engine = GameEngine(
        black_player=black_player,
        red_player=red_player,
        interface=interface,
        target_score=options["target_score"],
        ai_delay=options["ai_delay"]
    )
    
    # Welcome message
    print("\n=== WELCOME TO KULIBRAT ===")
    print(f"Game Mode: {options['mode']}")
    print(f"Target Score: {options['target_score']}")
    
    if "ai" in options["mode"]:
        print(f"AI Algorithm: {options['ai_algorithm']}")
        print(f"AI Delay: {options['ai_delay']} seconds")
    
    print("\nStarting game...")
    time.sleep(1)
    
    # Start the game
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


if __name__ == "__main__":
    main()