#!/usr/bin/env python3
"""
Main entry point for the Kulibrat game.

This script initializes and runs the Kulibrat game with different modes and configurations.

Run this file from the project root:
python main.py

Command line options:
--interface: Choose game interface (console or pygame) (default: console)
--player-1-type: Player 1 type (human, minimax, random, mcts, rl) (default: human)
--player-2-type: Player 2 type (human, minimax, random, mcts, rl) (default: random)
--player-1-color: Color for player 1 (black or red) (default: black)
--target-score: Score needed to win the game (default: 5)
--rl-model-1: Path to the trained RL model file for player 1 (default: models/rl_model.pth)
--rl-model-2: Path to the trained RL model file for player 2 (default: models/rl_model.pth)
"""

import argparse
import os
import time
from typing import Any, Dict

from src.core.game_engine import GameEngine
from src.core.player_color import PlayerColor
from src.players.ai.mcts_strategy import MCTSStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.rl_player import RLPlayer
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.human_player import HumanPlayer
from src.ui.console_interface import ConsoleInterface


def setup_game_options() -> Dict[str, Any]:
    """
    Parse command line arguments and setup game options.

    Returns:
        Dictionary of game options
    """
    parser = argparse.ArgumentParser(description="Kulibrat Game")

    parser.add_argument(
        "--interface",
        choices=["console", "pygame"],
        default="console",
        help="Choose game interface (console or pygame)",
    )

    parser.add_argument(
        "--player-1-type",
        choices=["human", "minimax", "random", "mcts", "rl"],
        default="human",
        help="Player 1 type (human, minimax, random, mcts, rl)",
    )

    parser.add_argument(
        "--player-2-type",
        choices=["human", "minimax", "random", "mcts", "rl"],
        default="minimax",
        help="Player 2 type (human, minimax, random, mcts, rl)",
    )

    parser.add_argument(
        "--player-1-color",
        choices=["black", "red"],
        default="black",
        help="Color for player 1",
    )

    parser.add_argument(
        "--target-score", 
        type=int, 
        default=5, 
        help="Score needed to win the game"
    )

    parser.add_argument(
        "--rl-model-1",
        default="models/rl_model.pth",
        help="Path to the trained RL model file if using RL as player 1",
    )

    parser.add_argument(
        "--rl-model-2",
        default="models/rl_model.pth",
        help="Path to the trained RL model file if using RL as player 2",
    )

    return vars(parser.parse_args())


def create_player(player_type, player_color, interface, rl_model_path=None, player_name=None):
    """
    Create a player based on the specified type and color.

    Args:
        player_type: Type of player (human, minimax, random, mcts, rl)
        player_color: Player's color (PlayerColor.BLACK or PlayerColor.RED)
        interface: Game interface
        rl_model_path: Path to the RL model file (only needed for RL players)
        player_name: Name of the player (optional)

    Returns:
        Player object
    """
    if player_type == "human":
        return HumanPlayer(color=player_color, interface=interface, name="Human")
    elif player_type == "minimax":
        return SimpleAIPlayer(
            color=player_color,
            strategy=MinimaxStrategy(max_depth=3, use_alpha_beta=True),
            name="Minimax",
        )
    elif player_type == "mcts":
        return SimpleAIPlayer(
            color=player_color,
            strategy=MCTSStrategy(simulation_time=3, max_iterations=10000),
            name="MCTS",
        )
    elif player_type == "random":
        return SimpleAIPlayer(
            color=player_color, strategy=RandomStrategy(), name="Random"
        )
    elif player_type == "rl":
        return RLPlayer(
            color=player_color,
            model_path=rl_model_path,
            exploration_rate=0,
            temperature=0.5,
            name=player_name or "RL Player",
        )
    else:
        raise ValueError(f"Invalid player type: {player_type}")


def check_rl_model(model_path, player_type, player_num):
    """
    Check if the RL model file exists if RL player type is selected.

    Args:
        model_path: Path to the RL model file
        player_type: Type of player
        player_num: Player number (1 or 2)

    Returns:
        True if model exists or if player type is not RL, False otherwise
    """
    if player_type == "rl":
        if not os.path.exists(model_path):
            print(f"Error: RL model file '{model_path}' not found.")
            print(
                f"You can train a model using train_rl_agent.py or specify a different model with --rl-model-{player_num}."
            )
            return False
    return True


def get_game_mode(player_1_type, player_2_type):
    """
    Determine the game mode based on player types.
    
    Args:
        player_1_type: Type of player 1
        player_2_type: Type of player 2
        
    Returns:
        String describing the game mode
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
    # Setup game options
    options = setup_game_options()

    # Create interface based on user selection
    interface = None
    if options["interface"] == "pygame":
        try:
            # Only import pygame when needed to avoid dependency if not used
            from src.ui.pygame_interface import KulibratGUI
            interface = KulibratGUI()
            print("Using PyGame interface")
        except ImportError:
            print("PyGame not installed. Falling back to console interface.")
            interface = ConsoleInterface()
    else:
        interface = ConsoleInterface()

    # Get player types and model paths
    player_1_type = options["player_1_type"]  # Use underscore instead of hyphen (argparse converts hyphens to underscores)
    player_2_type = options["player_2_type"]
    rl_model_1 = options["rl_model_1"]
    rl_model_2 = options["rl_model_2"]
    target_score = options["target_score"]

    # Get player colors
    player_1_color = PlayerColor.BLACK if options["player_1_color"] == "black" else PlayerColor.RED
    player_2_color = PlayerColor.RED if player_1_color == PlayerColor.BLACK else PlayerColor.BLACK

    # Check if RL model files exist and are valid if using RL
    if not check_rl_model(rl_model_1, player_1_type, 1) or not check_rl_model(rl_model_2, player_2_type, 2):
        return  # Exit the program if models are not available

    # Create players
    player_1 = create_player(player_1_type, player_1_color, interface, rl_model_1, "RL Player 1")
    player_2 = create_player(player_2_type, player_2_color, interface, rl_model_2, "RL Player 2")

    # Assign players to their correct roles (black or red)
    black_player = player_1 if player_1_color == PlayerColor.BLACK else player_2
    red_player = player_2 if player_2_color == PlayerColor.RED else player_1

    # Create game engine
    engine = GameEngine(
        black_player=black_player,
        red_player=red_player,
        interface=interface,
        target_score=target_score,
        ai_delay=1,
    )

    # Get game mode
    game_mode = get_game_mode(player_1_type, player_2_type)
    
    # Console interface welcome message
    if options["interface"] == "console":
        print("\n=== WELCOME TO KULIBRAT ===")
        print(f"Game Mode: {game_mode}")
        print(f"Target Score: {target_score}")

        # Show player 1 color and name
        print(f"Player 1 ({player_1_type}): {player_1_color.name}")
        if player_1_type == "rl":
            print(f"RL Model: {rl_model_1}")

        # Show player 2 color and name
        print(f"Player 2 ({player_2_type}): {player_2_color.name}")
        if player_2_type == "rl":
            print(f"RL Model: {rl_model_2}")

        print("\nStarting game...")
        print("")
        time.sleep(2)

    # Start the game
    play_again = True

    while play_again:
        winner = engine.start_game()

        # For console interface, ask if the player wants to play again
        if options["interface"] == "console":
            play_again = input("\nDo you want to play again? (y/n): ").lower().startswith("y")
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