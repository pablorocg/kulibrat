#!/usr/bin/env python3
"""
Main entry point for the Kulibrat game.

This script initializes and runs the Kulibrat game with different modes and configurations.

Run this file from the project root:
python main.py

Command line options:
--interface: Choose game interface (console or pygame) (default: console)
--player-1-type: Player 1 type (human, minimax, random, mcts, rl, alphazero) (default: human)
--player-2-type: Player 2 type (human, minimax, random, mcts, rl, alphazero) (default: random)
--player-1-color: Color for player 1 (black or red) (default: black)
--target-score: Score needed to win the game (default: 5)
--rl-model-1: Path to the trained RL model file for player 1 (default: models/rl_model.pth)
--rl-model-2: Path to the trained RL model file for player 2 (default: models/rl_model.pth)
--az-model-1: Path to the trained AlphaZero model file for player 1 (default: models/alphazero_model_best.pt)
--az-model-2: Path to the trained AlphaZero model file for player 2 (default: models/alphazero_model_best.pt)
--az-simulations: Number of MCTS simulations for AlphaZero (default: 800)
"""

import argparse
import os
import time
from typing import Any, Dict

from src.core.game_engine import GameEngine
from src.core.player_color import PlayerColor
from src.players.ai.alphazero_player import AlphaZeroPlayer
from src.players.ai.mcts_strategy import MCTSStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.random_strategy import RandomStrategy
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
        choices=["human", "minimax", "random", "mcts", "alphazero"],
        default="human",
        help="Player 1 type (human, minimax, random, mcts, alphazero)",
    )

    parser.add_argument(
        "--player-2-type",
        choices=["human", "minimax", "random", "mcts", "alphazero"],
        default="minimax",
        help="Player 2 type (human, minimax, random, mcts, alphazero)",
    )

    parser.add_argument(
        "--player-1-color",
        choices=["black", "red"],
        default="black",
        help="Color for player 1",
    )

    parser.add_argument(
        "--target-score", type=int, default=5, help="Score needed to win the game"
    )

    parser.add_argument(
        "--az-model-1",
        default="models/alphazero_model_best.pt",
        help="Path to the trained AlphaZero model file if using AlphaZero as player 1",
    )

    parser.add_argument(
        "--az-model-2",
        default="models/alphazero_model_best.pt",
        help="Path to the trained AlphaZero model file if using AlphaZero as player 2",
    )

    parser.add_argument(
        "--az-simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations for AlphaZero players",
    )

    return vars(parser.parse_args())


def create_player(
    player_type,
    player_color,
    interface,
    az_model_path=None,
    az_simulations=800,
    player_name=None,
):
    """
    Create a player based on the specified type and color.

    Args:
        player_type: Type of player (human, minimax, random, mcts, rl, alphazero)
        player_color: Player's color (PlayerColor.BLACK or PlayerColor.RED)
        interface: Game interface
        rl_model_path: Path to the RL model file (only needed for RL players)
        az_model_path: Path to the AlphaZero model file (only needed for AlphaZero players)
        az_simulations: Number of MCTS simulations for AlphaZero players
        player_name: Name of the player (optional)

    Returns:
        Player object
    """
    if player_type == "human":
        return HumanPlayer(color=player_color, interface=interface, name="Human")
    elif player_type == "minimax":
        return SimpleAIPlayer(
            color=player_color,
            strategy=MinimaxStrategy(max_depth=6, use_alpha_beta=True),
            name="Minimax",
        )
    elif player_type == "mcts":
        return SimpleAIPlayer(
            color=player_color,
            strategy=MCTSStrategy(simulation_time=1.5, max_iterations=30000),
            name="MCTS",
        )
    elif player_type == "random":
        return SimpleAIPlayer(
            color=player_color, strategy=RandomStrategy(), name="Random"
        )

    elif player_type == "alphazero":
        return AlphaZeroPlayer(
            color=player_color,
            model_path=az_model_path,
            n_simulations=az_simulations,
            exploration_rate=0,
            temperature=0.1,
            name=player_name or "AlphaZero",
        )
    else:
        raise ValueError(f"Invalid player type: {player_type}")


def check_alphazero_model(model_path, player_type, player_num):
    """
    Check if the AlphaZero model file exists if AlphaZero player type is selected.

    Args:
        model_path: Path to the AlphaZero model file
        player_type: Type of player
        player_num: Player number (1 or 2)

    Returns:
        True if model exists or if player type is not AlphaZero, False otherwise
    """
    if player_type == "alphazero":
        if not os.path.exists(model_path):
            print(f"Error: AlphaZero model file '{model_path}' not found.")
            print(
                f"You can train a model using train_alphazero.py or specify a different model with --az-model-{player_num}."
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
    player_1_type = options[
        "player_1_type"
    ]  # Use underscore instead of hyphen (argparse converts hyphens to underscores)
    player_2_type = options["player_2_type"]

    az_model_1 = options["az_model_1"]
    az_model_2 = options["az_model_2"]
    az_simulations = options["az_simulations"]
    target_score = options["target_score"]

    # Get player colors
    player_1_color = (
        PlayerColor.BLACK if options["player_1_color"] == "black" else PlayerColor.RED
    )
    player_2_color = (
        PlayerColor.RED if player_1_color == PlayerColor.BLACK else PlayerColor.BLACK
    )

    # Check if AlphaZero model files exist and are valid if using AlphaZero
    if not check_alphazero_model(
        az_model_1, player_1_type, 1
    ) or not check_alphazero_model(az_model_2, player_2_type, 2):
        return  # Exit the program if models are not available

    # Create players
    player_1 = create_player(
        player_1_type, player_1_color, interface, az_model_1, az_simulations, "Player 1"
    )

    player_2 = create_player(
        player_2_type, player_2_color, interface, az_model_2, az_simulations, "Player 2"
    )

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

        if player_1_type == "alphazero":
            print(f"AlphaZero Model: {az_model_1}")
            print(f"MCTS Simulations: {az_simulations}")

        # Show player 2 color and name
        print(f"Player 2 ({player_2_type}): {player_2_color.name}")

        if player_2_type == "alphazero":
            print(f"AlphaZero Model: {az_model_2}")
            print(f"MCTS Simulations: {az_simulations}")

        print("\nStarting game...")
        print("")
        time.sleep(2)

    # Start the game
    play_again = True

    while play_again:
        winner = engine.start_game()

        # For console interface, ask if the player wants to play again
        if options["interface"] == "console":
            play_again = (
                input("\nDo you want to play again? (y/n): ").lower().startswith("y")
            )
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
