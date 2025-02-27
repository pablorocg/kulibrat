#!/usr/bin/env python3
"""
Main entry point for the Kulibrat game.

This script initializes and runs the Kulibrat game with different modes and configurations.

Run this file from the project root:
python main.py

Command line options:
--mode: Game mode (human-vs-human, human-vs-ai, ai-vs-ai, human-vs-rl, rl-vs-ai)
--target-score: Score needed to win the game (default: 5)
--ai-algorithm: AI algorithm to use (random, minimax)
--delay: Delay between AI moves in seconds (default: 0.5)
--rl-model: Path to the trained RL model file
"""

import argparse
import os
import sys
import time
from typing import Any, Dict

from src.core.game_engine import GameEngine
from src.core.player_color import PlayerColor
from src.players.ai.mcts_strategy import MCTSStrategy
from src.players.ai.minimax_strategy import MinimaxStrategy
from src.players.ai.random_strategy import RandomStrategy
from src.players.ai.rl_player import RLPlayer  # Import the RL player
from src.players.ai.simple_ai_player import SimpleAIPlayer
from src.players.human_player import HumanPlayer
from src.ui.console_interface import ConsoleInterface
from src.ui.pygame_interface import KulibratGUI

# # Add the project root to the Python path to make imports work correctly
# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, project_root)





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
        help="Choose game interface (console or pygame)",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "human-vs-human",
            "human-vs-ai",
            "ai-vs-ai",
            "human-vs-rl",
            "rl-vs-ai",
        ],
        default="human-vs-ai",
        help="Game mode (human-vs-human, human-vs-ai, ai-vs-ai, human-vs-rl, rl-vs-ai)",
    )

    parser.add_argument(
        "--ai-algorithm",
        choices=["random", "minimax", "mcts"],
        default="random",
        help="AI algorithm to use (random, minimax, mcts)",
    )

    # Add MCTS parameters to the argument parser
    parser.add_argument(
        "--mcts-time",
        type=float,
        default=1.0,
        help="Time in seconds for MCTS to run simulations (default: 1.0)",
    )

    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=1000,
        help="Maximum iterations for MCTS (default: 1000)",
    )

    parser.add_argument(
        "--target-score", type=int, default=5, help="Score needed to win the game"
    )

    parser.add_argument(
        "--ai-delay", type=float, default=0.5, help="Delay between AI moves in seconds"
    )

    parser.add_argument(
        "--ai-color",
        choices=["black", "red"],
        default="red",
        help="Color for the AI player in human-vs-ai mode",
    )

    parser.add_argument(
        "--rl-model",
        type=str,
        default="models/kulibrat_rl_model_best.pt",
        help="Path to the trained RL model file",
    )

    parser.add_argument(
        "--rl-color",
        choices=["black", "red"],
        default="red",
        help="Color for the RL player in human-vs-rl mode",
    )

    parser.add_argument(
        "--rl-exploration",
        type=float,
        default=0.0,
        help="Exploration rate for RL player (0.0 for deterministic play)",
    )

    parser.add_argument(
        "--rl-temperature",
        type=float,
        default=0.1,
        help="Temperature for RL policy sampling (lower for stronger play)",
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
    # In the main function, modify the AI strategy selection:
    # Set up AI strategy based on selected algorithm
    if options["ai_algorithm"] == "minimax":
        ai_strategy = MinimaxStrategy(max_depth=6, use_alpha_beta=True)
    elif options["ai_algorithm"] == "mcts":  # Add this condition
        ai_strategy = MCTSStrategy(
            simulation_time=options["mcts_time"],
            max_iterations=options["mcts_iterations"],
        )
    else:  # random
        ai_strategy = RandomStrategy()

    # Validate RL model path if using RL modes
    if "rl" in options["mode"]:
        if not os.path.exists(options["rl_model"]):
            print(f"Error: RL model file '{options['rl_model']}' not found.")
            print(
                "You can train a model using train_rl_agent.py or specify a different model with --rl-model."
            )
            return

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

    elif options["mode"] == "ai-vs-ai":
        black_player = SimpleAIPlayer(PlayerColor.BLACK, ai_strategy)
        red_player = SimpleAIPlayer(
            PlayerColor.RED,
            ai_strategy if options["ai_algorithm"] != "random" else RandomStrategy(),
        )

    elif options["mode"] == "human-vs-rl":
        # RL player with the trained model
        if options["rl_color"] == "black":
            black_player = RLPlayer(
                PlayerColor.BLACK,
                model_path=options["rl_model"],
                exploration_rate=options["rl_exploration"],
                temperature=options["rl_temperature"],
                name="RL BLACK",
            )
            red_player = HumanPlayer(PlayerColor.RED, interface)
        else:  # RL is red
            black_player = HumanPlayer(PlayerColor.BLACK, interface)
            red_player = RLPlayer(
                PlayerColor.RED,
                model_path=options["rl_model"],
                exploration_rate=options["rl_exploration"],
                temperature=options["rl_temperature"],
                name="RL RED",
            )

    elif options["mode"] == "rl-vs-ai":
        # RL player against AI player
        black_player = RLPlayer(
            PlayerColor.BLACK,
            model_path=options["rl_model"],
            exploration_rate=options["rl_exploration"],
            temperature=options["rl_temperature"],
            name="RL BLACK",
        )
        red_player = SimpleAIPlayer(
            PlayerColor.RED, ai_strategy, name=f"AI RED ({options['ai_algorithm']})"
        )

    # Create game engine
    engine = GameEngine(
        black_player=black_player,
        red_player=red_player,
        interface=interface,
        target_score=options["target_score"],
        ai_delay=options["ai_delay"],
    )

    # Welcome message (for console interface)
    if options["interface"] == "console":
        print("\n=== WELCOME TO KULIBRAT ===")
        print(f"Game Mode: {options['mode']}")
        print(f"Target Score: {options['target_score']}")

        if "ai" in options["mode"]:
            print(f"AI Algorithm: {options['ai_algorithm']}")
            print(f"AI Delay: {options['ai_delay']} seconds")

        if "rl" in options["mode"]:
            print(f"RL Model: {options['rl_model']}")
            print(f"RL Exploration Rate: {options['rl_exploration']}")
            print(f"RL Temperature: {options['rl_temperature']}")

        print("\nStarting game...")
        time.sleep(1)

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
