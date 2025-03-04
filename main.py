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

# import argparse
# import os
# import sys
# import time
# from typing import Any, Dict

# from src.core.game_engine import GameEngine
# from src.core.player_color import PlayerColor
# from src.players.ai.mcts_strategy import MCTSStrategy
# from src.players.ai.minimax_strategy import MinimaxStrategy
# from src.players.ai.random_strategy import RandomStrategy
# from src.players.ai.rl_player import RLPlayer  # Import the RL player
# from src.players.ai.simple_ai_player import SimpleAIPlayer
# from src.players.human_player import HumanPlayer
# from src.ui.console_interface import ConsoleInterface
# from src.ui.pygame_interface import KulibratGUI

# # # Add the project root to the Python path to make imports work correctly
# # project_root = os.path.dirname(os.path.abspath(__file__))
# # sys.path.insert(0, project_root)


# # Kulibrat Game

# # options:
# #   -h, --help            show this help message and exit

# #   --interface {console,pygame} -> Choose game interface (console or pygame)

# #   --player_1_type -> Player 1 and Player 2 types (human, minimax, random, mcts, rl)
# #   --player_2_type -> Player 1 and Player 2 types (human, minimax, random, mcts, rl)

# #   --player_1_color {black, red} -> Color for the player 1 in human-vs-ai mode


# #   --target-score TARGET_SCORE -> Score needed to win the game

# #   --rl-model-1 RL_MODEL -> Path to the trained RL model file if using RL as player 1 (any of the players)
# #   --rl-model-2 RL_MODEL -> Path to the trained RL model file if using RL as player 2 (any of the players)


# def setup_game_options() -> Dict[str, Any]:
#     """
#     Parse command line arguments and setup game options.

#     Returns:
#         Dictionary of game options
#     """
#     parser = argparse.ArgumentParser(description="Kulibrat Game")

#     parser.add_argument(
#         "--interface",
#         choices=["console", "pygame"],
#         default="console",
#         help="Choose game interface (console or pygame)",
#     )

#     parser.add_argument(
#         "--player-1-type",
#         choices=["human", "minimax", "random", "mcts", "rl"],
#         default="human",
#         help="Player 1 type (human, minimax, random, mcts, rl)",
#     )

#     parser.add_argument(
#         "--player-2-type",
#         choices=["human", "minimax", "random", "mcts", "rl"],
#         default="random",
#         help="Player 2 type (human, minimax, random, mcts, rl)",
#     )

#     parser.add_argument(
#         "--player-1-color",
#         choices=["black", "red"],
#         default="black",
#         help="Color for the player 1 in human-vs-ai mode",
#     )

#     parser.add_argument(
#         "--target-score", type=int, default=5, help="Score needed to win the game"
#     )

#     parser.add_argument(
#         "--rl-model-1",
#         default="models/rl_model.pth",
#         help="Path to the trained RL model file if using RL as player 1",
#     )

#     parser.add_argument(
#         "--rl-model-2",
#         default="models/rl_model.pth",
#         help="Path to the trained RL model file if using RL as player 2",
#     )

#     return vars(parser.parse_args())


# def main():
#     """Main entry point for the game."""
#     # Setup game options
#     options = setup_game_options()

#     # Create interface based on user selection
#     if options["interface"] == "pygame":
#         try:
#             interface = KulibratGUI()
#             print("Using PyGame interface")
#         except ImportError:
#             print("PyGame not installed. Falling back to console interface.")
#             interface = ConsoleInterface()
#     else:
#         interface = ConsoleInterface()

#     # Get player colors
#     player_1_color, player_2_color = (
#         (PlayerColor.BLACK, PlayerColor.RED)
#         if options["player_1_color"] == "black"
#         else (PlayerColor.RED, PlayerColor.BLACK)
#     )

#     # Check if RL model files exist and are valid if using RL
#     if options["player-1-type"] == "rl":
#         if not os.path.exists(options["rl_model_1"]):
#             print(f"Error: RL model file '{options['rl_model_1']}' not found.")
#             print(
#                 "You can train a model using train_rl_agent.py or specify a different model with --rl-model-1."
#             )
#             return  # Exit the program

#     if options["player-2-type"] == "rl":
#         if not os.path.exists(options["rl_model_2"]):
#             print(f"Error: RL model file '{options['rl_model_2']}' not found.")
#             print(
#                 "You can train a model using train_rl_agent.py or specify a different model with --rl-model-2."
#             )
#             return

#     # Create AI strategy based on selected algorithm
#     if options["player-1-type"] == "human":
#         player_1 = HumanPlayer(player_1_color, interface)

#     elif options["player-1-type"] == "minimax":
#         player_1 = SimpleAIPlayer(player_1_color, MinimaxStrategy())

#     elif options["player_1_type"] == "mcts":
#         player_1 = SimpleAIPlayer(player_1_color, MCTSStrategy())

#     elif options["player_1_type"] == "random":
#         player_1 = SimpleAIPlayer(player_1_color, RandomStrategy())

#     elif options["player_1_type"] == "rl":
#         player_1 = RLPlayer(
#             player_1_color,
#             model_path=options["rl_model_1"],
#             name="RL Player 1",
#         )
#     else:
#         raise ValueError(f"Invalid player 1 type: {options['player_1_type']}")


#     if options["player_2_type"] == "human":
#         player_2 = HumanPlayer(player_2_color, interface)

#     elif options["player_2_type"] == "minimax":
#         player_2 = SimpleAIPlayer(player_2_color, MinimaxStrategy())

#     elif options["player_2_type"] == "mcts":
#         player_2 = SimpleAIPlayer(player_2_color, MCTSStrategy())

#     elif options["player_2_type"] == "random":
#         player_2 = SimpleAIPlayer(player_2_color, RandomStrategy())

#     elif options["player_2_type"] == "rl":
#         player_2 = RLPlayer(
#             player_2_color,
#             model_path=options["rl_model_2"],
#             name="RL Player 2",
#         )
#     else:
#         raise ValueError(f"Invalid player 2 type: {options['player_2_type']}")

#     # Create game engine
#     engine = GameEngine(
#         black_player=black_player,
#         red_player=red_player,
#         interface=interface,
#         target_score=options["target_score"],
#         ai_delay=1,
#     )
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
        default="random",
        help="Player 2 type (human, minimax, random, mcts, rl)",
    )

    parser.add_argument(
        "--player-1-color",
        choices=["black", "red"],
        default="black",
        help="Color for the player 1 in human-vs-ai mode",
    )

    parser.add_argument(
        "--target-score", type=int, default=5, help="Score needed to win the game"
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


def create_player(
    player_type, player_color, interface, rl_model_path=None, player_name=None
):
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
            strategy=MinimaxStrategy(max_depth=5, use_alpha_beta=True),
            name="Minimax",
        )
    elif player_type == "mcts":
        return SimpleAIPlayer(
            color=player_color,
            strategy=MCTSStrategy(simulation_time=2, max_iterations=1e4),
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
            name="RL Player",
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

    # Fix option keys to ensure consistent naming convention
    player_1_type = options["player-1-type"]
    player_2_type = options["player-2-type"]
    rl_model_1 = (
        options["rl-model-1"] if "rl-model-1" in options else options["rl_model_1"]
    )
    rl_model_2 = (
        options["rl-model-2"] if "rl-model-2" in options else options["rl_model_2"]
    )

    # Get player colors
    player_1_color = (
        PlayerColor.BLACK if options["player-1-color"] == "black" else PlayerColor.RED
    )
    player_2_color = (
        PlayerColor.RED if player_1_color == PlayerColor.BLACK else PlayerColor.BLACK
    )

    # Check if RL model files exist and are valid if using RL
    if not check_rl_model(rl_model_1, player_1_type, 1) or not check_rl_model(
        rl_model_2, player_2_type, 2
    ):
        return  # Exit the program if models are not available

    # Create players
    player_1 = create_player(
        player_1_type, player_1_color, interface, rl_model_1, "RL Player 1"
    )

    player_2 = create_player(
        player_2_type, player_2_color, interface, rl_model_2, "RL Player 2"
    )

    # Assign players to their correct roles (black or red)
    black_player = player_1 if player_1_color == PlayerColor.BLACK else player_2
    red_player = player_2 if player_2_color == PlayerColor.RED else player_1

    # Create game engine
    engine = GameEngine(
        black_player=black_player,
        red_player=red_player,
        interface=interface,
        target_score=options["target-score"]
        if "target-score" in options
        else options["target_score"],
        ai_delay=1,
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
