#!/usr/bin/env python3
"""
Main entry point for the Kulibrat game with enhanced UI.

This script initializes and runs the Kulibrat game with a configuration screen
and responsive fullscreen interface.

Run this file from the project root:
python main.py
"""

import sys
import os
import time
from typing import Dict, Any

# Add the project root to the Python path to make imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import core components
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
from src.players.ai.rl_player import RLPlayer

# Import UI components
from src.ui.game_interface import GameInterface
from src.ui.console_interface import ConsoleInterface
from src.ui.pygame_interface import KulibratGUI


def main():
    """Main entry point for the game."""
    # Initialize the GUI
    interface = KulibratGUI()
    
    # The interface will show a configuration screen first
    # and the game_config will be populated with user selections
    game_config = interface.game_config
    
    print("\n=== KULIBRAT GAME CONFIGURATION ===")
    print(f"Black Player: {game_config['black_player']['name']} ({game_config['black_player']['type']})")
    print(f"Red Player: {game_config['red_player']['name']} ({game_config['red_player']['type']})")
    print(f"Target Score: {game_config['target_score']}")
    print(f"AI Delay: {game_config['ai_delay']} seconds")
    print(f"Fullscreen: {game_config['fullscreen']}")
    if 'RL' in game_config['black_player']['type'] or 'RL' in game_config['red_player']['type']:
        print(f"RL Model Path: {game_config['rl_model_path']}")
    print("\nStarting game...")
    
    # Create players based on configuration
    black_player = create_player(
        PlayerColor.BLACK, 
        game_config['black_player']['type'],
        game_config['black_player']['name'],
        interface,
        game_config
    )
    
    red_player = create_player(
        PlayerColor.RED, 
        game_config['red_player']['type'],
        game_config['red_player']['name'],
        interface,
        game_config
    )
    
    # Create game engine
    engine = GameEngine(
        black_player=black_player,
        red_player=red_player,
        interface=interface,
        target_score=game_config['target_score'],
        ai_delay=game_config['ai_delay']
    )
    
    # Start the game
    play_again = True
    
    while play_again:
        winner = engine.start_game()
        
        # The pygame interface handles the play_again logic
        play_again = False
        
        if play_again:
            # Reset the game
            engine.reset_game()
    
    print("\nThanks for playing Kulibrat!")


def create_player(
    color: PlayerColor,
    player_type: str,
    name: str,
    interface: GameInterface,
    config: Dict[str, Any]
) -> Player:
    """
    Create a player based on configuration.
    
    Args:
        color: Player color
        player_type: Type of player (Human, AI, RL)
        name: Player name
        interface: Game interface
        config: Game configuration
        
    Returns:
        Configured player object
    """
    if player_type == "Human":
        return HumanPlayer(color, interface, name=name)
    
    elif player_type == "AI (Random)":
        strategy = RandomStrategy()
        return SimpleAIPlayer(color, strategy, name=f"{name} (Random AI)")
    
    elif player_type == "AI (Minimax)":
        strategy = MinimaxStrategy(max_depth=3, use_alpha_beta=True)
        return SimpleAIPlayer(color, strategy, name=f"{name} (Minimax AI)")
    
    elif player_type == "RL Model":
        # Check if RL model file exists
        model_path = config['rl_model_path']
        if not os.path.exists(model_path):
            print(f"Warning: RL model file not found at {model_path}. Using a random strategy instead.")
            strategy = RandomStrategy()
            return SimpleAIPlayer(color, strategy, name=f"{name} (Random Fallback)")
        
        return RLPlayer(
            color,
            model_path=model_path,
            exploration_rate=0.0,  # Use 0 for deterministic play
            temperature=0.1,       # Low temperature for stronger play
            name=f"{name} (RL)"
        )
    
    # Default to Random AI as fallback
    strategy = RandomStrategy()
    return SimpleAIPlayer(color, strategy, name=f"{name} (Random AI)")


if __name__ == "__main__":
    main()