#!/usr/bin/env python3
"""
Debug version of the Kulibrat game to identify player turn issues.
"""

import os
import sys
import time
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import from project modules
from src.core.player_color import PlayerColor
from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.players.human_player import HumanPlayer
from src.ui.console_interface import ConsoleInterface


class DebugConsoleInterface(ConsoleInterface):
    """Extended console interface with debugging info."""
    
    def display_state(self, game_state: GameState) -> None:
        """Display the game state with debugging info."""
        super().display_state(game_state)
        print("\n--- DEBUG INFO ---")
        print(f"Current Player: {game_state.current_player.name}")
        print(f"BLACK pieces off board: {game_state.pieces_off_board[PlayerColor.BLACK]}")
        print(f"RED pieces off board: {game_state.pieces_off_board[PlayerColor.RED]}")
        print(f"Scores: BLACK={game_state.scores[PlayerColor.BLACK]}, RED={game_state.scores[PlayerColor.RED]}")
        
        # Check valid moves for both players
        current_player = game_state.current_player
        
        # Check BLACK moves
        game_state.current_player = PlayerColor.BLACK
        black_moves = game_state.get_valid_moves()
        print(f"Valid moves for BLACK: {len(black_moves)}")
        
        # Check RED moves
        game_state.current_player = PlayerColor.RED
        red_moves = game_state.get_valid_moves()
        print(f"Valid moves for RED: {len(red_moves)}")
        
        # Restore original current player
        game_state.current_player = current_player
        print("------------------\n")
    
    def get_human_move(self, game_state: GameState, player_color: PlayerColor, 
                       valid_moves: List[Move]) -> Move:
        """Get a human move with added debugging."""
        print(f"\n[DEBUG] Getting move for {player_color.name} player")
        print(f"[DEBUG] There are {len(valid_moves)} valid moves")
        
        # Continue with normal move selection
        return super().get_human_move(game_state, player_color, valid_moves)


class DebugGameEngine:
    """Debug version of the game engine to identify player turn issues."""
    
    def __init__(self, black_player, red_player, interface):
        """Initialize the debug game engine."""
        self.state = GameState(target_score=5)
        self.players = {
            PlayerColor.BLACK: black_player,
            PlayerColor.RED: red_player
        }
        self.interface = interface
        self.moves_history = []
    
    def start_game(self):
        """Start the game with explicit player turns."""
        # Display initial state
        self.interface.display_state(self.state)
        
        # Main game loop
        while not self.state.is_game_over():
            # Get current player
            current_color = self.state.current_player
            current_player = self.players[current_color]
            opponent_color = current_color.opposite()
            
            print(f"\n[DEBUG] Turn for {current_color.name} player")
            
            # Get valid moves
            valid_moves = self.state.get_valid_moves()
            
            if not valid_moves:
                print(f"[DEBUG] No valid moves for {current_color.name}")
                self.interface.show_message(f"{current_color.name} has no valid moves. Skipping turn.")
                # Switch player and continue
                self.state.current_player = opponent_color
                continue
            
            # Get move from current player
            move = current_player.get_move(self.state)
            
            if not move:
                print(f"[DEBUG] Player {current_color.name} returned no move")
                self.interface.show_message(f"{current_color.name} couldn't make a move.")
                # Switch player and continue
                self.state.current_player = opponent_color
                continue
            
            # Apply the move
            print(f"[DEBUG] Applying move: {move}")
            previous_player = self.state.current_player
            success = self.state.apply_move(move)
            
            if success:
                print(f"[DEBUG] Move successful")
                print(f"[DEBUG] Player before: {previous_player.name}, after: {self.state.current_player.name}")
                
                # Save move to history
                self.moves_history.append(move)
                
                # Display the new state
                self.interface.display_state(self.state)
                
                # Explicitly switch player if needed
                if self.state.current_player == previous_player:
                    print(f"[DEBUG] Forcing player change from {previous_player.name} to {opponent_color.name}")
                    self.state.current_player = opponent_color
            else:
                print(f"[DEBUG] Move failed")
                self.interface.show_message("Invalid move!")
        
        # Game over
        winner = self.state.get_winner()
        self.interface.show_winner(winner, self.state)
        
        return winner


def main():
    """Run the debug version of the game."""
    print("\n=== KULIBRAT GAME DEBUG MODE ===")
    print("This version will print extra information to help diagnose turn issues.")
    
    # Create interface
    interface = DebugConsoleInterface()
    
    # Create human players
    black_player = HumanPlayer(PlayerColor.BLACK, interface)
    red_player = HumanPlayer(PlayerColor.RED, interface)
    
    # Create game engine
    engine = DebugGameEngine(black_player, red_player, interface)
    
    # Start the game
    print("\nStarting game...")
    time.sleep(1)
    
    # Run the game
    winner = engine.start_game()
    
    print("\nGame finished!")


if __name__ == "__main__":
    main()