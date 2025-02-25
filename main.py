#!/usr/bin/env python3
"""
Kulibrat - Main entry point
A board game with simple rules but complex strategy
"""
import sys
import argparse
from gui.ui import KulibratGUI
from src.game import Game
from src.player import HumanPlayer, RandomAIPlayer

def play_gui_game(win_score: int = 5):
    """Launch the game with a graphical user interface"""
    gui = KulibratGUI(win_score=win_score)
    gui.run()

def play_cli_game(win_score: int = 5, black_ai: bool = False, red_ai: bool = False):
    """Play the game in command-line interface mode"""
    game = Game(win_score=win_score)
    
    # Setup players
    black_player = RandomAIPlayer("black") if black_ai else HumanPlayer("black")
    red_player = RandomAIPlayer("red") if red_ai else HumanPlayer("red")
    
    print("=== Kulibrat ===")
    print(f"First to {win_score} points wins!\n")
    
    while not game.is_game_over():
        print("\n" + str(game) + "\n")
        
        current_player = black_player if game.current_player == "black" else red_player
        
        if isinstance(current_player, HumanPlayer):
            print(f"{game.current_player.capitalize()}'s turn (Human)")
        else:
            print(f"{game.current_player.capitalize()}'s turn (AI)")
        
        # Get player's move
        start_row, start_col, end_row, end_col = current_player.get_move(game)
        
        if start_row is None and start_col is None and end_row == -1 and end_col == -1:
            print(f"No legal moves for {game.current_player.capitalize()}")
            continue
        
        # Store the current player before making the move
        current_player_color = game.current_player
        
        # Make the move
        success = game.make_move(start_row, start_col, end_row, end_col)
        
        if success:
            # Print the move
            if start_row is None and start_col is None:
                print(f"{current_player_color.capitalize()} inserted a piece at column {end_col+1}")
            else:
                print(f"{current_player_color.capitalize()} moved a piece from ({start_row+1},{start_col+1}) to ({end_row+1},{end_col+1})")
        else:
            print("Invalid move! Try again.")
    
    # Game over
    print("\n" + str(game) + "\n")
    print("=== Game Over ===")
    
    if game.black_score > game.red_score:
        print("Black wins!")
    else:
        print("Red wins!")
    
    print(f"Final score - Black: {game.black_score}, Red: {game.red_score}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Kulibrat - A board game with simple rules but complex strategy")
    
    parser.add_argument("--cli", action="store_true", help="Run in command-line interface mode")
    parser.add_argument("--win-score", type=int, default=5, help="Number of points needed to win (default: 5)")
    parser.add_argument("--black-ai", action="store_true", help="Use AI for black player (CLI mode only)")
    parser.add_argument("--red-ai", action="store_true", help="Use AI for red player (CLI mode only)")
    
    args = parser.parse_args()
    
    if args.cli:
        play_cli_game(win_score=args.win_score, black_ai=args.black_ai, red_ai=args.red_ai)
    else:
        play_gui_game(win_score=args.win_score)

if __name__ == "__main__":
    main()