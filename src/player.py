# src/player.py
from typing import Tuple, Optional, List
import random
from .game import Game

class Player:
    """Base class for player implementations"""
    
    def __init__(self, color: str):
        self.color = color  # "red" or "black"
    
    def get_move(self, game: Game) -> Tuple[Optional[int], Optional[int], int, int]:
        """
        Get the player's move
        Should return (start_row, start_col, end_row, end_col)
        """
        raise NotImplementedError("Subclasses must implement get_move")


class HumanPlayer(Player):
    """Human player implementation (for command-line interface)"""
    
    def get_move(self, game: Game) -> Tuple[Optional[int], Optional[int], int, int]:
        """Get move from human input"""
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            print("No legal moves available!")
            return None, None, -1, -1
        
        # Display available moves
        print("\nAvailable moves:")
        for i, move in enumerate(legal_moves):
            move_type, start_row, start_col, end_row, end_col = move
            if move_type == "insert":
                print(f"{i+1}. Insert at column {end_col+1}")
            elif move_type == "diagonal":
                print(f"{i+1}. Move from ({start_row+1},{start_col+1}) to ({end_row+1},{end_col+1})")
            elif move_type == "attack":
                print(f"{i+1}. Attack from ({start_row+1},{start_col+1}) to ({end_row+1},{end_col+1})")
            elif move_type == "jump":
                print(f"{i+1}. Jump from ({start_row+1},{start_col+1}) to ({end_row+1},{end_col+1})")
        
        # Get user choice
        choice = -1
        while choice < 1 or choice > len(legal_moves):
            try:
                choice = int(input(f"\nEnter move number (1-{len(legal_moves)}): "))
            except ValueError:
                print("Please enter a valid number")
        
        # Return the selected move
        move_type, start_row, start_col, end_row, end_col = legal_moves[choice-1]
        print(f"DEBUG: Selected move: {move_type} from ({start_row}, {start_col}) to ({end_row}, {end_col})")
        return start_row, start_col, end_row, end_col


class RandomAIPlayer(Player):
    """AI player that makes random legal moves"""
    
    def get_move(self, game: Game) -> Tuple[Optional[int], Optional[int], int, int]:
        """Get a random legal move"""
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            return None, None, -1, -1
        
        move_type, start_row, start_col, end_row, end_col = random.choice(legal_moves)
        return start_row, start_col, end_row, end_col