"""
Reinforcement Learning strategy for Kulibrat.
"""

import os
import random
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple, Any

from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy
from src.players.ai.rl_model import KulibratNet, encode_board


class RLStrategy(AIStrategy):
    """
    AI strategy using a trained neural network for move selection.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 exploration_rate: float = 0.05,
                 temperature: float = 1.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the reinforcement learning strategy.
        
        Args:
            model_path: Path to the trained model file (None for random play)
            exploration_rate: Probability of selecting a random move
            temperature: Temperature for softmax sampling (higher -> more exploration)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        
        # Initialize the model
        self.model = KulibratNet().to(device)
        
        # Load pretrained model if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()  # Set to evaluation mode
            print(f"Loaded RL model from {model_path}")
        else:
            print("No model loaded. Using random initialization.")
        
        # Action encoding/decoding maps
        self.action_to_move = {}  # Maps action indices to corresponding moves
        self.move_to_action = {}  # Maps move descriptions to action indices
    
    def select_move(self, game_state: GameState, player_color: PlayerColor) -> Optional[Move]:
        """
        Select a move using the neural network.
        
        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move
            
        Returns:
            Selected move or None if no valid moves
        """
        valid_moves = game_state.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # With some probability, choose a random move (exploration)
        if random.random() < self.exploration_rate:
            return random.choice(valid_moves)
        
        # Encode the current board state
        encoded_state = encode_board(game_state.board, player_color.value).to(self.device)
        
        # Build the action mapping for this state
        self._build_action_mapping(game_state, valid_moves)
        
        # Get policy and value predictions from the model
        with torch.no_grad():
            try:
                policy_logits, value = self.model(encoded_state)
            except Exception as e:
                print(f"Error during model inference: {e}")
                # Fall back to random selection
                return random.choice(valid_moves)
            
        # Check if valid actions can be mapped
        try:
            # Filter to only valid actions
            valid_actions = []
            for move in valid_moves:
                move_key = self._move_to_key(move)
                if move_key in self.move_to_action:
                    valid_actions.append(self.move_to_action[move_key])
            
            # If no valid actions could be mapped, fall back to random
            if not valid_actions:
                return random.choice(valid_moves)
            
            # Convert to numpy array for safety
            valid_actions = np.array(valid_actions)
            
            # Check if indices are within bounds
            max_index = policy_logits.shape[1] - 1
            valid_actions = [idx for idx in valid_actions if 0 <= idx <= max_index]
            
            if not valid_actions:
                return random.choice(valid_moves)
            
            # Get logits for valid actions
            valid_logits = policy_logits[0, valid_actions].cpu().numpy()
            
            # Apply temperature and convert to probabilities
            if self.temperature > 0:
                valid_logits = valid_logits / self.temperature
                valid_probs = np.exp(valid_logits - np.max(valid_logits))
                valid_probs = valid_probs / np.sum(valid_probs)
            else:
                # If temperature is 0, just take the argmax
                best_idx = np.argmax(valid_logits)
                valid_probs = np.zeros_like(valid_logits)
                valid_probs[best_idx] = 1.0
            
            # Sample from the probability distribution
            chosen_idx = np.random.choice(len(valid_actions), p=valid_probs)
            action_idx = valid_actions[chosen_idx]
            
            # Find the move that corresponds to this action
            for i, move in enumerate(valid_moves):
                move_key = self._move_to_key(move)
                if move_key in self.move_to_action and self.move_to_action[move_key] == action_idx:
                    return move
                    
            # If for some reason we can't find the move, fall back to random
            return random.choice(valid_moves)
            
        except Exception as e:
            print(f"Error during move selection: {e}")
            # Fall back to random selection in case of any error
            return random.choice(valid_moves)
    
    def _build_action_mapping(self, game_state: GameState, valid_moves: List[Move]) -> None:
        """
        Build a mapping between neural network actions and game moves.
        
        Args:
            game_state: Current game state
            valid_moves: List of valid moves
        """
        self.action_to_move = {}
        self.move_to_action = {}
        
        # Board dimensions
        rows, cols = game_state.board.shape
        
        # Build action space
        action_idx = 0
        
        # INSERT moves
        for r in range(rows):
            for c in range(cols):
                self.action_to_move[action_idx] = (MoveType.INSERT, None, (r, c))
                self.move_to_action[f"{MoveType.INSERT.name}:None:{r},{c}"] = action_idx
                action_idx += 1
        
        # DIAGONAL moves
        for r in range(rows):
            for c in range(cols):
                for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    end_r, end_c = r + dr, c + dc
                    self.action_to_move[action_idx] = (MoveType.DIAGONAL, (r, c), (end_r, end_c))
                    self.move_to_action[f"{MoveType.DIAGONAL.name}:{r},{c}:{end_r},{end_c}"] = action_idx
                    action_idx += 1
        
        # ATTACK moves
        for r in range(rows):
            for c in range(cols):
                for dr, dc in [(1, 0), (-1, 0)]:  # Attacks are only straight ahead
                    end_r, end_c = r + dr, c + dc
                    self.action_to_move[action_idx] = (MoveType.ATTACK, (r, c), (end_r, end_c))
                    self.move_to_action[f"{MoveType.ATTACK.name}:{r},{c}:{end_r},{end_c}"] = action_idx
                    action_idx += 1
        
        # JUMP moves
        for r in range(rows):
            for c in range(cols):
                for jump_length in [2, 3, 4]:  # Possible jump lengths
                    for dr_direction in [1, -1]:  # Up or down
                        end_r = r + dr_direction * jump_length
                        self.action_to_move[action_idx] = (MoveType.JUMP, (r, c), (end_r, c))
                        self.move_to_action[f"{MoveType.JUMP.name}:{r},{c}:{end_r},{c}"] = action_idx
                        action_idx += 1
    
    def _move_to_key(self, move: Move) -> str:
        """
        Convert a Move object to a string key for the move_to_action dictionary.
        
        Args:
            move: The Move object
            
        Returns:
            String key representing the move
        """
        move_type = move.move_type.name
        start_pos = "None" if move.start_pos is None else f"{move.start_pos[0]},{move.start_pos[1]}"
        end_pos = f"{move.end_pos[0]},{move.end_pos[1]}"
        
        return f"{move_type}:{start_pos}:{end_pos}"