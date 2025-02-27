#!/usr/bin/env python3
"""
Train a reinforcement learning agent for Kulibrat using self-play.
"""

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

from src.core.player_color import PlayerColor
from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.players.ai.rl_strategy import RLStrategy
from src.players.ai.rl_model import KulibratNet, encode_board
from src.players.ai.simple_ai_player import SimpleAIPlayer


class SelfPlayWorker:
    """
    Worker that generates self-play games for reinforcement learning.
    """
    
    def __init__(self, model_path: str = None, 
                 num_games: int = 100,
                 exploration_rate: float = 0.25, 
                 temperature: float = 1.0,
                 target_score: int = 5):
        """
        Initialize the self-play worker.
        
        Args:
            model_path: Path to the trained model file (None for random play)
            num_games: Number of self-play games to generate
            exploration_rate: Probability of selecting a random move
            temperature: Temperature for softmax sampling
            target_score: Score needed to win the game
        """
        self.model_path = model_path
        self.num_games = num_games
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.target_score = target_score
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize training data
        self.states = []  # Board states
        self.policies = []  # Target policies
        self.values = []  # Target values
    
    def generate_games(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Generate self-play games for training.
        
        Returns:
            Tuple of (states, policies, values) for training
        """
        print(f"Generating {self.num_games} self-play games...")
        
        # Create AI players with the same strategy
        rl_strategy = RLStrategy(
            model_path=self.model_path,
            exploration_rate=self.exploration_rate,
            temperature=self.temperature,
            device=self.device
        )
        
        black_player = SimpleAIPlayer(PlayerColor.BLACK, rl_strategy)
        red_player = SimpleAIPlayer(PlayerColor.RED, rl_strategy)
        
        # Statistics tracking
        black_wins = 0
        red_wins = 0
        draws = 0
        
        # Generate games
        for game_idx in tqdm(range(self.num_games)):
            # Reset game state
            game_state = GameState(target_score=self.target_score)
            
            # Game history for this game
            game_history = []
            
            # Play until game is over
            while not game_state.is_game_over():
                # Get current player
                current_color = game_state.current_player
                current_player = black_player if current_color == PlayerColor.BLACK else red_player
                
                # Store current state
                current_board = game_state.board.copy()
                
                # Get valid moves
                valid_moves = game_state.get_valid_moves()
                
                if not valid_moves:
                    # No valid moves, skip turn
                    game_state.current_player = game_state.current_player.opposite()
                    continue
                
                # Get move from the current player
                move = current_player.get_move(game_state)
                
                if not move:
                    # Player couldn't make a move, skip turn
                    game_state.current_player = game_state.current_player.opposite()
                    continue
                
                # Create one-hot policy vector
                policy = np.zeros(74)  # Large enough for all possible actions
                
                # Get action index for the chosen move (if available)
                if hasattr(rl_strategy, 'move_to_action'):
                    move_key = rl_strategy._move_to_key(move)
                    if move_key in rl_strategy.move_to_action:
                        action_idx = rl_strategy.move_to_action[move_key]
                        policy[action_idx] = 1.0
                
                # Add state and policy to game history
                game_history.append({
                    'state': current_board,
                    'player': current_color.value,
                    'policy': policy,
                })
                
                # Apply move
                success = game_state.apply_move(move)
                
                if success:
                    # Switch to the other player
                    game_state.current_player = game_state.current_player.opposite()
            
            # Game is over, determine winner
            winner = game_state.get_winner()
            
            if winner == PlayerColor.BLACK:
                reward = 1.0
                black_wins += 1
            elif winner == PlayerColor.RED:
                reward = -1.0
                red_wins += 1
            else:
                reward = 0.0
                draws += 1
            
            # Update game history with rewards
            for i, step in enumerate(game_history):
                player = step['player']
                
                # Value is the reward from the perspective of the player
                if player == PlayerColor.BLACK.value:
                    value = reward
                else:  # RED
                    value = -reward
                
                # Encode state
                encoded_state = encode_board(step['state'], player).squeeze(0).numpy()
                
                # Add to training data
                self.states.append(encoded_state)
                self.policies.append(step['policy'])
                self.values.append(value)
        
        # Print statistics
        print(f"Games completed: BLACK wins: {black_wins}, RED wins: {red_wins}, Draws: {draws}")
        
        return self.states, self.policies, self.values


def train_network(states: List[np.ndarray], 
                  policies: List[np.ndarray], 
                  values: List[float],
                  model_path: str = None,
                  batch_size: int = 64,
                  epochs: int = 10,
                  learning_rate: float = 0.001,
                  output_dir: str = "models"):
    """
    Train the neural network on self-play data with robust policy handling.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    model = KulibratNet().to(device)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded initial model from {model_path}")
    
    # Convert data to PyTorch tensors
    X = torch.FloatTensor(np.array(states)).to(device)
    
    # Policy processing: find first index where policy is 1.0
    def process_policy(policy):
        idx = np.where(policy == 1.0)[0]
        return idx[0] if len(idx) > 0 else 0
    
    # Process policies and values
    policy_indices = torch.LongTensor([process_policy(p) for p in policies]).to(device)
    value_targets = torch.FloatTensor(values).unsqueeze(1).to(device)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(X, policy_indices, value_targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for states_batch, policy_batch, value_batch in progress_bar:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_preds = model(states_batch)
            
            # Compute losses
            policy_loss = policy_criterion(policy_logits, policy_batch)
            value_loss = value_criterion(value_preds, value_batch)
            
            # Combined loss
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Total Loss': loss.item(), 
                'Policy Loss': policy_loss.item(), 
                'Value Loss': value_loss.item()
            })
        
        # Print epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = total_policy_loss / len(dataloader)
        avg_value_loss = total_value_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Total Loss: {avg_loss:.4f}")
        print(f"Average Policy Loss: {avg_policy_loss:.4f}")
        print(f"Average Value Loss: {avg_value_loss:.4f}")
        
        # Save model checkpoint
        model_save_path = os.path.join(output_dir, f"kulibrat_rl_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_save_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "kulibrat_rl_model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    
    return final_model_path


def main():
    import os 

    """Main function for training the RL agent."""
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent for Kulibrat")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to an existing model to continue training (default: None, start from scratch)"
    )
    
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of self-play games to generate per iteration (default: 100)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of training iterations (default: 5)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs per iteration (default: 10)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--exploration-rate",
        type=float,
        default=0.25,
        help="Exploration rate for self-play (default: 0.25)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for policy sampling (default: 1.0)"
    )
    
    parser.add_argument(
        "--target-score",
        type=int,
        default=5,
        help="Target score for games (default: 5)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save models (default: 'models')"
    )
    
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Force using CPU even if CUDA is available"
    )
    
    args = parser.parse_args()
    
    print("=== Kulibrat Reinforcement Learning Training ===")
    
    # Force CPU if requested
    if args.use_cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Forcing CPU use as requested")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Training Configuration:")
    print(f"- Self-play games per iteration: {args.num_games}")
    print(f"- Training iterations: {args.iterations}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Epochs per iteration: {args.epochs}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Exploration rate: {args.exploration_rate}")
    print(f"- Temperature: {args.temperature}")
    print(f"- Target score: {args.target_score}")
    print(f"- Output directory: {args.output_dir}")
    print()
    
    # Initialize model path
    current_model_path = args.model_path
    
    # Training iterations
    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration+1}/{args.iterations} ===")
        
        # Generate self-play games
        worker = SelfPlayWorker(
            model_path=current_model_path,
            num_games=args.num_games,
            exploration_rate=args.exploration_rate,
            temperature=args.temperature,
            target_score=args.target_score
        )
        
        states, policies, values = worker.generate_games()
        
        print(f"Generated {len(states)} training examples")
        
        # Train network
        iteration_output_dir = os.path.join(args.output_dir, f"iteration_{iteration+1}")
        current_model_path = train_network(
            states=states,
            policies=policies,
            values=values,
            model_path=current_model_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            output_dir=iteration_output_dir
        )
        
        print(f"Iteration {iteration+1} complete. Current model: {current_model_path}")
    
    print("\n=== Training Complete ===")
    print(f"Final model saved to: {current_model_path}")


if __name__ == "__main__":
    main()