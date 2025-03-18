#!/usr/bin/env python3
"""
Train an AlphaZero model for Kulibrat using self-play with parallelization.
"""

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional, Iterator
import concurrent.futures
import multiprocessing
import json
import pickle
from datetime import datetime

from src.core.player_color import PlayerColor
from src.core.game_state import GameState
from src.core.move import Move
from src.core.move_type import MoveType
from src.players.ai.alphazero_strategy import AlphaZeroStrategy
from src.players.ai.alphazero_player import AlphaZeroPlayer
from src.players.ai.rl_model import KulibratNet, encode_board


# Global device variable to ensure all processes use the same device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class GameResult:
    """Simple container for game results to facilitate parallel processing."""
    
    def __init__(self, states: List[np.ndarray], policies: List[np.ndarray], values: List[float]):
        self.states = states
        self.policies = policies
        self.values = values


def play_single_game(
    game_idx: int,
    model_path: str,
    n_simulations: int, 
    c_puct: float, 
    exploration_rate: float,
    temperature: float,
    target_score: int,
    random_seed: Optional[int] = None
) -> GameResult:
    """
    Play a single self-play game and return training data. 
    This function is designed to be called in parallel.
    
    Args:
        game_idx: Index of this game (for logging)
        model_path: Path to the AlphaZero model file
        n_simulations: Number of MCTS simulations per move
        c_puct: Exploration constant for MCTS
        exploration_rate: Probability of selecting a random move
        temperature: Temperature for policy sampling
        target_score: Score needed to win
        random_seed: Random seed for reproducibility
        
    Returns:
        A GameResult object containing states, policies, and values
    """
    # Set random seed if provided
    if random_seed is not None:
        # Use different seeds for each game
        game_seed = random_seed + game_idx
        random.seed(game_seed)
        np.random.seed(game_seed)
        torch.manual_seed(game_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(game_seed)
    
    # Create AlphaZero players with the same strategy
    black_player = AlphaZeroPlayer(
        color=PlayerColor.BLACK,
        model_path=model_path,
        n_simulations=n_simulations,
        c_puct=c_puct,
        exploration_rate=exploration_rate,
        temperature=temperature
    )
    
    red_player = AlphaZeroPlayer(
        color=PlayerColor.RED,
        model_path=model_path,
        n_simulations=n_simulations,
        c_puct=c_puct,
        exploration_rate=exploration_rate,
        temperature=temperature
    )
    
    # Initialize game state
    game_state = GameState(target_score=target_score)
    
    # Game history
    game_history = []
    turn_count = 0
    
    # Play until game is over (with turn limit to prevent infinite games)
    max_turns = 200  # Reasonable limit for Kulibrat
    
    while not game_state.is_game_over() and turn_count < max_turns:
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
        
        # Create one-hot policy vector based on the selected move
        policy = np.zeros(51)  # Match model's action space
        
        # Try to get action index for the chosen move
        move_key = current_player.strategy._move_to_key(move)
        if move_key in current_player.strategy.move_to_action:
            action_idx = current_player.strategy.move_to_action[move_key]
            if 0 <= action_idx < len(policy):
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
            turn_count += 1
    
    # Game is over, determine winner
    winner = game_state.get_winner()
    
    # If we hit the turn limit, consider it a draw
    if turn_count >= max_turns:
        winner = None
    
    if winner == PlayerColor.BLACK:
        reward = 1.0
    elif winner == PlayerColor.RED:
        reward = -1.0
    else:
        reward = 0.0
    
    # Collect training data
    states = []
    policies = []
    values = []
    
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
        states.append(encoded_state)
        policies.append(step['policy'])
        values.append(value)
    
    return GameResult(states, policies, values)


class ParallelSelfPlayWorker:
    """
    Worker that generates self-play games in parallel for AlphaZero training.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        num_games: int = 100,
        n_simulations: int = 100,
        c_puct: float = 1.0,
        exploration_rate: float = 0.25, 
        temperature: float = 1.0,
        target_score: int = 5,
        num_workers: int = None,
        chunk_size: int = 10,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the parallel self-play worker.
        
        Args:
            model_path: Path to the trained model file (None for random initialization)
            num_games: Number of self-play games to generate
            n_simulations: Number of MCTS simulations per move during self-play
            c_puct: Exploration constant for MCTS
            exploration_rate: Probability of selecting a random move
            temperature: Temperature for softmax sampling
            target_score: Score needed to win the game
            num_workers: Number of worker processes (default: number of CPU cores)
            chunk_size: Number of games per chunk for parallel processing
            random_seed: Random seed for reproducibility
        """
        self.model_path = model_path
        self.num_games = num_games
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.target_score = target_score
        
        # Determine number of workers
        if num_workers is None:
            # Default to number of available CPU cores - 1 (to keep one free for system)
            self.num_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_workers = num_workers
            
        # Adjust chunk size based on number of games and workers
        self.chunk_size = min(chunk_size, max(1, num_games // self.num_workers))
        
        # Random seed for reproducibility
        self.random_seed = random_seed
        
        # Initialize training data
        self.states = []  # Board states
        self.policies = []  # Target policies
        self.values = []  # Target values
    
    def generate_games(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Generate self-play games in parallel.
        
        Returns:
            Tuple of (states, policies, values) for training
        """
        print(f"Generating {self.num_games} self-play games using {self.num_workers} workers...")
        print(f"MCTS simulations per move: {self.n_simulations}")
        print(f"Exploration rate: {self.exploration_rate}, Temperature: {self.temperature}")
        
        # Check if model exists
        if self.model_path and not os.path.exists(self.model_path):
            print(f"Warning: Model file not found at {self.model_path}")
            print("Using random initialization instead.")
            self.model_path = None
        
        try:
            # Calculate number of chunks
            num_chunks = (self.num_games + self.chunk_size - 1) // self.chunk_size
            
            # Create task list with chunk indices
            tasks = []
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, self.num_games)
                num_games_in_chunk = end_idx - start_idx
                
                # Add each game in this chunk as a separate task
                for game_offset in range(num_games_in_chunk):
                    game_idx = start_idx + game_offset
                    tasks.append((game_idx, self.model_path, self.n_simulations, 
                                 self.c_puct, self.exploration_rate, self.temperature, 
                                 self.target_score, self.random_seed))
            
            # Process chunks in parallel
            results = []
            
            with tqdm(total=self.num_games, desc="Self-play games") as pbar:
                # Use ProcessPoolExecutor for parallel execution
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    # Submit all tasks
                    future_to_game_idx = {
                        executor.submit(play_single_game, *task): task[0] 
                        for task in tasks
                    }
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(future_to_game_idx):
                        game_idx = future_to_game_idx[future]
                        try:
                            result = future.result()
                            results.append((game_idx, result))
                            # Update progress bar
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error in game {game_idx}: {e}")
            
            # Sort results by game index to ensure deterministic order
            results.sort(key=lambda x: x[0])
            
            # Combine all results
            for _, result in results:
                self.states.extend(result.states)
                self.policies.extend(result.policies)
                self.values.extend(result.values)
            
            print(f"Generated {len(self.states)} training examples from {len(results)} games")
            
            # Statistics
            black_wins = sum(1 for _, result in results if len(result.values) > 0 and result.values[0] > 0)
            red_wins = sum(1 for _, result in results if len(result.values) > 0 and result.values[0] < 0)
            draws = len(results) - black_wins - red_wins
            
            print(f"Game outcomes: BLACK wins: {black_wins}, RED wins: {red_wins}, Draws: {draws}")
            print(f"Win rates: BLACK: {black_wins/len(results)*100:.1f}%, RED: {red_wins/len(results)*100:.1f}%, Draws: {draws/len(results)*100:.1f}%")
            
            return self.states, self.policies, self.values
            
        except Exception as e:
            print(f"Error during parallel self-play: {e}")
            # Return whatever data we collected up to this point
            if not self.states:
                raise RuntimeError("No training data could be generated")
            return self.states, self.policies, self.values


def train_network(states: List[np.ndarray], 
                  policies: List[np.ndarray], 
                  values: List[float],
                  model_path: Optional[str] = None,
                  batch_size: int = 64,
                  epochs: int = 10,
                  learning_rate: float = 0.001,
                  output_dir: str = "models",
                  validation_split: float = 0.1):
    """
    Train the neural network on self-play data.
    
    Args:
        states: List of encoded board states
        policies: List of target policy vectors
        values: List of target values
        model_path: Path to initial model (None for random initialization)
        batch_size: Mini-batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        output_dir: Directory to save trained models
        validation_split: Fraction of data to use for validation
        
    Returns:
        Path to the trained model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    device = DEVICE
    print(f"Training on {device}")
    
    # Initialize model
    model = KulibratNet().to(device)
    
    # Load initial model if provided
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded initial model from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Training with a fresh model instead.")
    
    # Convert data to NumPy arrays
    states_np = np.array(states)
    policies_np = np.array(policies)
    values_np = np.array(values)
    
    # Split data into training and validation sets
    num_samples = len(states)
    indices = np.random.permutation(num_samples)
    val_size = int(num_samples * validation_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create tensors
    X_train = torch.FloatTensor(states_np[train_indices]).to(device)
    policy_train = torch.FloatTensor(policies_np[train_indices]).to(device)
    value_train = torch.FloatTensor(values_np[train_indices]).unsqueeze(1).to(device)
    
    if val_size > 0:
        X_val = torch.FloatTensor(states_np[val_indices]).to(device)
        policy_val = torch.FloatTensor(policies_np[val_indices]).to(device)
        value_val = torch.FloatTensor(values_np[val_indices]).unsqueeze(1).to(device)
    
    # Create dataset and dataloader for training
    train_dataset = TensorDataset(X_train, policy_train, value_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training history for plotting
    train_losses = []
    val_losses = []
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    
    print(f"Training on {len(train_indices)} examples, validating on {len(val_indices)} examples")
    
    model.train()
    for epoch in range(epochs):
        # Training phase
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for states_batch, policy_batch, value_batch in progress_bar:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_preds = model(states_batch)
            
            # Convert policy targets to action indices
            policy_targets = torch.argmax(policy_batch, dim=1)
            
            # Compute losses
            policy_loss = policy_criterion(policy_logits, policy_targets)
            value_loss = value_criterion(value_preds, value_batch)
            
            # Combined loss with weighting
            loss = policy_loss + value_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimize
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Policy': f"{policy_loss.item():.4f}",
                'Value': f"{value_loss.item():.4f}"
            })
        
        # Average training losses
        avg_train_loss = total_loss / len(train_dataloader)
        avg_policy_loss = total_policy_loss / len(train_dataloader)
        avg_value_loss = total_value_loss / len(train_dataloader)
        
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_size > 0:
            model.eval()
            val_loss = 0.0
            val_policy_loss = 0.0
            val_value_loss = 0.0
            
            with torch.no_grad():
                policy_logits, value_preds = model(X_val)
                
                # Convert policy targets to action indices
                policy_targets = torch.argmax(policy_val, dim=1)
                
                # Compute validation losses
                val_p_loss = policy_criterion(policy_logits, policy_targets)
                val_v_loss = value_criterion(value_preds, value_val)
                
                val_total_loss = val_p_loss + val_v_loss
                
                val_loss = val_total_loss.item()
                val_policy_loss = val_p_loss.item()
                val_value_loss = val_v_loss.item()
            
            val_losses.append(val_loss)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, "alphazero_model_best.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
            
            model.train()
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training - Loss: {avg_train_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")
            print(f"Validation - Loss: {val_loss:.4f}, Policy: {val_policy_loss:.4f}, Value: {val_value_loss:.4f}")
        else:
            # Print epoch summary (training only)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training - Loss: {avg_train_loss:.4f}, Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f"alphazero_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "alphazero_model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Use the best validation model if available, otherwise use the final model
    result_model_path = best_model_path if best_model_path else final_model_path
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    if val_losses:
        plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    return result_model_path


def main():
    """Main function for training the AlphaZero agent with parallelization."""
    parser = argparse.ArgumentParser(description="Train an AlphaZero agent for Kulibrat with parallel self-play")
    
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
        "--n-simulations",
        type=int,
        default=100,
        help="Number of MCTS simulations per move during self-play (default: 100)"
    )
    
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.0,
        help="Exploration constant for MCTS (default: 1.0)"
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
        default="models/alphazero",
        help="Directory to save models (default: 'models/alphazero')"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel self-play (default: CPU count - 1)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Number of games per worker batch (default: 10)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    parser.add_argument(
        "--save-memory",
        action="store_true",
        help="Save memory by processing games in smaller batches"
    )
    
    args = parser.parse_args()
    
    print("=== Kulibrat AlphaZero Training with Parallel Self-Play ===")
    
    # Set random seed if provided
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Print configuration
    print("Training Configuration:")
    print(f"- Device: {DEVICE}")
    print(f"- Self-play games per iteration: {args.num_games}")
    print(f"- MCTS simulations per move: {args.n_simulations}")
    print(f"- MCTS exploration constant (c_puct): {args.c_puct}")
    print(f"- Training iterations: {args.iterations}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Epochs per iteration: {args.epochs}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Exploration rate: {args.exploration_rate}")
    print(f"- Temperature: {args.temperature}")
    print(f"- Target score: {args.target_score}")
    print(f"- Validation split: {args.validation_split}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Parallel workers: {args.num_workers if args.num_workers is not None else 'auto'}")
    print(f"- Chunk size: {args.chunk_size}")
    print(f"- Memory-saving mode: {'on' if args.save_memory else 'off'}")
    print()
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    current_model_path = args.model_path
    
    try:
        # Training iterations
        for iteration in range(args.iterations):
            print(f"\n=== Iteration {iteration+1}/{args.iterations} ===")
            
            # Create iteration directory
            iteration_dir = os.path.join(run_dir, f"iteration_{iteration+1}")
            os.makedirs(iteration_dir, exist_ok=True)
            
            # Generate self-play games in parallel
            worker = ParallelSelfPlayWorker(
                model_path=current_model_path,
                num_games=args.num_games,
                n_simulations=args.n_simulations,
                c_puct=args.c_puct,
                exploration_rate=args.exploration_rate,
                temperature=args.temperature,
                target_score=args.target_score,
                num_workers=args.num_workers,
                chunk_size=args.chunk_size,
                random_seed=args.seed
            )
            
            states, policies, values = worker.generate_games()
            
            print(f"Generated {len(states)} training examples")
            
            # If memory-saving mode is enabled, save data to disk
            if args.save_memory:
                # Save data to disk temporarily
                data_file = os.path.join(iteration_dir, "training_data.pkl")
                with open(data_file, 'wb') as f:
                    pickle.dump((states, policies, values), f)
                
                # Clear memory
                states, policies, values = None, None, None
                
                # Load data back for training
                with open(data_file, 'rb') as f:
                    states, policies, values = pickle.load(f)
            
            # Train network
            current_model_path = train_network(
                states=states,
                policies=policies,
                values=values,
                model_path=current_model_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                output_dir=iteration_dir,
                validation_split=args.validation_split
            )
            
            print(f"Iteration {iteration+1} complete. Current model: {current_model_path}")
            
            # Save the model to the iteration directory with a different name
            final_iteration_model = os.path.join(run_dir, f"alphazero_model_iteration_{iteration+1}.pt")
            torch.save(torch.load(current_model_path, map_location=DEVICE), final_iteration_model)
            print(f"Saved iteration model to: {final_iteration_model}")
            
            # Make the current model the latest one for the next iteration
            current_model_path = final_iteration_model
        
        print("\n=== Training Complete ===")
        print(f"Final model saved to: {current_model_path}")
        
        # Also save the final model to the main output directory
        final_model_path = os.path.join(args.output_dir, "alphazero_model_final.pt")
        torch.save(torch.load(current_model_path, map_location=DEVICE), final_model_path)
        print(f"Copied final model to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if current_model_path:
            print(f"Latest model saved at: {current_model_path}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if current_model_path:
            print(f"Latest model saved at: {current_model_path}")


def worker_init():
    """
    Initialize worker process properly for CUDA usage.
    This must be called at the start of any worker function.
    """
    # Set torch multiprocessing sharing strategy
    if hasattr(torch.multiprocessing, 'set_sharing_strategy'):
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Ensure CUDA is properly initialized in this process
    if torch.cuda.is_available():
        torch.cuda.init()
        # Print unique identifier to verify isolated CUDA contexts
        torch.cuda.empty_cache()


def play_single_game_wrapper(args):
    """
    Wrapper function for play_single_game to handle process initialization.
    
    Args:
        args: Arguments tuple for play_single_game
        
    Returns:
        Result of play_single_game
    """
    # Initialize worker
    worker_init()
    
    # Call the actual game function
    try:
        return play_single_game(*args)
    except Exception as e:
        print(f"Error in game {args[0]}: {e}")
        # Return empty result
        return GameResult([], [], [])


if __name__ == "__main__":
    # Must set multiprocessing start method BEFORE creating any processes
    # This needs to happen at the top level, before importing torch or any other modules
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("Warning: Could not set multiprocessing start method to 'spawn'.")
        print("This might cause CUDA errors in subprocesses.")
    
    main()