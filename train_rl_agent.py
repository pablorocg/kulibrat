#!/usr/bin/env python3
"""
Train a reinforcement learning agent for Kulibrat using self-play.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Optional

from src.core.player_color import PlayerColor
from src.core.game_state_cy import GameState
from src.players.ai.rl_strategy import RLStrategy
from src.players.ai.rl_model import KulibratNet, encode_board
from src.players.ai.simple_ai_player import SimpleAIPlayer


class SelfPlayWorker:
    """
    Worker that generates self-play games for reinforcement learning.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
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
        
        try:
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
            for game_idx in tqdm(range(self.num_games), desc="Self-play games"):
                # Reset game state
                game_state = GameState(target_score=self.target_score)
                
                # Game history for this game
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
                    
                    # Create one-hot policy vector
                    policy = np.zeros(51)  # Match model's action space
                    
                    # Get action index for the chosen move (if available)
                    if hasattr(rl_strategy, 'move_to_action'):
                        move_key = rl_strategy._move_to_key(move)
                        if move_key in rl_strategy.move_to_action:
                            action_idx = rl_strategy.move_to_action[move_key]
                            if 0 <= action_idx < len(policy):  # Bounds check
                                policy[action_idx] = 1.0
                            else:
                                print(f"Warning: Action index {action_idx} out of bounds")
                    
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
            
        except Exception as e:
            print(f"Error during self-play: {e}")
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")
    
    # Initialize model variable outside try block
    model = None
    
    try:
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
        
        # Create training tensors
        X_train = torch.FloatTensor(states_np[train_indices]).to(device)
        
        # Process policies: find index with value 1.0
        def process_policy(policy):
            idx = np.where(policy == 1.0)[0]
            return idx[0] if len(idx) > 0 else 0
        
        policy_indices_train = torch.LongTensor([process_policy(p) for p in policies_np[train_indices]]).to(device)
        value_targets_train = torch.FloatTensor(values_np[train_indices]).unsqueeze(1).to(device)
        
        # Create validation tensors if we have validation data
        if val_size > 0:
            X_val = torch.FloatTensor(states_np[val_indices]).to(device)
            policy_indices_val = torch.LongTensor([process_policy(p) for p in policies_np[val_indices]]).to(device)
            value_targets_val = torch.FloatTensor(values_np[val_indices]).unsqueeze(1).to(device)
        
        # Create dataset and dataloader for training
        train_dataset = torch.utils.data.TensorDataset(X_train, policy_indices_train, value_targets_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss functions
        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        
        # Optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
                
                # Compute losses
                policy_loss = policy_criterion(policy_logits, policy_batch)
                value_loss = value_criterion(value_preds, value_batch)
                
                # Combined loss with weighting (policy is usually harder to learn)
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
                    
                    # Compute validation losses
                    val_p_loss = policy_criterion(policy_logits, policy_indices_val)
                    val_v_loss = value_criterion(value_preds, value_targets_val)
                    
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
                    best_model_path = os.path.join(output_dir, "kulibrat_rl_model_best.pt")
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
            checkpoint_path = os.path.join(output_dir, f"kulibrat_rl_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        
        # Save final model
        final_model_path = os.path.join(output_dir, "kulibrat_rl_model_final.pt")
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
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Save model even if training crashed
        try:
            if model is not None:
                emergency_save_path = os.path.join(output_dir, "kulibrat_rl_model_emergency.pt")
                torch.save(model.state_dict(), emergency_save_path)
                print(f"Emergency model save at: {emergency_save_path}")
                return emergency_save_path
        except Exception as save_error:
            print(f"Failed to save emergency model: {save_error}")
        
        raise


def main():
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
        default=1000,
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
        default=128,
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
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1)"
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
    print(f"- Validation split: {args.validation_split}")
    print(f"- Output directory: {args.output_dir}")
    print()
    
    current_model_path = args.model_path
    
    try:
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
                output_dir=iteration_output_dir,
                validation_split=args.validation_split
            )
            
            print(f"Iteration {iteration+1} complete. Current model: {current_model_path}")
        
        print("\n=== Training Complete ===")
        print(f"Final model saved to: {current_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if current_model_path:
            print(f"Latest model saved at: {current_model_path}")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if current_model_path:
            print(f"Latest model saved at: {current_model_path}")


if __name__ == "__main__":
    main()