"""
Optimized AlphaZero-style search strategy for Kulibrat.
Combines neural network evaluation with Monte Carlo Tree Search.
"""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from functools import lru_cache

import numpy as np
import torch

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy
from src.players.ai.rl_model import AttentionKulibratNet as KulibratNet
from src.players.ai.rl_model import encode_board_for_attention as encode_board


class MCTSNode:
    """Node in the AlphaZero MCTS tree."""
    
    __slots__ = ('prior_p', 'children', 'move_keys', 'visit_count', 'value_sum', 'is_expanded')

    def __init__(self, prior_p: float):
        """
        Initialize a node with prior probability from policy network.

        Args:
            prior_p: Prior probability of selecting this node
        """
        self.prior_p = prior_p  # Prior probability from policy network
        self.children: Dict[str, "MCTSNode"] = {}  # Child nodes (indexed by move key)
        self.move_keys: Dict[str, Move] = {}  # Mapping from move keys to Move objects
        self.visit_count = 0  # Number of visits
        self.value_sum = 0.0  # Sum of values from this node
        self.is_expanded = False  # Whether this node has been expanded

    def expanded(self) -> bool:
        """Check if the node has been expanded."""
        return self.is_expanded

    def value(self) -> float:
        """Get the value estimate of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct: float, sum_visits: Optional[int] = None) -> Tuple[str, "MCTSNode", Move]:
        """
        Select the child with the highest UCT value.

        Args:
            c_puct: Exploration constant
            sum_visits: Pre-calculated sum of visits (optional)

        Returns:
            A tuple of (move_key, child_node, move_object)
        """
        # Select the child with the highest UCT value
        best_score = -float("inf")
        best_move_key = None
        best_child = None

        # Calculate the sum of visit counts across all children
        if sum_visits is None:
            sum_visits = sum(child.visit_count for child in self.children.values())
        
        # Pre-compute square root for efficiency
        sqrt_sum_visits = math.sqrt(sum_visits + 1e-8)

        for move_key, child in self.children.items():
            # Calculate UCT value
            if child.visit_count > 0:
                # Exploitation term
                q_value = child.value()

                # Exploration term (AlphaZero's formula)
                u_value = (
                    c_puct
                    * child.prior_p
                    * sqrt_sum_visits
                    / (1 + child.visit_count)
                )

                # UCT value
                uct_value = q_value + u_value
            else:
                # If not visited, prioritize based on prior and exploration constant
                uct_value = c_puct * child.prior_p * sqrt_sum_visits

            # Update best move if this one has a higher UCT value
            if uct_value > best_score:
                best_score = uct_value
                best_move_key = move_key
                best_child = child

        if best_move_key is None:
            # This should not happen in normal circumstances
            raise ValueError("No child found in select_child")

        return best_move_key, best_child, self.move_keys[best_move_key]

    def expand(self, move_probs: Dict[Move, float], move_to_key: Callable):
        """
        Expand the node with children according to move probabilities.

        Args:
            move_probs: Dictionary mapping moves to probabilities
            move_to_key: Function to convert Move objects to string keys
        """
        for move, prob in move_probs.items():
            move_key = move_to_key(move)
            if move_key not in self.children:
                self.children[move_key] = MCTSNode(prior_p=prob)
                self.move_keys[move_key] = move
        self.is_expanded = True

    def update(self, value: float):
        """
        Update node statistics after a simulation.

        Args:
            value: The value to update with
        """
        self.visit_count += 1
        self.value_sum += value


class AlphaZeroStrategy(AIStrategy):
    """
    AI strategy using AlphaZero-style MCTS with neural network guidance.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_simulations: int = 800,
        c_puct: float = 4.0,
        exploration_rate: float = 0.0,
        temperature: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,  # Added batch size parameter
        cache_size: int = 10000,  # Added cache size parameter
    ):
        """
        Initialize the AlphaZero strategy.

        Args:
            model_path: Path to the trained model file (None for random play)
            n_simulations: Number of MCTS simulations per move
            c_puct: Exploration constant for MCTS
            exploration_rate: Probability of selecting a random move
            temperature: Temperature for move selection (higher -> more exploration)
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Size of batches for network inference
            cache_size: Size of the LRU cache for network evaluations
        """
        self.device = device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.exploration_rate = exploration_rate
        self.temperature = temperature
        self.batch_size = batch_size
        self.root = None
        
        # Caches for better performance
        self.encoded_state_cache = {}  # Cache for encoded states
        self.move_key_cache = {}  # Cache for move keys
        
        # Batch inference queues
        self.inference_queue = []  # States waiting for neural network evaluation
        self.inference_nodes = []  # Corresponding nodes
        self.inference_states = []  # Valid moves for each state

        # Initialize the model
        self.model = KulibratNet().to(device)

        # Load pretrained model if provided
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path} to {device}")
        else:
            print("No model loaded. Using random initialization.")

        # Set model to evaluation mode
        self.model.eval()

        # Action encoding/decoding maps
        self.action_to_move = {}  # Maps action indices to corresponding moves
        self.move_to_action = {}  # Maps move descriptions to action indices
        
        # Set up LRU cache for neural network evaluations
        self._evaluate_state = lru_cache(maxsize=cache_size)(self._evaluate_state_impl)

    def select_move(
        self, game_state: GameState, player_color: PlayerColor
    ) -> Optional[Move]:
        """
        Select a move using AlphaZero-style MCTS.

        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move

        Returns:
            The selected move or None if no valid moves
        """
        start_time = time.time()
        
        if game_state.current_player != player_color:
            return None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None

        # With some probability, choose a random move (exploration)
        if random.random() < self.exploration_rate:
            return random.choice(valid_moves)

        # Build the action mapping
        self._build_action_mapping(game_state, valid_moves)

        # Use MCTS to select the best move
        if self.root is None:
            self.root = MCTSNode(prior_p=1.0)
        
        # Clear caches for new search
        self.encoded_state_cache = {}
        self.inference_queue = []
        self.inference_nodes = []
        self.inference_states = []

        # Run MCTS simulations
        for i in range(self.n_simulations):
            state_copy = game_state.copy()
            visited_states = set()  # Track visited states to detect loops
            self._search(state_copy, self.root, visited_states)
            
            # Process any remaining states in the queue on the last iteration
            if i == self.n_simulations - 1 and self.inference_queue:
                self._process_inference_batch()

        # Optimization for zero temperature: directly select most visited move
        if self.temperature <= 0.01:
            max_visits = 0
            best_move = None
            
            for move_key, child in self.root.children.items():
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    best_move = self.root.move_keys[move_key]
            
            # Reset the search tree for the next move
            self.root = None
            
            print(f"Move selection took {time.time() - start_time:.3f} seconds")
            return best_move

        # Select move based on visit counts and temperature for non-zero temperature
        move_visits = {}
        for move_key, child in self.root.children.items():
            move = self.root.move_keys[move_key]
            # Apply temperature
            visits = child.visit_count ** (1.0 / self.temperature)
            move_visits[move] = visits

        # Convert visits to probabilities using numpy for better performance
        total_visits = sum(move_visits.values())
        if total_visits == 0:
            # Fallback to random selection if no visits
            return random.choice(valid_moves)

        # Use numpy arrays for faster operations
        moves = list(move_visits.keys())
        probs = np.array([move_visits[move] for move in moves])
        probs = probs / probs.sum()  # Normalize probabilities

        # Use numpy's choice function for weighted random selection
        selected_idx = np.random.choice(len(moves), p=probs)
        selected_move = moves[selected_idx]

        # Reset the search tree for the next move
        self.root = None
        
        print(f"Move selection took {time.time() - start_time:.3f} seconds")
        return selected_move

    def _search(self, game_state: GameState, node: MCTSNode, visited_states: Set[str]) -> float:
        """
        Run a single MCTS simulation from the given node.

        Args:
            game_state: Current game state
            node: Current node in the search tree
            visited_states: Set of state string representations visited in this path

        Returns:
            The value of the leaf node (from the perspective of the current player)
        """
        # Check for repeated states (loop detection)
        state_key = self._get_state_key(game_state)
        if state_key in visited_states:
            return 0.0  # Draw value for loops
        
        # Add current state to visited states
        visited_states.add(state_key)
        
        # Check if game is over
        if game_state.is_game_over():
            # Game is over, determine value based on winner
            winner = game_state.get_winner()
            if winner is None:
                # Draw
                value = 0.0
            else:
                # Win or loss from the perspective of the current player
                value = 1.0 if winner == game_state.current_player else -1.0

            # Update the node
            node.update(value)
            return value  # Return value from the perspective of the current player

        # If node is not expanded, expand it
        if not node.expanded():
            # Get valid moves
            valid_moves = game_state.get_valid_moves()

            if not valid_moves:
                # No valid moves, return a neutral value
                node.update(0.0)
                return 0.0

            # Add to batch queue for neural network evaluation
            self.inference_queue.append(game_state)
            self.inference_nodes.append(node)
            self.inference_states.append(valid_moves)
            
            # Process batch if it's full
            if len(self.inference_queue) >= self.batch_size:
                self._process_inference_batch()
                
            # If node wasn't expanded (not processed in batch yet), evaluate it directly
            if not node.expanded():
                self._evaluate_and_expand_node(game_state, node, valid_moves)
            
            # Return negated value (from opponent's perspective)
            return -node.value()

        # Node is already expanded, select the best child node
        # Pre-calculate sum of visits for efficiency
        sum_visits = sum(child.visit_count for child in node.children.values())
        move_key, child_node, move = node.select_child(self.c_puct, sum_visits)

        # Apply the move
        success = game_state.apply_move(move)

        if not success:
            # Move application failed, return a neutral value
            return 0.0

        # Recursively search from the child node
        value = self._search(game_state, child_node, visited_states)

        # Update the current node
        node.update(-value)  # Negate value because it's from the opponent's perspective

        return value  # Return value from the current player's perspective
    
    def _process_inference_batch(self):
        """Process a batch of states for neural network inference."""
        if not self.inference_queue:
            return
        
        try:
            # Create batch of encoded states
            batch_encoded_states = []
            for state in self.inference_queue:
                # Get cached encoding or create new one
                state_key = self._get_state_key(state)
                if state_key in self.encoded_state_cache:
                    encoded = self.encoded_state_cache[state_key]
                else:
                    # Encode board without adding extra dimensions
                    encoded = encode_board(state.board, state.current_player.value)
                    # Ensure encoded state has correct shape before caching
                    if len(encoded.shape) > 3:
                        # Squeeze any extra dimensions if needed
                        encoded = encoded.squeeze(0)
                    self.encoded_state_cache[state_key] = encoded
                batch_encoded_states.append(encoded)
            
            # Stack tensors into batch and move to device
            # Make sure each tensor in batch_encoded_states has the same shape
            shapes = [t.shape for t in batch_encoded_states]
            if not all(s == shapes[0] for s in shapes):
                raise ValueError(f"Inconsistent tensor shapes in batch: {shapes}")
            
            batch_tensor = torch.stack(batch_encoded_states).to(self.device)
            
            # Check shape before passing to model
            if len(batch_tensor.shape) == 4:
                # For attention models, reshape to match expected dimensions 
                # (batch_size, seq_len, features)
                batch_size = batch_tensor.shape[0]
                batch_tensor = batch_tensor.view(batch_size, -1, batch_tensor.shape[-1])
            
            # Run batch inference
            with torch.no_grad():
                policy_logits, values = self.model(batch_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
                values = values.cpu().numpy().flatten()
            
            # Process each state in the batch
            for i in range(len(self.inference_queue)):
                state = self.inference_queue[i]
                node = self.inference_nodes[i]
                valid_moves = self.inference_states[i]
                probs = policy_probs[i]
                value = values[i]
                
                # Create move probabilities
                move_probs = {}
                for move in valid_moves:
                    move_key = self._move_to_key(move)
                    if move_key in self.move_to_action:
                        action_idx = self.move_to_action[move_key]
                        
                        # Ensure index is within bounds
                        if action_idx < len(probs):
                            prob = probs[action_idx]
                        else:
                            # Fallback to uniform distribution
                            prob = 1.0 / len(valid_moves)
                    else:
                        # Fallback to uniform distribution
                        prob = 1.0 / len(valid_moves)
                    
                    move_probs[move] = prob
                
                # Normalize probabilities
                total_prob = sum(move_probs.values())
                if total_prob > 0:
                    move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
                else:
                    # Fallback to uniform distribution
                    move_probs = {move: 1.0 / len(valid_moves) for move in valid_moves}
                
                # Expand the node
                node.expand(move_probs, self._move_to_key)
                
                # Update the node with the value
                node.update(float(value))
        
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Fall back to individual processing of states
            for i in range(len(self.inference_queue)):
                self._evaluate_and_expand_node(
                    self.inference_queue[i], 
                    self.inference_nodes[i], 
                    self.inference_states[i]
                )
        
        # Clear the batch data
        self.inference_queue.clear()
        self.inference_nodes.clear()
        self.inference_states.clear()
    
    def _evaluate_and_expand_node(self, state: GameState, node: MCTSNode, valid_moves: List[Move]):
        """
        Evaluate a single state and expand the node.
        
        Args:
            state: Game state to evaluate
            node: Node to expand
            valid_moves: List of valid moves
        """
        # Use cached evaluation function
        state_key = self._get_state_key(state)
        policy_probs, value = self._evaluate_state(state_key, state)
        
        # Create move probabilities
        move_probs = {}
        for move in valid_moves:
            move_key = self._move_to_key(move)
            if move_key in self.move_to_action:
                action_idx = self.move_to_action[move_key]
                
                # Ensure index is within bounds
                if action_idx < len(policy_probs):
                    prob = policy_probs[action_idx]
                else:
                    # Fallback to uniform distribution
                    prob = 1.0 / len(valid_moves)
            else:
                # Fallback to uniform distribution
                prob = 1.0 / len(valid_moves)
            
            move_probs[move] = prob
        
        # Normalize probabilities
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            # Fallback to uniform distribution
            move_probs = {move: 1.0 / len(valid_moves) for move in valid_moves}
        
        # Expand the node
        node.expand(move_probs, self._move_to_key)
        
        # Update the node
        node.update(value)
    
    def _evaluate_state_impl(self, state_key: str, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Implementation for the cached state evaluation function.
        
        Args:
            state_key: String key for the state
            state: Game state to evaluate
            
        Returns:
            Tuple of (policy_probs, value)
        """
        try:
            # Encode the state
            encoded_state = encode_board(state.board, state.current_player.value)
            
            # Ensure shape is correct for attention model
            if len(encoded_state.shape) == 4:
                # Reshape to 3D for attention (sequence_length, features)
                encoded_state = encoded_state.squeeze(0)
            
            # For a single state, we may need to add batch dimension
            if len(encoded_state.shape) == 2:
                encoded_state = encoded_state.unsqueeze(0)
            
            # Move to device
            encoded_state = encoded_state.to(self.device)
            
            # Run neural network inference
            with torch.no_grad():
                policy_logits, value_tensor = self.model(encoded_state)
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy().flatten()
                value = value_tensor.item()
            
            return policy_probs, value
            
        except Exception as e:
            print(f"Error in state evaluation: {e}")
            # Return uniform policy and neutral value as fallback
            valid_moves = state.get_valid_moves()
            num_actions = 51  # Same as in original code
            policy = np.ones(num_actions) / num_actions
            return policy, 0.0

    def _build_action_mapping(
        self, game_state: GameState, valid_moves: List[Move]
    ) -> None:
        """
        Build a mapping between neural network actions and valid game moves.

        Args:
            game_state: Current game state
            valid_moves: List of valid moves
        """
        self.action_to_move = {}
        self.move_to_action = {}

        # Map each valid move to an action index
        action_idx = 0

        # Process each valid move
        for move in valid_moves:
            move_key = self._move_to_key(move)

            # Only add if not already mapped
            if move_key not in self.move_to_action:
                self.action_to_move[action_idx] = (
                    move.move_type,
                    move.start_pos,
                    move.end_pos,
                )
                self.move_to_action[move_key] = action_idx
                action_idx += 1

    def _move_to_key(self, move: Move) -> str:
        """
        Convert a Move object to a string key with caching.

        Args:
            move: The Move object

        Returns:
            String key representing the move
        """
        # Check if already in cache
        move_id = id(move)
        if move_id in self.move_key_cache:
            return self.move_key_cache[move_id]
        
        # Create new key if not in cache
        move_type = move.move_type.name
        start_pos = (
            "None"
            if move.start_pos is None
            else f"{move.start_pos[0]},{move.start_pos[1]}"
        )
        end_pos = (
            "None" 
            if move.end_pos is None 
            else f"{move.end_pos[0]},{move.end_pos[1]}"
        )

        key = f"{move_type}:{start_pos}:{end_pos}"
        self.move_key_cache[move_id] = key
        return key
    
    def _get_state_key(self, state: GameState) -> str:
        """
        Get a unique string key for a game state.
        
        Args:
            state: The game state
            
        Returns:
            String key representing the state
        """
        # Convert board to string representation and combine with player
        board_str = np.array2string(state.board, separator=',')
        return f"{board_str}:{state.current_player.value}"