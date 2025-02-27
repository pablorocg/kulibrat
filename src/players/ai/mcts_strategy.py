"""
Monte Carlo Tree Search strategy implementation for Kulibrat AI.
"""

import math
import time
import random
from typing import Optional, Dict, List, Tuple, Any

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    Each node represents a game state and contains statistics about simulations run through it.
    """
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, move: Optional[Move] = None):
        """
        Initialize a new node.
        
        Args:
            state: The game state this node represents
            parent: The parent node (None for root)
            move: The move that led to this state from the parent
        """
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this state
        self.children: List['MCTSNode'] = []
        self.wins = 0
        self.visits = 0
        self.untried_moves: List[Move] = state.get_valid_moves()
        self.player = state.current_player  # Player who made the move leading to this state
    
    def uct_value(self, exploration_weight: float = 1.0) -> float:
        """
        Calculate the UCT value of this node.
        UCT balances exploitation (win rate) with exploration (nodes with few visits).
        
        Args:
            exploration_weight: Controls the balance between exploration and exploitation
            
        Returns:
            The UCT value
        """
        if self.visits == 0:
            return float('inf')  # Nodes not yet visited have highest priority
        
        # Exploitation term: win rate
        exploitation = self.wins / self.visits
        
        # Exploration term: nodes with fewer visits are favored
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def select_child(self) -> 'MCTSNode':
        """
        Select the child with the highest UCT value.
        
        Returns:
            The selected child node
        """
        return max(self.children, key=lambda child: child.uct_value())
    
    def expand(self) -> Optional['MCTSNode']:
        """
        Expand the tree by adding a child node for an untried move.
        
        Returns:
            The new child node or None if no untried moves
        """
        if not self.untried_moves:
            return None
        
        # Select a random untried move
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        # Create a new state by applying the move
        new_state = self.state.copy()
        new_state.apply_move(move)
        
        # Create and link the new child node
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        
        return child
    
    def update(self, result: float):
        """
        Update node statistics after a simulation.
        
        Args:
            result: Result of the simulation (1 for win, 0 for loss, 0.5 for draw)
        """
        self.visits += 1
        self.wins += result
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this state have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        return self.state.is_game_over()


class MCTSStrategy(AIStrategy):
    """
    Monte Carlo Tree Search strategy for Kulibrat.
    This AI makes decisions by building a tree and evaluating moves through random simulations.
    """
    
    def __init__(self, 
                 simulation_time: float = 1.0, 
                 max_iterations: int = 1000,
                 exploration_weight: float = 1.41):
        """
        Initialize the MCTS strategy.
        
        Args:
            simulation_time: Maximum time in seconds for running simulations
            max_iterations: Maximum number of MCTS iterations to run
            exploration_weight: Controls the trade-off between exploration and exploitation
        """
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.stats = {
            "iterations": 0,
            "simulation_time": 0.0,
            "nodes_created": 0,
            "avg_depth": 0.0,
            "max_depth": 0
        }
    
    def select_move(self, game_state: GameState, player_color: PlayerColor) -> Optional[Move]:
        """
        Select the best move using Monte Carlo Tree Search.
        
        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move
            
        Returns:
            The selected move or None if no valid moves
        """
        # Reset stats for this move selection
        self.stats = {
            "iterations": 0,
            "simulation_time": 0.0,
            "nodes_created": 0,
            "avg_depth": 0.0,
            "max_depth": 0
        }
        
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        
        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Create root node
        root = MCTSNode(game_state.copy())
        self.stats["nodes_created"] += 1
        
        # Run MCTS algorithm
        start_time = time.time()
        iterations = 0
        depths = []
        
        while iterations < self.max_iterations:
            # Check if we've exceeded our time budget
            if time.time() - start_time > self.simulation_time:
                break
            
            # Phase 1: Selection - traverse the tree to find a node to expand
            node = root
            depth = 0
            
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child()
                depth += 1
            
            # Phase 2: Expansion - unless we've reached a terminal state
            if not node.is_terminal():
                child = node.expand()
                if child:
                    node = child
                    depth += 1
                    self.stats["nodes_created"] += 1
            
            # Phase 3: Simulation - play out a random game from this point
            simulation_state = node.state.copy()
            simulation_result = self._simulate(simulation_state, player_color)
            
            # Phase 4: Backpropagation - update the statistics up the tree
            while node:
                # If this is from the perspective of the player we want to maximize
                if node.player == player_color:
                    node.update(simulation_result)
                else:
                    # For the opponent's moves, we invert the result
                    node.update(1.0 - simulation_result)
                node = node.parent
            
            iterations += 1
            depths.append(depth)
        
        # Calculate statistics
        self.stats["iterations"] = iterations
        self.stats["simulation_time"] = time.time() - start_time
        self.stats["avg_depth"] = sum(depths) / len(depths) if depths else 0
        self.stats["max_depth"] = max(depths) if depths else 0
        
        # Choose the move with the highest win rate
        if not root.children:
            # If no simulations were run, choose randomly
            return random.choice(valid_moves)
        
        # Select best child based on visit count (more robust than win ratio)
        best_child = max(root.children, key=lambda c: c.visits)
        
        # Log statistics
        self._log_stats(root)
        
        return best_child.move
    
    def _simulate(self, state: GameState, player_color: PlayerColor) -> float:
        """
        Run a random simulation from the given state to determine an outcome.
        
        Args:
            state: The game state to simulate from
            player_color: The player to evaluate from (we want to know if this player wins)
            
        Returns:
            1.0 for a win, 0.0 for a loss, 0.5 for a draw
        """
        # Make a copy to avoid modifying the original state
        simulation_state = state.copy()
        max_moves = 100  # Prevent infinite loop in case of a bug
        move_count = 0
        
        # Play until the game is over or we hit the move limit
        while not simulation_state.is_game_over() and move_count < max_moves:
            # Get valid moves for the current player
            valid_moves = simulation_state.get_valid_moves()
            
            if not valid_moves:
                # If no valid moves, switch to the other player
                simulation_state.current_player = simulation_state.current_player.opposite()
                continue
            
            # Select a random move
            move = random.choice(valid_moves)
            
            # Apply the move
            simulation_state.apply_move(move)
            
            # The current player is changed in apply_move if needed
            move_count += 1
        
        # Evaluate the final state
        if not simulation_state.is_game_over():
            # If we hit the move limit without ending, consider it a draw
            return 0.5
        
        winner = simulation_state.get_winner()
        
        if winner == player_color:
            return 1.0  # Win
        elif winner is None:
            return 0.5  # Draw
        else:
            return 0.0  # Loss
    
    def _log_stats(self, root: MCTSNode):
        """Log statistics about the search."""
        print("\n=== MCTS Stats ===")
        print(f"Iterations: {self.stats['iterations']}")
        print(f"Time: {self.stats['simulation_time']:.3f}s")
        print(f"Nodes created: {self.stats['nodes_created']}")
        print(f"Average tree depth: {self.stats['avg_depth']:.2f}")
        print(f"Maximum tree depth: {self.stats['max_depth']}")
        
        # Print top moves with their statistics
        print("\nTop moves:")
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        for i, child in enumerate(sorted_children[:min(5, len(sorted_children))]):
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            print(f"{i+1}. {child.move} - Visits: {child.visits}, Win rate: {win_rate:.2f}")
        print("")