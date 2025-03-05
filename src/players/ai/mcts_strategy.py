"""
Monte Carlo Tree Search strategy implementation for Kulibrat AI.
Fixed to handle edge cases with empty child lists and added logging.
"""

import math
import time
import random
import logging
import os
from typing import Optional, Dict, List, Tuple, Any

from src.core.game_state_cy import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.ai.ai_strategy import AIStrategy


# Setup logging
logger = logging.getLogger('mcts')
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler('mcts.log', mode='w')
file_handler.setLevel(logging.DEBUG)

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)

# Make sure the logger propagates to root logger
logger.propagate = False

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
        
        logger.debug(f"Created node with {len(self.untried_moves)} untried moves, player={self.player}")
        if move:
            logger.debug(f"Node created from move: {move}")
    
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
            logger.debug("UCT value is infinity (node not visited)")
            return float('inf')  # Nodes not yet visited have highest priority
        
        # Exploitation term: win rate
        exploitation = self.wins / self.visits
        
        # Exploration term: nodes with fewer visits are favored
        if self.parent and self.parent.visits > 0:
            exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        else:
            exploration = exploration_weight * math.sqrt(1.0 / self.visits)
        
        uct = exploitation + exploration
        logger.debug(f"UCT value: {uct} (wins={self.wins}, visits={self.visits})")
        return uct
    
    def select_child(self) -> Optional['MCTSNode']:
        """
        Select the child with the highest UCT value.
        
        Returns:
            The selected child node or None if no children
        """
        if not self.children:
            logger.warning("Attempted to select child from node with no children")
            return None
        
        best_child = max(self.children, key=lambda child: child.uct_value())
        logger.debug(f"Selected child with move: {best_child.move}")
        return best_child
    
    def expand(self) -> Optional['MCTSNode']:
        """
        Expand the tree by adding a child node for an untried move.
        
        Returns:
            The new child node or None if no untried moves
        """
        if not self.untried_moves:
            logger.debug("Cannot expand: no untried moves")
            return None
        
        # Select a random untried move
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        logger.debug(f"Expanding with move: {move}")
        
        # Create a new state by applying the move
        new_state = self.state.copy()
        success = new_state.apply_move(move)
        
        # If move application failed, return None
        if not success:
            logger.warning(f"Move application failed: {move}")
            return None
        
        # Create and link the new child node
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        
        logger.debug(f"Added child node with move: {move}")
        return child
    
    def update(self, result: float):
        """
        Update node statistics after a simulation.
        
        Args:
            result: Result of the simulation (1 for win, 0 for loss, 0.5 for draw)
        """
        self.visits += 1
        self.wins += result
        logger.debug(f"Updated node: visits={self.visits}, wins={self.wins}, result={result}")
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this state have been tried."""
        is_expanded = len(self.untried_moves) == 0
        logger.debug(f"Node fully expanded: {is_expanded}")
        return is_expanded
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        is_terminal = self.state.is_game_over()
        logger.debug(f"Node is terminal: {is_terminal}")
        return is_terminal

    def has_children(self) -> bool:
        """Check if the node has any children."""
        has_children = len(self.children) > 0
        logger.debug(f"Node has children: {has_children}")
        return has_children


class MCTSStrategy(AIStrategy):
    """
    Monte Carlo Tree Search strategy for Kulibrat.
    This AI makes decisions by building a tree and evaluating moves through random simulations.
    """
    
    def __init__(self, 
                 simulation_time: float = 1.0, 
                 max_iterations: int = 10000,
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
        
        logger.info(f"Initialized MCTS with simulation_time={simulation_time}s, "
                   f"max_iterations={max_iterations}, exploration_weight={exploration_weight}")
    
    def select_move(self, game_state: GameState, player_color: PlayerColor) -> Optional[Move]:
        """
        Select the best move using Monte Carlo Tree Search.
        
        Args:
            game_state: Current state of the game
            player_color: Color of the player making the move
            
        Returns:
            The selected move or None if no valid moves
        """
        logger.info(f"Selecting move for player {player_color}")
        
        # Reset stats for this move selection
        self.stats = {
            "iterations": 0,
            "simulation_time": 0.0,
            "nodes_created": 0,
            "avg_depth": 0.0,
            "max_depth": 0
        }
        
        valid_moves = game_state.get_valid_moves()
        logger.info(f"Found {len(valid_moves)} valid moves: {valid_moves}")
        
        if not valid_moves:
            logger.warning("No valid moves available")
            return None
        
        # If there's only one valid move, return it immediately
        if len(valid_moves) == 1:
            logger.info(f"Only one valid move available: {valid_moves[0]}")
            return valid_moves[0]
        
        # Create root node
        root = MCTSNode(game_state.copy())
        self.stats["nodes_created"] += 1
        
        # Run MCTS algorithm
        start_time = time.time()
        iterations = 0
        depths = []
        
        logger.info(f"Starting MCTS search with time limit: {self.simulation_time}s")
        
        while iterations < self.max_iterations:
            # Check if we've exceeded our time budget
            elapsed = time.time() - start_time
            if elapsed > self.simulation_time:
                logger.info(f"Time budget exceeded: {elapsed:.3f}s > {self.simulation_time}s")
                break
            
            # Phase 1: Selection - traverse the tree to find a node to expand
            node = root
            depth = 0
            
            logger.debug(f"---- Iteration {iterations + 1} ----")
            logger.debug("Starting selection phase")
            
            selection_path = []
            
            while node.is_fully_expanded() and not node.is_terminal():
                selected_child = node.select_child()
                
                if selected_child is None:
                    # No valid children available, break out of the loop
                    logger.warning("Selection phase - no valid child to select")
                    break
                
                node = selected_child
                selection_path.append(str(node.move) if node.move else "root")
                depth += 1
            
            logger.debug(f"Selection path: {' -> '.join(selection_path) if selection_path else 'root only'}")
            logger.debug(f"Selection depth: {depth}")
            
            # Phase 2: Expansion - unless we've reached a terminal state
            logger.debug("Starting expansion phase")
            
            if not node.is_terminal():
                child = node.expand()
                if child:
                    node = child
                    depth += 1
                    self.stats["nodes_created"] += 1
                    logger.debug(f"Expanded node with move: {child.move}")
                else:
                    logger.debug("Expansion failed - no valid moves to try")
            else:
                logger.debug("No expansion needed - reached terminal state")
            
            # Phase 3: Simulation - play out a random game from this point
            logger.debug("Starting simulation phase")
            
            simulation_state = node.state.copy()
            simulation_result = self._simulate(simulation_state, player_color)
            
            logger.debug(f"Simulation result: {simulation_result}")
            
            # Phase 4: Backpropagation - update the statistics up the tree
            logger.debug("Starting backpropagation phase")
            
            backprop_path = []
            current_node = node
            
            while current_node:
                backprop_path.append(str(current_node.move) if current_node.move else "root")
                current_node = current_node.parent
                
            logger.debug(f"Backpropagation path: {' <- '.join(reversed(backprop_path))}")
            
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
            
            if iterations % 100 == 0:
                logger.info(f"Completed {iterations} iterations, elapsed time: {time.time() - start_time:.3f}s")
        
        # Calculate statistics
        self.stats["iterations"] = iterations
        self.stats["simulation_time"] = time.time() - start_time
        self.stats["avg_depth"] = sum(depths) / len(depths) if depths else 0
        self.stats["max_depth"] = max(depths) if depths else 0
        
        logger.info(f"MCTS search completed: {iterations} iterations in {self.stats['simulation_time']:.3f}s")
        
        # Choose the move with the highest win rate
        if not root.has_children():
            # If no simulations were run or no children were created, choose randomly
            selected_move = random.choice(valid_moves)
            logger.warning(f"No children created in search, selecting random move: {selected_move}")
            return selected_move
        
        # Select best child based on visit count (more robust than win ratio)
        best_child = max(root.children, key=lambda c: c.visits)
        selected_move = best_child.move
        
        logger.info(f"Selected best move: {selected_move}")
        logger.info(f"Best move stats: visits={best_child.visits}, wins={best_child.wins}, "
                   f"win_rate={best_child.wins/best_child.visits:.3f}")
        
        # Log statistics
        self._log_stats(root)
        
        return selected_move
    
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
        
        logger.debug("Starting random simulation")
        
        # Play until the game is over or we hit the move limit
        while not simulation_state.is_game_over() and move_count < max_moves:
            # Get valid moves for the current player
            valid_moves = simulation_state.get_valid_moves()
            
            if not valid_moves:
                # If no valid moves, switch to the other player
                logger.debug(f"No valid moves for {simulation_state.current_player}, switching players")
                simulation_state.current_player = simulation_state.current_player.opposite()
                continue
            
            # Select a random move
            move = random.choice(valid_moves)
            
            # Apply the move
            success = simulation_state.apply_move(move)
            if not success:
                logger.warning(f"Failed to apply move in simulation: {move}")
                # If move application failed, try another move or end simulation
                continue
                
            logger.debug(f"Simulation move {move_count + 1}: {move}")
            
            # The current player is changed in apply_move if needed
            move_count += 1
        
        # Evaluate the final state
        if not simulation_state.is_game_over():
            # If we hit the move limit without ending, consider it a draw
            logger.debug(f"Simulation hit move limit ({max_moves} moves) - considering as draw")
            return 0.5
        
        winner = simulation_state.get_winner()
        
        if winner == player_color:
            logger.debug(f"Simulation result: Win for {player_color}")
            return 1.0  # Win
        elif winner is None:
            logger.debug("Simulation result: Draw")
            return 0.5  # Draw
        else:
            logger.debug(f"Simulation result: Loss (winner is {winner})")
            return 0.0  # Loss
    
    def _log_stats(self, root: MCTSNode):
        """Log statistics about the search."""
        logger.info("\n=== MCTS Stats ===")
        logger.info(f"Iterations: {self.stats['iterations']}")
        logger.info(f"Time: {self.stats['simulation_time']:.3f}s")
        logger.info(f"Nodes created: {self.stats['nodes_created']}")
        logger.info(f"Average tree depth: {self.stats['avg_depth']:.2f}")
        logger.info(f"Maximum tree depth: {self.stats['max_depth']}")
        
        # Print top moves with their statistics
        logger.info("\nTop moves:")
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        for i, child in enumerate(sorted_children[:min(5, len(sorted_children))]):
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            logger.info(f"{i+1}. {child.move} - Visits: {child.visits}, Win rate: {win_rate:.2f}")
        
        # Also print to console for convenience
        print("\n=== MCTS Stats ===")
        print(f"Iterations: {self.stats['iterations']}")
        print(f"Time: {self.stats['simulation_time']:.3f}s")
        print(f"Nodes created: {self.stats['nodes_created']}")
        print(f"Average tree depth: {self.stats['avg_depth']:.2f}")
        print(f"Maximum tree depth: {self.stats['max_depth']}")
        
        # Print top moves with their statistics
        print("\nTop moves:")
        for i, child in enumerate(sorted_children[:min(5, len(sorted_children))]):
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            print(f"{i+1}. {child.move} - Visits: {child.visits}, Win rate: {win_rate:.2f}")
        print("")