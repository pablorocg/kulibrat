"""
Classic Monte Carlo Tree Search strategy for Kulibrat with parallelized simulations.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from src.players.mcts_player.mcts_node import MCTSNode
from src.core import GameState, Move, PlayerColor
from src.players.player import Player



class MCTSPlayer(Player):
    """
    Classic Monte Carlo Tree Search strategy for Kulibrat with:
    - No domain-specific knowledge or heuristics
    - Parallel simulations for better performance
    - Tree reuse between moves
    """
    
    def __init__(self, 
                 simulation_time: float = 2.0,
                 max_iterations: int = 15000,
                 exploration_weight: float = 1.41,
                 num_threads: int = 4):
        """Initialize the parallelized classic MCTS strategy."""
        super().__init__(name="MCTS Player", color=None)
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.num_threads = num_threads
        self.previous_root = None  # For tree reuse
        self.stats = {}
    
    def select_move(self, game_state: GameState, player_color: PlayerColor) -> Optional[Move]:
        """Select the best move using parallelized classic MCTS."""
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Try to reuse the tree from previous search
        root = self._try_reuse_tree(game_state)
        if not root:
            root = MCTSNode(game_state.copy())
        
        # Run MCTS algorithm with parallel simulations
        start_time = time.time()
        iterations = 0
        
        # Pre-expand root for parallel processing
        if not root.is_fully_expanded() and not root.is_terminal():
            for _ in range(min(len(root.untried_moves), 8)):  # Expand up to 8 children initially
                root.expand()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_node = {}
            
            while iterations < self.max_iterations:
                # Check if we've exceeded our time budget
                if time.time() - start_time > self.simulation_time:
                    break
                
                # Use available threads for parallel simulations
                while len(future_to_node) < self.num_threads and iterations < self.max_iterations:
                    # Phase 1 & 2: Selection & Expansion (must be serial)
                    selected_node = self._select_and_expand(root)
                    
                    # Phase 3: Submit simulation to thread pool
                    future = executor.submit(
                        self._simulate, 
                        selected_node.state.copy(), 
                        player_color
                    )
                    future_to_node[future] = selected_node
                    iterations += 1
                
                # Process completed simulations
                for future in list(as_completed(future_to_node)):
                    node = future_to_node.pop(future)
                    result = future.result()
                    
                    # Phase 4: Backpropagation
                    self._backpropagate(node, result)
            
            # Wait for any remaining simulations to complete
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                result = future.result()
                self._backpropagate(node, result)
        
        # Choose the move with the highest visit count
        if not root.children:
            return random.choice(valid_moves)
        
        # Select the best child based primarily on visits
        best_child = max(root.children, key=lambda c: c.visits)
        
        # Store the tree for potential reuse
        self.previous_root = best_child
        
        # Log statistics
        self.stats = {
            "iterations": iterations,
            "time": time.time() - start_time,
            "best_move_visits": best_child.visits,
            "best_move_win_rate": best_child.wins/best_child.visits if best_child.visits > 0 else 0
        }
        logger.info(f"MCTS completed {iterations} iterations in {self.stats['time']:.2f}s")
        
        return best_child.move
    
    
    
    def _try_reuse_tree(self, game_state: GameState) -> Optional[MCTSNode]:
        """Try to reuse the search tree from a previous move."""
        if not self.previous_root:
            return None
        
        # Check if any child of the previous root matches the current state
        for child in self.previous_root.children:
            if self._states_equivalent(child.state, game_state):
                # Found a matching state - reuse this subtree
                child.parent = None  # Detach from previous parent
                return child
        
        return None
    
    def _states_equivalent(self, state1: GameState, state2: GameState) -> bool:
        """Check if two game states are equivalent (ignoring move history)."""
        # This is a simplified version - enhance based on GameState implementation
        if state1.current_player != state2.current_player:
            return False
        
        # Check if the board configurations match
        if not np.array_equal(state1.board, state2.board):
            return False
            
        # Check if scores match
        if state1.scores != state2.scores:
            return False
            
        return True
    
    def _select_and_expand(self, root: MCTSNode) -> MCTSNode:
        """Combined selection and expansion phase."""
        node = root
        
        # Phase 1: Selection - traverse the tree to find a node to expand
        while node.is_fully_expanded() and not node.is_terminal():
            child = node.select_child()
            if child is None:
                break
            node = child
        
        # Phase 2: Expansion - unless we've reached a terminal state
        if not node.is_terminal():
            child = node.expand()
            if child:
                node = child
        
        return node
    
    def _simulate(self, state: GameState, player_color: PlayerColor) -> float:
        """Run a completely random simulation (classic MCTS)."""
        simulation_state = state  # No need to copy, we already have a copy
        max_moves = 80  # Prevent infinite loops
        move_count = 0
        
        # Play until the game is over or we hit the move limit
        while not simulation_state.is_game_over() and move_count < max_moves:
            valid_moves = simulation_state.get_valid_moves()
            
            if not valid_moves:
                # If no valid moves, the rules of Kulibrat say other player gets extra moves
                simulation_state.current_player = simulation_state.current_player.opposite()
                continue
            
            # Pure random move selection
            move = random.choice(valid_moves)
            
            if not simulation_state.apply_move(move):
                # If move application failed, try another move
                continue
                
            move_count += 1
        
        # Evaluate the final state
        return self._evaluate_terminal(simulation_state, player_color)
    
    def _evaluate_terminal(self, state: GameState, player_color: PlayerColor) -> float:
        """Evaluate a terminal state with a simple win/draw/loss result."""
        if state.is_game_over():
            winner = state.get_winner()
            if winner == player_color:
                return 1.0  # Win
            elif winner is None:
                return 0.5  # Draw
            else:
                return 0.0  # Loss
        
        # If we reached move limit but game isn't over, use score difference
        # to determine a winner, scaled to [0, 1]
        opponent_color = player_color.opposite()
        player_score = state.scores[player_color]
        opponent_score = state.scores[opponent_color]
        
        # If one player has a higher score, they win
        if player_score > opponent_score:
            return 1.0
        elif opponent_score > player_score:
            return 0.0
        else:
            return 0.5  # Draw
    
    def _backpropagate(self, node: MCTSNode, result: float):
        """Update statistics for all nodes in the path from leaf to root."""
        current = node
        while current:
            # If this is from the perspective of the maximizing player
            if current.player == node.player:
                current.update(result)
            else:
                # For the opponent's moves, we invert the result
                current.update(1.0 - result)
            current = current.parent