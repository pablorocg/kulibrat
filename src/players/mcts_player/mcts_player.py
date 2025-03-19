"""
Classic Monte Carlo Tree Search strategy for Kulibrat with parallelized simulations.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
from src.players.mcts_player.mcts_node import MCTSNode
from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.player import Player


class MCTSPlayer(Player):
    """
    Classic Monte Carlo Tree Search strategy for Kulibrat with:
    - No domain-specific knowledge or heuristics
    - Parallel simulations for better performance
    - Tree reuse between moves
    """
    
    def __init__(self, 
                 color: PlayerColor,
                 simulation_time: float = 2.0,
                 max_iterations: int = 15000,
                 exploration_weight: float = 1.41,
                 num_threads: int = 4):
        """Initialize the parallelized classic MCTS strategy."""
        super().__init__(color, name="MCTS Player")
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight
        self.num_threads = num_threads
        self.previous_root = None  # For tree reuse
        self.stats = {}
    
    def get_move(self, game_state: GameState) -> Optional[Move]:
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
        
        start_time = time.time()
        iterations = 0
        
        # Pre-expand root for parallel processing
        if not root.is_fully_expanded() and not root.is_terminal():
            for _ in range(min(len(root.untried_moves), 8)):  # Expand up to 8 children initially
                root.expand()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_node = {}
            
            while iterations < self.max_iterations:
                if time.time() - start_time > self.simulation_time:
                    break
                
                while len(future_to_node) < self.num_threads and iterations < self.max_iterations:
                    selected_node = self._select_and_expand(root)
                    future = executor.submit(
                        self._simulate, 
                        selected_node.state.copy(), 
                        self.color
                    )
                    future_to_node[future] = selected_node
                    iterations += 1
                
                for future in list(as_completed(future_to_node)):
                    node = future_to_node.pop(future)
                    result = future.result()
                    self._backpropagate(node, result)
            
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                result = future.result()
                self._backpropagate(node, result)
        
        if not root.children:
            return random.choice(valid_moves)
        
        best_child = max(root.children, key=lambda c: c.visits)
        
        self.previous_root = best_child
        
        self.stats = {
            "iterations": iterations,
            "time": time.time() - start_time,
            "best_move_visits": best_child.visits,
            "best_move_win_rate": best_child.wins / best_child.visits if best_child.visits > 0 else 0
        }
        
        return best_child.move
    
    def _try_reuse_tree(self, game_state: GameState) -> Optional[MCTSNode]:
        """Try to reuse the search tree from a previous move."""
        if not self.previous_root:
            return None
        
        for child in self.previous_root.children:
            if self._states_equivalent(child.state, game_state):
                child.parent = None  # Detach from previous parent
                return child
        
        return None
    
    def _states_equivalent(self, state1: GameState, state2: GameState) -> bool:
        """Check if two game states are equivalent (ignoring move history)."""
        if state1.current_player != state2.current_player:
            return False
        
        if not np.array_equal(state1.board, state2.board):
            return False
            
        if state1.scores != state2.scores:
            return False
            
        return True
    
    def _select_and_expand(self, root: MCTSNode) -> MCTSNode:
        """Combined selection and expansion phase."""
        node = root
        
        # Selection: traverse the tree to find a node to expand
        while node.is_fully_expanded() and not node.is_terminal():
            child = node.select_child()
            if child is None:
                break
            node = child
        
        # Expansion: unless we've reached a terminal state
        if not node.is_terminal():
            child = node.expand()
            if child:
                node = child
        
        return node
    
    def _simulate(self, state: GameState, player_color: PlayerColor) -> float:
        """Run a completely random simulation (classic MCTS)."""
        simulation_state = state
        max_moves = 80  # Prevent infinite loops
        move_count = 0
        
        while not simulation_state.is_game_over() and move_count < max_moves:
            valid_moves = simulation_state.get_valid_moves()
            
            if not valid_moves:
                simulation_state.current_player = simulation_state.current_player.opposite()
                continue
            
            move = random.choice(valid_moves)
            
            if not simulation_state.apply_move(move):
                continue
                
            move_count += 1
        
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
        
        opponent_color = player_color.opposite()
        player_score = state.scores[player_color]
        opponent_score = state.scores[opponent_color]
        
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
            if current.player == node.player:
                current.update(result)
            else:
                current.update(1.0 - result)
            current = current.parent
