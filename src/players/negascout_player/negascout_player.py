"""
NegaScout (Principal Variation Search) implementation for Kulibrat AI.
This player extends the Minimax player with a more efficient search algorithm.
"""

import time
from typing import Dict, List, Optional, Tuple

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.minimax_player.heuristics import HeuristicRegistry
from src.players.player import Player


class NegaScoutPlayer(Player):
    """
    NegaScout (Principal Variation Search) implementation for Kulibrat.
    NegaScout is an enhanced version of alpha-beta pruning that uses a "scout" search
    with a null window to quickly determine if a node might fail high, potentially
    saving significant search time.
    """

    def __init__(
        self,
        color: PlayerColor,
        name: str = "NegaScout Player",
        time_limit: float = 1.0,
        max_depth: int = 20,
        heuristic: str = "score_diff",
        tt_size: int = 500000,
    ):
        """
        Initialize the NegaScout player.

        Args:
            color: The player's color.
            name: The player's name.
            time_limit: Maximum time in seconds to spend on a move.
            max_depth: Maximum search depth (as a safety limit).
            heuristic: Name of the heuristic function to use.
            tt_size: Size of the transposition table.
        """
        super().__init__(color, name)
        self.time_limit = time_limit
        self.max_depth = max_depth
        
        # Set the heuristic function
        try:
            self.heuristic_name = heuristic
            self.heuristic_func = HeuristicRegistry.get(heuristic)
        except ValueError:
            print(f"Warning: Heuristic '{heuristic}' not found, using 'score_diff' instead")
            self.heuristic_name = "score_diff"
            self.heuristic_func = HeuristicRegistry.get("score_diff")
        
        # Simple transposition table with state hash -> (value, depth, flag)
        # where flag is: 0=exact, 1=lower bound, 2=upper bound
        self.tt = {}
        self.tt_size = tt_size
        
        # Statistics
        self.stats = {
            "nodes_evaluated": 0,
            "tt_hits": 0,
            "tt_stores": 0,
            "cutoffs": 0,
            "scout_cutoffs": 0,  # For NegaScout-specific statistics
            "full_window_searches": 0,
            "time_taken": 0,
            "depth_reached": 0,
        }
        
        # Time management
        self.start_time = 0
        self.time_up = False
    
    def get_move(self, game_state: GameState) -> Optional[Move]:
        """
        Find the best move using NegaScout with iterative deepening.

        Args:
            game_state: Current state of the game

        Returns:
            The best move according to NegaScout, or None if no move is possible.
        """
        self.start_time = time.time()
        self.time_up = False
        
        # Reset statistics
        self.stats = {
            "nodes_evaluated": 0,
            "tt_hits": 0,
            "tt_stores": 0,
            "cutoffs": 0,
            "scout_cutoffs": 0,
            "full_window_searches": 0,
            "time_taken": 0,
            "depth_reached": 0,
        }
        
        # Get valid moves
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return None
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # First do a light evaluation to order moves
        ordered_moves = self._order_moves(valid_moves, game_state)
        
        # Iterative deepening
        current_depth = 1
        best_move = ordered_moves[0]  # Fallback move
        
        while current_depth <= self.max_depth and not self.time_up:
            try:
                # Run NegaScout to current depth
                current_best_move, current_best_score = self._search_at_depth(
                    game_state, ordered_moves, current_depth
                )
                
                # If we completed the search at this depth, update best move
                if not self.time_up:
                    best_move = current_best_move
                    self.stats["depth_reached"] = current_depth
                    
                    # Update move ordering based on results at this depth
                    ordered_moves = self._reorder_after_search(ordered_moves, game_state, current_depth)
                    
                    # If we found a winning move, we can stop
                    if current_best_score > 900:
                        break
                        
                    # Check if we're out of time
                    if time.time() - self.start_time > self.time_limit * 0.8:
                        break
                
                # Increment depth for next iteration
                current_depth += 1
                
            except TimeoutError:
                # Time ran out during search, use the best move from last completed depth
                break
        
        # Update statistics
        elapsed = time.time() - self.start_time
        self.stats["time_taken"] = elapsed
        
        print(f"Depth reached: {self.stats['depth_reached']}")
        print(f"Evaluated {self.stats['nodes_evaluated']} nodes in {elapsed:.2f}s")
        print(f"TT hits: {self.stats['tt_hits']}, Scout cutoffs: {self.stats['scout_cutoffs']}")
        print(f"Full window searches: {self.stats['full_window_searches']}")
        print(f"Best move: {best_move}")
        
        return best_move
    
    def _search_at_depth(self, game_state: GameState, moves: List[Move], depth: int) -> Tuple[Move, float]:
        """
        Search to a specific depth with NegaScout.
        
        Args:
            game_state: Current game state
            moves: List of moves to search
            depth: Depth to search to
            
        Returns:
            Tuple of (best_move, best_score)
            
        Raises:
            TimeoutError: If time limit is exceeded
        """
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        # NegaScout: first node is searched with full window
        first_move = True
        
        # Search each move to the specified depth
        for move in moves:
            # Check if we're out of time
            if time.time() - self.start_time > self.time_limit:
                self.time_up = True
                if best_move is None:
                    return moves[0], 0  # Return first move as fallback
                raise TimeoutError("Time limit exceeded during search")
            
            # Apply move to new state
            new_state = game_state.copy()
            if not new_state.apply_move(move):
                continue
                
            # For the first move, use full window search
            if first_move:
                score = -self._negascout(new_state, depth - 1, -beta, -alpha, True)
                first_move = False
                self.stats["full_window_searches"] += 1
            else:
                # For subsequent moves, use a null window scout first
                # If the scout indicates a potentially better move (fails high),
                # then re-search with full window
                score = -self._negascout(new_state, depth - 1, -alpha - 1, -alpha, True)
                
                if alpha < score < beta:
                    # Scout indicated a promising move, re-search with full window
                    self.stats["full_window_searches"] += 1
                    score = -self._negascout(new_state, depth - 1, -beta, -score, True)
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
            
            # Update alpha
            alpha = max(alpha, best_score)
            
        return best_move, best_score
    
    def _negascout(self, state: GameState, depth: int, alpha: float, beta: float, is_pv_node: bool) -> float:
        """
        NegaScout search algorithm with transposition table.

        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value
            beta: Beta value
            is_pv_node: Whether this is a principal variation node
            
        Returns:
            NegaScout evaluation score
            
        Raises:
            TimeoutError: If time limit is exceeded
        """
        # Check for timeout periodically
        if self.stats["nodes_evaluated"] % 1000 == 0:
            if time.time() - self.start_time > self.time_limit:
                self.time_up = True
                raise TimeoutError("Time limit exceeded during search")
        
        self.stats["nodes_evaluated"] += 1
        
        # Transposition table lookup
        state_hash = self._hash_state(state)
        tt_entry = self.tt.get(state_hash)
        
        if tt_entry and tt_entry['depth'] >= depth:
            self.stats["tt_hits"] += 1
            flag = tt_entry['flag']
            value = tt_entry['value']
            
            # Use the TT value based on its flag
            if flag == 0:  # Exact value
                return value
            elif flag == 1 and value > alpha:  # Lower bound
                alpha = value
            elif flag == 2 and value < beta:   # Upper bound
                beta = value
                
            # If bounds are crossed, we can use the TT value
            if alpha >= beta:
                return value
        
        # Check for terminal state
        if state.is_game_over():
            winner = state.get_winner()
            if winner == self.color:
                return 1000.0
            elif winner is None:
                return 0.0
            else:
                return -1000.0
                
        # Check depth limit
        if depth <= 0:
            return self._evaluate(state)
        
        # Get valid moves
        valid_moves = state.get_valid_moves()
        
        # Handle no moves (switch player)
        if not valid_moves:
            state.current_player = state.current_player.opposite()
            return -self._negascout(state, depth, -beta, -alpha, is_pv_node)
        
        # Move ordering for better pruning
        if depth > 2:
            valid_moves = self._order_moves(valid_moves, state)
        
        # Initialize for NegaScout
        best_value = float('-inf')
        first_child = True
        original_alpha = alpha
        
        # NegaScout main loop
        for move in valid_moves:
            new_state = state.copy()
            if not new_state.apply_move(move):
                continue
            
            # Search first child with full window
            if first_child:
                value = -self._negascout(new_state, depth - 1, -beta, -alpha, True)
                first_child = False
            else:
                # Null window scout search for subsequent children
                value = -self._negascout(new_state, depth - 1, -alpha - 1, -alpha, False)
                
                # Re-search with full window if the scout shows promise
                if alpha < value < beta:
                    self.stats["full_window_searches"] += 1
                    value = -self._negascout(new_state, depth - 1, -beta, -value, True)
                else:
                    # Scout search provided sufficient info, no need for full window
                    self.stats["scout_cutoffs"] += 1
            
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            
            # Alpha-beta pruning
            if alpha >= beta:
                self.stats["cutoffs"] += 1
                break
        
        # Store in transposition table
        flag = 0  # Exact value
        if best_value <= original_alpha:
            flag = 2  # Upper bound
        elif best_value >= beta:
            flag = 1  # Lower bound
            
        self._store_transposition(state_hash, best_value, depth, flag)
        
        return best_value
    
    def _evaluate(self, state: GameState) -> float:
        """
        Evaluate a state using the heuristic function.

        Args:
            state: Game state to evaluate

        Returns:
            Evaluation score
        """
        return self.heuristic_func(state, self.color)
    
    def _order_moves(self, moves: List[Move], state: GameState) -> List[Move]:
        """
        Order moves based on 1-ply evaluation.

        Args:
            moves: List of valid moves
            state: Current game state

        Returns:
            Ordered list of moves
        """
        move_scores = []
        
        for move in moves:
            # Apply move to new state
            new_state = state.copy()
            if not new_state.apply_move(move):
                continue
            
            # Light evaluation of the resulting position
            score = -self._evaluate(new_state)
            move_scores.append((move, score))
        
        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in move_scores]
    
    def _reorder_after_search(self, moves: List[Move], state: GameState, depth: int) -> List[Move]:
        """
        Reorder moves based on previous search results.
        
        Args:
            moves: Previously ordered list of moves
            state: Current game state
            depth: Last completed search depth
            
        Returns:
            Reordered list of moves
        """
        # Simple implementation: perform a very short search to better order moves
        # We don't need to go as deep since we just need relative ordering
        return self._order_moves(moves, state)
    
    def _hash_state(self, state: GameState) -> int:
        """
        Simple state hashing function.

        Args:
            state: Game state to hash

        Returns:
            Hash value
        """
        # Convert board to bytes
        board_bytes = state.board.tobytes()
        
        # Combine with scores and current player
        player_hash = 1 if state.current_player == PlayerColor.BLACK else 2
        black_score = state.scores[PlayerColor.BLACK]
        red_score = state.scores[PlayerColor.RED]
        
        # Simple hash combining
        return hash((board_bytes, black_score, red_score, player_hash))
    
    def _store_transposition(self, state_hash: int, value: float, depth: int, flag: int):
        """
        Store position in transposition table with simple LRU replacement.

        Args:
            state_hash: Hash of the game state
            value: Evaluation value
            depth: Search depth
            flag: Entry type (0=exact, 1=lower bound, 2=upper bound)
        """
        # Manage table size - simple LRU strategy
        if len(self.tt) >= self.tt_size:
            # Remove oldest 10% of entries
            keys_to_remove = list(self.tt.keys())[:self.tt_size // 10]
            for key in keys_to_remove:
                self.tt.pop(key, None)
        
        # Store value
        self.tt[state_hash] = {'value': value, 'depth': depth, 'flag': flag}
        self.stats["tt_stores"] += 1
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the search.

        Returns:
            Dictionary of statistics
        """
        return {
            "algorithm": "NegaScout (Principal Variation Search)",
            "time_limit": self.time_limit,
            "depth_reached": self.stats["depth_reached"],
            "heuristic": self.heuristic_name,
            "nodes_evaluated": self.stats["nodes_evaluated"],
            "tt_hits": self.stats["tt_hits"],
            "tt_stores": self.stats["tt_stores"],
            "cutoffs": self.stats["cutoffs"],
            "scout_cutoffs": self.stats["scout_cutoffs"],
            "full_window_searches": self.stats["full_window_searches"],
            "time_taken": self.stats["time_taken"],
        }