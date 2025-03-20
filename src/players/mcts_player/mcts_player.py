"""
Simplified and fixed Monte Carlo Tree Search implementation for Kulibrat.
"""

import logging
import random
import time
from typing import Dict, Optional, Tuple

from src.core.game_state import GameState
from src.core.move import Move
from src.core.player_color import PlayerColor
from src.players.mcts_player.mcts_node import MCTSNode
from src.players.player import Player

# Configure logger
logger = logging.getLogger(__name__)


class MCTSPlayer(Player):
    """
    Simplified Monte Carlo Tree Search player for Kulibrat.
    """

    def __init__(
        self,
        color: PlayerColor,
        name: str = "MCTS",
        simulation_time: float = 1.0,
        max_iterations: int = 10000,
        exploration_weight: float = 1.5,
    ):
        """Initialize the MCTS player."""
        super().__init__(color, name or "Optimized MCTS")
        self.simulation_time = simulation_time
        self.max_iterations = max_iterations
        self.exploration_weight = exploration_weight

        # Statistics tracking
        self.stats = {"simulations": 0, "time_taken": 0, "depths": []}

    def get_move(self, game_state: GameState) -> Optional[Move]:
        """Find the best move using Monte Carlo Tree Search."""
        # Reset statistics
        self.stats = {"simulations": 0, "time_taken": 0, "depths": []}

        start_time = time.time()
        root = MCTSNode(game_state)

        # Log the start of search
        logger.info(f"Starting MCTS search for player {self.color.name}")

        # Progress tracking
        simulations = 0
        simulation_stats = {"sim_lengths": [], "max_depth": 0}

        # Main MCTS loop
        while (
            time.time() - start_time < self.simulation_time
            and simulations < self.max_iterations
        ):
            # Phase 1: Selection - select node to expand
            node, depth = self._select(root)
            simulation_stats["max_depth"] = max(simulation_stats["max_depth"], depth)

            # Phase 2: Expansion - expand selected node
            if not node.is_terminal():
                expanded_node = node.expand()
                if expanded_node:
                    node = expanded_node
                    depth += 1
                    simulation_stats["max_depth"] = max(
                        simulation_stats["max_depth"], depth
                    )

            # Phase 3: Simulation - simulate random play
            winner, sim_length = self._simulate(node.state.copy())
            simulations += 1

            # Record simulation length
            simulation_stats["sim_lengths"].append(sim_length)

            # Phase 4: Backpropagation - update statistics
            self._backpropagate(node, winner)

            # Track progress
            self.stats["depths"].append(depth)

            # Periodic log for large simulations
            if simulations % 20000 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Progress: {simulations} simulations, "
                    f"{elapsed:.2f}s elapsed, "
                    f"{simulations / elapsed:.1f} sims/sec"
                )
                # Quick check on simulation length
                avg_length = (
                    sum(simulation_stats["sim_lengths"])
                    / len(simulation_stats["sim_lengths"])
                    if simulation_stats["sim_lengths"]
                    else 0
                )
                logger.info(
                    f"Current avg simulation length: {avg_length:.1f} moves after {simulations} simulations"
                )

        # Log results and get best move
        elapsed_time = time.time() - start_time
        self.stats["simulations"] = simulations
        self.stats["time_taken"] = elapsed_time

        # Choose the child with the most visits (most robust)
        if not root.children:
            logger.warning("No moves explored - selecting randomly")
            valid_moves = game_state.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else None

        best_child = max(root.children, key=lambda c: c.visits)

        # Log detailed statistics
        self._log_detailed_stats(root, simulations, elapsed_time, simulation_stats)

        return best_child.move

    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, int]:
        """
        Select a node to expand using UCB1.
        Returns selected node and depth.
        """
        depth = 0

        # Apply first visit initialization: always visit unvisited nodes first
        while not node.is_terminal():
            # If node is not fully expanded, return it for expansion
            if not node.is_fully_expanded():
                return node, depth

            # Otherwise, select child with highest UCB value
            child = node.select_child(self.exploration_weight)
            if child is None:
                # If no children, return current node
                return node, depth

            node = child
            depth += 1

            # Add randomness to prevent getting stuck in local optima
            if random.random() < 0.05:  # 5% chance
                child_options = [c for c in node.children if c.visits > 0]
                if child_options:
                    node = random.choice(child_options)

        return node, depth

    def _simulate(self, state: GameState) -> Tuple[Optional[PlayerColor], int]:
        """
        Simulate a random playout from the given state.
        Returns the winner and number of moves in the simulation.
        """
        # Track simulation length
        moves_made = 0
        max_moves = 40  # Prevent infinite loops

        # Store original current player for result perspective
        original_player = state.current_player

        # Debug flag to print simulation details (for very complex bugs)
        debug_sim = False  # Set to True only for debugging specific moves

        if debug_sim:
            logger.info(
                f"Starting simulation from state with current player: {state.current_player}"
            )
            logger.info(f"Board state: {state.board}")

        while not state.is_game_over() and moves_made < max_moves:
            valid_moves = state.get_valid_moves()

            if not valid_moves:
                # No moves - switch player
                state.current_player = state.current_player.opposite()
                if debug_sim:
                    logger.info(
                        f"No valid moves for {state.current_player.opposite()}, switching to {state.current_player}"
                    )
                continue

            # Choose a random move
            move = random.choice(valid_moves)
            success = state.apply_move(move)

            if success:
                moves_made += 1
                if debug_sim:
                    logger.info(f"Applied move: {move}, new moves count: {moves_made}")
            else:
                if debug_sim:
                    logger.info(f"Failed to apply move: {move}")

        # Get final winner
        winner = state.get_winner()

        if debug_sim:
            logger.info(f"Simulation ended. Winner: {winner}, Moves made: {moves_made}")

        # Important: ensure we always return a non-negative move count
        return winner, max(0, moves_made)

    def _backpropagate(self, node: MCTSNode, winner: Optional[PlayerColor]):
        """Update node statistics based on simulation result."""
        while node:
            node.visits += 1

            # Award wins: 1 for victory, 0.5 for draw, 0 for loss
            current_player = node.state.current_player

            if winner is None:
                # Draw - award half win
                node.wins += 0.5
            elif winner == current_player:
                # Current player won
                node.wins += 1
            # Otherwise it's a loss, no wins awarded

            node = node.parent

    def _log_detailed_stats(self, root, simulations, elapsed_time, simulation_stats):
        """Log detailed statistics about the search."""
        # Calculate average depth
        avg_depth = (
            sum(self.stats["depths"]) / len(self.stats["depths"])
            if self.stats["depths"]
            else 0
        )

        # Calculate depth distribution
        depth_dist = {}
        for d in self.stats["depths"]:
            depth_dist[d] = depth_dist.get(d, 0) + 1

        # Log move options with statistics
        logger.info(f"Move options after {simulations} simulations:")
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        for i, child in enumerate(sorted_children[:5]):  # Show top 5 moves
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            logger.info(
                f"  {i + 1}. Move: {child.move}, "
                f"Visits: {child.visits}, "
                f"Win rate: {win_rate:.2%}"
            )

        # Log depth distribution
        logger.info("Depth distribution:")
        for depth in sorted(depth_dist.keys()):
            count = depth_dist[depth]
            pct = (count / simulations) * 100 if simulations > 0 else 0
            logger.info(f"  Depth {depth}: {count} nodes ({pct:.1f}%)")

        # Log simulation statistics
        sim_lengths = simulation_stats.get("sim_lengths", [])
        avg_sim_length = sum(sim_lengths) / len(sim_lengths) if sim_lengths else 0

        # Debug simulation lengths
        if sim_lengths:
            lengths_sample = sim_lengths[:5]  # Just show a few for debugging
            logger.info(f"Debug: First few simulation lengths: {lengths_sample}")

        logger.info("Simulation statistics:")
        logger.info(f"  Simulations: {simulations}")
        logger.info(
            f"  Time: {elapsed_time:.3f}s ({simulations / elapsed_time:.1f} sims/sec)"
        )
        logger.info(f"  Max depth: {simulation_stats.get('max_depth', 0)}")
        logger.info(f"  Avg depth: {avg_depth:.1f}")
        logger.info(f"  Avg simulation length: {avg_sim_length:.1f} moves")
        logger.info(f"  Total simulation moves: {sum(sim_lengths)}")

        best_child = (
            max(root.children, key=lambda c: c.visits) if root.children else None
        )
        if best_child:
            win_rate = (
                best_child.wins / best_child.visits if best_child.visits > 0 else 0
            )
            logger.info(
                f"Selected move: {best_child.move} with {best_child.visits} visits "
                f"({win_rate:.2%} win rate)"
            )

    def get_stats(self) -> Dict:
        """Get statistics about the MCTS search."""
        return {
            "algorithm": "Simplified MCTS",
            "exploration_weight": self.exploration_weight,
            "simulations": self.stats.get("simulations", 0),
            "time_taken": f"{self.stats.get('time_taken', 0):.3f}s",
            "avg_depth": f"{sum(self.stats.get('depths', [0])) / len(self.stats.get('depths', [1])) if self.stats.get('depths', []) else 0:.1f}",
        }

    def setup(self, game_state: GameState) -> None:
        """Initialize for a new game."""
        self.stats = {"simulations": 0, "time_taken": 0, "depths": []}
        logger.info(f"MCTS player {self.name} ({self.color.name}) initialized")

    def notify_move(self, move: Move, game_state: GameState) -> None:
        """Handle move notification."""
        pass  # Simplified version doesn't use tree reuse

    def game_over(self, game_state: GameState) -> None:
        """Handle game over."""
        logger.info(f"Game over for MCTS player {self.name}")
