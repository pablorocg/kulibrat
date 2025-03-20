"""
Monte Carlo Tree Search Node implementation for Kulibrat.
"""

import math
import random

from src.core.game_state import GameState


class MCTSNode:
    """Node in the Monte Carlo Tree Search."""

    __slots__ = (
        "state",
        "parent",
        "move",
        "children",
        "wins",
        "visits",
        "untried_moves",
    )

    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_valid_moves()

    def select_child(self, exploration_weight=1.4):
        """Select a child node using the UCB1 formula."""
        if not self.children:
            return None

        # UCB1 formula: w/n + c * sqrt(ln(N)/n)
        return max(
            self.children,
            key=lambda c: (c.wins / c.visits)
            + exploration_weight * math.sqrt(math.log(self.visits) / c.visits)
            if c.visits > 0
            else float("inf"),
        )

    def expand(self):
        """Expand the node by adding a child node for an untried move."""
        if not self.untried_moves:
            return None

        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)

        new_state = self.state.copy()
        if not new_state.apply_move(move):
            # If move fails, try another move
            return self.expand() if self.untried_moves else None

        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        """Check if all possible moves from this state have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """Check if this node represents a terminal state."""
        return self.state.is_game_over()
