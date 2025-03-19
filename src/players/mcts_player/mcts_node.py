from typing import Optional
import random
import math

from src.core.game_state import GameState
from src.core.move import Move


class MCTSNode:
    """Node in the Monte Carlo Tree Search representing a game state."""
    __slots__ = ('state', 'parent', 'move', 'children', 'wins', 'visits', 
                 'untried_moves', 'player')
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, move: Optional[Move] = None):
        """Initialize a new node with minimal overhead."""
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = state.get_valid_moves()
        self.player = state.current_player
    
    def uct_value(self, exploration_weight: float = 1.41) -> float:
        """Calculate the UCT value for node selection."""
        if self.visits == 0:
            return float('inf')
        
        # Exploitation term (win rate)
        exploitation = self.wins / self.visits
        
        # Exploration term
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        
        return exploitation + exploration
    
    def select_child(self) -> Optional['MCTSNode']:
        """Select the child with the highest UCT value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.uct_value())
    
    def expand(self) -> Optional['MCTSNode']:
        """Expand the tree by adding a child for an untried move."""
        if not self.untried_moves:
            return None
        
        # Simple random move selection for expansion
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        # Apply the move
        new_state = self.state.copy()
        if not new_state.apply_move(move):
            return None
        
        # Create and link the child node
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def update(self, result: float):
        """Update node statistics with simulation result."""
        self.visits += 1
        self.wins += result
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        return self.state.is_game_over()