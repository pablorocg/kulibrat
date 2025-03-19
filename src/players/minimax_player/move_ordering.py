from typing import List, Optional
from src.core import GameState, Move, PlayerColor, MoveType




class MoveOrdering:
    """
    Implements move ordering to improve alpha-beta pruning efficiency.
    """
    
    def __init__(self):
        """Initialize move ordering with history heuristic."""
        # History heuristic table
        self.history_table = {}
        
        # Killer moves storage (two killer moves per ply depth)
        self.killer_moves = {}
        for i in range(20):  # Support up to depth 20
            self.killer_moves[i] = [None, None]
        
        # Move type priorities (higher is better)
        self.move_type_priority = {
            MoveType.ATTACK: 4,    # Generally good to capture
            MoveType.JUMP: 5,      # Bumped up - tournament showed underutilization
            MoveType.DIAGONAL: 2,  # Regular moves
            MoveType.INSERT: 1     # Insertions are default
        }
    
    def order_moves(self, moves: List[Move], tt_move: Optional[Move] = None, depth: int = 0, state: Optional[GameState] = None) -> List[Move]:
        """
        Order moves to maximize alpha-beta pruning efficiency.
        
        Args:
            moves: List of legal moves
            tt_move: Best move from transposition table, if available
            depth: Current search depth
            state: Current game state (optional, for more context-aware ordering)
            
        Returns:
            Ordered list of moves
        """
        # Score each move
        move_scores = []
        
        # Get current player color for context-aware ordering
        current_player = state.current_player if state else None
        
        for move in moves:
            score = 0
            
            # Transposition table move gets highest priority
            if tt_move and self._moves_equal(move, tt_move):
                score += 10000
            
            # Killer move bonus (first killer gets higher priority)
            killer1, killer2 = self.killer_moves.get(depth, [None, None])
            if killer1 and self._moves_equal(move, killer1):
                score += 9000
            elif killer2 and self._moves_equal(move, killer2):
                score += 8000
            
            # Move type priority
            score += self.move_type_priority.get(move.move_type, 0) * 100
            
            # Context-aware scoring if state is provided
            if state and current_player:
                # Prioritize scoring moves
                if self._is_scoring_move(move, current_player):
                    score += 6000
                
                # Prioritize moves that block opponent's scoring opportunities
                if move.move_type in [MoveType.ATTACK, MoveType.JUMP] and self._blocks_scoring(move, state):
                    score += 3000
                
                # Add positional bonus for moves that advance pieces toward scoring
                if move.end_pos and move.start_pos:
                    if current_player == PlayerColor.BLACK and move.end_pos[0] > move.start_pos[0]:
                        # BLACK advancing down
                        score += 50 * move.end_pos[0]  # More advanced = higher score
                    elif current_player == PlayerColor.RED and move.end_pos[0] < move.start_pos[0]:
                        # RED advancing up
                        score += 50 * (state.BOARD_ROWS - 1 - move.end_pos[0])
            
            # History heuristic
            move_key = self._get_move_key(move)
            score += self.history_table.get(move_key, 0)
                
            move_scores.append((move, score))
        
        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return ordered moves
        return [m[0] for m in move_scores]
    
    def update_history(self, move: Move, depth: int):
        """
        Update history heuristic.
        
        Args:
            move: Move that caused a beta cutoff
            depth: Search depth
        """
        move_key = self._get_move_key(move)
        # Use depth^2 to give more weight to moves that cause cutoffs at deeper levels
        self.history_table[move_key] = self.history_table.get(move_key, 0) + depth * depth
    
    def update_killer_move(self, move: Move, depth: int):
        """
        Update killer moves.
        
        Args:
            move: Move that caused a beta cutoff
            depth: Search depth
        """
        # Only store non-capturing moves as killer moves (better for Kulibrat)
        if move.move_type in [MoveType.DIAGONAL, MoveType.INSERT, MoveType.JUMP]:
            killer1, killer2 = self.killer_moves.get(depth, [None, None])
            # Don't duplicate killer moves
            if killer1 is None or not self._moves_equal(move, killer1):
                self.killer_moves[depth] = [move, killer1]
    
    def _get_move_key(self, move: Move) -> str:
        """
        Convert a move to a unique string key for the history table.
        
        Args:
            move: Move to convert
            
        Returns:
            String key
        """
        return f"{move.move_type.name}:{move.start_pos}:{move.end_pos}"
    
    def _moves_equal(self, move1: Move, move2: Move) -> bool:
        """
        Check if two moves are equal (even if they are different objects).
        
        Args:
            move1: First move
            move2: Second move
            
        Returns:
            True if moves are equal
        """
        return (move1.move_type == move2.move_type and 
                move1.start_pos == move2.start_pos and 
                move1.end_pos == move2.end_pos)
                
    def _is_scoring_move(self, move: Move, player_color: PlayerColor) -> bool:
        """
        Check if a move is a scoring move.
        
        Args:
            move: Move to check
            player_color: Player making the move
            
        Returns:
            True if the move scores a point
        """
        if not move.end_pos or not move.start_pos:
            return False
            
        # Diagonal moves from row 3 for BLACK or row 0 for RED can score
        if move.move_type == MoveType.DIAGONAL:
            if (player_color == PlayerColor.BLACK and move.start_pos[0] == 3 and 
                (move.end_pos[0] >= 4 or move.end_pos[0] < 0)):
                return True  # BLACK scoring
            if (player_color == PlayerColor.RED and move.start_pos[0] == 0 and 
                (move.end_pos[0] >= 4 or move.end_pos[0] < 0)):
                return True  # RED scoring
                
        # Jump moves that land off the board can score
        if move.move_type == MoveType.JUMP:
            if move.end_pos[0] < 0 or move.end_pos[0] >= 4:
                return True
                
        return False
    
    def _blocks_scoring(self, move: Move, state: GameState) -> bool:
        """
        Check if a move blocks opponent from scoring.
        
        Args:
            move: Move to check
            state: Current game state
            
        Returns:
            True if the move blocks a potential score
        """
        if not move.end_pos or move.move_type not in [MoveType.ATTACK, MoveType.JUMP]:
            return False
            
        opponent_color = state.current_player.opposite()
        
        # Check if the move prevents a scoring opportunity
        # For BLACK opponent, check if we're removing a piece from row 3
        # For RED opponent, check if we're removing a piece from row 0
        if (opponent_color == PlayerColor.BLACK and move.end_pos[0] == 3) or \
           (opponent_color == PlayerColor.RED and move.end_pos[0] == 0):
            return True
            
        return False