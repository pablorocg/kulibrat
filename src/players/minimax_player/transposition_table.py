from typing import Any, Dict, Optional
import random
from typing import Any, Dict

from src.core import PlayerColor, Move, GameState
import numpy as np




class TranspositionTable:
    """Hash table for storing evaluated positions to avoid recalculating."""
    
    # Node types for transposition table
    EXACT = 0    # Exact evaluation
    ALPHA = 1    # Upper bound (alpha cutoff)
    BETA = 2     # Lower bound (beta cutoff)
    
    def __init__(self, max_size: int = 1000000):
        """
        Initialize the transposition table.
        
        Args:
            max_size: Maximum number of entries in the table
        """
        self.max_size = max_size
        self.table = {}
        self.hits = 0
        self.stores = 0
    
    def store(self, zobrist_hash: int, depth: int, value: float, node_type: int, best_move: Optional[Move] = None):
        """
        Store a position evaluation in the table.
        
        Args:
            zobrist_hash: Zobrist hash of the position
            depth: Depth of the search
            value: Evaluation value
            node_type: Type of node (EXACT, ALPHA, BETA)
            best_move: Best move found at this position
        """
        if len(self.table) >= self.max_size:
            # Improved replacement strategy - prioritize keeping deeper evaluations and exact nodes
            keys_to_check = random.sample(list(self.table.keys()), min(100, len(self.table)))
            best_key_to_replace = None
            lowest_priority = float('inf')
            
            for key in keys_to_check:
                entry = self.table[key]
                # Calculate priority: depth + bonus for exact nodes
                priority = entry['depth'] + (2 if entry['type'] == self.EXACT else 0)
                if priority < lowest_priority:
                    lowest_priority = priority
                    best_key_to_replace = key
            
            if best_key_to_replace:
                self.table.pop(best_key_to_replace)
            else:
                # Fallback to removing a random entry
                self.table.pop(random.choice(list(self.table.keys())))
            
        self.table[zobrist_hash] = {
            'depth': depth,
            'value': value,
            'type': node_type,
            'best_move': best_move,
        }
        self.stores += 1
    
    def lookup(self, zobrist_hash: int) -> Optional[Dict[str, Any]]:
        """
        Look up a position in the table.
        
        Args:
            zobrist_hash: Zobrist hash of the position
            
        Returns:
            Entry data or None if not found
        """
        entry = self.table.get(zobrist_hash)
        if entry:
            self.hits += 1
        return entry
    
    def get_hit_rate(self) -> float:
        """
        Calculate the cache hit rate.
        
        Returns:
            Hit rate as a percentage
        """
        lookups = self.hits + (len(self.table) - self.stores)
        return self.hits / lookups * 100 if lookups > 0 else 0


