from typing import Any, Dict, Optional, Tuple

from src.core.move_type import MoveType


class Move:
    def __init__(self, 
                 move_type: MoveType, 
                 start_pos: Optional[Tuple[int, int]] = None, 
                 end_pos: Optional[Tuple[int, int]] = None):
        self.move_type = move_type
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def serialize(self) -> Dict[str, Any]:
        """Convert move to a dictionary for JSON serialization."""
        return {
            "move_type": self.move_type.name,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Move':
        """Create a Move object from serialized data."""
        return cls(
            move_type=MoveType[data["move_type"]],
            start_pos=tuple(data["start_pos"]) if data["start_pos"] else None,
            end_pos=tuple(data["end_pos"]) if data["end_pos"] else None
        )
    
    def __str__(self) -> str:
        return f"{self.move_type.name}: {self.start_pos} -> {self.end_pos}"