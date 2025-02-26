from enum import Enum, auto

class MoveType(Enum):
    """
    Enum class representing different types of moves.

    Attributes:
        INSERT: Represents an insert move.
        DIAGONAL: Represents a diagonal move.
        ATTACK: Represents an attack move.
        JUMP: Represents a jump move.
    """
    INSERT = auto()
    DIAGONAL = auto()
    ATTACK = auto()
    JUMP = auto()
