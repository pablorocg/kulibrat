from enum import Enum


class PlayerColor(Enum):
    """
    An enumeration representing player colors in a game, with associated properties and methods.
    Attributes:
        BLACK (int): Represents the black player with a value of 1.
        RED (int): Represents the red player with a value of -1.
    Methods:
        opposite() -> PlayerColor:
            Returns the opposite player color.
        start_row() -> int:
            Property that returns the starting row for the player color.
            Returns 0 for BLACK and 3 for RED.
        direction() -> int:
            Property that returns the movement direction for the player color.
            Returns 1 for BLACK (positive direction) and -1 for RED (negative direction).
    """

    BLACK = 1 # Player 1
    RED = -1 # Player 2

    def opposite(self):
        return PlayerColor.RED if self == PlayerColor.BLACK else PlayerColor.BLACK

    @property
    def start_row(self) -> int:
        return 0 if self == PlayerColor.BLACK else 3

    @property
    def direction(self) -> int:
        """Movement direction: positive for BLACK, negative for RED."""
        return 1 if self == PlayerColor.BLACK else -1
