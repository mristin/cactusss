"""Model the commands in the game."""
import enum


class Command(enum.Enum):
    """Represent a command given to the balloon."""

    GO_LEFT = 0
    GO_RIGHT = 1
