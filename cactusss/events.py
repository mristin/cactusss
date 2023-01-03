"""Define the actions."""
import abc
import enum
from typing import Union

from icontract import DBC


class Event(DBC):
    """Represent an abstract event in the game."""

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()


class Tick(Event):
    """Mark a tick in the (irregular) game clock."""

    def __str__(self) -> str:
        return self.__class__.__name__


class ReceivedQuit(Event):
    """Signal that we have to exit the game."""

    def __str__(self) -> str:
        return self.__class__.__name__


class GameOverKind(enum.Enum):
    """Model different game endings."""

    HAPPY_END = 0
    PLOP = 1


class GameOver(Event):
    """Signal that we have to exit the game."""

    def __init__(self, kind: GameOverKind) -> None:
        """Initialize with the given values."""
        self.kind = kind

    def __str__(self) -> str:
        return self.__class__.__name__


class ReceivedRestart(Event):
    """Capture the event that we want to restart the game."""

    def __str__(self) -> str:
        return self.__class__.__name__


EventUnion = Union[Tick, ReceivedQuit, GameOver, ReceivedRestart]
