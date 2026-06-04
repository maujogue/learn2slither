from enum import Enum
from dataclasses import dataclass


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def move(self, direction: Direction) -> "Point":
        """Returns a new Point moved in the given direction."""
        dx, dy = direction.value
        return Point(self.x + dx, self.y + dy)


class GameOverReason(Enum):
    WALL_COLLISION = "wall_collision"
    TAIL_COLLISION = "tail_collision"
    STARVATION = "starvation"  # Length drops to 0


@dataclass
class Snake:
    body: list[Point]  # body[0] represents the head of the snake
    direction: Direction

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


@dataclass
class BoardConfig:
    width: int
    height: int
