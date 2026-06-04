from typing import Literal

from learn2slither.core import Direction


ACTION_TO_DIRECTION: tuple[Direction, ...] = (
    Direction.UP,
    Direction.LEFT,
    Direction.DOWN,
    Direction.RIGHT,
)
EngineName = Literal["q", "nn"]


def action_to_direction(action: int) -> Direction:
    """Converts an absolute action id to a Direction."""
    return ACTION_TO_DIRECTION[action]
