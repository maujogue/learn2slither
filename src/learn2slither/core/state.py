from dataclasses import dataclass, field

from learn2slither.core import (
    BoardConfig,
    Direction,
    GameOverReason,
    Point,
    Snake,
)


@dataclass
class GameState:
    config: BoardConfig
    snake: Snake

    # Using sets for O(1) lookup when checking for collisions with apples
    green_apples: set[Point] = field(default_factory=set)
    red_apples: set[Point] = field(default_factory=set)

    is_game_over: bool = False
    game_over_reason: GameOverReason | None = None

    def is_within_bounds(self, point: Point) -> bool:
        """Checks if a point is within the board boundaries."""
        return (
            0 <= point.x < self.config.width
            and 0 <= point.y < self.config.height
        )

    def get_empty_cells(self) -> list[Point]:
        """Return cells not occupied by the snake or any apples."""
        occupied = set(self.snake.body) | self.green_apples | self.red_apples
        empty = []
        for x in range(self.config.width):
            for y in range(self.config.height):
                p = Point(x, y)
                if p not in occupied:
                    empty.append(p)
        return empty

    def _spawn_green_apple(self) -> None:
        """Spawns a green apple in a random empty cell."""
        empty_cells = self.get_empty_cells()
        if empty_cells:
            import random

            self.green_apples.add(random.choice(empty_cells))

    def _spawn_red_apple(self) -> None:
        """Spawns a red apple in a random empty cell."""
        empty_cells = self.get_empty_cells()
        if empty_cells:
            import random

            self.red_apples.add(random.choice(empty_cells))

    def change_direction(self, new_direction: Direction) -> None:
        """Changes the snake's direction."""
        self.snake.direction = new_direction

    def step(self) -> None:
        """Move the snake one step; handle collisions and apples."""
        if self.is_game_over:
            return

        new_head = self.snake.head.move(self.snake.direction)

        # 1. Check wall collision
        if not self.is_within_bounds(new_head):
            self.is_game_over = True
            self.game_over_reason = GameOverReason.WALL_COLLISION
            return

        # 2. Check collision with apples
        ate_green = new_head in self.green_apples
        ate_red = new_head in self.red_apples

        if ate_green:
            # Eat green apple: snake grows!
            self.green_apples.remove(new_head)
            self.snake.body.insert(0, new_head)

            # Body collision check (excluding the head we just added)
            if new_head in self.snake.body[1:]:
                self.is_game_over = True
                self.game_over_reason = GameOverReason.TAIL_COLLISION
                return

            self._spawn_green_apple()

        elif ate_red:
            # Eat red apple: snake shrinks!
            self.red_apples.remove(new_head)

            # If length is 1 and it shrinks, it drops to 0 -> starvation!
            if len(self.snake.body) <= 1:
                self.snake.body = []
                self.is_game_over = True
                self.game_over_reason = GameOverReason.STARVATION
                return

            # Prepend new head, drop last two segments (net length -1)
            new_body = [new_head] + self.snake.body[:-2]

            if len(new_body) == 0:
                self.snake.body = []
                self.is_game_over = True
                self.game_over_reason = GameOverReason.STARVATION
                return

            # Check collision with remaining body segments
            if new_head in self.snake.body[:-2]:
                self.is_game_over = True
                self.game_over_reason = GameOverReason.TAIL_COLLISION
                return

            self.snake.body = new_body
            self._spawn_red_apple()

        else:
            # Normal move: prepend new head, remove last element
            # Check collision with remaining body segments that won't move out
            if new_head in self.snake.body[:-1]:
                self.is_game_over = True
                self.game_over_reason = GameOverReason.TAIL_COLLISION
                return

            self.snake.body = [new_head] + self.snake.body[:-1]
