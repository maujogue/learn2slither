from enum import Enum
from dataclasses import dataclass, field


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
        return 0 <= point.x < self.config.width and 0 <= point.y < self.config.height

    def get_empty_cells(self) -> list[Point]:
        """Returns a list of all cells that are not occupied by the snake or any apples."""
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
        """Changes the snake's direction, preventing direct reversals."""
        if len(self.snake.body) > 1:
            neck = self.snake.body[1]
            head = self.snake.head
            next_point = head.move(new_direction)
            if next_point == neck:
                return
        self.snake.direction = new_direction

    def step(self) -> None:
        """Moves the snake forward one step and processes collisions and apples."""
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

            # Check collision with own body (excluding the new head we just added)
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

            # Compute new body: prepend new head, remove last two elements (net length -1)
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


def create_initial_game(width: int = 20, height: int = 20) -> GameState:
    """Creates a new game state with a contiguous snake of length 3 and spawned apples."""
    import random

    config = BoardConfig(width, height)

    while True:
        # 1. Pick a random head cell
        head_x = random.randint(0, width - 1)
        head_y = random.randint(0, height - 1)
        head = Point(head_x, head_y)

        body = [head]

        # 2. Grow the snake body to length 3 contiguously using a simple DFS
        def find_body_path(current_path: list[Point]) -> list[Point] | None:
            if len(current_path) == 3:
                return current_path

            last = current_path[-1]
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = Point(last.x + dx, last.y + dy)
                if 0 <= neighbor.x < width and 0 <= neighbor.y < height:
                    if neighbor not in current_path:
                        neighbors.append(neighbor)

            if not neighbors:
                return None

            random.shuffle(neighbors)
            for n in neighbors:
                result = find_body_path(current_path + [n])
                if result is not None:
                    return result
            return None

        path = find_body_path(body)
        if path is not None:
            # Snake body created contiguously!
            # Align head direction: pointing from neck (path[1]) to head (path[0])
            dx = path[0].x - path[1].x
            dy = path[0].y - path[1].y

            initial_direction = Direction.RIGHT
            for d in Direction:
                if d.value == (dx, dy):
                    initial_direction = d
                    break

            snake = Snake(body=path, direction=initial_direction)
            break

    state = GameState(config=config, snake=snake)

    # Spawn 2 green apples
    state._spawn_green_apple()
    state._spawn_green_apple()

    # Spawn 1 red apple
    state._spawn_red_apple()

    return state
