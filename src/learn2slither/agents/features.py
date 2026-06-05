from dataclasses import dataclass

from learn2slither.core import Direction, GameState, Point


@dataclass(frozen=True)
class StateFeatures:
    """A list of state features extracted from the GameState.

    To keep the Q-learning state space compact and enable extremely fast,
    generalized learning, we use 12 boolean absolute features.
    These features describe immediate surroundings and direction to the closest
    apples in absolute grid directions (UP, LEFT, DOWN, RIGHT).
    """

    danger_up: bool
    danger_left: bool
    danger_down: bool
    danger_right: bool

    green_apple_up: bool
    green_apple_left: bool
    green_apple_down: bool
    green_apple_right: bool

    red_apple_up: bool
    red_apple_left: bool
    red_apple_down: bool
    red_apple_right: bool

    def to_tuple(self) -> tuple[bool, ...]:
        """Converts features to a hashable tuple for Q-table indexing."""
        return (
            self.danger_up,
            self.danger_left,
            self.danger_down,
            self.danger_right,
            self.green_apple_up,
            self.green_apple_left,
            self.green_apple_down,
            self.green_apple_right,
            self.red_apple_up,
            self.red_apple_left,
            self.red_apple_down,
            self.red_apple_right,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Returns a list of all feature names in order."""
        return [
            "danger_up",
            "danger_left",
            "danger_down",
            "danger_right",
            "green_apple_up",
            "green_apple_left",
            "green_apple_down",
            "green_apple_right",
            "red_apple_up",
            "red_apple_left",
            "red_apple_down",
            "red_apple_right",
        ]

    @classmethod
    def from_game_state(cls, state: GameState) -> "StateFeatures":
        """Extracts absolute state features from the current GameState."""
        if state.is_game_over or not state.snake.body:
            return cls(
                danger_up=True,
                danger_left=True,
                danger_down=True,
                danger_right=True,
                green_apple_up=False,
                green_apple_left=False,
                green_apple_down=False,
                green_apple_right=False,
                red_apple_up=False,
                red_apple_left=False,
                red_apple_down=False,
                red_apple_right=False,
            )

        head = state.snake.head

        # Helper to check for obstacles
        def is_obstacle(p: Point) -> bool:
            if not state.is_within_bounds(p):
                return True
            # Colliding with the snake's own body is an obstacle
            if p in state.snake.body:
                return True
            return False

        danger_up = is_obstacle(head.move(Direction.UP))
        danger_left = is_obstacle(head.move(Direction.LEFT))
        danger_down = is_obstacle(head.move(Direction.DOWN))
        danger_right = is_obstacle(head.move(Direction.RIGHT))

        # Raycast function that stops at obstacles (wall/body)
        def raycast_apples(
            start: Point, direction: Direction
        ) -> tuple[bool, bool]:
            curr = start.move(direction)
            while (
                state.is_within_bounds(curr) and curr not in state.snake.body
            ):
                if curr in state.green_apples:
                    return True, False
                if curr in state.red_apples:
                    return False, True
                curr = curr.move(direction)
            return False, False

        green_up, red_up = raycast_apples(head, Direction.UP)
        green_left, red_left = raycast_apples(head, Direction.LEFT)
        green_down, red_down = raycast_apples(head, Direction.DOWN)
        green_right, red_right = raycast_apples(head, Direction.RIGHT)

        return cls(
            danger_up=danger_up,
            danger_left=danger_left,
            danger_down=danger_down,
            danger_right=danger_right,
            green_apple_up=green_up,
            green_apple_left=green_left,
            green_apple_down=green_down,
            green_apple_right=green_right,
            red_apple_up=red_up,
            red_apple_left=red_left,
            red_apple_down=red_down,
            red_apple_right=red_right,
        )


@dataclass(frozen=True)
class NeuralStateFeatures:
    """Numeric ray-distance features for the DQN engine.

    The agent sees along the vertical and horizontal rays from its head. Wall
    distances and body distances are encoded separately because walls are fixed
    terminal obstacles while body segments are dynamic obstacles. All distances
    use inverse encoding: adjacent danger is 1.0 and farther objects decay
    toward 0.0. A body value of 0.0 means no body segment is visible before the
    wall in that direction.
    """

    wall_up: float
    wall_left: float
    wall_down: float
    wall_right: float

    body_up: float
    body_left: float
    body_down: float
    body_right: float

    green_apple_up: float
    green_apple_left: float
    green_apple_down: float
    green_apple_right: float

    red_apple_up: float
    red_apple_left: float
    red_apple_down: float
    red_apple_right: float

    def to_tuple(self) -> tuple[float, ...]:
        return (
            self.wall_up,
            self.wall_left,
            self.wall_down,
            self.wall_right,
            self.body_up,
            self.body_left,
            self.body_down,
            self.body_right,
            self.green_apple_up,
            self.green_apple_left,
            self.green_apple_down,
            self.green_apple_right,
            self.red_apple_up,
            self.red_apple_left,
            self.red_apple_down,
            self.red_apple_right,
        )

    @classmethod
    def from_game_state(cls, state: GameState) -> "NeuralStateFeatures":
        if state.is_game_over or not state.snake.body:
            return cls(
                wall_up=1.0,
                wall_left=1.0,
                wall_down=1.0,
                wall_right=1.0,
                body_up=0.0,
                body_left=0.0,
                body_down=0.0,
                body_right=0.0,
                green_apple_up=0.0,
                green_apple_left=0.0,
                green_apple_down=0.0,
                green_apple_right=0.0,
                red_apple_up=0.0,
                red_apple_left=0.0,
                red_apple_down=0.0,
                red_apple_right=0.0,
            )

        def raycast(direction: Direction) -> tuple[float, float, float, float]:
            distance = 1
            body_distance = 0
            green_distance = 0
            red_distance = 0
            current = state.snake.head.move(direction)
            while state.is_within_bounds(current):
                if current in state.snake.body:
                    if body_distance == 0:
                        body_distance = distance
                elif body_distance == 0:
                    if green_distance == 0 and current in state.green_apples:
                        green_distance = distance
                    if red_distance == 0 and current in state.red_apples:
                        red_distance = distance
                current = current.move(direction)
                distance += 1

            wall = 1.0 / distance
            body = 0.0 if body_distance == 0 else 1.0 / body_distance
            green = 0.0 if green_distance == 0 else 1.0 / green_distance
            red = 0.0 if red_distance == 0 else 1.0 / red_distance
            return wall, body, green, red

        wall_up, body_up, green_up, red_up = raycast(Direction.UP)
        wall_left, body_left, green_left, red_left = raycast(Direction.LEFT)
        wall_down, body_down, green_down, red_down = raycast(Direction.DOWN)
        wall_right, body_right, green_right, red_right = raycast(
            Direction.RIGHT
        )

        return cls(
            wall_up=wall_up,
            wall_left=wall_left,
            wall_down=wall_down,
            wall_right=wall_right,
            body_up=body_up,
            body_left=body_left,
            body_down=body_down,
            body_right=body_right,
            green_apple_up=green_up,
            green_apple_left=green_left,
            green_apple_down=green_down,
            green_apple_right=green_right,
            red_apple_up=red_up,
            red_apple_left=red_left,
            red_apple_down=red_down,
            red_apple_right=red_right,
        )
