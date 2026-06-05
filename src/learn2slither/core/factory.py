from learn2slither.core import BoardConfig, Direction, Point, Snake, GameState


def create_initial_game(width: int = 10, height: int = 10) -> GameState:
    """Create a game state with a length-3 contiguous snake and apples."""
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
            # Align head direction: neck (path[1]) toward head (path[0])
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
