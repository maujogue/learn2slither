import sys
from learn2slither.models import Direction, GameOverReason, Point, create_initial_game


def clear_terminal() -> None:
    """Clears the terminal screen in-place using ANSI escape codes for a smooth TUI."""
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()


def print_vision_grid(state) -> None:
    """Computes and prints the Snake's vision.

    Only shows the vertical and horizontal lines of sight that cross at the snake's head.
    Displays a beautiful visual ASCII cross representation and flat arrays for agent parsing.
    """
    clear_terminal()
    print("\n")
    print("=" * 60)
    print("  Legend: H=Head, S=Body, W=Wall, G=Green(+1), R=Red(-1), 0=Empty")
    print("=" * 60)
    width = state.config.width
    height = state.config.height
    head = state.snake.head

    # 1. Compute Horizontal Line characters (from x = -1 to width)
    horiz_chars = []
    for x in range(-1, width + 1):
        is_border = x == -1 or x == width or head.y == -1 or head.y == height
        if is_border:
            horiz_chars.append("W")
        else:
            p = Point(x, head.y)
            if p == head:
                horiz_chars.append("H")
            elif p in state.snake.body:
                horiz_chars.append("S")
            elif p in state.green_apples:
                horiz_chars.append("G")
            elif p in state.red_apples:
                horiz_chars.append("R")
            else:
                horiz_chars.append("0")

    # 2. Compute Vertical Line characters (from y = -1 to height)
    vert_chars = []
    for y in range(-1, height + 1):
        is_border = head.x == -1 or head.x == width or y == -1 or y == height
        if is_border:
            vert_chars.append("W")
        else:
            p = Point(head.x, y)
            if p == head:
                vert_chars.append("H")
            elif p in state.snake.body:
                vert_chars.append("S")
            elif p in state.green_apples:
                vert_chars.append("G")
            elif p in state.red_apples:
                vert_chars.append("R")
            else:
                vert_chars.append("0")

    # 3. Print the gorgeous Visual Sight Cross
    for y in range(-1, height + 1):
        if y == head.y:
            # Print the entire horizontal row
            print(" ".join(horiz_chars))
        else:
            # Print spaces leading up to the head column, then print the vertical character
            # Each character is separated by a space (2 chars per column width)
            spaces = " " * (2 * (head.x + 1))
            print(f"{spaces}{vert_chars[y + 1]}")


def run_cli_game(width: int = 10, height: int = 10, speed: int = 6) -> None:
    """Runs the pure headless command-line loop of the snake game."""
    print(
        f"\n🐍 Initializing CLI Snake Game: {width}x{height} Grid | Speed: {speed} steps/sec"
    )
    state = create_initial_game(width=width, height=height)

    # Setup input steering
    input_to_dir = {
        "W": Direction.UP,
        "UP": Direction.UP,
        "S": Direction.DOWN,
        "DOWN": Direction.DOWN,
        "A": Direction.LEFT,
        "LEFT": Direction.LEFT,
        "D": Direction.RIGHT,
        "RIGHT": Direction.RIGHT,
    }

    while not state.is_game_over:
        # Show State Vision
        print_vision_grid(state)
        print(f"Snake Length: {len(state.snake.body)}")

        # Prompt for next move
        try:
            prompt = "\nNext Move (W/A/S/D or UP/DOWN/LEFT/RIGHT): "
            user_input = input(prompt).strip().upper()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting game.")
            sys.exit(0)

        if user_input in input_to_dir:
            new_dir = input_to_dir[user_input]
            state.change_direction(new_dir)
        else:
            print("⚠️ Invalid input! Continuing in the current direction.")

        state.step()

    # Final Display
    print("\n" + "=" * 60)
    print_vision_grid(state)
    print("=" * 60)
    print("GAME OVER")

    if state.game_over_reason == GameOverReason.WALL_COLLISION:
        print("Reason: The snake crashed into a Wall [W]!")
    elif state.game_over_reason == GameOverReason.TAIL_COLLISION:
        print("Reason: The snake bit its own Body [S]!")
    elif state.game_over_reason == GameOverReason.STARVATION:
        print("Reason: The snake starved to death (length became 0)!")

    print(f"Final Score (Length): {len(state.snake.body)}")
    print("=" * 60)


def run_headless_autopilot(
    width: int = 10,
    height: int = 10,
    qtable_path: str | None = None,
    runs: int = 1,
) -> list[int]:
    """Runs the snake game headlessly using the Q-table agent.

    Plays the specified number of games and returns the list of scores.
    """
    from learn2slither.agent import QLearningAgent, StateFeatures
    from learn2slither.models import create_initial_game, Direction

    agent = QLearningAgent()
    if qtable_path:
        if agent.q_table.load_from_file(qtable_path):
            print(f"Loaded Q-table from '{qtable_path}'")
        else:
            print(
                f"⚠️ Warning: Could not load Q-table from '{qtable_path}'. Using untrained agent."
            )
    else:
        print("⚠️ Warning: No Q-table path provided. Using untrained agent.")

    scores = []
    max_steps = width * height * 10
    for i in range(1, runs + 1):
        state = create_initial_game(width=width, height=height)
        done = False
        steps = 0
        while not done and steps < max_steps:
            curr_features = StateFeatures.from_game_state(state)
            action = agent.get_action(curr_features, training=False)

            # Map absolute action to direction
            action_to_dir = {
                0: Direction.UP,
                1: Direction.LEFT,
                2: Direction.DOWN,
                3: Direction.RIGHT,
            }
            new_dir = action_to_dir[action]

            state.change_direction(new_dir)
            state.step()
            done = state.is_game_over
            steps += 1

        progress_str = f"Game {i}/{runs} - Current Score: {len(state.snake.body)}"
        print(f"{progress_str:<50}", end="\r")

        score = len(state.snake.body)
        scores.append(score)

    # Clear the progress line so subsequent prints are clean
    print(" " * 60, end="\r")
    return scores
