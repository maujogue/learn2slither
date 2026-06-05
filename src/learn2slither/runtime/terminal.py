import sys
from learn2slither.core import (
    Direction,
    GameOverReason,
    Point,
    create_initial_game,
)


def clear_terminal() -> None:
    """Clear the terminal in-place using ANSI escape codes for a smooth TUI."""
    sys.stdout.write("\033[H\033[2J")
    sys.stdout.flush()


def print_vision_grid(state) -> None:
    """Compute and print the snake's vision.

    Shows the vertical and horizontal sight lines that cross at the head.
    Renders an ASCII cross and flat arrays for agent parsing.
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
            # Print spaces leading up to the head column,
            # then print the vertical character
            # Each character is separated by a space (2 chars per column width)
            spaces = " " * (2 * (head.x + 1))
            print(f"{spaces}{vert_chars[y + 1]}")


def run_cli_game(width: int = 10, height: int = 10, speed: int = 6) -> None:
    """Runs the pure headless command-line loop of the snake game."""
    print(
        f"\n🐍 Initializing CLI Snake Game: {width}x{height} Grid "
        f"| Speed: {speed} steps/sec"
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
