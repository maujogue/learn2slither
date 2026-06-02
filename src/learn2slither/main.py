import sys
import argparse


def main() -> None:
    """Launcher entry point for the snake game.

    Accepts the '--headless' command-line flag, as well as --width, --height, and --speed.
    - If '--headless' is provided, runs strictly as a command-line utility.
    - If not provided, opens the graphical Pygame interface AND outputs the state
      vision to the terminal console at the same time on every step.
    """
    parser = argparse.ArgumentParser(description="Learn2Slither - Playable Snake Game")
    parser.add_argument(
        "--headless", action="store_true", help="Run strictly in CLI mode"
    )
    parser.add_argument(
        "--width", type=int, default=10, help="Initial board width (5-25)"
    )
    parser.add_argument(
        "--height", type=int, default=10, help="Initial board height (5-20)"
    )
    parser.add_argument(
        "--speed", type=int, default=6, help="Snake speed in steps per second (1-20)"
    )

    args, unknown = parser.parse_known_args()

    # Enforce constraints and safe defaults
    width = max(5, min(25, args.width))
    height = max(5, min(25, args.height))
    speed = max(1, min(20, args.speed))

    if args.headless or "--headless" in sys.argv:
        from learn2slither.cli import run_cli_game

        run_cli_game(width=width, height=height, speed=speed)
    else:
        from learn2slither.board import run_game

        run_game(initial_width=width, initial_height=height, initial_speed=speed)


if __name__ == "__main__":
    main()
