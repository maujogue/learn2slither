import os
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
    parser.add_argument(
        "--train", action="store_true", help="Train the Q-learning agent headlessly"
    )
    parser.add_argument(
        "--episodes", type=int, required=True, help="Number of training episodes"
    )
    parser.add_argument(
        "--qtable",
        type=str,
        default=None,
        help="Path to save/load the Q-table",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Play manually (reverts autopilot default)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run in headless autopilot",
    )

    args, unknown = parser.parse_known_args()

    # Enforce constraints and safe defaults
    width = max(5, min(25, args.width))
    height = max(5, min(25, args.height))
    speed = max(1, min(1000, args.speed))

    # Resolve models directory at the root of the project
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Determine Q-table path
    qtable_path = args.qtable
    if qtable_path is None:
        # If user explicitly trained, or specified --episodes, or if they are training,
        # use models/q_table{args.episodes}.json
        if args.train or "--episodes" in sys.argv:
            path = os.path.join(models_dir, f"q_table_{args.episodes}.json")
            # If playing and underscore version exists, load it
            if not args.train and os.path.exists(path):
                qtable_path = path
        else:
            # Default fallbacks when playing without --episodes or --qtable:
            # Check if models/q_table.json exists
            default_path = os.path.join(models_dir, "q_table.json")
            if os.path.exists(default_path):
                qtable_path = default_path
            else:
                # Otherwise, fallback to q_table_15000.json
                qtable_path = os.path.join(models_dir, f"q_table_{args.episodes}.json")

    if args.train or "--train" in sys.argv:
        if args.headless or "--headless" in sys.argv:
            from learn2slither.agent import train_agent

            train_agent(
                episodes=args.episodes,
                width=width,
                height=height,
                save_path=qtable_path,
            )
        else:
            from learn2slither.board import run_game

            run_game(
                initial_width=width,
                initial_height=height,
                initial_speed=speed,
                qtable_path=qtable_path,
                autopilot=True,
                training=True,
                episodes=args.episodes,
            )
    elif args.headless or "--headless" in sys.argv:
        if args.manual or "--manual" in sys.argv:
            from learn2slither.cli import run_cli_game

            run_cli_game(width=width, height=height, speed=speed)
        else:
            from learn2slither.cli import run_headless_autopilot

            scores = run_headless_autopilot(
                width=width,
                height=height,
                qtable_path=qtable_path,
                iterations=args.iterations,
            )
            if args.iterations == 1:
                score = scores[0]
                print(f"Score: {score}")
            else:
                mean_score = sum(scores) / len(scores)
                print(f"Scores: {scores}")
                print(f"Mean Score: {mean_score:.2f}")
    else:
        from learn2slither.board import run_game

        autopilot = not (args.manual or "--manual" in sys.argv)
        run_game(
            initial_width=width,
            initial_height=height,
            initial_speed=speed,
            qtable_path=qtable_path,
            autopilot=autopilot,
        )


if __name__ == "__main__":
    main()
