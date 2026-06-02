import os
import argparse


def setup_train_args(parser: argparse.ArgumentParser) -> None:
    """Helper to add options specific to training."""
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to save the Q-table",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        required=True,
        help="Number of training sessions (episodes) to do",
    )


def setup_test_args(parser: argparse.ArgumentParser) -> None:
    """Helper to add options specific to testing."""
    parser.add_argument(
        "path",
        type=str,
        default=None,
        help="Path to load the Q-table",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of experiments (runs) to do, returning the results and mean",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Play manually (reverts autopilot default)",
    )


def train(args: argparse.Namespace) -> None:
    """Sub-function to handle the training flow."""
    width = max(5, min(25, args.width))
    height = max(5, min(25, args.height))
    speed = max(1, min(1000, args.speed))

    # Resolve models directory at the root of the project
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    qtable_path = args.path
    if qtable_path is None:
        qtable_path = os.path.join(models_dir, f"q_table_{args.sessions}.json")

    # If path already exists, prompt the user if they want to continue training or reset the file.
    if os.path.exists(qtable_path):
        while True:
            try:
                response = (
                    input(
                        f"⚠️ Q-table file '{qtable_path}' already exists.\n"
                        "Do you want to [C]ontinue training or [R]eset/overwrite the file? (c/r): "
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                print("\nTraining aborted.")
                return

            if response in ("c", "continue"):
                print("Continuing training from the existing Q-table...")
                import re

                basename = os.path.basename(qtable_path)
                match = re.search(r"(\d+)", basename)
                if match:
                    existing_sessions = int(match.group(1))
                    new_sessions = existing_sessions + args.sessions
                    new_basename = basename.replace(
                        str(existing_sessions), str(new_sessions), 1
                    )
                    new_qtable_path = os.path.join(
                        os.path.dirname(qtable_path), new_basename
                    )
                else:
                    new_sessions = args.sessions
                    name, ext = os.path.splitext(basename)
                    new_basename = f"{name}_{args.sessions}{ext}"
                    new_qtable_path = os.path.join(
                        os.path.dirname(qtable_path), new_basename
                    )

                if new_qtable_path != qtable_path:
                    if os.path.exists(new_qtable_path):
                        print(
                            f"⚠️ Note: Destination file '{new_qtable_path}' already exists and will be overwritten."
                        )
                    try:
                        os.rename(qtable_path, new_qtable_path)
                        print(
                            f"Renamed Q-table file to '{new_qtable_path}' to reflect final sessions ({new_sessions})."
                        )
                        qtable_path = new_qtable_path
                    except OSError as e:
                        print(
                            f"⚠️ Warning: Could not rename file to reflect new sessions: {e}"
                        )
                break
            elif response in ("r", "reset"):
                print(
                    "Resetting Q-table file. A new Q-table will be trained from scratch."
                )
                try:
                    os.remove(qtable_path)
                except OSError as e:
                    print(f"Error removing existing file: {e}")
                break
            else:
                print("Invalid input. Please enter 'c' or 'r'.")

    if args.headless:
        from learn2slither.agent import train_agent

        train_agent(
            episodes=args.sessions,
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
            episodes=args.sessions,
        )


def test(args: argparse.Namespace) -> None:
    """Sub-function to handle the testing flow."""
    width = max(5, min(25, args.width))
    height = max(5, min(25, args.height))
    speed = max(1, min(1000, args.speed))

    # Resolve models directory at the root of the project
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Warn if Q-table does not exist when not manual testing
    if not args.manual and not os.path.exists(args.path):
        print(
            f"⚠️ Warning: Q-table file '{args.path}' does not exist."
        )
        return

    if args.headless:
        if args.manual:
            from learn2slither.cli import run_cli_game

            run_cli_game(width=width, height=height, speed=speed)
        else:
            from learn2slither.cli import run_headless_autopilot

            scores = run_headless_autopilot(
                width=width,
                height=height,
                qtable_path=args.path,
                runs=args.runs,
            )
            mean_score = sum(scores) / len(scores)
            print(f"Scores: {scores}")
            print(f"Mean Score: {mean_score:.2f}")
    else:
        from learn2slither.board import run_game

        autopilot = not args.manual
        run_game(
            initial_width=width,
            initial_height=height,
            initial_speed=speed,
            qtable_path=args.path,
            autopilot=autopilot,
        )


def main() -> None:
    """Launcher entry point for the snake game.

    Accepts subcommands 'train' or 'test'.
    """
    parser = argparse.ArgumentParser(description="Learn2Slither - Playable Snake Game")

    # Common arguments shared by both subparsers
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--headless", action="store_true", help="Run strictly in CLI mode"
    )
    parent_parser.add_argument(
        "--width", type=int, default=10, help="Initial board width (5-25)"
    )
    parent_parser.add_argument(
        "--height", type=int, default=10, help="Initial board height (5-20)"
    )
    parent_parser.add_argument(
        "--speed", type=int, default=6, help="Snake speed in steps per second (1-20)"
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Sub-commands"
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        parents=[parent_parser],
        help="Train the Q-learning agent",
    )
    setup_train_args(train_parser)

    # Test subcommand
    test_parser = subparsers.add_parser(
        "test",
        parents=[parent_parser],
        help="Test/run the Q-learning agent or play manually",
    )
    setup_test_args(test_parser)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)


if __name__ == "__main__":
    main()
