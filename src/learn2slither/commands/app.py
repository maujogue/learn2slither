import argparse

from learn2slither.commands.handlers import benchmark, test, train
from learn2slither.commands.options import (
    setup_benchmark_args,
    setup_test_args,
    setup_train_args,
)


def main() -> None:
    """Launcher entry point for the snake game.

    Accepts subcommands 'train', 'test', or 'benchmark'.
    """
    parser = argparse.ArgumentParser(
        description="Learn2Slither - Playable Snake Game"
    )

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
        "--speed",
        type=int,
        default=6,
        help="Snake speed in steps per second (1-20)",
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

    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        parents=[parent_parser],
        help="Run every saved model of one type or all types and rank the best models",
    )
    setup_benchmark_args(benchmark_parser)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    elif args.command == "benchmark":
        benchmark(args)


if __name__ == "__main__":
    main()
