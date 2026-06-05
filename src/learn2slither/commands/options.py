import argparse


def setup_train_args(parser: argparse.ArgumentParser) -> None:
    """Helper to add options specific to training."""
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--engine",
        choices=("q", "nn"),
        default="q",
        help="Training engine: q for Q-table, nn for custom MLP DQN",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        required=True,
        help="Number of training sessions/iterations to do",
    )


def setup_test_args(parser: argparse.ArgumentParser) -> None:
    """Helper to add options specific to testing."""
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to load the trained model",
    )
    parser.add_argument(
        "--engine",
        choices=("q", "nn"),
        default="q",
        help="Testing engine: q for Q-table, nn for custom MLP DQN",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help=(
            "Number of experiments (runs) to do,"
            " returning the results and summary metrics"
        ),
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Play manually (reverts autopilot default)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each autopilot action during headless model testing",
    )


def setup_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Helper to add options specific to benchmarking all saved models."""
    parser.add_argument(
        "--engine",
        choices=("all", "q", "nn"),
        default="all",
        help="Model type to benchmark, or all for every known model type",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=300,
        help="Number of headless games to run per model",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Number of best models to print",
    )
