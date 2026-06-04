import argparse
import os


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
        help="Number of experiments (runs) to do, returning the results and summary metrics",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Play manually (reverts autopilot default)",
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


def get_models_dir() -> str:
    """Return the project-level models directory."""
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(root_dir, "models")


def first_model_file(models_dir: str) -> str | None:
    """Return the first JSON model file in models_dir, sorted by name."""
    try:
        names = sorted(os.listdir(models_dir))
    except OSError:
        return None

    for name in names:
        path = os.path.join(models_dir, name)
        if name.endswith(".json") and os.path.isfile(path):
            return path
    return None


MODEL_ENGINE_SPECS = {
    "q": (".json", ("q_table", "q-")),
    "nn": (".json", ("dqn", "nn-")),
}


def model_engine_for_path(path: str) -> str | None:
    """Return the engine implied by a model file name, if it is known."""
    name = os.path.basename(path).lower()
    for engine, (extension, prefixes) in MODEL_ENGINE_SPECS.items():
        if name.endswith(extension) and name.startswith(prefixes):
            return engine
    return None


def find_model_files(models_dir: str, engine: str = "all") -> list[tuple[str, str]]:
    """Return discovered model files as (engine, path), sorted for stable output."""
    try:
        names = sorted(os.listdir(models_dir))
    except OSError:
        return []

    models = []
    for name in names:
        path = os.path.join(models_dir, name)
        if not os.path.isfile(path):
            continue
        detected_engine = model_engine_for_path(path)
        if detected_engine is None:
            continue
        if engine != "all" and detected_engine != engine:
            continue
        models.append((detected_engine, path))
    return models


def _mean_score(scores: list[int]) -> float:
    """Return the arithmetic mean for a non-empty score list."""
    if not scores:
        raise ValueError("At least one completed run is required to calculate score metrics.")
    return sum(scores) / len(scores)

def _median_sorted(values: list[int]) -> float:
    """Return the median for an already sorted non-empty list."""
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return float(values[midpoint])
    return (values[midpoint - 1] + values[midpoint]) / 2


def _quartiles_sorted(values: list[int]) -> tuple[float, float, float]:
    """Return Q1, median, and Q3 using Tukey's hinges."""
    median = _median_sorted(values)
    midpoint = len(values) // 2
    lower_half = values[:midpoint]
    upper_half = values[midpoint + (len(values) % 2) :]
    q1 = _median_sorted(lower_half) if lower_half else median
    q3 = _median_sorted(upper_half) if upper_half else median
    return q1, median, q3


def _score_summary_lines(scores: list[int]) -> list[str]:
    """Return human-readable summary statistics for completed test runs."""
    if not scores:
        raise ValueError("At least one completed run is required to calculate score metrics.")
    sorted_scores = sorted(scores)
    n_scores = len(sorted_scores)
    mean_score = sum(sorted_scores) / n_scores
    q1, median_score, q3 = _quartiles_sorted(sorted_scores)
    minimum_score = sorted_scores[0]
    maximum_score = sorted_scores[-1]
    iqr = q3 - q1
    low_outlier_limit = q1 - 1.5 * iqr
    high_outlier_limit = q3 + 1.5 * iqr
    outliers = [
        score
        for score in sorted_scores
        if score < low_outlier_limit or score > high_outlier_limit
    ]
    top_count = max(1, (n_scores + 9) // 10)
    top_scores = sorted_scores[-top_count:]
    return [
        f"Runs: {n_scores}",
        f"Scores: {scores}",
        f"Mean Score: {mean_score:.2f}",
        f"Median Score: {median_score:.2f}",
        f"Min Score: {minimum_score}",
        f"Q1 Score: {q1:.2f}",
        f"Q3 Score: {q3:.2f}",
        f"Max Score: {maximum_score}",
        f"IQR: {iqr:.2f}",
        f"Outlier Bounds: < {low_outlier_limit:.2f} or > {high_outlier_limit:.2f}",
        f"Outliers ({len(outliers)}): {outliers}",
        f"Top 10% Count: {top_count}",
        f"Top 10% Scores: {top_scores}",
    ]


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

    model_path = args.path
    if model_path is None:
        if args.engine == "q":
            prefix, ext = "q_table", "json"
        else:
            prefix, ext = "dqn", "json"
        model_path = os.path.join(models_dir, f"{prefix}_{args.sessions}.{ext}")

    # If path already exists, prompt the user if they want to continue training or reset the file.
    if os.path.exists(model_path):
        while True:
            try:
                response = (
                    input(
                        f"⚠️ model file '{model_path}' already exists.\n"
                        "Do you want to [C]ontinue training or [R]eset/overwrite the file? (c/r): "
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                print("\nTraining aborted.")
                return

            if response in ("c", "continue"):
                print("Continuing training from the existing model...")
                import re

                basename = os.path.basename(model_path)
                match = re.search(r"(\d+)", basename)
                if match:
                    existing_sessions = int(match.group(1))
                    new_sessions = existing_sessions + args.sessions
                    new_basename = basename.replace(
                        str(existing_sessions), str(new_sessions), 1
                    )
                    new_model_path = os.path.join(
                        os.path.dirname(model_path), new_basename
                    )
                else:
                    new_sessions = args.sessions
                    name, ext = os.path.splitext(basename)
                    new_basename = f"{name}_{args.sessions}{ext}"
                    new_model_path = os.path.join(
                        os.path.dirname(model_path), new_basename
                    )

                if new_model_path != model_path:
                    if os.path.exists(new_model_path):
                        print(
                            f"⚠️ Note: Destination file '{new_model_path}' already exists and will be overwritten."
                        )
                    try:
                        os.rename(model_path, new_model_path)
                        print(
                            f"Renamed model file to '{new_model_path}' to reflect final sessions ({new_sessions})."
                        )
                        model_path = new_model_path
                    except OSError as e:
                        print(
                            f"⚠️ Warning: Could not rename file to reflect new sessions: {e}"
                        )
                break
            elif response in ("r", "reset"):
                print(
                    "Resetting model file. A new model will be trained from scratch."
                )
                try:
                    os.remove(model_path)
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
            save_path=model_path,
            engine=args.engine,
        )
    else:
        from learn2slither.board import run_game

        run_game(
            initial_width=width,
            initial_height=height,
            initial_speed=speed,
            model_path=model_path,
            engine=args.engine,
            autopilot=True,
            training=True,
            episodes=args.sessions,
        )


def test(args: argparse.Namespace) -> None:
    """Sub-function to handle the testing flow."""
    width = max(5, min(25, args.width))
    height = max(5, min(25, args.height))
    speed = max(1, min(1000, args.speed))

    models_dir = get_models_dir()
    os.makedirs(models_dir, exist_ok=True)

    model_path = args.path
    if args.headless and not args.manual and model_path is None:
        raise SystemExit("test --headless requires a model path")
    if not args.headless and not args.manual and model_path is None:
        model_path = first_model_file(models_dir)
        if model_path is not None:
            print(f"Loaded first model from '{model_path}'")

    # Warn if model does not exist when not manual testing and no GUI fallback is possible.
    if not args.manual and model_path is not None and not os.path.exists(model_path):
        print(f"⚠️ Warning: model file '{model_path}' does not exist.")
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
                model_path=model_path,
                engine=args.engine,
                runs=args.runs,
            )
            for line in _score_summary_lines(scores):
                print(line)
    else:
        from learn2slither.board import run_game

        autopilot = not args.manual
        run_game(
            initial_width=width,
            initial_height=height,
            initial_speed=speed,
            model_path=model_path,
            engine=args.engine,
            autopilot=autopilot,
        )



def benchmark(args: argparse.Namespace) -> None:
    """Run every matching saved model headlessly and print the best performers."""
    width = max(5, min(25, args.width))
    height = max(5, min(25, args.height))
    runs = max(1, args.runs)
    top = max(1, args.top)

    models = find_model_files(get_models_dir(), args.engine)
    if not models:
        raise SystemExit(f"No saved models found for engine '{args.engine}'.")

    from learn2slither.cli import run_headless_autopilot

    results = []
    for engine, model_path in models:
        print(f"Benchmarking [{engine}] {model_path}")
        scores = run_headless_autopilot(
            width=width,
            height=height,
            model_path=model_path,
            engine=engine,
            runs=runs,
        )
        mean_score = _mean_score(scores)
        max_score = max(scores)
        results.append((mean_score, max_score, engine, model_path, scores))
        print(f"Mean Score: {mean_score:.2f} | Max Score: {max_score} | Scores: {scores}")

    results.sort(key=lambda result: (-result[0], -result[1], result[3]))

    print("\nBest Models")
    for rank, (mean_score, max_score, engine, model_path, scores) in enumerate(
        results[:top], start=1
    ):
        print(
            f"{rank}. [{engine}] {model_path} "
            f"(mean={mean_score:.2f}, max={max_score}, scores={scores})"
        )


def main() -> None:
    """Launcher entry point for the snake game.

    Accepts subcommands 'train', 'test', or 'benchmark'.
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
