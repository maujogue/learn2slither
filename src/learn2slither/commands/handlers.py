import argparse
import os

from learn2slither.agents.model_format import detect_model_engine
from learn2slither.commands.stats import _mean_score, _score_summary_lines


def get_models_dir() -> str:
    """Return the project-level models directory."""
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(root_dir, "models")


def first_model_file(models_dir: str, engine: str | None = None) -> str | None:
    """Return the first JSON model file in models_dir, sorted by name."""
    try:
        names = sorted(os.listdir(models_dir))
    except OSError:
        return None

    for name in names:
        path = os.path.join(models_dir, name)
        if not name.endswith(".json") or not os.path.isfile(path):
            continue
        if engine is not None and model_engine_for_path(path) != engine:
            continue
        return path
    return None


def model_engine_for_path(path: str) -> str | None:
    """Return the engine implied by a model file's JSON payload, if known."""
    return detect_model_engine(path)


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
        from learn2slither.agents.training import train_agent

        train_agent(
            episodes=args.sessions,
            width=width,
            height=height,
            save_path=model_path,
            engine=args.engine,
        )
    else:
        from learn2slither.pygame_ui.app import run_game

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
        model_path = first_model_file(models_dir, engine=args.engine)
        if model_path is not None:
            print(f"Loaded first model from '{model_path}'")

    # Warn if model does not exist when not manual testing and no GUI fallback is possible.
    if not args.manual and model_path is not None and not os.path.exists(model_path):
        print(f"⚠️ Warning: model file '{model_path}' does not exist.")
        return

    if args.headless:
        if args.manual:
            from learn2slither.runtime.terminal import run_cli_game

            run_cli_game(width=width, height=height, speed=speed)
        else:
            from learn2slither.runtime.headless import run_headless_autopilot

            scores = run_headless_autopilot(
                width=width,
                height=height,
                model_path=model_path,
                engine=args.engine,
                runs=args.runs,
                verbose=args.verbose,
            )
            for line in _score_summary_lines(scores):
                print(line)
    else:
        from learn2slither.pygame_ui.app import run_game

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

    from learn2slither.runtime.headless import run_headless_autopilot

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
