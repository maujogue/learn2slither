import os

from learn2slither.agents.model_format import detect_model_engine


def _get_default_model_paths(model_path: str | None, engine: str) -> list[str]:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_name = "q_table.json" if engine == "q" else "dqn.json"
    candidates = [
        os.path.join(root_dir, "models", default_name),
        os.path.join(os.path.dirname(__file__), default_name),
    ]
    if model_path:
        candidates.insert(0, model_path)

    seen: set[str] = set()
    paths: list[str] = []
    for path in candidates:
        normalized = os.path.abspath(path)
        if normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
    return paths


def _matches_engine(path: str, engine: str) -> bool:
    return detect_model_engine(path) == engine


def _discover_model_files(model_path: str | None, engine: str) -> list[str]:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dirs = [os.path.join(root_dir, "models"), os.path.dirname(__file__)]
    paths = _get_default_model_paths(model_path, engine)
    seen = {os.path.abspath(path) for path in paths}

    for directory in dirs:
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in sorted(names):
            if not name.endswith(".json"):
                continue
            path = os.path.abspath(os.path.join(directory, name))
            if path in seen or not _matches_engine(path, engine):
                continue
            seen.add(path)
            paths.append(path)

    return [path for path in paths if os.path.exists(path)]


def _model_label(path: str) -> str:
    return os.path.basename(path) or path
