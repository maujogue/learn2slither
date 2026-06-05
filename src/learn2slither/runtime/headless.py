def run_headless_autopilot(
    width: int = 10,
    height: int = 10,
    model_path: str | None = None,
    engine: str = "q",
    runs: int | None = 1,
    qtable_path: str | None = None,
    verbose: bool = False,
) -> list[int]:
    """Runs snake games headlessly using the selected trained agent."""
    from learn2slither.agents import (
        NeuralStateFeatures,
        StateFeatures,
        action_to_direction,
        create_agent,
    )
    from learn2slither.core import create_initial_game

    if model_path is None:
        model_path = qtable_path
    if runs is None:
        runs = 1

    agent = create_agent(engine, training=False)
    model_label = "Q-table" if engine == "q" else "DQN model"
    if model_path:
        if agent.load_from_file(model_path):
            print(f"Loaded {model_label} from '{model_path}'")
        else:
            print(
                f"⚠️ Warning: Could not load {model_label}"
                f" from '{model_path}'. Using untrained agent."
            )
    else:
        print(
            f"⚠️ Warning: No {model_label} path provided."
            " Using untrained agent."
        )

    scores = []
    max_steps = width * height * 10
    for i in range(1, runs + 1):
        state = create_initial_game(width=width, height=height)
        done = False
        steps = 0
        while not done and steps < max_steps:
            if engine == "nn":
                curr_features = NeuralStateFeatures.from_game_state(state)
            else:
                curr_features = StateFeatures.from_game_state(state)
            action = agent.get_action(
                curr_features, training=False, verbose=verbose
            )
            state.change_direction(action_to_direction(action))
            state.step()
            done = state.is_game_over
            steps += 1

        progress_str = (
            f"Game {i}/{runs} - Current Score: {len(state.snake.body)}"
        )
        if verbose:
            print(progress_str)
        else:
            print(f"{progress_str:<50}", end="\r")

        score = len(state.snake.body)
        scores.append(score)

    # Clear the progress line so subsequent prints are clean.
    if not verbose:
        print(" " * 60, end="\r")
    return scores
