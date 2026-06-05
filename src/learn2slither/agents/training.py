import os
import re
import subprocess
import sys
import tempfile
import time

from learn2slither.agents import create_agent
from learn2slither.agents.actions import EngineName, action_to_direction
from learn2slither.agents.features import NeuralStateFeatures, StateFeatures
from learn2slither.agents.plots import (
    _loss_plot_path,
    _validation_plot_path,
    _write_loss_plot,
    _write_validation_plot,
)
from learn2slither.agents.rewards import compute_reward, get_min_green_dist
from learn2slither.core import create_initial_game


def _run_validation_command(
    model_path: str, engine: EngineName, width: int, height: int
) -> float:
    command = [
        "uv",
        "run",
        "l2s",
        "test",
        "--headless",
        "--width",
        str(width),
        "--height",
        str(height),
        "--engine",
        engine,
        "--runs",
        "500",
        model_path,
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except FileNotFoundError:
        fallback = [
            sys.executable,
            "-m",
            "learn2slither.commands.app",
            "test",
            "--headless",
            "--width",
            str(width),
            "--height",
            str(height),
            "--engine",
            engine,
            "--runs",
            "100",
            model_path,
        ]
        result = subprocess.run(
            fallback, check=True, capture_output=True, text=True
        )
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"Mean Score:\s*([0-9]+(?:\.[0-9]+)?)", output)
    if match is None:
        raise RuntimeError(
            "Could not parse validation mean score from command"
            f" output:\n{output}"
        )
    return float(match.group(1))


def train_agent(
    save_path: str | None,
    episodes: int = 15000,
    width: int = 10,
    height: int = 10,
    engine: EngineName = "q",
) -> None:
    """Trains the selected agent headlessly and saves the trained model."""
    engine_label = "Q-Learning" if engine == "q" else "DQN"
    model_label = "Q-table" if engine == "q" else "DQN model"
    print(
        f"\n🚀 Starting Headless {engine_label} Training"
        f" ({episodes} episodes) on {width}x{height} Grid..."
    )
    agent = create_agent(engine, training=True)

    if save_path and agent.load_from_file(save_path):
        print(
            f"Loaded existing {model_label} from '{save_path}'"
            " to continue training."
        )

    scores: list[int] = []
    steps_list: list[int] = []
    validation_points: list[tuple[int, float, float]] = []
    loss_points: list[tuple[int, float]] = []
    temp_validation_dir = (
        tempfile.TemporaryDirectory() if save_path is None else None
    )
    validation_model_path = save_path
    if validation_model_path is None and temp_validation_dir is not None:
        filename = (
            "q_table_validation.json"
            if engine == "q"
            else "dqn_validation.json"
        )
        validation_model_path = os.path.join(
            temp_validation_dir.name, filename
        )
    plot_path = (
        _validation_plot_path(save_path)
        if save_path is not None
        else os.path.abspath("validation_performance.svg")
    )
    loss_plot_path = (
        _loss_plot_path(save_path)
        if save_path is not None
        else os.path.abspath("dqn_training_loss.svg")
    )

    started_at = time.perf_counter()
    for ep in range(1, episodes + 1):
        state = create_initial_game(width=width, height=height)
        score = len(state.snake.body)
        steps = 0
        done = False
        last_dist = get_min_green_dist(state)
        episode_loss_sum = 0.0
        episode_loss_count = 0

        while not done:
            curr_features = (
                NeuralStateFeatures.from_game_state(state)
                if engine == "nn"
                else StateFeatures.from_game_state(state)
            )
            action = agent.get_action(curr_features, training=True)
            state.change_direction(action_to_direction(action))
            state.step()
            steps += 1

            next_features = (
                NeuralStateFeatures.from_game_state(state)
                if engine == "nn"
                else StateFeatures.from_game_state(state)
            )
            done = state.is_game_over
            reward, score, last_dist = compute_reward(
                state, score, last_dist, engine
            )
            training_steps_before = agent.training_steps
            agent.update(curr_features, action, reward, next_features, done)
            if (
                engine == "nn"
                and agent.training_steps != training_steps_before
                and agent.last_loss is not None
            ):
                episode_loss_sum += agent.last_loss
                episode_loss_count += 1

        agent.decay_epsilon()
        scores.append(score)
        steps_list.append(steps)
        if episode_loss_count > 0:
            loss_points.append((ep, episode_loss_sum / episode_loss_count))

        if ep % 10 == 0 or ep == 1:
            mean_score = sum(scores[-100:]) / len(scores[-100:])
            mean_steps = sum(steps_list[-100:]) / len(steps_list[-100:])
            print(
                f"Episode {ep:5d}/{episodes} |"
                f" Avg Score (last 100): {mean_score:4.1f} |"
                f" Avg Steps: {mean_steps:5.1f}"
                f" | Epsilon: {agent.epsilon:5.3f}"
                f" | {agent.training_status()}"
            )

        validation_interval = max(1, episodes // 5)
        if ep % validation_interval == 0:
            if validation_model_path is None:
                raise RuntimeError("validation_model_path was not initialized")
            agent.save_to_file(validation_model_path)
            validation_mean = _run_validation_command(
                validation_model_path,
                engine,
                width,
                height,
            )
            elapsed = time.perf_counter() - started_at
            validation_points.append((ep, validation_mean, elapsed))
            print(
                f"Validation after {ep:5d}/{episodes} episodes | "
                f"Mean Score (100 runs): {validation_mean:.2f}"
                f" | Training Time: {elapsed:.2f}s"
            )

    if episodes > 0 and (
        not validation_points or validation_points[-1][0] != episodes
    ):
        if validation_model_path is None:
            raise RuntimeError("validation_model_path was not initialized")
        agent.save_to_file(validation_model_path)
        validation_mean = _run_validation_command(
            validation_model_path, engine, width, height
        )
        elapsed = time.perf_counter() - started_at
        validation_points.append((episodes, validation_mean, elapsed))
        print(
            f"Final validation after {episodes:5d}/{episodes} episodes | "
            f"Mean Score (100 runs): {validation_mean:.2f}"
            f" | Training Time: {elapsed:.2f}s"
        )

    if validation_points:
        _write_validation_plot(validation_points, plot_path)
        print(f"Validation performance plot saved to '{plot_path}'.")
    if loss_points:
        _write_loss_plot(loss_points, loss_plot_path)
        print(f"DQN training loss plot saved to '{loss_plot_path}'.")

    if save_path:
        agent.save_to_file(save_path)
        print(
            f"🎉 Training completed successfully!"
            f" Trained {model_label} saved to '{save_path}'."
        )
    else:
        print(
            f"🎉 Training completed successfully!"
            f" ({model_label} was not saved since save_path is None)"
        )
    if temp_validation_dir is not None:
        temp_validation_dir.cleanup()
