import os
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from mlp_core import DQN
from learn2slither.models import GameState, Direction, Point, create_initial_game


@dataclass(frozen=True)
class StateFeatures:
    """A list of state features extracted from the GameState.

    To keep the Q-learning state space compact and enable extremely fast,
    generalized learning, we use 12 boolean absolute features.
    These features describe immediate surroundings and direction to the closest
    apples in absolute grid directions (UP, LEFT, DOWN, RIGHT).
    """

    danger_up: bool
    danger_left: bool
    danger_down: bool
    danger_right: bool

    green_apple_up: bool
    green_apple_left: bool
    green_apple_down: bool
    green_apple_right: bool

    red_apple_up: bool
    red_apple_left: bool
    red_apple_down: bool
    red_apple_right: bool

    def to_tuple(self) -> tuple[bool, ...]:
        """Converts features to a hashable tuple for Q-table indexing."""
        return (
            self.danger_up,
            self.danger_left,
            self.danger_down,
            self.danger_right,
            self.green_apple_up,
            self.green_apple_left,
            self.green_apple_down,
            self.green_apple_right,
            self.red_apple_up,
            self.red_apple_left,
            self.red_apple_down,
            self.red_apple_right,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Returns a list of all feature names in order."""
        return [
            "danger_up",
            "danger_left",
            "danger_down",
            "danger_right",
            "green_apple_up",
            "green_apple_left",
            "green_apple_down",
            "green_apple_right",
            "red_apple_up",
            "red_apple_left",
            "red_apple_down",
            "red_apple_right",
        ]

    @classmethod
    def from_game_state(cls, state: GameState) -> "StateFeatures":
        """Extracts absolute state features from the current GameState."""
        if state.is_game_over or not state.snake.body:
            return cls(
                danger_up=True,
                danger_left=True,
                danger_down=True,
                danger_right=True,
                green_apple_up=False,
                green_apple_left=False,
                green_apple_down=False,
                green_apple_right=False,
                red_apple_up=False,
                red_apple_left=False,
                red_apple_down=False,
                red_apple_right=False,
            )

        head = state.snake.head

        # Helper to check for obstacles
        def is_obstacle(p: Point) -> bool:
            if not state.is_within_bounds(p):
                return True
            # Colliding with the snake's own body is an obstacle
            if p in state.snake.body:
                return True
            return False

        danger_up = is_obstacle(head.move(Direction.UP))
        danger_left = is_obstacle(head.move(Direction.LEFT))
        danger_down = is_obstacle(head.move(Direction.DOWN))
        danger_right = is_obstacle(head.move(Direction.RIGHT))

        # Raycast function that stops at obstacles (wall/body)
        def raycast_apples(start: Point, direction: Direction) -> tuple[bool, bool]:
            curr = start.move(direction)
            while state.is_within_bounds(curr) and curr not in state.snake.body:
                if curr in state.green_apples:
                    return True, False
                if curr in state.red_apples:
                    return False, True
                curr = curr.move(direction)
            return False, False

        green_up, red_up = raycast_apples(head, Direction.UP)
        green_left, red_left = raycast_apples(head, Direction.LEFT)
        green_down, red_down = raycast_apples(head, Direction.DOWN)
        green_right, red_right = raycast_apples(head, Direction.RIGHT)

        return cls(
            danger_up=danger_up,
            danger_left=danger_left,
            danger_down=danger_down,
            danger_right=danger_right,
            green_apple_up=green_up,
            green_apple_left=green_left,
            green_apple_down=green_down,
            green_apple_right=green_right,
            red_apple_up=red_up,
            red_apple_left=red_left,
            red_apple_down=red_down,
            red_apple_right=red_right,
        )


@dataclass(frozen=True)
class NeuralStateFeatures:
    """Numeric ray-distance features for the DQN engine.

    The agent sees along the vertical and horizontal rays from its head. Wall
    distances and body distances are encoded separately because walls are fixed
    terminal obstacles while body segments are dynamic obstacles. All distances
    use inverse encoding: adjacent danger is 1.0 and farther objects decay
    toward 0.0. A body value of 0.0 means no body segment is visible before the
    wall in that direction.
    """

    wall_up: float
    wall_left: float
    wall_down: float
    wall_right: float

    body_up: float
    body_left: float
    body_down: float
    body_right: float

    green_apple_up: float
    green_apple_left: float
    green_apple_down: float
    green_apple_right: float

    red_apple_up: float
    red_apple_left: float
    red_apple_down: float
    red_apple_right: float

    def to_tuple(self) -> tuple[float, ...]:
        return (
            self.wall_up,
            self.wall_left,
            self.wall_down,
            self.wall_right,
            self.body_up,
            self.body_left,
            self.body_down,
            self.body_right,
            self.green_apple_up,
            self.green_apple_left,
            self.green_apple_down,
            self.green_apple_right,
            self.red_apple_up,
            self.red_apple_left,
            self.red_apple_down,
            self.red_apple_right,
        )

    @classmethod
    def from_game_state(cls, state: GameState) -> "NeuralStateFeatures":
        if state.is_game_over or not state.snake.body:
            return cls(
                wall_up=1.0,
                wall_left=1.0,
                wall_down=1.0,
                wall_right=1.0,
                body_up=0.0,
                body_left=0.0,
                body_down=0.0,
                body_right=0.0,
                green_apple_up=0.0,
                green_apple_left=0.0,
                green_apple_down=0.0,
                green_apple_right=0.0,
                red_apple_up=0.0,
                red_apple_left=0.0,
                red_apple_down=0.0,
                red_apple_right=0.0,
            )

        def raycast(direction: Direction) -> tuple[float, float, float, float]:
            distance = 1
            body_distance = 0
            green_distance = 0
            red_distance = 0
            current = state.snake.head.move(direction)
            while state.is_within_bounds(current):
                if current in state.snake.body:
                    if body_distance == 0:
                        body_distance = distance
                elif body_distance == 0:
                    if green_distance == 0 and current in state.green_apples:
                        green_distance = distance
                    if red_distance == 0 and current in state.red_apples:
                        red_distance = distance
                current = current.move(direction)
                distance += 1

            wall = 1.0 / distance
            body = 0.0 if body_distance == 0 else 1.0 / body_distance
            green = 0.0 if green_distance == 0 else 1.0 / green_distance
            red = 0.0 if red_distance == 0 else 1.0 / red_distance
            return wall, body, green, red

        wall_up, body_up, green_up, red_up = raycast(Direction.UP)
        wall_left, body_left, green_left, red_left = raycast(Direction.LEFT)
        wall_down, body_down, green_down, red_down = raycast(Direction.DOWN)
        wall_right, body_right, green_right, red_right = raycast(Direction.RIGHT)

        return cls(
            wall_up=wall_up,
            wall_left=wall_left,
            wall_down=wall_down,
            wall_right=wall_right,
            body_up=body_up,
            body_left=body_left,
            body_down=body_down,
            body_right=body_right,
            green_apple_up=green_up,
            green_apple_left=green_left,
            green_apple_down=green_down,
            green_apple_right=green_right,
            red_apple_up=red_up,
            red_apple_left=red_left,
            red_apple_down=red_down,
            red_apple_right=red_right,
        )


@dataclass
class QValue:
    """A Q-value entity representing the expected cumulative reward.

    Tracks additional metadata such as the number of updates to monitor training density.
    """

    value: float = 0.0
    n_updates: int = 0

    def update(self, target: float, alpha: float) -> None:
        """Applies the Q-learning temporal difference update."""
        self.value += alpha * (target - self.value)
        self.n_updates += 1


class QTable:
    """A Q-table class wrapping a dictionary of state-action mappings.

    Actions are represented as absolute actions:
    0 = UP, 1 = LEFT, 2 = DOWN, 3 = RIGHT
    """

    def __init__(self) -> None:
        # Dictionary mapping: state_tuple (tuple of 12 booleans) -> {action_id (0..3) -> QValue}
        self.table: dict[tuple[bool, ...], dict[int, QValue]] = {}

    def get_q_value(self, state: StateFeatures, action: int) -> QValue:
        """Retrieves a QValue entity for a given state and action, initializing it if absent."""
        state_tuple = state.to_tuple()
        if state_tuple not in self.table:
            self.table[state_tuple] = {
                0: QValue(0.0),
                1: QValue(0.0),
                2: QValue(0.0),
                3: QValue(0.0),
            }
        return self.table[state_tuple][action]

    def get_best_action_value(self, state: StateFeatures) -> tuple[int, float]:
        """Returns the best absolute action and its corresponding Q-value."""
        state_tuple = state.to_tuple()
        if state_tuple not in self.table:
            # If state not visited, default to UP (0) and 0.0 Q-value
            return 0, 0.0

        best_action = 0
        best_val = -float("inf")
        actions = list(self.table[state_tuple].items())
        # Shuffle actions to break ties randomly and prevent bias
        random.shuffle(actions)
        for act, q_val in actions:
            if q_val.value > best_val:
                best_val = q_val.value
                best_action = act
        return best_action, best_val

    def save_to_file(self, filepath: str) -> None:
        """Serializes the Q-table to a JSON file."""
        serialized = {}
        for state_tuple, action_dict in self.table.items():
            # Convert boolean tuple key to a comma-separated string
            key_str = ",".join(str(int(b)) for b in state_tuple)
            serialized[key_str] = {
                str(act): {"value": q_val.value, "n_updates": q_val.n_updates}
                for act, q_val in action_dict.items()
            }
        with open(filepath, "w") as f:
            json.dump(serialized, f, indent=2)

    def load_from_file(self, filepath: str) -> bool:
        """Deserializes the Q-table from a JSON file. Returns True if successful."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()
                if not content:
                    return False
                serialized = json.loads(content)
        except Exception:
            return False

        self.table.clear()
        for key_str, action_dict in serialized.items():
            state_tuple = tuple(bool(int(x)) for x in key_str.split(","))
            self.table[state_tuple] = {}
            for act_str, val_dict in action_dict.items():
                act = int(act_str)
                self.table[state_tuple][act] = QValue(
                    value=float(val_dict["value"]),
                    n_updates=int(val_dict.get("n_updates", 0)),
                )
        return True


ACTION_TO_DIRECTION: tuple[Direction, ...] = (
    Direction.UP,
    Direction.LEFT,
    Direction.DOWN,
    Direction.RIGHT,
)
EngineName = Literal["q", "nn"]


def action_to_direction(action: int) -> Direction:
    """Converts an absolute action id to a Direction."""
    return ACTION_TO_DIRECTION[action]


def get_min_green_dist(state: GameState) -> float:
    """Returns Manhattan distance from snake head to the closest green apple."""
    if not state.green_apples:
        return 0.0
    head = state.snake.head
    return min(abs(apple.x - head.x) + abs(apple.y - head.y) for apple in state.green_apples)


def compute_reward(
    state: GameState,
    previous_score: int,
    last_green_dist: float,
    engine: str = "q",
) -> tuple[float, int, float]:
    """Computes engine-specific reward shaping.

    Q-learning keeps the original large tabular rewards. DQN uses the same
    events but scaled to small magnitudes so the MLP does not chase unstable
    targets; its survival reward is intentionally tiny so apple-seeking remains
    dominant even with multi-step returns.
    """
    if state.is_game_over:
        crash_penalty = -100.0 if engine == "q" else -1.0
        return crash_penalty, previous_score, last_green_dist

    new_score = len(state.snake.body)
    if new_score > previous_score:
        grow_reward = 100.0 if engine == "q" else 1.0
        return grow_reward, new_score, last_green_dist
    if new_score < previous_score:
        shrink_penalty = -30.0 if engine == "q" else -0.3
        return shrink_penalty, new_score, last_green_dist

    reward = 1.0 if engine == "q" else 0.01
    new_dist = get_min_green_dist(state)
    if new_dist < last_green_dist:
        reward += 10.0 if engine == "q" else 0.1
    elif new_dist > last_green_dist:
        reward -= 15.0 if engine == "q" else 0.15
    return reward, previous_score, new_dist


class QLearningAgent:
    """Orchestrates Q-learning training and action selection for the Snake game."""

    engine_name: EngineName = "q"

    def __init__(
        self,
        alpha: float = 0.1,  # Learning rate
        gamma: float = 0.95,  # Discount factor
        epsilon: float = 0.9,  # Exploration rate
        epsilon_decay: float = 0.995,  # Epsilon decay rate per episode
        min_epsilon: float = 0.005,  # Minimum exploration rate
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = QTable()
        self.last_action: int = 0
        self.last_state: StateFeatures | None = None
        self.training_steps = 0

    def get_action(
        self, state: StateFeatures, training: bool = True, verbose: bool | None = None
    ) -> int:
        """Selects an action using an epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            action, _ = self.q_table.get_best_action_value(state)

        if verbose is None:
            verbose = not training

        if verbose:
            action_names = {0: "UP", 1: "LEFT", 2: "DOWN", 3: "RIGHT"}
            print(f"Agent choice: {action_names[action]}")

        return action

    def update(
        self,
        state: StateFeatures,
        action: int,
        reward: float,
        next_state: StateFeatures,
        done: bool,
    ) -> None:
        """Performs the Q-learning temporal difference update."""
        q_entity = self.q_table.get_q_value(state, action)

        if done:
            target = reward
        else:
            _, best_next_q = self.q_table.get_best_action_value(next_state)
            target = reward + self.gamma * best_next_q

        q_entity.update(target, self.alpha)
        self.training_steps += 1

    def decay_epsilon(self) -> None:
        """Decays the exploration rate after an episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def load_from_file(self, filepath: str) -> bool:
        return self.q_table.load_from_file(filepath)

    def save_to_file(self, filepath: str) -> None:
        self.q_table.save_to_file(filepath)
    def training_status(self) -> str:
        return f"States: {len(self.q_table.table)}"

    def telemetry_summary(self) -> str:
        return f"Unique States: {len(self.q_table.table)}"

    def best_action_value(self, state: StateFeatures) -> tuple[int, float]:
        return self.q_table.get_best_action_value(state)


@dataclass(frozen=True)
class ReplayTransition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool


class DQNAgent:
    """DQN adapter exposing the same surface as QLearningAgent."""

    engine_name: EngineName = "nn"
    state_size = 16
    n_actions = 4

    def __init__(
        self,
        epsilon: float = 0.9,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.005,
        gamma: float = 0.95,
        learning_rate: float = 0.0005,
        hidden_layers: list[int] | None = None,
        batch_size: int = 64,
        replay_capacity: int = 10_000,
        target_sync_interval: int = 200,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers if hidden_layers is not None else [64, 64]
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.target_sync_interval = target_sync_interval
        self.seed = seed
        self.training_steps = 0
        self.last_loss: float | None = None
        self.replay_memory: deque[ReplayTransition] = deque(maxlen=replay_capacity)
        self.model = DQN(
            self.state_size,
            self.n_actions,
            self.hidden_layers,
            gamma=gamma,
            learning_rate=learning_rate,
            optimizer="rmsprop",
            activation="relu",
            seed=seed,
            use_target_network=True,
        )

    @staticmethod
    def vectorize(state: StateFeatures | NeuralStateFeatures) -> tuple[float, ...]:
        if isinstance(state, NeuralStateFeatures):
            return state.to_tuple()
        return tuple(1.0 if value else 0.0 for value in state.to_tuple())

    @staticmethod
    def _array(rows: list[tuple[float, ...]]) -> np.ndarray:
        return np.asarray(rows, dtype=np.float64)

    def get_action(
        self, state: StateFeatures | NeuralStateFeatures, training: bool = True, verbose: bool | None = None
    ) -> int:
        epsilon = self.epsilon if training else 0.0
        states = self._array([self.vectorize(state)])
        action = int(self.model.select_actions(states, epsilon=epsilon)[0])

        if verbose is None:
            verbose = not training
        if verbose:
            action_names = {0: "UP", 1: "LEFT", 2: "DOWN", 3: "RIGHT"}
            print(f"Agent choice: {action_names[action]}")

        return action

    def update(
        self,
        state: StateFeatures | NeuralStateFeatures,
        action: int,
        reward: float,
        next_state: StateFeatures | NeuralStateFeatures,
        done: bool,
    ) -> None:
        self.replay_memory.append(
            ReplayTransition(
                state=self.vectorize(state),
                action=action,
                reward=reward,
                next_state=self.vectorize(next_state),
                done=done,
            )
        )
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(list(self.replay_memory), self.batch_size)
        states = self._array([transition.state for transition in batch])
        actions = np.asarray([transition.action for transition in batch], dtype=np.intp)
        rewards = np.asarray([transition.reward for transition in batch], dtype=np.float64)
        next_states = self._array([transition.next_state for transition in batch])
        dones = np.asarray([transition.done for transition in batch], dtype=bool)
        self.last_loss = float(
            self.model.train_batch(states, actions, rewards, next_states, dones, loss="huber")
        )
        self.training_steps += 1
        if self.training_steps % self.target_sync_interval == 0:
            self.model.sync_target_network()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_to_file(self, filepath: str) -> None:
        payload = {
            "engine": "nn",
            "version": 2,
            "feature_encoding": "ray_wall_body_inverse_v1",
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "hidden_layers": self.hidden_layers,
            "batch_size": self.batch_size,
            "replay_capacity": self.replay_capacity,
            "target_sync_interval": self.target_sync_interval,
            "seed": self.seed,
            "training_steps": self.training_steps,
            "last_loss": self.last_loss,
            "network": self._network_payload(self.model.network),
            "target_network": self._network_payload(self.model.target_network),
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

    def load_from_file(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r") as f:
                content = f.read().strip()
                if not content:
                    return False
                payload = json.loads(content)
            if payload.get("engine") != "nn" or payload.get("feature_encoding") != "ray_wall_body_inverse_v1":
                return False
            self.epsilon = float(payload.get("epsilon", self.epsilon))
            self.epsilon_decay = float(payload.get("epsilon_decay", self.epsilon_decay))
            self.min_epsilon = float(payload.get("min_epsilon", self.min_epsilon))
            self.gamma = float(payload.get("gamma", self.gamma))
            self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
            self.hidden_layers = [int(v) for v in payload.get("hidden_layers", self.hidden_layers)]
            self.batch_size = int(payload.get("batch_size", self.batch_size))
            self.replay_capacity = int(payload.get("replay_capacity", self.replay_capacity))
            self.target_sync_interval = int(
                payload.get("target_sync_interval", self.target_sync_interval)
            )
            self.seed = int(payload.get("seed", self.seed))
            self.training_steps = int(payload.get("training_steps", 0))
            last_loss = payload.get("last_loss")
            self.last_loss = None if last_loss is None else float(last_loss)
            self.replay_memory = deque(maxlen=self.replay_capacity)
            self.model = DQN(
                self.state_size,
                self.n_actions,
                self.hidden_layers,
                gamma=self.gamma,
                learning_rate=self.learning_rate,
                optimizer="rmsprop",
                activation="relu",
                seed=self.seed,
                use_target_network=True,
            )
            self._load_network_payload(self.model.network, payload["network"])
            self._load_network_payload(self.model.target_network, payload["target_network"])
        except Exception:
            return False
        return True

    def training_status(self) -> str:
        loss = "n/a" if self.last_loss is None else f"{self.last_loss:.4f}"
        return f"Replay: {len(self.replay_memory)} | Loss: {loss}"

    def telemetry_summary(self) -> str:
        return f"Replay Samples: {len(self.replay_memory)}"

    def best_action_value(self, state: StateFeatures | NeuralStateFeatures) -> tuple[int, float]:
        q_values = self.model.q_values(self._array([self.vectorize(state)]))[0]
        action = int(np.argmax(q_values))
        return action, float(q_values[action])

    @staticmethod
    def _network_payload(network) -> dict[str, object]:
        layers = [
            {"weights": weights.tolist(), "biases": biases.tolist()}
            for weights, biases in network.parameters()
        ]
        return {
            "layers": layers,
            "rms_weights": [value.tolist() for value in network._rms_W],
            "rms_biases": [value.tolist() for value in network._rms_b],
        }

    @staticmethod
    def _load_network_payload(network, payload: dict[str, Any]) -> None:
        layers = payload["layers"]
        for index, layer in enumerate(layers):
            weights, biases = network._layers[index]
            np.copyto(weights, np.asarray(layer["weights"], dtype=np.float64))
            np.copyto(biases, np.asarray(layer["biases"], dtype=np.float64))

        rms_weights = payload.get("rms_weights", [])
        rms_biases = payload.get("rms_biases", [])
        if rms_weights and rms_biases:
            if not network._rms_W:
                network._rms_W = [np.zeros_like(weights) for weights, _ in network._layers]
                network._rms_b = [np.zeros_like(biases) for _, biases in network._layers]
            for index, rms_weight in enumerate(rms_weights):
                np.copyto(network._rms_W[index], np.asarray(rms_weight, dtype=np.float64))
            for index, rms_bias in enumerate(rms_biases):
                np.copyto(network._rms_b[index], np.asarray(rms_bias, dtype=np.float64))




def create_agent(engine: str, *, training: bool = False):
    if engine == "q":
        if training:
            return QLearningAgent()
        return QLearningAgent()
    if engine == "nn":
        if training:
            return DQNAgent()
        return DQNAgent()
    raise ValueError(f"Unknown engine: {engine}")


def _validation_plot_path(model_path: str) -> str:
    root, _ = os.path.splitext(model_path)
    return f"{root}_validation.svg"


def _loss_plot_path(model_path: str) -> str:
    root, _ = os.path.splitext(model_path)
    return f"{root}_loss.svg"


def _write_loss_plot(points: list[tuple[int, float]], filepath: str) -> None:
    if not points:
        return

    width = 900
    height = 500
    margin_left = 70
    margin_right = 30
    margin_top = 30
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_episode = max(episode for episode, _ in points)
    max_loss = max(loss for _, loss in points)
    y_loss_max = max(1e-9, max_loss)

    def x_pos(episode: int) -> float:
        if max_episode <= 1:
            return margin_left + plot_width
        return margin_left + (episode / max_episode) * plot_width

    def y_pos(loss: float) -> float:
        return margin_top + plot_height - (loss / y_loss_max) * plot_height

    loss_polyline = " ".join(
        f"{x_pos(episode):.2f},{y_pos(loss):.2f}" for episode, loss in points
    )
    circles = "\n".join(
        f'<circle cx="{x_pos(episode):.2f}" cy="{y_pos(loss):.2f}" r="3" fill="#dc2626">'
        f"<title>Episode {episode}: mean Huber loss {loss:.6f}</title></circle>"
        for episode, loss in points
    )
    loss_mid = y_pos(y_loss_max / 2)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.0f}" y="20" text-anchor="middle" font-size="16" font-family="sans-serif" fill="#111827">DQN training loss</text>
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151"/>
<text x="{margin_left + plot_width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#374151">Training episodes</text>
<text x="18" y="{margin_top + plot_height / 2:.0f}" transform="rotate(-90 18 {margin_top + plot_height / 2:.0f})" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#dc2626">Mean Huber loss per episode</text>
<text x="{margin_left - 8}" y="{margin_top + plot_height + 4}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#6b7280">0</text>
<text x="{margin_left - 8}" y="{loss_mid + 4:.2f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#6b7280">{y_loss_max / 2:.6f}</text>
<text x="{margin_left - 8}" y="{margin_top + 4}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#6b7280">{y_loss_max:.6f}</text>
<text x="{margin_left}" y="{height - 40}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#6b7280">0</text>
<text x="{margin_left + plot_width}" y="{height - 40}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#6b7280">{max_episode}</text>
<polyline points="{loss_polyline}" fill="none" stroke="#dc2626" stroke-width="3"/>
{circles}
</svg>
'''
    with open(filepath, "w") as f:
        f.write(svg)


def _write_validation_plot(points: list[tuple[int, float, float]], filepath: str) -> None:
    if not points:
        return

    width = 900
    height = 500
    margin_left = 70
    margin_right = 70
    margin_top = 30
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_episode = max(episode for episode, _, _ in points)
    max_score = max(score for _, score, _ in points)
    max_seconds = max(seconds for _, _, seconds in points)
    y_score_max = max(1.0, max_score)
    y_seconds_max = max(1.0, max_seconds)

    def x_pos(episode: int) -> float:
        if max_episode <= 1:
            return margin_left + plot_width
        return margin_left + (episode / max_episode) * plot_width

    def score_y_pos(score: float) -> float:
        return margin_top + plot_height - (score / y_score_max) * plot_height

    def time_y_pos(seconds: float) -> float:
        return margin_top + plot_height - (seconds / y_seconds_max) * plot_height

    score_polyline = " ".join(
        f"{x_pos(episode):.2f},{score_y_pos(score):.2f}" for episode, score, _ in points
    )
    time_polyline = " ".join(
        f"{x_pos(episode):.2f},{time_y_pos(seconds):.2f}" for episode, _, seconds in points
    )
    circles = "\n".join(
        f'<circle cx="{x_pos(episode):.2f}" cy="{score_y_pos(score):.2f}" r="4" fill="#2563eb">'
        f"<title>Iteration {episode}: mean score {score:.2f}, training time {seconds:.2f}s</title></circle>"
        f'<circle cx="{x_pos(episode):.2f}" cy="{time_y_pos(seconds):.2f}" r="4" fill="#f97316">'
        f"<title>Iteration {episode}: training time {seconds:.2f}s, mean score {score:.2f}</title></circle>"
        for episode, score, seconds in points
    )
    labels = "\n".join(
        f'<text x="{x_pos(episode):.2f}" y="{score_y_pos(score) - 10:.2f}" '
        f'text-anchor="middle" font-size="12" fill="#111827">{score:.2f}</text>'
        for episode, score, _ in points
    )
    score_mid = score_y_pos(y_score_max / 2)
    time_mid = time_y_pos(y_seconds_max / 2)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.0f}" y="20" text-anchor="middle" font-size="16" font-family="sans-serif" fill="#111827">Validation mean score and training time</text>
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#374151"/>
<line x1="{margin_left + plot_width}" y1="{margin_top}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#374151"/>
<text x="{margin_left + plot_width / 2:.0f}" y="{height - 18}" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#374151">Training iterations</text>
<text x="18" y="{margin_top + plot_height / 2:.0f}" transform="rotate(-90 18 {margin_top + plot_height / 2:.0f})" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#2563eb">Mean score over 100 validation runs</text>
<text x="{width - 18}" y="{margin_top + plot_height / 2:.0f}" transform="rotate(90 {width - 18} {margin_top + plot_height / 2:.0f})" text-anchor="middle" font-size="13" font-family="sans-serif" fill="#f97316">Training time (seconds)</text>
<text x="{margin_left - 8}" y="{margin_top + plot_height + 4}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#6b7280">0</text>
<text x="{margin_left - 8}" y="{score_mid + 4:.2f}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#6b7280">{y_score_max / 2:.1f}</text>
<text x="{margin_left - 8}" y="{margin_top + 4}" text-anchor="end" font-size="12" font-family="sans-serif" fill="#6b7280">{y_score_max:.1f}</text>
<text x="{margin_left + plot_width + 8}" y="{margin_top + plot_height + 4}" text-anchor="start" font-size="12" font-family="sans-serif" fill="#6b7280">0</text>
<text x="{margin_left + plot_width + 8}" y="{time_mid + 4:.2f}" text-anchor="start" font-size="12" font-family="sans-serif" fill="#6b7280">{y_seconds_max / 2:.1f}</text>
<text x="{margin_left + plot_width + 8}" y="{margin_top + 4}" text-anchor="start" font-size="12" font-family="sans-serif" fill="#6b7280">{y_seconds_max:.1f}</text>
<text x="{margin_left}" y="{height - 40}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#6b7280">0</text>
<text x="{margin_left + plot_width}" y="{height - 40}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#6b7280">{max_episode}</text>
<polyline points="{score_polyline}" fill="none" stroke="#2563eb" stroke-width="3"/>
<polyline points="{time_polyline}" fill="none" stroke="#f97316" stroke-width="3"/>
{circles}
{labels}
</svg>
'''
    with open(filepath, "w") as f:
        f.write(svg)

def _run_validation_command(model_path: str, engine: EngineName, width: int, height: int) -> float:
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
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        fallback = [
            sys.executable,
            "-m",
            "learn2slither.main",
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
        result = subprocess.run(fallback, check=True, capture_output=True, text=True)
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"Mean Score:\s*([0-9]+(?:\.[0-9]+)?)", output)
    if match is None:
        raise RuntimeError(f"Could not parse validation mean score from command output:\n{output}")
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
        f"\n🚀 Starting Headless {engine_label} Training ({episodes} episodes) on {width}x{height} Grid..."
    )
    agent = create_agent(engine, training=True)

    if save_path and agent.load_from_file(save_path):
        print(f"Loaded existing {model_label} from '{save_path}' to continue training.")

    scores: list[int] = []
    steps_list: list[int] = []
    validation_points: list[tuple[int, float, float]] = []
    loss_points: list[tuple[int, float]] = []
    temp_validation_dir = tempfile.TemporaryDirectory() if save_path is None else None
    validation_model_path = save_path
    if validation_model_path is None and temp_validation_dir is not None:
        filename = "q_table_validation.json" if engine == "q" else "dqn_validation.json"
        validation_model_path = os.path.join(temp_validation_dir.name, filename)
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
            reward, score, last_dist = compute_reward(state, score, last_dist, engine)
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
                f"Episode {ep:5d}/{episodes} | Avg Score (last 100): {mean_score:4.1f} | "
                f"Avg Steps: {mean_steps:5.1f} | Epsilon: {agent.epsilon:5.3f} | {agent.training_status()}"
            )

        if ep % 1000 == 0:
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
                f"Mean Score (100 runs): {validation_mean:.2f} | Training Time: {elapsed:.2f}s"
            )

    if episodes > 0 and (not validation_points or validation_points[-1][0] != episodes):
        if validation_model_path is None:
            raise RuntimeError("validation_model_path was not initialized")
        agent.save_to_file(validation_model_path)
        validation_mean = _run_validation_command(validation_model_path, engine, width, height)
        elapsed = time.perf_counter() - started_at
        validation_points.append((episodes, validation_mean, elapsed))
        print(
            f"Final validation after {episodes:5d}/{episodes} episodes | "
            f"Mean Score (100 runs): {validation_mean:.2f} | Training Time: {elapsed:.2f}s"
        )

    if validation_points:
        _write_validation_plot(validation_points, plot_path)
        print(f"Validation performance plot saved to '{plot_path}'.")
    if loss_points:
        _write_loss_plot(loss_points, loss_plot_path)
        print(f"DQN training loss plot saved to '{loss_plot_path}'.")


    if save_path:
        agent.save_to_file(save_path)
        print(f"🎉 Training completed successfully! Trained {model_label} saved to '{save_path}'.")
    else:
        print(
            f"🎉 Training completed successfully! ({model_label} was not saved since save_path is None)"
        )
    if temp_validation_dir is not None:
        temp_validation_dir.cleanup()
