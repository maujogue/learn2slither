import json
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from mlp_core import DQN

from learn2slither.agents.actions import EngineName
from learn2slither.agents.features import NeuralStateFeatures, StateFeatures


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
        epsilon_decay: float = 0.995,
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
