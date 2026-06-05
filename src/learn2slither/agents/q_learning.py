import json
import os
import random
from dataclasses import dataclass

from learn2slither.agents.actions import EngineName
from learn2slither.agents.features import StateFeatures


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
        # Shuffle actions to break ties randomly (if multiple values are equal) and prevent bias
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
