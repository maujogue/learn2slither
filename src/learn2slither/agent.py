import os
import json
import random
from dataclasses import dataclass
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


class QLearningAgent:
    """Orchestrates Q-learning training and action selection for the Snake game."""

    def __init__(
        self,
        alpha: float = 0.1,  # Learning rate
        gamma: float = 0.9,  # Discount factor
        epsilon: float = 0.1,  # Exploration rate
        epsilon_decay: float = 0.9995,  # Epsilon decay rate per episode
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

    def get_action(
        self, state: StateFeatures, training: bool = True, verbose: bool | None = None
    ) -> int:
        """Selects an action using an epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Explore: random choice from 0 (UP), 1 (LEFT), 2 (DOWN), 3 (RIGHT)
            action = random.randint(0, 3)
        else:
            # Exploit: best action from Q-table
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

    def decay_epsilon(self) -> None:
        """Decays the exploration rate after an episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


def train_agent(
    save_path: str | None,
    episodes: int = 15000,
    width: int = 10,
    height: int = 10,
) -> None:
    """Trains the Q-learning agent headlessly in CLI and saves the trained Q-table."""
    print(
        f"\n🚀 Starting Headless Q-Learning Training ({episodes} episodes) on {width}x{height} Grid..."
    )
    agent = QLearningAgent(epsilon=0.9, min_epsilon=0.01, epsilon_decay=0.9997)

    # Load existing Q-table if available to continue training
    if save_path and agent.q_table.load_from_file(save_path):
        print(f"Loaded existing Q-table from '{save_path}' to continue training.")

    scores: list[int] = []
    steps_list: list[int] = []

    for ep in range(1, episodes + 1):
        state = create_initial_game(width=width, height=height)
        score = 3
        steps = 0
        done = False

        # Keep track of distance to closest green apple to compute dense rewards
        def get_min_green_dist(s: GameState) -> float:
            if not s.green_apples:
                return 0.0
            h = s.snake.head
            return min(abs(a.x - h.x) + abs(a.y - h.y) for a in s.green_apples)

        last_dist = get_min_green_dist(state)

        while not done:
            curr_features = StateFeatures.from_game_state(state)
            action = agent.get_action(curr_features, training=True)

            # Map absolute action to direction
            action_to_dir = {
                0: Direction.UP,
                1: Direction.LEFT,
                2: Direction.DOWN,
                3: Direction.RIGHT,
            }
            new_dir = action_to_dir[action]

            # Apply action
            state.change_direction(new_dir)

            # Move one step
            state.step()
            steps += 1

            # Observe new state and reward
            next_features = StateFeatures.from_game_state(state)
            done = state.is_game_over

            # Compute reward
            reward = 0.0
            if done:
                reward = -100.0  # Heavy crash penalty
            else:
                new_len = len(state.snake.body)
                if new_len > score:
                    reward = 100.0  # High reward for growing
                    score = new_len
                elif new_len < score:
                    reward = -30.0  # Penalty for shrinking (eating a red apple)
                    score = new_len
                else:
                    # Survival reward
                    reward = 1.0

                    # Dense distance-based reward
                    new_dist = get_min_green_dist(state)
                    if new_dist < last_dist:
                        reward += 10.0  # Reward for moving closer to green apple
                    elif new_dist > last_dist:
                        reward -= 15.0  # Penalty for moving away from green apple
                    last_dist = new_dist

            # Update Q-table
            agent.update(curr_features, action, reward, next_features, done)

        agent.decay_epsilon()
        scores.append(score)
        steps_list.append(steps)

        # Log training progress
        if ep % 10 == 0 or ep == 1:
            mean_score = sum(scores[-100:]) / len(scores[-100:])
            mean_steps = sum(steps_list[-100:]) / len(steps_list[-100:])
            print(
                f"Episode {ep:5d}/{episodes} | Avg Score (last 100): {mean_score:4.1f} | "
                f"Avg Steps: {mean_steps:5.1f} | Epsilon: {agent.epsilon:5.3f} | States: {len(agent.q_table.table)}"
            )

    # Save Q-table
    if save_path:
        agent.q_table.save_to_file(save_path)
        print(
            f"🎉 Training completed successfully! Trained Q-table saved to '{save_path}'."
        )
    else:
        print(
            "🎉 Training completed successfully! (Q-table was not saved since save_path is None)"
        )
