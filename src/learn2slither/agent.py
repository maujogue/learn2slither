import os
import json
import random
from dataclasses import dataclass
from learn2slither.models import GameState, Direction, Point, create_initial_game


@dataclass(frozen=True)
class StateFeatures:
    """A list of state features extracted from the GameState.

    To keep the Q-learning state space compact and enable extremely fast,
    generalized learning, we use 11 boolean relative features.
    These features describe immediate surroundings and direction to the closest
    apples, relative to the snake's current heading (Straight, Left, Right).
    """

    danger_straight: bool
    danger_left: bool
    danger_right: bool

    green_apple_front: bool
    green_apple_back: bool
    green_apple_left: bool
    green_apple_right: bool

    red_apple_front: bool
    red_apple_back: bool
    red_apple_left: bool
    red_apple_right: bool

    def to_tuple(self) -> tuple[bool, ...]:
        """Converts features to a hashable tuple for Q-table indexing."""
        return (
            self.danger_straight,
            self.danger_left,
            self.danger_right,
            self.green_apple_front,
            self.green_apple_back,
            self.green_apple_left,
            self.green_apple_right,
            self.red_apple_front,
            self.red_apple_back,
            self.red_apple_left,
            self.red_apple_right,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        """Returns a list of all feature names in order."""
        return [
            "danger_straight",
            "danger_left",
            "danger_right",
            "green_apple_front",
            "green_apple_back",
            "green_apple_left",
            "green_apple_right",
            "red_apple_front",
            "red_apple_back",
            "red_apple_left",
            "red_apple_right",
        ]

    @classmethod
    def from_game_state(cls, state: GameState) -> "StateFeatures":
        """Extracts relative state features from the current GameState."""
        if state.is_game_over or not state.snake.body:
            return cls(
                danger_straight=True,
                danger_left=True,
                danger_right=True,
                green_apple_front=False,
                green_apple_back=False,
                green_apple_left=False,
                green_apple_right=False,
                red_apple_front=False,
                red_apple_back=False,
                red_apple_left=False,
                red_apple_right=False,
            )

        head = state.snake.head
        current_dir = state.snake.direction
        dx, dy = current_dir.value

        # Helper to check for obstacles
        def is_obstacle(p: Point) -> bool:
            if not state.is_within_bounds(p):
                return True
            # Colliding with the snake's own body is an obstacle
            if p in state.snake.body:
                return True
            return False

        # Define relative directions
        dir_front = current_dir
        dir_left = next(d for d in Direction if d.value == (dy, -dx))
        dir_right = next(d for d in Direction if d.value == (-dy, dx))
        dir_back = next(d for d in Direction if d.value == (-dx, -dy))

        # Define points in straight, left, and right directions relative to current_dir
        pt_straight = head.move(dir_front)
        pt_left = head.move(dir_left)
        pt_right = head.move(dir_right)

        danger_straight = is_obstacle(pt_straight)
        danger_left = is_obstacle(pt_left)
        danger_right = is_obstacle(pt_right)

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

        green_front, red_front = raycast_apples(head, dir_front)
        green_left, red_left = raycast_apples(head, dir_left)
        green_right, red_right = raycast_apples(head, dir_right)
        green_back, red_back = raycast_apples(head, dir_back)

        return cls(
            danger_straight=danger_straight,
            danger_left=danger_left,
            danger_right=danger_right,
            green_apple_front=green_front,
            green_apple_back=green_back,
            green_apple_left=green_left,
            green_apple_right=green_right,
            red_apple_front=red_front,
            red_apple_back=red_back,
            red_apple_left=red_left,
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

    Actions are represented as relative actions:
    0 = STRAIGHT, 1 = LEFT, 2 = RIGHT
    """

    def __init__(self) -> None:
        # Dictionary mapping: state_tuple (tuple of 11 booleans) -> {action_id (0..2) -> QValue}
        self.table: dict[tuple[bool, ...], dict[int, QValue]] = {}

    def get_q_value(self, state: StateFeatures, action: int) -> QValue:
        """Retrieves a QValue entity for a given state and action, initializing it if absent."""
        state_tuple = state.to_tuple()
        if state_tuple not in self.table:
            self.table[state_tuple] = {
                0: QValue(0.0),
                1: QValue(0.0),
                2: QValue(0.0),
            }
        return self.table[state_tuple][action]

    def get_best_action_value(self, state: StateFeatures) -> tuple[int, float]:
        """Returns the best relative action and its corresponding Q-value."""
        state_tuple = state.to_tuple()
        if state_tuple not in self.table:
            # If state not visited, default to STRAIGHT (0) and 0.0 Q-value
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

    def get_action(self, state: StateFeatures, training: bool = True) -> int:
        """Selects an action using an epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Explore: random choice from 0 (STRAIGHT), 1 (LEFT), 2 (RIGHT)
            return random.randint(0, 2)
        else:
            # Exploit: best action from Q-table
            action, _ = self.q_table.get_best_action_value(state)
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
    save_path: str,
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
    if agent.q_table.load_from_file(save_path):
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

            # Map relative action to absolute direction
            dx, dy = state.snake.direction.value
            if action == 0:  # STRAIGHT
                new_dir = state.snake.direction
            elif action == 1:  # LEFT
                new_dir = next(d for d in Direction if d.value == (dy, -dx))
            else:  # RIGHT
                new_dir = next(d for d in Direction if d.value == (-dy, dx))

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
        if ep % 500 == 0 or ep == 1:
            mean_score = sum(scores[-100:]) / len(scores[-100:])
            mean_steps = sum(steps_list[-100:]) / len(steps_list[-100:])
            print(
                f"Episode {ep:5d}/{episodes} | Avg Score (last 100): {mean_score:4.1f} | "
                f"Avg Steps: {mean_steps:5.1f} | Epsilon: {agent.epsilon:5.3f} | States: {len(agent.q_table.table)}"
            )

    # Save Q-table
    agent.q_table.save_to_file(save_path)
    print(
        f"🎉 Training completed successfully! Trained Q-table saved to '{save_path}'."
    )
