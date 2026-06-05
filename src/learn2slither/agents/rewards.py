from learn2slither.core import GameState


def get_min_green_dist(state: GameState) -> float:
    """Return Manhattan distance from snake head to nearest green apple."""
    if not state.green_apples:
        return 0.0
    head = state.snake.head
    return min(
        abs(apple.x - head.x) + abs(apple.y - head.y)
        for apple in state.green_apples
    )


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
