# Learn2Slither

Learn2Slither is a Snake-style reinforcement-learning project. The snake plays on a configurable board, learns from green and red apples, and can be driven manually, by a Q-table agent, or by a small DQN model.

## Game rules

- Default board size is 10x10; CLI options clamp boards to 5..25 cells per side.
- The snake starts as a contiguous length-3 body.
- Each game contains two green apples and one red apple.
- Green apples grow the snake and reward the agent.
- Red apples shrink the snake and penalize the agent.
- Hitting a wall, hitting the snake body, or shrinking to length zero ends the game.

## Agent state, actions, and rewards

Agents do not receive the whole board as private input. They receive head-centered vision features:

- Q-learning uses danger plus green/red apple visibility in the four cardinal directions.
- DQN uses numeric ray-distance features from the snake head.

Actions are absolute directions in this order: `UP`, `LEFT`, `DOWN`, `RIGHT`.

Rewards encourage survival and moving toward green apples, strongly reward eating green apples, penalize red apples, and strongly penalize death. The Q-table and DQN engines use the same reward shape with different numeric scales.

## Models

- `--engine q` uses a JSON Q-table.
- `--engine nn` uses a JSON DQN model backed by the project MLP implementation.
- Saved models and validation plots live in `src/models/`.

## Commands

Run commands through `uv`:

```bash
uv run l2s train --headless --engine q --sessions 1000
uv run l2s train --headless --engine nn --sessions 7000
uv run l2s test --headless --engine nn --runs 20 src/models/dqn_7000.json
uv run l2s benchmark --engine all --runs 300 --top 3
```

Manual or GUI play:

```bash
uv run l2s test --manual
uv run l2s test --engine nn src/models/dqn_7000.json
```

Print each headless autopilot decision with `--verbose`:

```bash
uv run l2s test --headless --verbose --engine nn --runs 1 src/models/dqn_7000.json
```

The quiet default is preserved for benchmarks and batch evaluation.

## Evaluation notes

Headless tests print per-run scores and summary metrics. The included trained DQN models are intended for non-training evaluation, while benchmark mode compares saved Q-table and DQN JSON files under the same board settings.
