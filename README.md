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

In the GUI, a step-by-step toggle lets you pause between moves and advance with `SPACE`. This allows careful session inspection at human speed.

Print each headless autopilot decision with `--verbose`:

```bash
uv run l2s test --headless --verbose --engine nn --runs 1 src/models/dqn_7000.json
```

The quiet default is preserved for benchmarks and batch evaluation.

## Evaluation notes

Headless tests print per-run scores and summary metrics. The included trained DQN models are intended for non-training evaluation, while benchmark mode compares saved Q-table and DQN JSON files under the same board settings.

## Feature set benchmark

Results below were measured on a 10×10 board after 2100 training episodes and 1000 headless evaluation runs per configuration (sorted by `eval_mean`).

| Rank | Engine | Profile <br/> <sub>(see labels below)</sub> | Dim | Eval mean | Eval median | Eval max | Eval min | Train (s) | Q states | ε | DQN loss |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | nn | full | 16 | 26.58 | 26.00 | 50 | 3 | 96.1 | — | 0.005 | 0.0557 |
| 2 | nn | full_wall_body | 16 | 26.21 | 27.00 | 51 | 3 | 91.7 | — | 0.005 | 0.0422 |
| 3 | nn | green_wall_body | 12 | 25.31 | 25.00 | 48 | 7 | 90.5 | — | 0.005 | 0.0495 |
| 4 | nn | green_danger | 8 | 24.89 | 25.00 | 48 | 0 | 81.9 | — | 0.005 | 0.0568 |
| 5 | q | green_danger | 8 | 23.50 | 23.00 | 48 | 6 | 13.8 | 66 | 0.900 | — |
| 6 | q | green_wall_body | 12 | 21.05 | 20.00 | 52 | 1 | 14.0 | 158 | 0.900 | — |
| 7 | q | full | 12 | 20.72 | 21.00 | 50 | 1 | 14.1 | 146 | 0.900 | — |
| 8 | q | full_wall_body | 16 | 15.30 | 15.00 | 41 | 2 | 10.8 | 287 | 0.900 | — |

<details>
<summary><strong>Profile descriptions</strong></summary>

DQN features are distances from the head to the object, normalized to the board size.
Q-table features are immediate contact with danger, or sight of an apple.

- <strong>full</strong>: Danger (wall or self), green apple, red apple &mdash; single boolean for each, in all 4 directions.
- <strong>full_wall_body</strong>: Separate booleans for wall, body, green, red in all 4 directions.
- <strong>green_wall_body</strong>: Separate booleans for wall, body, and green apple in all 4 directions.
- <strong>green_danger</strong>: Combined danger (max of wall/body), plus green apple in all 4 directions.

</details>

DQN variants rank highest overall; for Q-tables, the smaller `green_danger` profile beats the richer `full` and `full_wall_body` encodings on mean score.
