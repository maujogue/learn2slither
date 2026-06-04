# Learnings

## Reinforcement learning loop

The project reinforced the basic loop of observing state, choosing an action, applying the game transition, receiving a reward, and updating the policy. Snake is useful for this because every decision has immediate consequences but long-term planning still matters.

## Q-values and Q-tables

A Q-table stores one value per `(state, action)` pair. Higher values mean the agent expects better future reward from that action in that state. This is simple and inspectable, but it depends on keeping the state representation compact enough that repeated states occur during training.

## Epsilon-greedy exploration

The agents use epsilon-greedy exploration during training: sometimes they choose a random action instead of the current best action. This prevents the policy from locking onto early bad habits before it has explored enough alternatives.

## Reward shaping

Reward design changes what behavior the agent discovers. Green apples, survival, and moving closer to food create positive pressure; red apples, moving away from food, and death create negative pressure. The strongest rewards and penalties define the behavior the agent treats as most important.

## Head-only state representation

The agents use features derived from the snake head's visible rays instead of direct full-board access. This keeps the input smaller and makes the learned behavior closer to a local-vision player: avoid immediate danger, notice apples in visible directions, and move based on local evidence.

## Q-table vs DQN

The Q-table is easier to understand and save, but it scales poorly as states become more detailed. The DQN replaces table lookup with a neural network that estimates action values from numeric features, which can generalize across similar states instead of memorizing each exact state separately.

## Model persistence

Saving models to JSON makes training reusable. A model can be trained once, loaded later, benchmarked, or used in the GUI without retraining. This also makes different training runs comparable because their artifacts can be evaluated under the same command.

## Validation and benchmarking

Single games are noisy, so validation needs multiple runs and summary metrics. Mean, median, min, max, and success counts are more informative than one score because random initial boards can make individual outcomes unusually easy or hard.

## Board-size generalization

A head-centered feature representation helps models run on different board sizes because the input describes local geometry rather than absolute board coordinates. Larger or smaller boards can still change difficulty, so resized-board performance must be validated rather than assumed.
