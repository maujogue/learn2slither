from typing import Any

from learn2slither.agents.actions import (
    ACTION_TO_DIRECTION,
    EngineName,
    action_to_direction,
)
from learn2slither.agents.dqn import DQNAgent, ReplayTransition
from learn2slither.agents.features import NeuralStateFeatures, StateFeatures
from learn2slither.agents.q_learning import QLearningAgent, QTable, QValue
from learn2slither.agents.rewards import compute_reward, get_min_green_dist


def create_agent(engine: str, *, training: bool = False) -> Any:
    if engine == "q":
        if training:
            return QLearningAgent()
        return QLearningAgent()
    if engine == "nn":
        if training:
            return DQNAgent()
        return DQNAgent()
    raise ValueError(f"Unknown engine: {engine}")


__all__ = [
    "ACTION_TO_DIRECTION",
    "DQNAgent",
    "EngineName",
    "NeuralStateFeatures",
    "QLearningAgent",
    "QTable",
    "QValue",
    "ReplayTransition",
    "StateFeatures",
    "action_to_direction",
    "compute_reward",
    "create_agent",
    "get_min_green_dist",
]
