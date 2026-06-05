import json
import os
import re
from typing import Any

_DQN_HEAD_RE = re.compile(r'^\{\s*"engine"\s*:\s*"nn"')
_Q_TABLE_HEAD_RE = re.compile(r'^\{\s*"[01](?:,[01])+":\s*\{')


def is_dqn_payload(data: dict[str, Any]) -> bool:
    return (
        isinstance(data, dict)
        and data.get("engine") == "nn"
        and isinstance(data.get("network"), dict)
    )


def is_q_table_payload(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    if "engine" in data or "network" in data:
        return False
    for key, action_dict in data.items():
        if not isinstance(key, str) or not isinstance(action_dict, dict):
            return False
        parts = key.split(",")
        if not parts or not all(part in ("0", "1") for part in parts):
            return False
        for act, val in action_dict.items():
            if act not in {"0", "1", "2", "3"}:
                return False
            if not isinstance(val, dict) or "value" not in val:
                return False
    return True


def peek_model_engine(path: str) -> str | None:
    """Fast header-based detection for model discovery."""
    try:
        with open(path, "r") as f:
            head = f.read(512).lstrip()
    except OSError:
        return None

    if _DQN_HEAD_RE.search(head) or (
        '"engine"' in head and '"nn"' in head and '"network"' in head
    ):
        return "nn"
    if _Q_TABLE_HEAD_RE.search(head):
        return "q"
    return None


def detect_model_engine(path: str) -> str | None:
    """Detect whether a JSON model file is a Q-table or DQN payload."""
    if not os.path.isfile(path):
        return None

    peeked = peek_model_engine(path)
    if peeked is not None:
        return peeked

    try:
        with open(path, "r") as f:
            content = f.read().strip()
        if not content:
            return None
        data = json.loads(content)
    except (OSError, json.JSONDecodeError):
        return None

    if is_dqn_payload(data):
        return "nn"
    if is_q_table_payload(data):
        return "q"
    return None
