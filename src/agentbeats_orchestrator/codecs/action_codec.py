from __future__ import annotations

from typing import Any


ENV_NULL_ACTION = {
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,
    "sprint": 0,
    "sneak": 0,
    "use": 0,
    "drop": 0,
    "attack": 0,
    "jump": 0,
    "inventory": 0,
    "camera": [0.0, 0.0],
}


def noop_env_action() -> dict[str, Any]:
    return dict(ENV_NULL_ACTION)
