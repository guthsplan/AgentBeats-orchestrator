from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml


def validate_mcu_config(config: dict[str, Any]) -> bool:
    required_fields = ["text", "reward_cfg"]
    for field in required_fields:
        if field not in config:
            return False
    if not isinstance(config["reward_cfg"], list):
        return False
    return True


def list_mcu_tasks(tasks_dir: str, category: str | None = None) -> list[str]:
    tasks_path = Path(tasks_dir)
    if category:
        category_dir = tasks_path / category
        if not category_dir.exists():
            return []
        return sorted(str(path) for path in category_dir.glob("*.yaml"))

    yaml_files: list[str] = []
    for category_dir in tasks_path.iterdir():
        if category_dir.is_dir():
            yaml_files.extend(str(path) for path in category_dir.glob("*.yaml"))
    return sorted(yaml_files)


def env_init_mcu(
    yaml_path: str,
    rollout_path: str,
    obs_size: tuple[int, int] = (224, 224),
    render_size: tuple[int, int] = (640, 360),
    max_steps: int = 600,
    fps: int = 30,
    **kwargs,
):
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.callbacks import (
        CommandsCallback,
        JudgeResetCallback,
        RecordCallback,
        RewardsCallback,
    )

    with open(yaml_path, "r", encoding="utf-8") as handle:
        task_config = yaml.safe_load(handle)
    if not validate_mcu_config(task_config):
        raise ValueError(f"Invalid MCU task configuration: {yaml_path}")

    os.makedirs(rollout_path, exist_ok=True)
    callbacks = []
    commands = task_config.get("custom_init_commands", [])
    reward_cfg = task_config.get("reward_cfg", [])
    if commands:
        callbacks.append(CommandsCallback(commands))
    callbacks.append(JudgeResetCallback(max_steps))
    if reward_cfg:
        callbacks.append(RewardsCallback(reward_cfg))
    callbacks.append(RecordCallback(record_path=rollout_path, fps=fps, frame_type="pov"))

    env = MinecraftSim(
        action_type="env",
        obs_size=obs_size,
        render_size=render_size,
        preferred_spawn_biome=None,
        callbacks=callbacks,
        **kwargs,
    )

    with open(os.path.join(rollout_path, "task_config.json"), "w", encoding="utf-8") as handle:
        json.dump(task_config, handle, indent=2, ensure_ascii=False)
    return env, task_config


def save_episode_results(rollout_path: str, results: dict[str, Any]) -> str:
    result_path = os.path.join(rollout_path, "episode_results.json")
    payload = dict(results)
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return result_path
