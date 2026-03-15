"""Configuration helpers for orchestrator runtime."""

from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    tasks_dir: str
    output_dir: str
    max_steps: int = 600
    obs_width: int = 224
    obs_height: int = 224
