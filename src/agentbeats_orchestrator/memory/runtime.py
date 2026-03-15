from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentbeats_orchestrator.types import ObservationSummary


@dataclass
class RuntimeEntry:
    step_index: int
    subgoal_index: int
    raw_action: dict[str, Any]
    executed_action: dict[str, Any]
    reward: float = 0.0
    done: bool = False
    obs_summary: ObservationSummary | None = None
    exec_info: dict[str, Any] = field(default_factory=dict)
    eval_summary: dict[str, Any] = field(default_factory=dict)


class RuntimeMemory:
    def __init__(self, maxlen: int = 20):
        self.maxlen = maxlen
        self.entries: list[RuntimeEntry] = []

    def append(self, entry: RuntimeEntry) -> None:
        self.entries.append(entry)
        self.entries = self.entries[-self.maxlen :]

    def recent_action_repeat_ratio(self) -> float:
        if len(self.entries) < 2:
            return 0.0
        actions = [str(entry.executed_action) for entry in self.entries]
        same = sum(1 for idx in range(1, len(actions)) if actions[idx] == actions[idx - 1])
        return same / (len(actions) - 1)

    def no_progress_streak(self) -> int:
        streak = 0
        for entry in reversed(self.entries):
            if entry.reward > 0:
                break
            streak += 1
        return streak
