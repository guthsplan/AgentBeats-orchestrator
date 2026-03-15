from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class Subgoal:
    name: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Decision(str, Enum):
    KEEP = "KEEP"
    RECOVER = "RECOVER"
    SWITCH_SUBGOAL = "SWITCH_SUBGOAL"
    REPLAN = "REPLAN"


@dataclass
class ObservationSummary:
    frame_hash: str | None = None
    image_embedding: np.ndarray | None = None
    inventory: dict[str, Any] | None = None
    entity_counts: dict[str, int] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class InventorySnapshot:
    items: dict[str, int]
    timestamp: int
    confidence: float


@dataclass
class ToolSnapshot:
    equipped_item: str | None = None
    timestamp: int = 0
    confidence: float = 0.0


@dataclass
class StateCache:
    inventory: InventorySnapshot | None = None
    tool: ToolSnapshot | None = None


@dataclass
class EvaluationResult:
    progress_score: float
    status: str
    reasons: list[str] = field(default_factory=list)
    detected_events: list[dict[str, Any]] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionResult:
    decision: Decision
    next_subgoal_index: int | None = None
    reason: str = ""


@dataclass
class ExecutionResult:
    raw_action: dict[str, Any]
    executed_action: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class PendingAction:
    step_index: int
    subgoal_index: int
    execution: ExecutionResult


@dataclass
class RuntimeState:
    task_text: str
    subgoals: list[Subgoal]
    current_subgoal_index: int = 0
    step_index: int = 0
    recover_mode: bool = False
    state_cache: StateCache = field(default_factory=StateCache)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_subgoal(self) -> Subgoal:
        return self.subgoals[self.current_subgoal_index]
