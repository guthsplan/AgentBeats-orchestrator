from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from agentbeats_orchestrator.memory.event import EventMemory
from agentbeats_orchestrator.memory.runtime import RuntimeMemory
from agentbeats_orchestrator.types import EvaluationResult, RuntimeState


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        frame: np.ndarray,
        reward: float,
        done: bool,
        state: RuntimeState,
        runtime_memory: RuntimeMemory,
        event_memory: EventMemory,
    ) -> EvaluationResult:
        raise NotImplementedError
