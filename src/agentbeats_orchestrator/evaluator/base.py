from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from agentbeats_orchestrator.memory.event import EventMemory
from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.memory.runtime import RuntimeMemory
from agentbeats_orchestrator.types import EvaluationResult, RuntimeState


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        frame: np.ndarray,
        state: RuntimeState,
        runtime_memory: RuntimeMemory,
        event_memory: EventMemory,
        failure_memory: FailureMemory,
    ) -> EvaluationResult:
        raise NotImplementedError
