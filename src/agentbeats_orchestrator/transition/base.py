from __future__ import annotations

from abc import ABC, abstractmethod

from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.memory.runtime import RuntimeMemory
from agentbeats_orchestrator.types import EvaluationResult, RuntimeState, TransitionResult


class TransitionPolicy(ABC):
    @abstractmethod
    def decide(
        self,
        state: RuntimeState,
        evaluation: EvaluationResult,
        runtime_memory: RuntimeMemory,
        failure_memory: FailureMemory,
    ) -> TransitionResult:
        raise NotImplementedError
