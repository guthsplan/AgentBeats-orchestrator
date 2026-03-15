from __future__ import annotations

from abc import ABC, abstractmethod

from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.types import RuntimeState, Subgoal


class Planner(ABC):
    @abstractmethod
    def plan(
        self,
        task_text: str,
        failure_memory: FailureMemory | None = None,
        state: RuntimeState | None = None,
    ) -> list[Subgoal]:
        raise NotImplementedError

    @abstractmethod
    def replan(
        self,
        task_text: str,
        failure_memory: FailureMemory,
        state: RuntimeState,
    ) -> list[Subgoal]:
        raise NotImplementedError
