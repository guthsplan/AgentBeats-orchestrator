from __future__ import annotations

from abc import ABC, abstractmethod

from agentbeats_orchestrator.types import Subgoal


class Planner(ABC):
    @abstractmethod
    def plan(self, task_text: str) -> list[Subgoal]:
        raise NotImplementedError
