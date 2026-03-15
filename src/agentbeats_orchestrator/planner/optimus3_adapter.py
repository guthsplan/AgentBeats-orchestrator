from __future__ import annotations

from agentbeats_orchestrator.planner.base import Planner
from agentbeats_orchestrator.types import Subgoal


class Optimus3PlannerAdapter(Planner):
    def __init__(self, optimus_agent):
        self.optimus_agent = optimus_agent

    def plan(self, task_text: str) -> list[Subgoal]:
        _, steps, _ = self.optimus_agent.plan(task_text)
        if steps:
            return [
                Subgoal(name=f"step_{idx}", description=step)
                for idx, step in enumerate(steps)
            ]
        return [Subgoal(name="task", description=task_text)]
