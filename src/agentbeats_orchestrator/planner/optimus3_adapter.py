from __future__ import annotations

from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.planner.base import Planner
from agentbeats_orchestrator.types import RuntimeState, Subgoal


class Optimus3PlannerAdapter(Planner):
    def __init__(self, optimus_agent):
        self.optimus_agent = optimus_agent

    def plan(
        self,
        task_text: str,
        failure_memory: FailureMemory | None = None,
        state: RuntimeState | None = None,
    ) -> list[Subgoal]:
        _, steps, _ = self.optimus_agent.plan(task_text)
        if steps:
            return [
                Subgoal(name=f"step_{idx}", description=step)
                for idx, step in enumerate(steps)
            ]
        return [Subgoal(name="task", description=task_text)]

    def replan(
        self,
        task_text: str,
        failure_memory: FailureMemory,
        state: RuntimeState,
    ) -> list[Subgoal]:
        failure_context = failure_memory.as_prompt_context(subgoal_index=state.current_subgoal_index)
        replanning_task = task_text
        if failure_context:
            replanning_task = (
                f"{task_text}\n"
                f"Previous failure patterns for the current subgoal:\n{failure_context}\n"
                "Revise the plan to avoid these failures."
            )
        return self.plan(replanning_task, failure_memory=failure_memory, state=state)
