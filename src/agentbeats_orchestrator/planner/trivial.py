from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.planner.base import Planner
from agentbeats_orchestrator.types import RuntimeState, Subgoal


class TrivialPlanner(Planner):
    def plan(
        self,
        task_text: str,
        failure_memory: FailureMemory | None = None,
        state: RuntimeState | None = None,
    ) -> list[Subgoal]:
        return [Subgoal(name="task", description=task_text)]

    def replan(
        self,
        task_text: str,
        failure_memory: FailureMemory,
        state: RuntimeState,
    ) -> list[Subgoal]:
        failure_context = failure_memory.as_prompt_context(subgoal_index=state.current_subgoal_index)
        description = task_text
        if failure_context:
            description = f"{task_text}\nAvoid repeating:\n{failure_context}"
        return [Subgoal(name="task_replan", description=description)]
