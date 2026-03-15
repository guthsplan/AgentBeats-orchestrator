from agentbeats_orchestrator.planner.base import Planner
from agentbeats_orchestrator.types import Subgoal


class TrivialPlanner(Planner):
    def plan(self, task_text: str) -> list[Subgoal]:
        return [Subgoal(name="task", description=task_text)]
