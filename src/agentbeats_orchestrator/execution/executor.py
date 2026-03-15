from __future__ import annotations

from agentbeats_orchestrator.execution.guard import ActionGuard
from agentbeats_orchestrator.types import ExecutionResult


class ActionExecutor:
    def __init__(self, guard: ActionGuard | None = None):
        self.guard = guard or ActionGuard()

    def execute(self, raw_action: dict | None) -> ExecutionResult:
        executed_action, info = self.guard.sanitize(raw_action)
        return ExecutionResult(
            raw_action=raw_action or {},
            executed_action=executed_action,
            info=info,
        )
