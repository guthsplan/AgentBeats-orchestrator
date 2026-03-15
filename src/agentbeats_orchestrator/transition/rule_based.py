from __future__ import annotations

from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.memory.runtime import RuntimeMemory
from agentbeats_orchestrator.transition.base import TransitionPolicy
from agentbeats_orchestrator.types import Decision, EvaluationResult, RuntimeState, TransitionResult


class RuleBasedTransitionPolicy(TransitionPolicy):
    def decide(
        self,
        state: RuntimeState,
        evaluation: EvaluationResult,
        runtime_memory: RuntimeMemory,
        failure_memory: FailureMemory,
    ) -> TransitionResult:
        no_progress = runtime_memory.no_progress_streak()

        if evaluation.progress_score >= 0.5:
            return TransitionResult(decision=Decision.KEEP, reason="progress_ok")
        if no_progress >= 12 and state.current_subgoal_index + 1 < len(state.subgoals):
            return TransitionResult(
                decision=Decision.SWITCH_SUBGOAL,
                next_subgoal_index=state.current_subgoal_index + 1,
                reason="stalled_switch",
            )
        if no_progress >= 20:
            return TransitionResult(decision=Decision.REPLAN, reason="long_stall")
        return TransitionResult(decision=Decision.RECOVER, reason="local_stall")
