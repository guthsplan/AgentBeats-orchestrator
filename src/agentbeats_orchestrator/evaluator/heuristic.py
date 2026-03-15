from __future__ import annotations

import numpy as np

from agentbeats_orchestrator.evaluator.base import Evaluator
from agentbeats_orchestrator.evaluator.detectors import detect_events
from agentbeats_orchestrator.types import EvaluationResult


class HeuristicEvaluator(Evaluator):
    def evaluate(
        self,
        frame: np.ndarray,
        state,
        runtime_memory,
        event_memory,
        failure_memory,
    ) -> EvaluationResult:
        repeated = runtime_memory.recent_action_repeat_ratio()
        no_progress = runtime_memory.no_progress_streak()
        reasons: list[str] = []
        score = 1.0

        if repeated > 0.8:
            score -= 0.4
            reasons.append("high_action_repeat")
        if no_progress > 8:
            score -= 0.4
            reasons.append("no_progress_streak")

        detected_events = detect_events(state, runtime_memory)
        status = "progress" if score >= 0.5 else "stalled"
        return EvaluationResult(
            progress_score=score,
            status=status,
            reasons=reasons,
            detected_events=detected_events,
            signals={
                "repeat_ratio": repeated,
                "no_progress_streak": no_progress,
            },
        )
