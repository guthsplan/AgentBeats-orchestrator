from __future__ import annotations

import numpy as np

from agentbeats_orchestrator.evaluator.base import Evaluator
from agentbeats_orchestrator.evaluator.detectors import detect_events
from agentbeats_orchestrator.types import EvaluationResult


class HeuristicEvaluator(Evaluator):
    INVENTORY_KEYWORDS = ("craft", "smelt", "furnace", "pickaxe", "plank", "stick", "tool")

    def evaluate(
        self,
        frame: np.ndarray,
        reward: float,
        done: bool,
        state,
        runtime_memory,
        event_memory,
    ) -> EvaluationResult:
        repeated = runtime_memory.recent_action_repeat_ratio()
        no_progress = runtime_memory.no_progress_streak()
        need_inventory_check = self._need_inventory_check(state)
        reasons: list[str] = []
        score = 1.0

        if reward > 0:
            score += 0.3
            reasons.append("positive_reward")
        if repeated > 0.8:
            score -= 0.4
            reasons.append("high_action_repeat")
        if no_progress > 8:
            score -= 0.4
            reasons.append("no_progress_streak")
        if done:
            reasons.append("episode_done")
        if need_inventory_check:
            reasons.append("inventory_unknown_for_inventory_sensitive_subgoal")

        score = max(0.0, min(1.0, score))
        detected_events = detect_events(state, runtime_memory, reward, done)
        status = "progress" if score >= 0.5 else "stalled"
        return EvaluationResult(
            progress_score=score,
            status=status,
            reasons=reasons,
            detected_events=detected_events,
            signals={
                "repeat_ratio": repeated,
                "no_progress_streak": no_progress,
                "reward": reward,
                "done": done,
                "need_inventory_check": need_inventory_check,
            },
        )

    def _need_inventory_check(self, state) -> bool:
        subgoal_text = state.current_subgoal.description.lower()
        inventory_relevant = any(keyword in subgoal_text for keyword in self.INVENTORY_KEYWORDS)
        if not inventory_relevant:
            return False
        snapshot = state.state_cache.inventory
        if snapshot is None:
            return True
        if snapshot.confidence < 0.5:
            return True
        if state.step_index - snapshot.timestamp > 15:
            return True
        return False
