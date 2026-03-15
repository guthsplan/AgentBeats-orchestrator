from __future__ import annotations

from agentbeats_orchestrator.memory.runtime import RuntimeMemory
from agentbeats_orchestrator.types import RuntimeState


def detect_events(state: RuntimeState, runtime_memory: RuntimeMemory, reward: float, done: bool) -> list[dict]:
    events: list[dict] = []
    if reward > 0:
        events.append(
            {
                "kind": "reward_progress",
                "summary": "Environment returned positive reward after the last action.",
                "subgoal_index": state.current_subgoal_index,
                "step_index": state.step_index,
                "reward": reward,
            }
        )
    if done:
        events.append(
            {
                "kind": "episode_done",
                "summary": "Episode terminated after the last action.",
                "subgoal_index": state.current_subgoal_index,
                "step_index": state.step_index,
            }
        )
    if runtime_memory.no_progress_streak() >= 8:
        events.append(
            {
                "kind": "stall",
                "summary": "No reward-bearing progress detected for several steps.",
                "subgoal_index": state.current_subgoal_index,
                "step_index": state.step_index,
            }
        )
    if runtime_memory.recent_action_repeat_ratio() >= 0.8:
        events.append(
            {
                "kind": "action_repeat",
                "summary": "High repeated-action ratio detected.",
                "subgoal_index": state.current_subgoal_index,
                "step_index": state.step_index,
            }
        )
    return events
