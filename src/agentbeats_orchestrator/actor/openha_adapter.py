from __future__ import annotations

from typing import Any

import numpy as np

from agentbeats_orchestrator.actor.base import Actor
from agentbeats_orchestrator.types import RuntimeState


class OpenHAActorAdapter(Actor):
    def __init__(self, openha_agent: Any, subgoal_as_instruction: bool = True):
        self.agent = openha_agent
        self.subgoal_as_instruction = subgoal_as_instruction

    def reset(self, task_text: str) -> None:
        self.agent.reset(instruction=task_text, task_name=task_text)

    def act(self, frame: np.ndarray, state: RuntimeState) -> dict[str, Any]:
        instruction = (
            state.current_subgoal.description
            if self.subgoal_as_instruction
            else state.task_text
        )
        action = self.agent.get_action(
            obs={"image": frame},
            info={"pov": frame},
            instruction=instruction,
            verbose=False,
        )
        if not isinstance(action, dict):
            raise ValueError("OpenHA returned invalid env action")
        return action
