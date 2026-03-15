from __future__ import annotations

from typing import Any

import numpy as np

from agentbeats_orchestrator.actor.base import Actor
from agentbeats_orchestrator.codecs.action_codec import PurpleActionCodec
from agentbeats_orchestrator.types import RuntimeState


class OpenHAEnvActorAdapter(Actor):
    """OpenHA adapter that returns env-style actions for benchmark execution."""

    def __init__(
        self,
        openha_agent: Any,
        instruction_mode: str = "task_and_subgoal",
        include_recover_hint: bool = True,
        reset_on_subgoal_switch: bool = False,
        verbose: bool = False,
    ):
        self.agent = openha_agent
        self.instruction_mode = instruction_mode
        self.include_recover_hint = include_recover_hint
        self.reset_on_subgoal_switch = reset_on_subgoal_switch
        self.verbose = verbose
        self._last_subgoal_index: int | None = None

    def reset(self, task_text: str) -> None:
        self.agent.reset(instruction=task_text, task_name=task_text)
        self._last_subgoal_index = None

    def act(self, frame: np.ndarray, state: RuntimeState) -> dict[str, Any]:
        normalized_frame = self._normalize_frame(frame)
        instruction = self._build_instruction(state)

        if (
            self.reset_on_subgoal_switch
            and self._last_subgoal_index is not None
            and self._last_subgoal_index != state.current_subgoal_index
        ):
            self.agent.reset(instruction=instruction, task_name=state.task_text)

        action = self.agent.get_action(
            obs={"image": normalized_frame},
            info={"pov": normalized_frame},
            instruction=instruction,
            verbose=self.verbose,
        )
        self._last_subgoal_index = state.current_subgoal_index

        if not isinstance(action, dict) or "camera" not in action:
            raise ValueError("OpenHA returned invalid env action")
        return action

    def on_transition(self, state: RuntimeState) -> None:
        self._last_subgoal_index = state.current_subgoal_index

    def _build_instruction(self, state: RuntimeState) -> str:
        if self.instruction_mode == "task_only":
            instruction = state.task_text
        elif self.instruction_mode == "subgoal_only":
            instruction = state.current_subgoal.description
        elif self.instruction_mode == "task_and_subgoal":
            instruction = (
                f"Task: {state.task_text}\n"
                f"Current subgoal: {state.current_subgoal.description}"
            )
        else:
            raise ValueError(f"Unsupported instruction_mode: {self.instruction_mode}")

        if state.recover_mode and self.include_recover_hint:
            instruction += "\nRecovery: the previous approach seems stuck. Try a different local approach."
        if state.metadata.get("need_inventory_check"):
            instruction += "\nBefore continuing, open the inventory and check item counts."
        return instruction

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy array")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must have shape (H, W, 3), got {frame.shape}")
        if np.issubdtype(frame.dtype, np.floating):
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame


class OpenHAPurpleActorAdapter(OpenHAEnvActorAdapter):
    """OpenHA adapter that returns compact Purple agent actions."""

    def __init__(self, openha_agent: Any, codec: PurpleActionCodec | None = None, **kwargs):
        super().__init__(openha_agent=openha_agent, **kwargs)
        self.codec = codec or PurpleActionCodec()

    def act(self, frame: np.ndarray, state: RuntimeState) -> dict[str, Any]:
        env_action = super().act(frame, state)
        return self.codec.env_to_compact_agent(env_action)


OpenHAActorAdapter = OpenHAEnvActorAdapter
