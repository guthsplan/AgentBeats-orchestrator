from __future__ import annotations

from agentbeats_orchestrator.codecs.image_codec import frame_hash
from agentbeats_orchestrator.execution.executor import ActionExecutor
from agentbeats_orchestrator.memory.event import Event, EventMemory
from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.memory.runtime import RuntimeEntry, RuntimeMemory
from agentbeats_orchestrator.state_cache.inventory import InventoryParser, NullInventoryParser
from agentbeats_orchestrator.types import ObservationSummary, PendingAction, RuntimeState


class PurpleRuntime:
    def __init__(
        self,
        planner,
        actor,
        evaluator,
        transition_policy,
        action_executor: ActionExecutor | None = None,
        inventory_parser: InventoryParser | None = None,
    ):
        self.planner = planner
        self.actor = actor
        self.evaluator = evaluator
        self.transition_policy = transition_policy
        self.action_executor = action_executor or ActionExecutor()
        self.inventory_parser = inventory_parser or NullInventoryParser()
        self.runtime_memory = RuntimeMemory()
        self.event_memory = EventMemory()
        self.failure_memory = FailureMemory()
        self._pending_action: PendingAction | None = None

    def initialize(self, task_text: str) -> RuntimeState:
        subgoals = self.planner.plan(task_text, failure_memory=self.failure_memory)
        self.actor.reset(task_text)
        self._pending_action = None
        return RuntimeState(task_text=task_text, subgoals=subgoals)

    def act(self, frame, state: RuntimeState):
        raw_action = self.actor.act(frame, state)
        execution = self.action_executor.execute(raw_action)
        self._pending_action = PendingAction(
            step_index=state.step_index,
            subgoal_index=state.current_subgoal_index,
            execution=execution,
        )
        return execution.executed_action

    def update(self, frame, reward: float, done: bool, state: RuntimeState):
        if self._pending_action is None:
            raise RuntimeError("update() called before act()")

        self._update_state_cache(frame, state)
        obs_summary = ObservationSummary(frame_hash=frame_hash(frame))
        if state.state_cache.inventory is not None:
            obs_summary.inventory = dict(state.state_cache.inventory.items)
        self.runtime_memory.append(
            RuntimeEntry(
                step_index=self._pending_action.step_index,
                subgoal_index=self._pending_action.subgoal_index,
                raw_action=self._pending_action.execution.raw_action,
                executed_action=self._pending_action.execution.executed_action,
                reward=reward,
                done=done,
                obs_summary=obs_summary,
                exec_info=self._pending_action.execution.info,
            )
        )

        evaluation = self.evaluator.evaluate(
            frame=frame,
            reward=reward,
            done=done,
            state=state,
            runtime_memory=self.runtime_memory,
            event_memory=self.event_memory,
        )
        state.metadata["need_inventory_check"] = bool(
            evaluation.signals.get("need_inventory_check", False)
        )
        for detected in evaluation.detected_events:
            self.event_memory.append(
                Event(
                    step_index=detected.get("step_index", state.step_index),
                    subgoal_index=detected.get("subgoal_index", state.current_subgoal_index),
                    kind=detected.get("kind", "event"),
                    summary=detected.get("summary", ""),
                    payload=detected,
                )
            )
        transition = self.transition_policy.decide(
            state=state,
            evaluation=evaluation,
            runtime_memory=self.runtime_memory,
            failure_memory=self.failure_memory,
        )
        if transition.decision.value in {"RECOVER", "REPLAN"}:
            self.failure_memory.record(
                subgoal_index=state.current_subgoal_index,
                summary=transition.reason or transition.decision.value,
            )
        if transition.decision.value == "REPLAN":
            replanned_subgoals = self.planner.replan(
                task_text=state.task_text,
                failure_memory=self.failure_memory,
                state=state,
            )
            if replanned_subgoals:
                state.subgoals = replanned_subgoals
                state.current_subgoal_index = 0
                state.recover_mode = False
                self.actor.reset(state.task_text)
        if transition.decision.value == "SWITCH_SUBGOAL" and transition.next_subgoal_index is not None:
            state.current_subgoal_index = transition.next_subgoal_index
        elif transition.decision.value == "RECOVER":
            state.recover_mode = True
        else:
            state.recover_mode = False
        self.actor.on_transition(state)

        last_entry = self.runtime_memory.entries[-1]
        last_entry.eval_summary = {
            "progress_score": evaluation.progress_score,
            "status": evaluation.status,
            "reasons": evaluation.reasons,
        }
        state.step_index += 1
        self._pending_action = None
        return transition, state

    def _update_state_cache(self, frame, state: RuntimeState) -> None:
        inventory = state.state_cache.inventory
        if inventory is not None:
            inventory.confidence *= 0.97
            if state.step_index - inventory.timestamp > 20:
                inventory.confidence *= 0.8

            raw_action = self._pending_action.execution.raw_action if self._pending_action else {}
            if any(raw_action.get(key, 0) for key in ("use", "drop", "attack")):
                inventory.confidence *= 0.7

        if self.inventory_parser.can_parse(frame, state):
            snapshot = self.inventory_parser.parse(frame, state)
            if snapshot is not None:
                state.state_cache.inventory = snapshot
