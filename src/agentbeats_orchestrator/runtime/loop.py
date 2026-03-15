from __future__ import annotations

from agentbeats_orchestrator.codecs.image_codec import frame_hash
from agentbeats_orchestrator.execution.executor import ActionExecutor
from agentbeats_orchestrator.memory.event import Event, EventMemory
from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.memory.runtime import RuntimeEntry, RuntimeMemory
from agentbeats_orchestrator.types import ObservationSummary, RuntimeState


class PurpleRuntime:
    def __init__(
        self,
        planner,
        actor,
        evaluator,
        transition_policy,
        action_executor: ActionExecutor | None = None,
    ):
        self.planner = planner
        self.actor = actor
        self.evaluator = evaluator
        self.transition_policy = transition_policy
        self.action_executor = action_executor or ActionExecutor()
        self.runtime_memory = RuntimeMemory()
        self.event_memory = EventMemory()
        self.failure_memory = FailureMemory()

    def initialize(self, task_text: str) -> RuntimeState:
        subgoals = self.planner.plan(task_text)
        self.actor.reset(task_text)
        return RuntimeState(task_text=task_text, subgoals=subgoals)

    def step(self, frame, state: RuntimeState):
        raw_action = self.actor.act(frame, state)
        execution = self.action_executor.execute(raw_action)
        evaluation = self.evaluator.evaluate(
            frame=frame,
            state=state,
            runtime_memory=self.runtime_memory,
            event_memory=self.event_memory,
            failure_memory=self.failure_memory,
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
        if transition.decision.value == "SWITCH_SUBGOAL" and transition.next_subgoal_index is not None:
            state.current_subgoal_index = transition.next_subgoal_index
        elif transition.decision.value == "RECOVER":
            state.recover_mode = True
        else:
            state.recover_mode = False

        obs_summary = ObservationSummary(frame_hash=frame_hash(frame))
        self.runtime_memory.append(
            RuntimeEntry(
                step_index=state.step_index,
                subgoal_index=state.current_subgoal_index,
                raw_action=execution.raw_action,
                executed_action=execution.executed_action,
                obs_summary=obs_summary,
                exec_info=execution.info,
                eval_summary={
                    "progress_score": evaluation.progress_score,
                    "status": evaluation.status,
                    "reasons": evaluation.reasons,
                },
            )
        )
        state.step_index += 1
        return execution.executed_action, transition, state

    def record_env_feedback(self, reward: float, done: bool) -> None:
        if not self.runtime_memory.entries:
            return
        last = self.runtime_memory.entries[-1]
        last.reward = reward
        last.done = done
