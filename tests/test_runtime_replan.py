import numpy as np

from agentbeats_orchestrator.evaluator.base import Evaluator
from agentbeats_orchestrator.memory.event import EventMemory
from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.memory.runtime import RuntimeMemory
from agentbeats_orchestrator.runtime.loop import PurpleRuntime
from agentbeats_orchestrator.transition.base import TransitionPolicy
from agentbeats_orchestrator.types import Decision, EvaluationResult, RuntimeState, Subgoal, TransitionResult


class DummyPlanner:
    def __init__(self):
        self.replan_calls = 0

    def plan(self, task_text, failure_memory=None, state=None):
        return [Subgoal(name="s0", description=task_text)]

    def replan(self, task_text, failure_memory, state):
        self.replan_calls += 1
        return [Subgoal(name="replanned", description="replanned task")]


class DummyActor:
    def __init__(self):
        self.reset_calls = 0

    def reset(self, task_text):
        self.reset_calls += 1

    def act(self, frame, state):
        return {"forward": 1, "camera": [0.0, 0.0]}

    def on_transition(self, state):
        pass


class DummyEvaluator(Evaluator):
    def evaluate(self, frame, reward, done, state, runtime_memory: RuntimeMemory, event_memory: EventMemory):
        return EvaluationResult(progress_score=0.0, status="stalled")


class ForceReplanPolicy(TransitionPolicy):
    def decide(self, state: RuntimeState, evaluation, runtime_memory: RuntimeMemory, failure_memory: FailureMemory):
        return TransitionResult(decision=Decision.REPLAN, reason="forced_replan")


def test_runtime_replan_calls_planner_and_resets_actor():
    planner = DummyPlanner()
    actor = DummyActor()
    runtime = PurpleRuntime(
        planner=planner,
        actor=actor,
        evaluator=DummyEvaluator(),
        transition_policy=ForceReplanPolicy(),
    )

    state = runtime.initialize("collect logs")
    runtime.act(np.zeros((4, 4, 3), dtype=np.uint8), state)
    transition, state = runtime.update(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        reward=0.0,
        done=False,
        state=state,
    )

    assert transition.decision == Decision.REPLAN
    assert planner.replan_calls == 1
    assert actor.reset_calls == 2
    assert state.subgoals[0].name == "replanned"
