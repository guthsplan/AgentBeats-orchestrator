from agentbeats_orchestrator.memory.failure import FailureMemory
from agentbeats_orchestrator.memory.runtime import RuntimeEntry, RuntimeMemory
from agentbeats_orchestrator.transition.rule_based import RuleBasedTransitionPolicy
from agentbeats_orchestrator.types import EvaluationResult, RuntimeState, Subgoal


def test_switch_subgoal_when_stalled():
    memory = RuntimeMemory(maxlen=20)
    for step in range(12):
        memory.append(
            RuntimeEntry(
                step_index=step,
                subgoal_index=0,
                raw_action={},
                executed_action={},
                reward=0.0,
            )
        )
    policy = RuleBasedTransitionPolicy()
    state = RuntimeState(
        task_text="collect logs",
        subgoals=[
            Subgoal(name="s0", description="find tree"),
            Subgoal(name="s1", description="mine logs"),
        ],
    )
    evaluation = EvaluationResult(progress_score=0.1, status="stalled")
    result = policy.decide(state, evaluation, memory, FailureMemory())
    assert result.next_subgoal_index == 1
