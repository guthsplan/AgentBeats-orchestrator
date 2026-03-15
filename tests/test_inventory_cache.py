from agentbeats_orchestrator.evaluator.heuristic import HeuristicEvaluator
from agentbeats_orchestrator.types import RuntimeState, Subgoal


def test_inventory_check_requested_for_inventory_sensitive_subgoal_without_cache():
    evaluator = HeuristicEvaluator()
    state = RuntimeState(
        task_text="craft wooden pickaxe",
        subgoals=[Subgoal(name="s0", description="craft planks")],
    )
    result = evaluator.evaluate(
        frame=None,
        reward=0.0,
        done=False,
        state=state,
        runtime_memory=type("RuntimeMemoryStub", (), {
            "recent_action_repeat_ratio": lambda self: 0.0,
            "no_progress_streak": lambda self: 0,
        })(),
        event_memory=None,
    )
    assert result.signals["need_inventory_check"] is True
