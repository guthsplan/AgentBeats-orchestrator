from agentbeats_orchestrator.memory.failure import FailureMemory


def test_failure_memory_prompt_context_aggregates_counts():
    memory = FailureMemory()
    memory.record(0, "local_stall")
    memory.record(0, "local_stall")
    memory.record(0, "repeated_subgoal_failures")

    context = memory.as_prompt_context(subgoal_index=0)

    assert "local_stall (count=2)" in context
    assert "repeated_subgoal_failures (count=1)" in context
