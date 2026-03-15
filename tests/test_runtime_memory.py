from agentbeats_orchestrator.memory.runtime import RuntimeEntry, RuntimeMemory


def test_recent_action_repeat_ratio():
    memory = RuntimeMemory(maxlen=5)
    action = {"forward": 1, "camera": [0.0, 0.0]}
    for step in range(3):
        memory.append(
            RuntimeEntry(
                step_index=step,
                subgoal_index=0,
                raw_action=action,
                executed_action=action,
            )
        )
    assert memory.recent_action_repeat_ratio() == 1.0
