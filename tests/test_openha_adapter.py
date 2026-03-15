import numpy as np

from agentbeats_orchestrator.actor.openha_adapter import OpenHAEnvActorAdapter
from agentbeats_orchestrator.types import RuntimeState, Subgoal


class DummyOpenHA:
    def __init__(self):
        self.reset_calls = []
        self.action_calls = []

    def reset(self, instruction=None, task_name=None):
        self.reset_calls.append((instruction, task_name))

    def get_action(self, obs=None, info=None, instruction=None, verbose=False):
        self.action_calls.append(
            {
                "obs": obs,
                "info": info,
                "instruction": instruction,
                "verbose": verbose,
            }
        )
        return {"forward": 1, "camera": [0.0, 0.0]}


def test_openha_adapter_builds_task_and_subgoal_instruction():
    agent = DummyOpenHA()
    adapter = OpenHAEnvActorAdapter(agent, instruction_mode="task_and_subgoal")
    state = RuntimeState(
        task_text="craft wooden pickaxe",
        subgoals=[Subgoal(name="s0", description="gather wood")],
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    action = adapter.act(frame, state)

    assert action["forward"] == 1
    assert "Task: craft wooden pickaxe" in agent.action_calls[0]["instruction"]
    assert "Current subgoal: gather wood" in agent.action_calls[0]["instruction"]


def test_openha_adapter_normalizes_float_frame():
    agent = DummyOpenHA()
    adapter = OpenHAEnvActorAdapter(agent)
    state = RuntimeState(
        task_text="collect logs",
        subgoals=[Subgoal(name="s0", description="find tree")],
    )
    frame = np.ones((4, 4, 3), dtype=np.float32)

    adapter.act(frame, state)

    sent = agent.action_calls[0]["obs"]["image"]
    assert sent.dtype == np.uint8
    assert sent.max() == 255
