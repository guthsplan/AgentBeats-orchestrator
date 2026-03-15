from agentbeats_orchestrator.codecs.action_codec import noop_env_action


def test_noop_env_action_has_camera():
    action = noop_env_action()
    assert "camera" in action
    assert action["camera"] == [0.0, 0.0]
