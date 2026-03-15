from __future__ import annotations

from typing import Any


ENV_NULL_ACTION = {
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,
    "sprint": 0,
    "sneak": 0,
    "use": 0,
    "drop": 0,
    "attack": 0,
    "jump": 0,
    "inventory": 0,
    "camera": [0.0, 0.0],
}


def noop_env_action() -> dict[str, Any]:
    return dict(ENV_NULL_ACTION)


def noop_agent_action() -> dict[str, Any]:
    return {
        "buttons": [0],
        "camera": [60],
    }


class PurpleActionCodec:
    """Convert env-style Minecraft actions into compact Purple agent actions."""

    def __init__(
        self,
        camera_binsize: int = 2,
        camera_maxval: int = 10,
        camera_mu: float = 10.0,
        camera_quantization_scheme: str = "mu_law",
    ):
        self.camera_binsize = camera_binsize
        self.camera_maxval = camera_maxval
        self.camera_mu = camera_mu
        self.camera_quantization_scheme = camera_quantization_scheme
        self._action_transformer = None
        self._action_mapper = None

    def _lazy_init(self) -> None:
        if self._action_transformer is not None and self._action_mapper is not None:
            return
        from minestudio.utils.vpt_lib.actions import ActionTransformer
        from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping

        self._action_transformer = ActionTransformer(
            camera_binsize=self.camera_binsize,
            camera_maxval=self.camera_maxval,
            camera_mu=self.camera_mu,
            camera_quantization_scheme=self.camera_quantization_scheme,
        )
        n_camera_bins = 2 * self.camera_maxval // self.camera_binsize + 1
        self._action_mapper = CameraHierarchicalMapping(n_camera_bins=n_camera_bins)

    def env_to_compact_agent(self, env_action: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(env_action, dict):
            return noop_agent_action()

        self._lazy_init()

        normalized: dict[str, Any] = {}
        for key, value in env_action.items():
            if key == "camera":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    return noop_agent_action()
                normalized[key] = [list(value)]
            else:
                normalized[key] = [int(bool(value))]

        try:
            factored = self._action_transformer.env2policy(normalized)
            joint = self._action_mapper.from_factored(factored)
            return {
                "buttons": [int(joint["buttons"][0, 0])],
                "camera": [int(joint["camera"][0, 0])],
            }
        except Exception:
            return noop_agent_action()
