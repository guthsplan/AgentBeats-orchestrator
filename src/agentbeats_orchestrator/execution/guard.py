from __future__ import annotations

from typing import Any

from agentbeats_orchestrator.codecs.action_codec import noop_env_action

REQUIRED_BUTTON_KEYS = [
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "sprint",
    "sneak",
    "use",
    "drop",
    "attack",
    "jump",
    "inventory",
]


class ActionGuard:
    def sanitize(self, action: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        info: dict[str, Any] = {"guard_applied": False, "reasons": []}
        if not isinstance(action, dict):
            info["guard_applied"] = True
            info["reasons"].append("invalid_action_type")
            return noop_env_action(), info

        sanitized = noop_env_action()
        for key in REQUIRED_BUTTON_KEYS:
            sanitized[key] = int(bool(action.get(key, 0)))

        camera = action.get("camera", [0.0, 0.0])
        if not isinstance(camera, (list, tuple)) or len(camera) != 2:
            info["guard_applied"] = True
            info["reasons"].append("invalid_camera")
            camera = [0.0, 0.0]
        pitch = float(max(-180.0, min(180.0, camera[0])))
        yaw = float(max(-180.0, min(180.0, camera[1])))
        if [pitch, yaw] != list(camera):
            info["guard_applied"] = True
            info["reasons"].append("camera_clamped")
        sanitized["camera"] = [pitch, yaw]
        return sanitized, info
