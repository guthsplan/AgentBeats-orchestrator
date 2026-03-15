from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from agentbeats_orchestrator.types import RuntimeState


class Actor(ABC):
    @abstractmethod
    def reset(self, task_text: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, frame: np.ndarray, state: RuntimeState) -> dict[str, Any]:
        raise NotImplementedError

    def on_transition(self, state: RuntimeState) -> None:
        """Optional hook for actors that need to react to state changes."""
        return None
