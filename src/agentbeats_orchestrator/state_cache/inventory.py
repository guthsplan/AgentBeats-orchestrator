from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from agentbeats_orchestrator.types import InventorySnapshot, RuntimeState


class InventoryParser(ABC):
    @abstractmethod
    def can_parse(self, frame: np.ndarray, state: RuntimeState) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, frame: np.ndarray, state: RuntimeState) -> InventorySnapshot | None:
        raise NotImplementedError


class NullInventoryParser(InventoryParser):
    def can_parse(self, frame: np.ndarray, state: RuntimeState) -> bool:
        return False

    def parse(self, frame: np.ndarray, state: RuntimeState) -> InventorySnapshot | None:
        return None
