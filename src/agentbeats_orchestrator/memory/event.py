from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    step_index: int
    subgoal_index: int
    kind: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


class EventMemory:
    def __init__(self):
        self.events: list[Event] = []

    def append(self, event: Event) -> None:
        self.events.append(event)

    def query(self, subgoal_index: int) -> list[Event]:
        return [event for event in self.events if event.subgoal_index == subgoal_index]
