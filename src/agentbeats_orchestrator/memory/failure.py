from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FailurePattern:
    subgoal_index: int
    summary: str
    count: int = 1


class FailureMemory:
    def __init__(self):
        self.patterns: list[FailurePattern] = []

    def record(self, subgoal_index: int, summary: str) -> None:
        for pattern in self.patterns:
            if pattern.subgoal_index == subgoal_index and pattern.summary == summary:
                pattern.count += 1
                return
        self.patterns.append(FailurePattern(subgoal_index=subgoal_index, summary=summary))

    def query(self, subgoal_index: int) -> list[FailurePattern]:
        return [pattern for pattern in self.patterns if pattern.subgoal_index == subgoal_index]

    def total_failures(self, subgoal_index: int | None = None) -> int:
        patterns = self.patterns if subgoal_index is None else self.query(subgoal_index)
        return sum(pattern.count for pattern in patterns)

    def as_prompt_context(self, subgoal_index: int | None = None) -> str:
        patterns = self.patterns if subgoal_index is None else self.query(subgoal_index)
        if not patterns:
            return ""
        return "\n".join(
            f"- {pattern.summary} (count={pattern.count})"
            for pattern in patterns
        )
