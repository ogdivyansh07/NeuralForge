"""
Conversational memory placeholder for multi-turn PDF QA.

Wire this to Streamlit `st.session_state` or a DB later. Module-level API is intentionally minimal.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class ConversationBuffer:
    """In-process buffer (optional); use session state in UI for persistence."""

    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 12

    def append(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def as_context_block(self) -> str:
        if not self.turns:
            return ""
        lines = [f"{t.role}: {t.content}" for t in self.turns[-6:]]
        return "\n".join(lines)


_GLOBAL_BUFFER = ConversationBuffer()


def append_turn(role: str, content: str) -> None:
    """Placeholder hook for future multi-turn conditioning."""
    _GLOBAL_BUFFER.append(role, content)


def get_recent_context() -> str:
    """Return a short string of recent turns for optional prompt injection."""
    return _GLOBAL_BUFFER.as_context_block()


def clear_memory() -> None:
    _GLOBAL_BUFFER.turns.clear()
