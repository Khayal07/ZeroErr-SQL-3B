"""Protocol shared by every inference backend."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMEngine(ABC):
    """Minimal interface consumed by the guardrail loop and the API."""

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float = 0.1, max_tokens: int = 512) -> str:
        """Generate a completion for ``prompt`` and return the raw text."""