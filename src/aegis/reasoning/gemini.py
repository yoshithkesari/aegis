"""
Gemini reasoner - hosted, free-tier, zero local weights.

Enforces a hard per-process call cap so a runaway loop cannot exceed the free
tier, and translates any API failure (429, timeout) into a ReasonerError so the
caller can fall back to a local reasoner instead of crashing.
"""

from __future__ import annotations

from typing import Optional

from .base import Reasoner, ReasonerError

try:
    import google.generativeai as genai

    _GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _GENAI_AVAILABLE = False


class GeminiReasoner(Reasoner):
    """Single hosted call per incident, capped and fail-safe."""

    name = "gemini"

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gemini-2.5-flash",
        max_calls: int = 20,
    ):
        self.model = model
        self.max_calls = max_calls
        self._calls = 0
        self._client = None

        if _GENAI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model)

    @property
    def available(self) -> bool:
        return self._client is not None and self._calls < self.max_calls

    def complete(self, prompt: str) -> str:
        if self._client is None:
            raise ReasonerError("Gemini client unavailable (no key or SDK missing)")
        if self._calls >= self.max_calls:
            raise ReasonerError(f"Gemini call cap reached ({self.max_calls})")
        try:
            self._calls += 1
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as exc:  # 429 / timeout / transport error
            raise ReasonerError(f"Gemini request failed: {exc}") from exc
