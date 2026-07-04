"""
Pluggable reasoning layer.

`build_reasoner()` picks the best available backend and always returns something
that works: hosted Gemini when a key is present, otherwise a local model,
otherwise a no-LLM heuristic. Swapping the demo's hosted model for a private
local one is a config change, not a code change.
"""

from __future__ import annotations

from typing import Optional

from .base import Reasoner, ReasonerError
from .gemini import GeminiReasoner
from .local import HeuristicReasoner, OllamaReasoner

__all__ = [
    "Reasoner",
    "ReasonerError",
    "GeminiReasoner",
    "OllamaReasoner",
    "HeuristicReasoner",
    "FallbackReasoner",
    "build_reasoner",
]


class FallbackReasoner(Reasoner):
    """Tries each reasoner in order; the last one should be always-available."""

    name = "fallback"

    def __init__(self, *reasoners: Reasoner):
        if not reasoners:
            raise ValueError("FallbackReasoner needs at least one backend")
        self.reasoners = list(reasoners)

    def complete(self, prompt: str) -> str:
        last_error: Optional[Exception] = None
        for reasoner in self.reasoners:
            if not reasoner.available:
                continue
            try:
                return reasoner.complete(prompt)
            except ReasonerError as exc:
                last_error = exc
                continue
        raise ReasonerError(f"All reasoners exhausted (last: {last_error})")


def build_reasoner(settings=None) -> Reasoner:
    """Construct the default reasoner chain from settings.

    Gemini (hosted, free tier) -> Ollama (local) -> Heuristic (no LLM).
    """
    if settings is None:
        from ..config import get_settings

        settings = get_settings()

    chain: list[Reasoner] = []
    if settings.gemini_api_key:
        chain.append(
            GeminiReasoner(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                max_calls=settings.max_llm_calls_per_incident,
            )
        )
    chain.append(
        OllamaReasoner(host=settings.ollama_host, model=settings.ollama_model)
    )
    chain.append(HeuristicReasoner())
    return FallbackReasoner(*chain)
