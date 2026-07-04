"""
Reasoner interface.

The one place the LLM is allowed to influence the system is root-cause
synthesis. That call sits behind this interface so the hosted free-tier model
(Gemini) can be swapped for a local model (Ollama) - or a no-LLM heuristic -
with a one-line change and no edits to the investigator.

A Reasoner is *read-only by contract*: it turns a prompt into text. It has no
handle on any store, registry, or deploy action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ReasonerError(RuntimeError):
    """Raised when a reasoner cannot produce a completion (e.g. 429, timeout)."""


class Reasoner(ABC):
    """Turns a prompt into a text completion. Nothing else."""

    #: human-readable identifier, surfaced in postmortems / audit logs
    name: str = "reasoner"

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return a completion for ``prompt`` or raise :class:`ReasonerError`."""

    @property
    def available(self) -> bool:
        """Whether this reasoner can currently serve a request."""
        return True
