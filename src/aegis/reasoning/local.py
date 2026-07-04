"""
Local reasoners - no data egress.

`OllamaReasoner` talks to a local Ollama server (features never leave the
machine). `HeuristicReasoner` uses no LLM at all: it is the always-available
last resort so the loop closes even offline, with no key, on a cold free host.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import Reasoner, ReasonerError


class OllamaReasoner(Reasoner):
    """Completions from a local Ollama model - private, zero egress."""

    name = "ollama"

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "llama3.2",
        timeout: float = 30.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
            return body.get("response", "")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ReasonerError(f"Ollama request failed: {exc}") from exc


class HeuristicReasoner(Reasoner):
    """No LLM. Synthesises a diagnosis directly from the tool evidence.

    Always available - this is what guarantees the loop closes when no model
    is reachable. It reads the same structured evidence the prompt is built
    from, so its output slots into the exact same parser.
    """

    name = "heuristic"

    def complete(self, prompt: str) -> str:
        # The heuristic path does not parse free text; the investigator calls
        # `diagnose_from_evidence` directly. `complete` exists to satisfy the
        # interface and returns a stable, parseable stub.
        return (
            "DIAGNOSIS: Performance degradation detected\n"
            "ROOT CAUSE: Distribution shift in the top drifting feature\n"
            "RECOMMENDED ACTION: Retrain challenger on a recent window\n"
            "CONFIDENCE: 0.5"
        )
