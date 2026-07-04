"""
Central configuration for AEGIS.

One settings object, populated from environment variables (optionally via a
`.env` file). Every tunable the report talks about - bounded-loop caps, deploy
thresholds, store locations, the local bind address - lives here so nothing is
hard-coded across the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:  # optional - .env is a convenience, not a requirement
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


REPO_ROOT = Path(__file__).resolve().parents[2]


def _flag(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable, process-wide configuration."""

    # --- reasoning (the pluggable LLM) ---
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2")
    )
    # cap keeps every deploy comfortably inside the free tier
    max_llm_calls_per_incident: int = field(
        default_factory=lambda: int(os.getenv("MAX_LLM_CALLS", "20"))
    )

    # --- bounded-loop safety (adversarial-drift defense) ---
    max_retrains_per_day: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRAINS_PER_DAY", "5"))
    )
    retrain_cooldown_seconds: int = field(
        default_factory=lambda: int(os.getenv("RETRAIN_COOLDOWN_SECONDS", "900"))
    )

    # --- deploy gate thresholds (deterministic, LLM-free) ---
    drift_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_THRESHOLD", "0.05"))
    )
    regression_threshold: float = field(
        default_factory=lambda: float(os.getenv("REGRESSION_THRESHOLD", "-0.05"))
    )

    # --- stores ---
    incidents_db: str = field(
        default_factory=lambda: os.getenv(
            "INCIDENTS_DB", str(REPO_ROOT / "artifacts" / "incidents.db")
        )
    )
    metrics_db: str = field(
        default_factory=lambda: os.getenv(
            "METRICS_DB", str(REPO_ROOT / "artifacts" / "metrics.duckdb")
        )
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI", str(REPO_ROOT / "artifacts" / "mlruns")
        )
    )

    # --- serving ---
    # bind to loopback by default: no egress, nothing exposed to the network
    host: str = field(default_factory=lambda: os.getenv("AEGIS_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("AEGIS_PORT", "8000")))

    # send only aggregated stats to a hosted LLM; never raw rows / PII
    redact_before_hosted_llm: bool = field(
        default_factory=lambda: _flag("REDACT_BEFORE_HOSTED_LLM", True)
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
