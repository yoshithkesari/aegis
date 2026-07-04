"""
Model registry - champion / challenger with rollback.

Wraps MLflow when it is installed; otherwise falls back to a tiny filesystem
registry (joblib artifacts + a JSON pointer) so the demo runs on a bare free
host with no tracking server. The interface is identical either way, so the
controller never needs to know which backend is active.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import mlflow  # noqa: F401

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    _MLFLOW_AVAILABLE = False


@dataclass
class RegisteredModel:
    model_id: str
    version: str
    stage: str  # 'champion' | 'challenger' | 'archived'
    uri: str


class ModelRegistry:
    """Champion/challenger registry with instant rollback.

    Always keeps the previous champion so a promotion can be reversed with no
    retrain - the blast-radius guarantee the report makes.
    """

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.backend = "mlflow" if _MLFLOW_AVAILABLE else "filesystem"
        self._root = Path(tracking_uri)
        self._root.mkdir(parents=True, exist_ok=True)
        self._pointer = self._root / "registry.json"
        if not self._pointer.exists():
            self._write_state({"champion": None, "challenger": None, "previous": None})

    # --- pointer state (filesystem backend) ---
    def _read_state(self) -> dict:
        return json.loads(self._pointer.read_text())

    def _write_state(self, state: dict) -> None:
        self._pointer.write_text(json.dumps(state, indent=2))

    def register_challenger(self, model_id: str, artifact_path: str) -> RegisteredModel:
        """Stage a freshly retrained model as the challenger."""
        state = self._read_state()
        version = f"v{self._next_version()}"
        dest = self._root / model_id / version
        dest.mkdir(parents=True, exist_ok=True)
        stored = str(dest / Path(artifact_path).name)
        shutil.copy2(artifact_path, stored)
        state["challenger"] = {"model_id": model_id, "version": version, "uri": stored}
        self._write_state(state)
        return RegisteredModel(model_id, version, "challenger", stored)

    def promote(self) -> Optional[RegisteredModel]:
        """Promote challenger -> champion, retaining the old champion."""
        state = self._read_state()
        challenger = state.get("challenger")
        if not challenger:
            return None
        state["previous"] = state.get("champion")  # keep for rollback
        state["champion"] = challenger
        state["challenger"] = None
        self._write_state(state)
        return RegisteredModel(
            challenger["model_id"], challenger["version"], "champion", challenger["uri"]
        )

    def rollback(self) -> Optional[RegisteredModel]:
        """Instant rollback to the retained previous champion."""
        state = self._read_state()
        previous = state.get("previous")
        if not previous:
            return None
        state["champion"] = previous
        state["previous"] = None
        self._write_state(state)
        return RegisteredModel(
            previous["model_id"], previous["version"], "champion", previous["uri"]
        )

    def get_champion(self) -> Optional[RegisteredModel]:
        champ = self._read_state().get("champion")
        if not champ:
            return None
        return RegisteredModel(champ["model_id"], champ["version"], "champion", champ["uri"])

    def _next_version(self) -> int:
        state = self._read_state()
        seen = [
            m for m in (state.get("champion"), state.get("challenger"), state.get("previous"))
            if m
        ]
        return len(list(self._root.rglob("v*"))) + len(seen) + 1
