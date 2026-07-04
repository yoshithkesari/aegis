"""
Metrics store - time-series of model health (DuckDB).

Records the health metrics the monitor emits so the dashboard and scoreboard can
query trends. Falls back to an in-memory list when DuckDB is not installed, so
nothing hard-fails on a minimal host.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import duckdb

    _DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional
    _DUCKDB_AVAILABLE = False


class MetricsStore:
    """Append-only time-series of (model, metric, value, ts)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backend = "duckdb" if _DUCKDB_AVAILABLE else "memory"
        self._mem: List[Dict[str, Any]] = []
        self._conn = None
        if _DUCKDB_AVAILABLE:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(db_path)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    ts        TIMESTAMP,
                    model_id  VARCHAR,
                    metric    VARCHAR,
                    value     DOUBLE
                )
                """
            )

    def record(self, model_id: str, metric: str, value: float, ts: str) -> None:
        if self._conn is not None:
            self._conn.execute(
                "INSERT INTO metrics VALUES (?, ?, ?, ?)",
                (ts, model_id, metric, float(value)),
            )
        else:
            self._mem.append(
                {"ts": ts, "model_id": model_id, "metric": metric, "value": float(value)}
            )

    def history(self, model_id: str, metric: str) -> List[Dict[str, Any]]:
        if self._conn is not None:
            rows = self._conn.execute(
                "SELECT ts, value FROM metrics WHERE model_id = ? AND metric = ? "
                "ORDER BY ts",
                (model_id, metric),
            ).fetchall()
            return [{"ts": str(ts), "value": v} for ts, v in rows]
        return [
            {"ts": r["ts"], "value": r["value"]}
            for r in self._mem
            if r["model_id"] == model_id and r["metric"] == metric
        ]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
