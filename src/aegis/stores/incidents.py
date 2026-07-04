"""
Incident store - durable, auditable, zero-ops (SQLite).

Every incident and every state transition is persisted here. This is what backs
the "full audit trail" guarantee: the record survives a restart and can be
replayed, which an in-memory list cannot.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class IncidentStore:
    """SQLite-backed incident + audit-event store."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id   TEXT PRIMARY KEY,
                model_id      TEXT NOT NULL,
                state         TEXT NOT NULL,
                severity      TEXT,
                drift_type    TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                payload       TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id   TEXT NOT NULL,
                ts            TEXT NOT NULL,
                actor         TEXT NOT NULL,   -- 'controller' or 'investigator'
                from_state    TEXT,
                to_state      TEXT,
                detail        TEXT
            );
            """
        )
        self._conn.commit()

    @staticmethod
    def _to_dict(incident: Any) -> Dict[str, Any]:
        if is_dataclass(incident):
            d = asdict(incident)
        elif isinstance(incident, dict):
            d = dict(incident)
        else:
            d = dict(incident.__dict__)
        # normalise enum-ish state to its value
        state = d.get("state")
        d["state"] = getattr(state, "value", state)
        return d

    def save(self, incident: Any) -> None:
        """Upsert an incident (idempotent on incident_id)."""
        d = self._to_dict(incident)
        self._conn.execute(
            """
            INSERT INTO incidents
                (incident_id, model_id, state, severity, drift_type,
                 created_at, updated_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(incident_id) DO UPDATE SET
                state=excluded.state,
                severity=excluded.severity,
                drift_type=excluded.drift_type,
                updated_at=excluded.updated_at,
                payload=excluded.payload
            """,
            (
                d["incident_id"],
                d["model_id"],
                d["state"],
                d.get("severity"),
                d.get("drift_type"),
                d["created_at"],
                d["updated_at"],
                json.dumps(d, default=str),
            ),
        )
        self._conn.commit()

    def record_event(
        self,
        incident_id: str,
        ts: str,
        actor: str,
        from_state: Optional[str],
        to_state: Optional[str],
        detail: str = "",
    ) -> None:
        """Append an immutable audit event."""
        self._conn.execute(
            """
            INSERT INTO audit_events
                (incident_id, ts, actor, from_state, to_state, detail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (incident_id, ts, actor, from_state, to_state, detail),
        )
        self._conn.commit()

    def get(self, incident_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT payload FROM incidents WHERE incident_id = ?", (incident_id,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def list_open(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT payload FROM incidents WHERE state NOT IN "
            "('healthy','promoted','rolled_back') ORDER BY created_at DESC"
        ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def audit_trail(self, incident_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT ts, actor, from_state, to_state, detail FROM audit_events "
            "WHERE incident_id = ? ORDER BY id",
            (incident_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
