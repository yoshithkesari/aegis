"""
Composition root - assemble the real components and run a full incident.

This is where the pieces stop being parts and become a system: a trained
champion, a drift-injected recent window, a registry-backed remediation, the
label-free CBPE gate, and a durable incident store, all driven by the
deterministic controller. `python -m aegis.system` runs one incident and prints
the diagnosis, the deploy-gate decision, the resolved state, and the persisted
audit trail.

Everything here is offline and deterministic (heuristic reasoner, fixed seeds)
so it runs anywhere with no API key.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from .agent.investigator import Investigator
from .control_plane.controller import Controller
from .control_plane.detectors import DetectorSuite
from .control_plane.monitor import StreamMonitor
from .control_plane.risk_gate import RiskGate
from .data_plane.replayer import StreamReplayer
from .reasoning.local import HeuristicReasoner
from .stores import IncidentStore, ModelRegistry
from .validation import DeployGate

FEATURE_COLS = [f"f{i}" for i in range(12)]

MODEL_ID = "fraud-classifier"


class RegistryRemediation:
    """Real write-side: trains a challenger, versions it, promotes/rolls back.

    On retrain it attaches the challenger's predictions + measured champion
    baseline to the incident so the label-free gate has real evidence to judge.
    Promote/rollback go through the model registry, which always retains the
    previous champion for instant rollback.
    """

    def __init__(self, registry: ModelRegistry, workdir: str,
                 recent_X, recent_y, reference_df: pd.DataFrame, champion):
        self.registry = registry
        self.workdir = workdir
        self.recent_X, self.recent_y = recent_X, recent_y
        self.reference_df = reference_df
        self.champion = champion
        self._challenger = None

    def retrain(self, incident) -> Dict[str, Any]:
        challenger = LogisticRegression(max_iter=500).fit(self.recent_X, self.recent_y)
        proba = challenger.predict_proba(self.recent_X)[:, 1]
        baseline = float((self.champion.predict(self.recent_X) == self.recent_y).mean())

        path = os.path.join(self.workdir, "challenger.pkl")
        joblib.dump(challenger, path)
        version = self.registry.register_challenger(MODEL_ID, path).version

        # evidence for the label-free deploy gate (no ground-truth labels used)
        incident.validation_context = {
            "challenger_proba": proba,
            "baseline": baseline,
            "reference": self.reference_df,
        }
        self._challenger = challenger
        return {"success": True, "job_id": version}

    def deploy_canary(self, incident) -> Dict[str, Any]:
        return {"success": True, "metrics": {"traffic_percentage": 0.1}}

    def promote(self, incident) -> Dict[str, Any]:
        promoted = self.registry.promote()
        if promoted and self._challenger is not None:
            self.champion = self._challenger
        return {"success": promoted is not None}

    def rollback(self, incident) -> Dict[str, Any]:
        return {"success": self.registry.rollback() is not None}


@dataclass
class System:
    controller: Controller
    registry: ModelRegistry
    incident_store: IncidentStore
    remediation: RegistryRemediation
    monitor: StreamMonitor
    healthy_stream: pd.DataFrame
    drift_stream: pd.DataFrame


def build_system(workdir: Optional[str] = None, seed: int = 1,
                 drift_scale: float = 1.1) -> System:
    """Assemble a fully-wired, real (offline) AEGIS system."""
    workdir = workdir or tempfile.mkdtemp(prefix="aegis_")

    # --- data: one world split into reference + recent, then drift the recent ---
    X, y = make_classification(20000, n_features=12, n_informative=6, random_state=seed)
    Xr, yr, Xp, yp = X[:10000], y[:10000], X[10000:], y[10000:]
    champion = LogisticRegression(max_iter=500).fit(Xr, yr)
    reference_df = pd.DataFrame(
        {"proba": champion.predict_proba(Xr)[:, 1], "is_fraud": yr}
    )
    rng = np.random.RandomState(seed + 3)
    recent_X = Xp + rng.normal(0, drift_scale, Xp.shape)  # covariate drift
    recent_y = yp

    # Feature-only frames for the stream monitor / detector.
    reference_features = pd.DataFrame(Xr, columns=FEATURE_COLS)
    healthy_stream = pd.DataFrame(Xp, columns=FEATURE_COLS)  # same dist as reference
    # Drift two features -> a small set flagged -> MEDIUM severity (auto-remediate).
    drifted = Xp.copy()
    drifted[:, :2] += 3.0
    drift_stream = pd.DataFrame(drifted, columns=FEATURE_COLS)

    # --- stores ---
    registry = ModelRegistry(os.path.join(workdir, "mlruns"))
    incident_store = IncidentStore(os.path.join(workdir, "incidents.db"))

    # register + promote the champion so the registry starts populated
    champ_path = os.path.join(workdir, "champion.pkl")
    joblib.dump(champion, champ_path)
    registry.register_challenger(MODEL_ID, champ_path)
    registry.promote()

    remediation = RegistryRemediation(
        registry, workdir, recent_X, recent_y, reference_df, champion
    )

    detector = DetectorSuite()
    controller = Controller(model_id=MODEL_ID)
    controller.set_dependencies(
        detectors=detector,
        risk_gate=RiskGate(),
        remediation=remediation,
        investigator=Investigator(reasoner=HeuristicReasoner()),  # offline
        validator=DeployGate(),                                   # real CBPE gate
        incident_store=incident_store,
    )
    monitor = StreamMonitor(controller, reference_features, detector=detector)
    return System(controller, registry, incident_store, remediation,
                  monitor, healthy_stream, drift_stream)


def run_stream(system: System, drifted: bool = True, batch_size: int = 200) -> Dict[str, Any]:
    """Replay a stream through the monitor; drift auto-opens an incident."""
    stream = system.drift_stream if drifted else system.healthy_stream
    replayer = StreamReplayer(stream, batch_size=batch_size)
    return system.monitor.run(replayer)


def run_incident(system: System, severity: str = "medium") -> Dict[str, Any]:
    """Open and run one incident end-to-end; return a structured summary."""
    incident = system.controller.handle_drift_detected(
        {"drift_type": "covariate", "severity": severity,
         "summary": "covariate drift on recent window"}
    )
    status = system.controller.get_status()
    champ = system.registry.get_champion()
    return {
        "final_state": status["current_state"],
        "incident_id": incident.incident_id,
        "diagnosis": incident.diagnosis,
        "root_cause": incident.root_cause,
        "gate": incident.validation_result,
        "champion_version": champ.version if champ else None,
        "audit_trail": system.incident_store.audit_trail(incident.incident_id),
    }


def demo_run(seed: int = 1) -> Dict[str, Any]:
    """Run the full narrative on real components; return everything a UI needs.

    Computes the *true* (label-computed) champion and challenger accuracy on the
    drifted window - which the live gate never sees - so a demo can show the
    label-free estimate against ground truth.
    """
    system = build_system(seed=seed)
    champion = system.remediation.champion            # capture before promotion
    rx, ry = system.remediation.recent_X, system.remediation.recent_y
    champion_acc = float((champion.predict(rx) == ry).mean())

    healthy = run_stream(system, drifted=False)       # no false interventions
    drift = run_stream(system, drifted=True)          # auto-opens + resolves

    inc = system.controller.incident_history[-1]
    challenger = system.remediation._challenger
    challenger_acc = (
        float((challenger.predict(rx) == ry).mean()) if challenger is not None else None
    )
    champ = system.registry.get_champion()
    return {
        "incident_id": inc.incident_id,
        "diagnosis": inc.diagnosis,
        "root_cause": inc.root_cause,
        "severity": (drift["detections"][0] if drift["detections"] else None),
        "gate": inc.validation_result or {},
        "champion_acc": champion_acc,        # true, before remediation
        "challenger_acc": challenger_acc,    # true, after remediation
        "healthy_incidents": healthy["incidents_opened"],
        "drift_incidents": drift["incidents_opened"],
        "final_state": drift["final_state"],
        "champion_version": champ.version if champ else None,
        "audit_trail": system.incident_store.audit_trail(inc.incident_id),
    }


def main():  # pragma: no cover - CLI entrypoint
    system = build_system()

    # 1) healthy stream -> detector stays quiet, no incident opened
    healthy = run_stream(system, drifted=False)
    print(f"[stream] healthy: {healthy['batches_seen']} batches, "
          f"{healthy['incidents_opened']} incidents opened")

    # 2) drifted stream -> detector auto-opens an incident, loop runs
    drifted = run_stream(system, drifted=True)
    print(f"[stream] drifted: {drifted['incidents_opened']} incident(s) auto-opened, "
          f"severities={drifted['detections'][:1]}, final state={drifted['final_state']}")

    inc_id = system.controller.incident_history[-1].incident_id
    trail = system.incident_store.audit_trail(inc_id)
    champ = system.registry.get_champion()
    result = {
        "incident_id": inc_id,
        "diagnosis": system.controller.incident_history[-1].diagnosis,
        "root_cause": system.controller.incident_history[-1].root_cause,
        "gate": system.controller.incident_history[-1].validation_result,
        "final_state": drifted["final_state"],
        "champion_version": champ.version if champ else None,
        "audit_trail": trail,
    }
    print("=" * 64)
    print(f"AEGIS incident {result['incident_id']}  (auto-opened from stream)")
    print("=" * 64)
    print(f"diagnosis     : {result['diagnosis']}")
    print(f"root cause    : {result['root_cause']}")
    gate = result["gate"] or {}
    print(f"deploy gate   : {'PROMOTE' if gate.get('passed') else 'HOLD'} "
          f"(est {gate.get('estimated_performance')}, base {gate.get('baseline_performance')})")
    print(f"final state   : {result['final_state']}")
    print(f"champion ver  : {result['champion_version']}")
    print(f"audit trail   : {len(result['audit_trail'])} persisted transitions")
    for e in result["audit_trail"]:
        print(f"   {e['from_state']:>16} -> {e['to_state']:<16} {e['detail']}")


if __name__ == "__main__":
    main()
