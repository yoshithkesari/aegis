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
from .control_plane.risk_gate import RiskGate
from .reasoning.local import HeuristicReasoner
from .stores import IncidentStore, ModelRegistry
from .validation import DeployGate

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

    controller = Controller(model_id=MODEL_ID)
    controller.set_dependencies(
        detectors=None,
        risk_gate=RiskGate(),
        remediation=remediation,
        investigator=Investigator(reasoner=HeuristicReasoner()),  # offline
        validator=DeployGate(),                                   # real CBPE gate
        incident_store=incident_store,
    )
    return System(controller, registry, incident_store, remediation)


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


def main():  # pragma: no cover - CLI entrypoint
    system = build_system()
    result = run_incident(system, severity="medium")
    print("=" * 64)
    print(f"AEGIS incident {result['incident_id']}")
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
