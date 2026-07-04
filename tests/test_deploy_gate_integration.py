"""
End-to-end: the controller's deploy gate is the *real* label-free validator.

The retrain step attaches challenger predictions to the incident; the DeployGate
runs CBPE on them (no labels) and the controller acts on the verdict. A strong
challenger is promoted; a regressive one is blocked - with zero ground truth.
"""

import numpy as np

from aegis.agent.investigator import InvestigationResult
from aegis.control_plane.controller import Controller, IncidentState
from aegis.control_plane.risk_gate import RiskGate
from aegis.validation import DeployGate


class _Investigator:
    def investigate(self, incident):
        return InvestigationResult("d", "rc", "ra", 0.8, {}, "x")


class _Remediation:
    """Retrain attaches the challenger's predictions for the gate to judge."""

    def __init__(self, challenger_proba, baseline):
        self.challenger_proba = challenger_proba
        self.baseline = baseline

    def retrain(self, incident):
        incident.validation_context = {
            "challenger_proba": self.challenger_proba,
            "baseline": self.baseline,
        }
        return {"success": True, "job_id": "job-1"}

    def deploy_canary(self, incident):
        return {"success": True, "metrics": {}}

    def promote(self, incident):
        return {"success": True}

    def rollback(self, incident):
        return {"success": True}


def _controller(challenger_proba, baseline):
    c = Controller(model_id="fraud")
    c.set_dependencies(
        detectors=None,
        risk_gate=RiskGate(),
        remediation=_Remediation(challenger_proba, baseline),
        investigator=_Investigator(),
        validator=DeployGate(),  # the REAL label-free gate
    )
    return c


def test_strong_challenger_is_promoted_by_the_real_gate():
    # confident, well-separated challenger -> CBPE estimates high accuracy
    strong = np.concatenate([np.full(400, 0.96), np.full(400, 0.04)])
    c = _controller(strong, baseline=0.70)
    c.handle_drift_detected({"drift_type": "concept", "severity": "medium"})
    assert c.current_state is IncidentState.HEALTHY   # loop closed
    assert c.current_incident is None


def test_regressive_challenger_is_blocked_by_the_real_gate():
    # near-coin-flip challenger, strong champion baseline -> gate must HOLD
    weak = np.random.RandomState(0).uniform(0.45, 0.55, 600)
    c = _controller(weak, baseline=0.90)
    c.handle_drift_detected({"drift_type": "concept", "severity": "medium"})
    assert c.current_state is IncidentState.ROLLED_BACK  # champion kept
    # incident stays open (not resolved) because nothing was promoted
    assert c.current_incident is not None


def test_gate_fails_closed_without_challenger_evidence():
    gate = DeployGate()

    class _Incident:
        validation_context = None

    verdict = gate.validate(_Incident())
    assert verdict["passed"] is False
